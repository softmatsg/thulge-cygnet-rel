# Copyright (c) 2026 Nikodem Tomczak, Thulge Labs. All rights reserved.

"""Cost gate: pre-execution cost estimation via Neo4j EXPLAIN.

The cost gate is the headline novel-claim contribution. Given a
structurally valid Cypher query, it runs ``EXPLAIN`` against the
configured Neo4j (typically the mirror), parses the plan, identifies
the most actionable operator above threshold, and returns a structured
``CostGateResult`` with row/db-hit estimates and one or more mitigation
suggestions tailored to the cost driver.

The gate is *pre*-execution: it does not run the query. PROFILE could
give real db-hits but defeats the purpose, so we use EXPLAIN and
proxy db-hits via summed operator row estimates (see
``cygnet.cost.explain_parser``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from neo4j import Query
from neo4j.exceptions import ClientError, CypherSyntaxError

from cygnet.cost.explain_parser import parse_explain_plan
from cygnet.models import CostGateResult, ExplainOperator, OperatorCost

if TYPE_CHECKING:
    from neo4j import Driver

    from cygnet.models import ExplainPlan

__all__ = ["CostGate"]


# ---------------------------------------------------------------------------
# Cost-driver vocabulary
#
# Strings here are part of the immutable artifact surface for this
# slice: they appear in CostGateResult.cost_driver and in the
# suggested_mitigations list. Treat as user-facing copy.
# ---------------------------------------------------------------------------

_EXPLAIN_FAILED: Final[str] = "explain_failed"
_DEFAULT_MITIGATION: Final[str] = "Consider adding `LIMIT` to bound the result set."

_MITIGATIONS: Final[dict[str, str]] = {
    "AllNodesScan": "Add a label filter to narrow the scan.",
    "NodeByLabelScan": (
        "Consider adding an index on the filtered property or restricting via WHERE."
    ),
    "CartesianProduct": ("Avoid disconnected patterns; add a relationship between the variables."),
}

_EXPAND_MITIGATION: Final[str] = (
    "Add a relationship type filter or reduce the path expansion depth."
)
_VARLEN_MITIGATION: Final[str] = "Constrain the path depth with `*..N`."


# Cost-driver priority ordering. Lower index = more user-actionable.
# When a query's plan contains several operators above the per-operator
# cutoff, the gate prefers the one whose category produces the most
# concrete mitigation advice.
#
# The brief's "deepest wins on ties" rule still applies — within a
# single priority bucket, the deeper operator wins — but bucket
# precedence trumps depth because Neo4j's planner propagates inflated
# row estimates DOWN through scan operators in cartesian/expand cases.
# Without this, a CartesianProduct that triples row counts would get
# attributed to its child label scan with a useless "add an index"
# mitigation.
_OPERATOR_PRIORITY: Final[list[str]] = [
    "CartesianProduct",
    "VarLengthExpand",
    "AllNodesScan",
    "Expand",
    "NodeByLabelScan",
]


def _operator_priority(name: str) -> int:
    for i, prefix in enumerate(_OPERATOR_PRIORITY):
        if name == prefix or name.startswith(prefix):
            return i
    return len(_OPERATOR_PRIORITY)


class CostGate:
    """Pre-execution cost gate built on Neo4j EXPLAIN.

    Args:
        driver: an already-connected ``neo4j.Driver``. The gate does
            not own the driver; ``Gate.close()`` is responsible for
            the lifecycle.
        threshold_rows: queries whose plan estimates more than this
            many rows for any single operator are rejected. The same
            value is the per-operator cutoff for cost-driver
            attribution (default heuristic: an operator contributes
            to the rejection if its row estimate is at least half
            ``threshold_rows``).
        threshold_dbhits: queries whose summed-operator row estimate
            (the EXPLAIN-time db-hit proxy) exceeds this value are
            rejected. EXPLAIN does not surface real db-hits;
            see :mod:`cygnet.cost.explain_parser` for the rationale.
        database: Neo4j database name. Defaults to ``"neo4j"``.
        timeout_seconds: per-query timeout passed to Neo4j via
            ``neo4j.Query(timeout=...)``. EXPLAIN typically returns
            in tens of milliseconds; 5 seconds is generous.
    """

    def __init__(
        self,
        driver: Driver,
        *,
        threshold_rows: int = 100_000,
        threshold_dbhits: int = 1_000_000,
        database: str = "neo4j",
        timeout_seconds: float = 5.0,
    ) -> None:
        self._driver = driver
        self._threshold_rows = threshold_rows
        self._threshold_dbhits = threshold_dbhits
        self._database = database
        self._timeout_seconds = timeout_seconds

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def estimate(self, query: str) -> CostGateResult:
        try:
            plan_dict = self._run_explain(query)
        except (CypherSyntaxError, ClientError):
            return CostGateResult(
                passed=False,
                estimated_rows=0,
                estimated_dbhits=0,
                threshold_used=self._threshold_rows,
                cost_driver=_EXPLAIN_FAILED,
                suggested_mitigations=[
                    "Neo4j refused the EXPLAIN call. The cost gate's "
                    "precondition is a structurally valid query; the "
                    "structural validator chain should have caught this "
                    "first. Re-run the query through the gate, or "
                    "validate via `Gate.validate(...)` to see the "
                    "underlying parse / schema error."
                ],
            )

        plan = parse_explain_plan(plan_dict)
        driver_op = self._select_cost_driver(plan)
        breakdown = _top_cost_breakdown(plan.operators)

        rows = plan.total_estimated_rows
        dbhits = plan.total_estimated_dbhits

        if rows > self._threshold_rows or dbhits > self._threshold_dbhits:
            cost_driver_label, mitigations = self._reject_driver(driver_op)
            return CostGateResult(
                passed=False,
                estimated_rows=rows,
                estimated_dbhits=dbhits,
                threshold_used=self._threshold_rows,
                cost_driver=cost_driver_label,
                suggested_mitigations=mitigations,
                estimated_cost_breakdown=breakdown,
            )

        return CostGateResult(
            passed=True,
            estimated_rows=rows,
            estimated_dbhits=dbhits,
            threshold_used=self._threshold_rows,
            cost_driver=None,
            suggested_mitigations=[],
            estimated_cost_breakdown=breakdown,
        )

    # ------------------------------------------------------------------
    # EXPLAIN execution
    # ------------------------------------------------------------------

    def _run_explain(self, query: str) -> dict[str, object]:
        with self._driver.session(database=self._database) as session:
            result = session.run(Query("EXPLAIN " + query, timeout=self._timeout_seconds))
            summary = result.consume()
        plan = getattr(summary, "plan", None)
        if plan is None:
            return {}
        return _plan_to_dict(plan)

    # ------------------------------------------------------------------
    # Cost-driver attribution
    # ------------------------------------------------------------------

    def _select_cost_driver(self, plan: ExplainPlan) -> ExplainOperator | None:
        """Pick the operator most responsible for the rejection.

        Three-key sort:
        1. ``_operator_priority`` (lower is more actionable — Cartesian
           and variable-length expansion win over label scans).
        2. ``-estimated_rows`` (higher is preferred — the big consumer
           is the rejection's true cause).
        3. ``index`` in post-order (smaller is deeper — same depth tie-
           break as the original "deepest wins" rule, but now applies
           only within a priority bucket).

        An operator contributes when its row estimate is at least half
        the threshold; if no operator clears that bar, returns ``None``
        and the gate emits the default "add LIMIT" mitigation.
        """
        cutoff = max(1, self._threshold_rows // 2)
        candidates = [op for op in plan.operators if op.estimated_rows >= cutoff]
        if not candidates:
            return None
        indexed = {id(op): i for i, op in enumerate(plan.operators)}
        return min(
            candidates,
            key=lambda op: (
                _operator_priority(op.name),
                -op.estimated_rows,
                indexed[id(op)],
            ),
        )

    def _reject_driver(self, driver_op: ExplainOperator | None) -> tuple[str, list[str]]:
        if driver_op is None:
            # Plan didn't surface a clear actionable operator (e.g.
            # estimates spread thin across many small operators).
            return ("aggregate", [_DEFAULT_MITIGATION])
        label = self._format_driver_label(driver_op)
        return (label, self._mitigations_for(driver_op))

    @staticmethod
    def _format_driver_label(op: ExplainOperator) -> str:
        if op.identifiers:
            return f"{op.name} on [{', '.join(op.identifiers)}]"
        return op.name

    @staticmethod
    def _mitigations_for(op: ExplainOperator) -> list[str]:
        name = op.name
        if name in _MITIGATIONS:
            return [_MITIGATIONS[name]]
        # Pattern matches that need substring checks.
        if name.startswith("VarLengthExpand"):
            return [_VARLEN_MITIGATION]
        if name.startswith("Expand") and "All" in name:
            return [_EXPAND_MITIGATION]
        return [_DEFAULT_MITIGATION]


# ---------------------------------------------------------------------------
# Plan-object → dict normaliser
#
# The Neo4j driver's plan accessor sometimes returns a ``Plan`` namedtuple
# and sometimes a plain dict, depending on the driver version. Normalise
# to the dict shape `parse_explain_plan` consumes.
# ---------------------------------------------------------------------------


_COST_BREAKDOWN_TOP_N: Final[int] = 5
"""Cap on operators surfaced in ``CostGateResult.estimated_cost_breakdown``.
Treat as part of the public contract: callers may rely on
``len(breakdown) <= 5``. Configurable raise is deliberately not exposed."""


def _top_cost_breakdown(operators: list[ExplainOperator]) -> list[OperatorCost]:
    """Pick the top-N operators by ``estimated_rows`` for the breakdown.

    Plan-order ties broken by post-order index (deeper first), matching
    the cost-driver heuristic's tie-break rule.
    """
    if not operators:
        return []
    indexed = list(enumerate(operators))
    ranked = sorted(indexed, key=lambda pair: (-pair[1].estimated_rows, pair[0]))
    top = [op for _, op in ranked[:_COST_BREAKDOWN_TOP_N]]
    return [
        OperatorCost(
            operator=op.name,
            identifiers=list(op.identifiers),
            estimated_rows=op.estimated_rows,
            estimated_dbhits=op.estimated_dbhits,
        )
        for op in top
    ]


def _plan_to_dict(plan: object) -> dict[str, object]:
    if isinstance(plan, dict):
        return plan
    # Namedtuple / attribute-style plan: copy known fields.
    return {
        "operatorType": getattr(plan, "operator_type", "Unknown"),
        "identifiers": list(getattr(plan, "identifiers", []) or []),
        "args": dict(getattr(plan, "arguments", {}) or {}),
        "children": [_plan_to_dict(child) for child in (getattr(plan, "children", []) or [])],
    }
