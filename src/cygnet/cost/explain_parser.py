# Copyright (c) 2026 Nikodem Tomczak, Thulge Labs. All rights reserved.

"""Parse Neo4j EXPLAIN plan dicts into the public ``ExplainPlan`` model.

The Neo4j 5 driver exposes the plan as ``summary.plan``, a nested
``dict`` with the shape::

    {
        "operatorType": "AllNodesScan@neo4j",
        "identifiers": ["n"],
        "args": {"Details": "n", "EstimatedRows": 130.0, "Id": 1},
        "children": [...],
    }

(verified empirically against Neo4j 5.26 — Neo4j 4.x uses a similar
shape with minor key spelling differences which we don't target).

This module flattens that tree into a list of :class:`ExplainOperator`
in depth-first post-order so the deepest data-source operator appears
first. The cost-driver heuristic in ``cost.gate`` walks this list to
attribute the rejection to the most actionable operator.

EXPLAIN does not provide real per-operator db-hit estimates; only
``EstimatedRows`` is populated. We mirror ``estimated_rows`` into
``estimated_dbhits`` so the model field is non-empty and downstream
threshold checks still have something to compare against. PROFILE would
give real db-hits but actually executes the query, defeating
pre-execution gating.
"""

from __future__ import annotations

from typing import Any

from cygnet.models import ExplainOperator, ExplainPlan

__all__ = ["parse_explain_plan"]


_OPERATOR_SUFFIX = "@neo4j"


def parse_explain_plan(plan: dict[str, Any] | str) -> ExplainPlan:
    """Build an :class:`ExplainPlan` from a Neo4j driver plan.

    Accepts the nested ``dict`` returned by
    ``summary.plan`` on a ``EXPLAIN <query>`` consume, or a raw text
    representation as a fallback. The string fallback returns an empty
    plan — text parsing is intentionally not implemented because the
    dict form is what every supported driver version returns.
    """
    if isinstance(plan, str):
        # Fallback: we don't parse the textual plan. Returning an
        # empty plan lets callers decide how to handle this case
        # without an exception.
        return ExplainPlan(
            operators=[],
            total_estimated_rows=0,
            total_estimated_dbhits=0,
        )

    operators: list[ExplainOperator] = []
    _walk(plan, operators)

    # Total row estimate: the root operator's value (top of the plan
    # tree, last in post-order). Falls back to 0 if the plan is empty.
    total_rows = operators[-1].estimated_rows if operators else 0
    total_dbhits = sum(op.estimated_rows for op in operators)
    return ExplainPlan(
        operators=operators,
        total_estimated_rows=total_rows,
        total_estimated_dbhits=total_dbhits,
    )


def _walk(node: dict[str, Any], out: list[ExplainOperator]) -> None:
    """Depth-first post-order traversal of the plan tree."""
    children = node.get("children", []) or []
    for child in children:
        if isinstance(child, dict):
            _walk(child, out)
    out.append(_to_operator(node))


def _to_operator(node: dict[str, Any]) -> ExplainOperator:
    op_type = str(node.get("operatorType", "Unknown"))
    name = op_type[: -len(_OPERATOR_SUFFIX)] if op_type.endswith(_OPERATOR_SUFFIX) else op_type
    args_raw = node.get("args") or node.get("arguments") or {}
    if not isinstance(args_raw, dict):
        args_raw = {}
    rows_f = _as_float(args_raw.get("EstimatedRows", 0.0))
    rows_int = max(0, round(rows_f))
    return ExplainOperator(
        name=name,
        estimated_rows=rows_int,
        estimated_dbhits=rows_int,
        identifiers=[str(i) for i in (node.get("identifiers") or [])],
        arguments={str(k): str(v) for k, v in args_raw.items()},
    )


def _as_float(value: Any) -> float:
    """Coerce a plan-argument value to float, tolerating strings."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
