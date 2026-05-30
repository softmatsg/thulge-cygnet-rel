# Copyright (c) 2026 Nikodem Tomczak, Thulge Labs. All rights reserved.

"""Mirror-execute validator backend.

Runs the query inside a rolled-back transaction against the mirror
Neo4j. Closes a real gap in the EXPLAIN backend's coverage that
the prompt audit surfaced: Neo4j's planner is permissive on
several classes of error that the runtime catches. Those classes
become catchable here without losing the
"no-execution-against-production" invariant the gate provides.

Specifically, this backend catches errors EXPLAIN does **not**:

- **Unknown procedure / function calls.** ``CALL apoc.foo.bar()``
  plans fine; the runtime raises ``ProcedureNotFound`` (or
  ``ProcedureNotRegistered``). Mapped to :class:`SchemaError`
  with the missing procedure name surfaced as the unknown
  reference.
- **Missing parameter bindings.** ``WHERE n.id = $missing``
  with no parameter supplied plans fine; the runtime raises
  ``ParameterMissing``. Mapped to :class:`PropertyError` with
  the parameter name surfaced.
- **Type-coercion errors.** Some implicit coercions (string
  arithmetic, type-incompatible WHERE comparisons) only fire at
  runtime. Mapped to :class:`PropertyError` with
  ``declared_type`` / ``used_type`` extracted from the Neo4j
  error text when possible.
- **Semantic errors** that EXPLAIN's plan-time check let through
  but the runtime rejects (e.g. variable scoping after ``WITH``).
  Mapped to :class:`SchemaError` or :class:`PropertyError`
  depending on the runtime message; we surface what we can
  extract and fall back to a generic :class:`ParseError` for the
  ones we can't categorise.

Transaction lifecycle
---------------------

Every ``validate(query)`` call:

1. Opens an explicit transaction on the configured mirror
   ``session``.
2. Runs the query with the configured ``timeout_seconds`` enforced
   server-side via :class:`neo4j.Query`.
3. Consumes the result, which forces full planning + execution
   (the gate's runtime-error catchment depends on consumption).
4. **Always** rolls back in a ``finally`` block — writes inside
   the query touch the mirror's transaction state but never
   commit. The rollback path is idempotent and safe even after a
   prior ``rollback`` call.

Write semantics caveat
----------------------

A query that contains an explicit ``COMMIT`` clause (Neo4j 5
allows ``CALL { ... } IN TRANSACTIONS`` sub-transactions that
commit independently of the outer transaction) defeats the
rollback protection. The validator does not pre-scan for this
case — it isn't realistic in agent-generated Cypher and a real
detection would need a Cypher parser. Documented here so future
readers know the limitation.

Why the mirror, not production
------------------------------

Even with rollback, execution-against-production-driver carries
risk: long-running queries, replication lag, query-log pollution,
read-vs-write resource accounting against your production graph.
The mirror is the safe target. The mirror's one-node-per-label
shape is enough for runtime error detection — the errors this
backend catches don't depend on real data cardinality.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Final

from neo4j.exceptions import ClientError, CypherSyntaxError

from cygnet._format import excerpt_with_caret
from cygnet.models import (
    ParseError,
    PropertyError,
    SchemaError,
    StructuralValidatorResult,
)

if TYPE_CHECKING:
    from neo4j import Driver

    from cygnet.models import Schema

__all__ = ["MirrorExecuteValidator"]


# ---------------------------------------------------------------------------
# Neo4j error-code substrings the backend recognises.
#
# Substring match rather than exact equality — Neo4j 5.x reuses the
# same .Statement.* / .Procedure.* / .Schema.* code families across
# minor versions but occasionally changes the trailing segment. Pin
# stable prefixes and document each one's surface.
# ---------------------------------------------------------------------------

_CODE_PROCEDURE_NOT_FOUND: Final[tuple[str, ...]] = (
    "ProcedureNotFound",
    "ProcedureRegistrationFailed",
    "ProcedureNotRegistered",
)
"""Procedure / function not registered. Surfaces as
``Neo.ClientError.Procedure.ProcedureNotFound`` on Neo4j 5.x; older
shapes used ``ProcedureNotRegistered``. Both map to a SchemaError —
the user referenced a procedure name not in the runtime's procedure
registry."""

_CODE_PARAMETER_MISSING: Final[tuple[str, ...]] = ("ParameterMissing",)
"""``Neo.ClientError.Statement.ParameterMissing`` — query references
``$param`` but the parameter set didn't include ``param``. Maps to
PropertyError (the missing parameter is functionally a missing
property binding from the LLM's perspective)."""

_CODE_TYPE_ERROR: Final[tuple[str, ...]] = (
    "TypeError",
    "ArgumentError",
)
"""``Neo.ClientError.Statement.TypeError`` and related. Implicit
coercion fails (``"5" + 1`` etc.). Maps to PropertyError when the
declared and used types can be extracted from the message."""

_CODE_SEMANTIC_ERROR: Final[tuple[str, ...]] = (
    "SemanticError",
    "InvalidSemantics",
)
"""``Neo.ClientError.Statement.SemanticError`` and ``InvalidSemantics``.
A broad bucket of "valid syntax, invalid semantics" failures (variable
out of scope, unknown function, malformed pattern, etc.). The mapping
inspects the message to pick property vs schema vs parse routing."""


# ---------------------------------------------------------------------------
# Message regexes for extracting structured info from Neo4j error text.
# ---------------------------------------------------------------------------

_BACKTICK_OR_QUOTED: Final[re.Pattern[str]] = re.compile(r"[`']([A-Za-z_][A-Za-z0-9_]*)[`']")
"""Capture the first backticked-or-single-quoted identifier in an
error message. Neo4j surrounds offending identifiers with backticks
in most modern messages and single quotes in some older ones."""

_TYPE_MISMATCH: Final[re.Pattern[str]] = re.compile(
    r"(?:expected|requires?|expects?)\s+([A-Z][a-z]+)"
    r"(?:[\s,]+(?:but\s+got|but\s+was|got|received|was|,\s*found)\s+([A-Z][a-z]+))?"
)
"""Best-effort extraction of declared-vs-used types from a type-error
message. Neo4j formats these as "expected X but got Y" or "requires X
but received Y"; the variation across versions is the reason for the
permissive non-capturing groups. Case sensitivity matters here:
Neo4j type names are always title-cased (``Integer``, ``String``,
``Float``, ``Boolean``, ``List``, ``Map``, ``Date``, ``DateTime``,
``Duration``, ``Point``). ``re.IGNORECASE`` would let the type
capture pick up lowercase glue tokens like ``got`` between the two
type names, which it must not."""

_PARAMETER_NAME: Final[re.Pattern[str]] = re.compile(
    r"parameter\(?s?\)?[:\s]*['\"`]?([A-Za-z_][A-Za-z0-9_]*)",
    re.IGNORECASE,
)
r"""Extract a parameter name from ``ParameterMissing`` messages.

Handles three observed shapes:

- ``Expected parameter(s): missing_param`` (Neo4j 5.x default).
- ``Missing parameter `missing_param``` (older formats; backticks
  delimit the identifier).
- ``Missing parameter missing_param`` (raw identifier).

The optional ``\(?s?\)?`` skips the parenthetical ``(s)`` Neo4j
prepends in its plural form; the optional quote/backtick wrapper
strips the identifier delimiter when present."""


class MirrorExecuteValidator:
    """Validator that runs the query inside a rolled-back transaction.

    Args:
        driver: a connected ``neo4j.Driver`` against the mirror
            instance. The validator does not own the driver;
            :meth:`cygnet.Gate.close` is responsible for lifecycle.
        schema: the loaded :class:`cygnet.models.Schema`. Used for
            ``available_in_scope`` enrichment on the errors that
            do reference schema entities; not consulted otherwise.
        database: Neo4j database name. Defaults to ``"neo4j"``.
        timeout_seconds: per-query timeout enforced server-side via
            :class:`neo4j.Query`. Default 5 s; matches the EXPLAIN
            backend.
    """

    def __init__(
        self,
        driver: Driver,
        schema: Schema,
        *,
        database: str = "neo4j",
        timeout_seconds: float = 5.0,
    ) -> None:
        self._driver = driver
        self._schema = schema
        self._database = database
        self._timeout_seconds = timeout_seconds

    def validate(self, query: str) -> StructuralValidatorResult:
        try:
            self._run(query)
        except CypherSyntaxError as exc:
            return _fail("parse", _parse_error_from_syntax(exc, query))
        except ClientError as exc:
            return self._classify_client_error(exc, query)
        # Empty results, populated results — both pass: the query
        # planned, executed, and produced valid output (even if
        # zero rows). Mirror data is sparse by design; absence of
        # rows is not a validation failure.
        return StructuralValidatorResult(passed=True, failed_stage="none")

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def _run(self, query: str) -> None:
        """Open a transaction, run the query, consume the result,
        then roll back. The rollback runs even when the query raises;
        the ``finally`` block is the safety net.

        Returns nothing — the validator's contract is "did the
        runtime accept this query", not "what did it return". We
        consume the result to force full execution; the rows are
        discarded.
        """
        # ``session`` is a context manager. ``begin_transaction``
        # opens an explicit transaction on the session; we use
        # explicit-mode rather than ``execute_read`` / ``execute_write``
        # because the latter auto-commit on success and we always
        # want to roll back. The timeout goes on the transaction
        # rather than the query — Neo4j's driver enforces that
        # ``Transaction.run`` accepts plain strings only and reserves
        # the :class:`neo4j.Query` wrapper for ``Session.run``
        # auto-commit calls.
        with self._driver.session(database=self._database) as session:
            tx = session.begin_transaction(timeout=self._timeout_seconds)
            try:
                result = tx.run(query)
                # Consume forces full planning + execution. Discarding
                # the rows is intentional.
                result.consume()
            finally:
                # ``rollback()`` is idempotent and safe to call even
                # after the transaction has been closed by an
                # exception path. The library raises on connection-
                # level failure (e.g. broken Bolt session) — we let
                # that propagate.
                tx.rollback()

    # ------------------------------------------------------------------
    # ClientError → typed error mapping
    # ------------------------------------------------------------------

    def _classify_client_error(
        self,
        exc: ClientError,
        query: str,
    ) -> StructuralValidatorResult:
        """Route a ``ClientError`` to one of ``parse`` / ``schema`` /
        ``property`` based on the error's code + message content.

        The code-string substring match is intentional — Neo4j 5.x
        re-uses the same code families (``Neo.ClientError.Procedure.*``,
        ``Neo.ClientError.Statement.*``) but the trailing segment
        sometimes shifts across minor versions. Each mapping
        documents its routing.
        """
        code = str(getattr(exc, "code", "") or "")
        message = str(getattr(exc, "message", None) or exc)

        if _any_substring_in(code, _CODE_PROCEDURE_NOT_FOUND):
            return _fail("schema", _schema_error_from_procedure_missing(message, query))

        if _any_substring_in(code, _CODE_PARAMETER_MISSING):
            return _fail("property", _property_error_from_parameter_missing(message, query))

        if _any_substring_in(code, _CODE_TYPE_ERROR):
            return _fail("property", _property_error_from_type_error(message, query))

        if _any_substring_in(code, _CODE_SEMANTIC_ERROR):
            return _fail("schema", _schema_error_from_semantic(message, query))

        # Unknown ClientError shape. Better to surface than to
        # silently pass — same defensive posture as the EXPLAIN
        # backend's generic-ClientError handler.
        return _fail(
            "parse",
            ParseError(
                message=f"{code or 'Neo.ClientError.Unknown'}: {message}",
                line=1,
                column=1,
                snippet=query[:80],
                excerpt_with_caret=excerpt_with_caret(query, 1, 1),
            ),
        )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _fail(stage: str, payload: object) -> StructuralValidatorResult:
    return StructuralValidatorResult(
        passed=False,
        failed_stage=stage,  # type: ignore[arg-type]
        error_payload=payload,  # type: ignore[arg-type]
    )


def _any_substring_in(haystack: str, needles: tuple[str, ...]) -> bool:
    return any(needle in haystack for needle in needles)


def _parse_error_from_syntax(exc: CypherSyntaxError, query: str) -> ParseError:
    """Map a ``CypherSyntaxError`` to ``ParseError``. Same shape as
    the EXPLAIN backend's version — Neo4j produces the same error
    structure whether ``EXPLAIN`` or actual execution surfaces the
    parse failure, so we re-use the line/column extraction logic
    via the ``explain_backend`` helpers.
    """
    from cygnet.validator.explain_backend import _line_col_from_message, _snippet_at

    raw = str(getattr(exc, "message", None) or exc)
    line, col_zero = _line_col_from_message(raw)
    col = max(1, col_zero + 1) if col_zero is not None else 1
    line = line or 1
    return ParseError(
        message=raw,
        line=line,
        column=col,
        snippet=_snippet_at(query, line, col_zero or 0),
        excerpt_with_caret=excerpt_with_caret(query, line, col),
    )


def _schema_error_from_procedure_missing(message: str, query: str) -> SchemaError:
    """Extract the missing procedure name from a ProcedureNotFound
    message and wrap as a SchemaError.

    Procedures aren't in :class:`cygnet.models.Schema`, so
    ``available_in_scope`` stays empty — the LLM can't pick from a
    list we don't have. ``reference_kind`` is set to ``"property"``
    as a pragmatic compromise: procedures aren't labels or
    relationships, and the property bucket is the most plausible
    routing for refinement-prompt purposes (the user supplied a
    name; the schema doesn't have it).
    """
    name_match = _BACKTICK_OR_QUOTED.search(message)
    unknown = name_match.group(1) if name_match else "<unknown procedure>"
    return SchemaError(
        unknown_reference=unknown,
        reference_kind="property",
        did_you_mean=[],
        query_context=_query_context_around(query, unknown),
        available_in_scope=[],
        available_in_scope_truncated=False,
    )


def _property_error_from_parameter_missing(message: str, query: str) -> PropertyError:
    """Map ``ParameterMissing`` to PropertyError.

    The parameter is a binding the LLM forgot to supply; surface
    its name as ``property_name``. Types are unknown at this stage
    — Neo4j just says "missing". ``declared_type`` and ``used_type``
    fall back to ``"any"`` / ``"missing"`` so the typed payload's
    contract holds.
    """
    name_match = _PARAMETER_NAME.search(message)
    if name_match is None:
        # Fall back to the backticked name (older messages format
        # parameter names that way).
        name_match = _BACKTICK_OR_QUOTED.search(message)
    name = name_match.group(1) if name_match else "<unknown parameter>"
    return PropertyError(
        property_name=name,
        declared_type="any",
        used_type="missing",
        query_context=_query_context_around(query, name),
        did_you_mean=[],
    )


def _property_error_from_type_error(message: str, query: str) -> PropertyError:
    """Map ``TypeError`` to PropertyError; best-effort extraction of
    declared and used types from the message.

    Neo4j 5.x type-error messages take many shapes. When we can
    extract ``expected X but got Y``, populate both fields. When we
    can't, fall back to ``"any"`` / ``"unknown"`` so the typed
    payload is still well-formed — the caller can still surface the
    raw message via ``query_context``.
    """
    type_match = _TYPE_MISMATCH.search(message)
    declared = type_match.group(1).upper() if type_match else "any"
    used = type_match.group(2).upper() if type_match and type_match.group(2) else "unknown"
    name_match = _BACKTICK_OR_QUOTED.search(message)
    prop_name = name_match.group(1) if name_match else "<expression>"
    return PropertyError(
        property_name=prop_name,
        declared_type=declared,
        used_type=used,
        query_context=message[:120],
        did_you_mean=[],
    )


def _schema_error_from_semantic(message: str, query: str) -> SchemaError:
    """Map a ``SemanticError`` to SchemaError.

    SemanticErrors are the catch-all bucket for "syntactically
    valid, semantically wrong" — variable not in scope after
    ``WITH``, unknown function inside an expression, malformed
    pattern, etc. We surface the offending identifier (when present
    in backticks) as ``unknown_reference`` and use ``"property"``
    as the reference kind, the most useful routing for refinement.
    """
    name_match = _BACKTICK_OR_QUOTED.search(message)
    unknown = name_match.group(1) if name_match else "<semantic error>"
    return SchemaError(
        unknown_reference=unknown,
        reference_kind="property",
        did_you_mean=[],
        query_context=message[:120],
        available_in_scope=[],
        available_in_scope_truncated=False,
    )


def _query_context_around(query: str, identifier: str) -> str:
    """Return a short query fragment around the offending identifier
    when locatable; otherwise a head excerpt. Keeps the
    PropertyError / SchemaError ``query_context`` field meaningful
    without forcing the caller to walk the message themselves."""
    idx = query.find(identifier)
    if idx < 0:
        return query[:80]
    start = max(0, idx - 20)
    end = min(len(query), idx + len(identifier) + 20)
    return query[start:end]
