# Copyright (c) 2026 Nikodem Tomczak, Thulge Labs. All rights reserved.

"""EXPLAIN-based validator backend.

Sends ``EXPLAIN <query>`` to Neo4j. The driver returns a result summary;
schema-reference issues surface as GQL status notifications (codes
``01N50``/``01N51``/``01N52`` for label/relationship/property), parse
errors surface as ``CypherSyntaxError`` exceptions.

**Capability scope** (discovered empirically against Neo4j 5 Community
during the slice's probe phase; recorded here so the gaps stay
documented):

- **Parse:** authoritative. Neo4j's parser is the canonical Cypher
  interpretation; this backend's ``ParseError`` is the strongest in
  the chain.
- **Schema reference:** authoritative for labels, relationship types,
  property keys. Notifications carry structured positions and the
  unknown reference in backticks.
- **Property type-mismatch:** NOT detected. Neo4j's planner is
  permissive on types; no notification surfaces a string-vs-int
  literal mismatch. The AST backend handles this case statically.
- **Constraint violations:** NOT detected. EXPLAIN does not execute,
  so a ``CREATE`` that would violate uniqueness plans fine. PROFILE
  could detect it but defeats pre-execution gating (it would actually
  create the row). The AST backend handles existence-constraint
  violations statically; uniqueness violations need runtime data
  that no static backend can carry.

The brief's "constraint violation produces ConstraintError" test case
is therefore not achievable via EXPLAIN and is documented in
``tests/integration/test_validator_explain.py``.
"""

from __future__ import annotations

import difflib
import re
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Final

from neo4j import Query
from neo4j.exceptions import ClientError, CypherSyntaxError

from cygnet._format import bound_available_in_scope, excerpt_with_caret
from cygnet.models import (
    ParseError,
    Schema,
    SchemaError,
    StructuralValidatorResult,
)

if TYPE_CHECKING:
    from neo4j import Driver

__all__ = ["ExplainValidator"]


# ---------------------------------------------------------------------------
# GQL status codes for schema-reference notifications.
#
# Neo4j 5 surfaces unknown labels/rels/properties as severity=WARNING
# notifications with these stable GQL status codes. Codes are part of
# Neo4j's public contract; message text is not.
# ---------------------------------------------------------------------------

_GQL_CODE_TO_KIND: Final[dict[str, str]] = {
    "01N50": "label",
    "01N51": "relationship",
    "01N52": "property",
}

_BACKTICK_REFERENCE_PATTERN: Final[re.Pattern[str]] = re.compile(r"`([^`]+)`")
"""Notification descriptions look like
``"The label `Ghost` does not exist. ..."`` — the backticked token is
the unknown reference name."""

_SYNTAX_LINE_COL_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\(line\s+(\d+),\s+column\s+(\d+)\s*\(offset:\s*(\d+)\)\)"
)
"""``CypherSyntaxError`` messages include a position trailer like
``(line 1, column 25 (offset: 24))``. Driver-level structured position
attributes aren't exposed by the exception, so we parse the trailer."""


class ExplainValidator:
    """Run ``EXPLAIN <query>`` and map the response to the error vocabulary.

    Args:
        schema: the loaded ``Schema``, used for ``did_you_mean``
            candidates when a schema-reference notification fires.
        driver: an already-constructed ``neo4j.Driver``. The validator
            does not own the driver; ``Gate.close()`` is responsible
            for the lifecycle.
        database: Neo4j database name. Defaults to ``"neo4j"`` (the
            default community database).
        timeout_seconds: per-query timeout passed to Neo4j via the
            driver's ``Query(timeout=)`` mechanism. EXPLAIN typically
            returns in tens of milliseconds; 5 seconds is generous.
        did_you_mean_max_suggestions: cap on suggestion list length.
        did_you_mean_cutoff: ``difflib`` similarity threshold; matches
            the builtin and AST backends' defaults.
    """

    def __init__(
        self,
        schema: Schema,
        driver: Driver,
        *,
        database: str = "neo4j",
        timeout_seconds: float = 5.0,
        did_you_mean_max_suggestions: int = 5,
        did_you_mean_cutoff: float = 0.6,
    ) -> None:
        self._schema = schema
        self._driver = driver
        self._database = database
        self._timeout_seconds = timeout_seconds
        self._max_suggestions = did_you_mean_max_suggestions
        self._cutoff = did_you_mean_cutoff
        self._label_names: set[str] = {nl.name for nl in schema.labels}
        self._rel_names: set[str] = {rt.name for rt in schema.relationship_types}

    def validate(self, query: str) -> StructuralValidatorResult:
        # 1. Parse stage via the driver's exception surface.
        try:
            summary = self._run_explain(query)
        except CypherSyntaxError as exc:
            return _fail("parse", _parse_error_from_syntax(exc, query))
        except ClientError as exc:
            return _fail("parse", _parse_error_from_client(exc, query))

        # 2. Schema-reference stage via GQL notifications.
        schema_err = self._schema_error_from_notifications(summary, query)
        if schema_err is not None:
            return _fail("schema", schema_err)

        return StructuralValidatorResult(passed=True, failed_stage="none")

    # ------------------------------------------------------------------
    # EXPLAIN execution
    # ------------------------------------------------------------------

    def _run_explain(self, query: str) -> Any:
        """Send ``EXPLAIN <query>`` and return the result summary.

        Uses ``neo4j.Query(query, timeout=...)`` so the timeout is
        enforced server-side rather than via Python wall-clock.
        """
        prefixed = "EXPLAIN " + query
        with self._driver.session(database=self._database) as session:
            result = session.run(Query(prefixed, timeout=self._timeout_seconds))
            return result.consume()

    # ------------------------------------------------------------------
    # Notification → SchemaError mapping
    # ------------------------------------------------------------------

    def _schema_error_from_notifications(self, summary: Any, query: str) -> SchemaError | None:
        """Inspect ``summary.gql_status_objects`` for schema-reference
        warnings; return the first match as a ``SchemaError``."""
        gql_objects = getattr(summary, "gql_status_objects", None) or ()
        for obj in gql_objects:
            gql_status = getattr(obj, "gql_status", None)
            if gql_status not in _GQL_CODE_TO_KIND:
                continue
            kind = _GQL_CODE_TO_KIND[gql_status]
            description = getattr(obj, "status_description", "") or ""
            match = _BACKTICK_REFERENCE_PATTERN.search(description)
            unknown = match.group(1) if match else "<unknown>"
            position = getattr(obj, "position", None)
            line, col = _position_to_line_col(position)
            vocab_list = list(self._vocabulary_for(kind))
            in_scope, truncated = bound_available_in_scope(vocab_list)
            return SchemaError(
                unknown_reference=unknown,
                reference_kind=kind,  # type: ignore[arg-type]
                did_you_mean=list(
                    difflib.get_close_matches(
                        unknown, vocab_list, n=self._max_suggestions, cutoff=self._cutoff
                    )
                ),
                query_context=_snippet_at(query, line, col),
                available_in_scope=in_scope,
                available_in_scope_truncated=truncated,
            )
        return None

    def _vocabulary_for(self, kind: str) -> Iterable[str]:
        """Vocabulary applicable to a schema-reference failure, shared
        by ``did_you_mean`` ranking and ``available_in_scope`` surfacing."""
        if kind == "label":
            return self._label_names
        if kind == "relationship":
            return self._rel_names
        if kind == "property":
            # Property suggestions need a label scope to be useful. We
            # don't have one from the notification; offer suggestions
            # across ALL declared properties as a best-effort.
            return {
                p.name for props in self._schema.properties_by_label.values() for p in props
            } | {p.name for props in self._schema.properties_by_rel_type.values() for p in props}
        return ()

    def _suggestions_for(self, unknown: str, kind: str) -> list[str]:
        return list(
            difflib.get_close_matches(
                unknown,
                list(self._vocabulary_for(kind)),
                n=self._max_suggestions,
                cutoff=self._cutoff,
            )
        )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _fail(stage: str, payload: object) -> StructuralValidatorResult:
    """Convenience constructor for a failed StructuralValidatorResult."""
    return StructuralValidatorResult(
        passed=False,
        failed_stage=stage,  # type: ignore[arg-type]
        error_payload=payload,  # type: ignore[arg-type]
    )


def _parse_error_from_syntax(exc: CypherSyntaxError, query: str) -> ParseError:
    """Convert a Neo4j CypherSyntaxError into our ParseError.

    The position trailer ``(line N, column M (offset: K))`` lives in the
    message text; we parse it and trim the trailer so the message we
    surface is the human-readable bit.
    """
    raw = str(getattr(exc, "message", None) or exc)
    # Strip the "EXPLAIN " prefix from the echoed query in the error
    # message so users see only their own input.
    raw = raw.replace('"EXPLAIN ', '"')
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


def _parse_error_from_client(exc: ClientError, query: str) -> ParseError:
    """Fallback: any other ClientError gets surfaced as a ParseError.

    Better to flag unknown errors visibly than to silently pass them.
    The brief asks for this behaviour explicitly.
    """
    code = getattr(exc, "code", "Neo.ClientError.Statement.Unknown") or ""
    raw = str(getattr(exc, "message", None) or exc)
    return ParseError(
        message=f"{code}: {raw}",
        line=1,
        column=1,
        snippet=query[:80],
        excerpt_with_caret=excerpt_with_caret(query, 1, 1),
    )


def _line_col_from_message(message: str) -> tuple[int | None, int | None]:
    """Extract the (line, column) position from a CypherSyntaxError
    message. Both are 1-based in Neo4j's message text; the AST-backend
    column convention (0-based-to-1-based) does not apply here because
    Neo4j reports columns as 1-based already."""
    match = _SYNTAX_LINE_COL_PATTERN.search(message)
    if match is None:
        return None, None
    return int(match.group(1)), int(match.group(2)) - 1
    # The -1 lets callers apply the same `max(1, col + 1)` normalisation
    # as the AST backend; we return a 0-based column for consistency
    # across backends.


def _position_to_line_col(position: object) -> tuple[int, int]:
    """Extract (line, column) from a Neo4j ``SummaryInputPosition`` (or
    plain dict). Both `.line` and `.column` are 1-based; we keep them so."""
    if position is None:
        return 1, 1
    line = getattr(position, "line", None)
    column = getattr(position, "column", None)
    if line is None or column is None:
        # Fall back to dict-style access for older driver versions.
        if isinstance(position, dict):
            line = position.get("line", 1)
            column = position.get("column", 1)
        else:
            return 1, 1
    return max(1, int(line)), max(1, int(column))


def _snippet_at(query: str, line: int, col: int, *, radius: int = 30) -> str:
    """Single-line context fragment centred on (line, col)."""
    lines = query.split("\n")
    if line < 1 or line > len(lines):
        return query[:60]
    target_line = lines[line - 1]
    start = max(0, col - radius)
    end = min(len(target_line), col + radius)
    return target_line[start:end].strip() or target_line.strip()
