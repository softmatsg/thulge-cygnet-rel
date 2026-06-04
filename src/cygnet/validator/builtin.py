# Copyright (c) 2026 Nikodem Tomczak, Thulge Labs. All rights reserved.

"""Builtin validator backend: pure-Python regex-based fast filter.

No native dependencies, no database. Validates against a loaded
``Schema`` Pydantic model. Designed as the first tier of the default
chain (``["builtin", "ast", "explain"]``): catches obvious errors at
zero I/O cost so the heavier backends only run on queries that survive.

Coverage:

- **Parse** stage: balanced ``()``/``[]``/``{}``, plus near-miss
  uppercase tokens that look like typo'd Cypher keywords (e.g.
  ``MTACH`` -> suggest ``MATCH``). Not a full grammar check.
- **Schema-reference** stage: extracts labels and relationship types
  from node and rel patterns, plus property accesses ``var.prop``
  through the variable's label binding. Unknown references produce a
  ``SchemaError`` carrying did-you-mean candidates from
  ``difflib.get_close_matches`` over the relevant vocabulary.
- **Property** stage (type-mismatch): skipped — requires expression
  analysis the regex layer cannot do. Lands in the AST/EXPLAIN backend.
- **Constraint** stage: skipped — same reason.

The five regex patterns are adapted from LangChain's
``CypherQueryCorrector`` (MIT, see ``docs/inspection_langchain.md``).
The variable-to-label binding pass and the multi-label union semantics
are CYGNET-specific.

Note on the brief vs architecture: the brief asked unknown property
names to be reported via ``PropertyError`` with did-you-mean candidates.
``PropertyError`` (per ``docs/architecture.md``) is scoped to type
mismatches and does not carry a ``did_you_mean`` field; the schema-
reference vocabulary (``SchemaError.reference_kind="property"``) is the
architecturally faithful carrier for "this property name does not
exist." We emit ``SchemaError(reference_kind="property")`` and report
``failed_stage="schema"`` for unknown property names. Type-mismatch
detection (the true ``PropertyError`` case) lands with the AST backend.
"""

from __future__ import annotations

import difflib
import re
from collections.abc import Iterable
from typing import Final

# PropertyError is intentionally NOT imported here: it is the
# architectural carrier for type-mismatch errors, which this backend
# does not produce (regexes cannot infer expression types). Type-
# mismatch detection is performed by the AST backend.
from cygnet._format import bound_available_in_scope
from cygnet.models import (
    ParseError,
    Schema,
    SchemaError,
    StructuralValidatorResult,
)

__all__ = ["BuiltinValidator"]


# ---------------------------------------------------------------------------
# Regex starter kit
#
# Five patterns adapted from LangChain's CypherQueryCorrector
# (libs/neo4j/langchain_neo4j/chains/graph_qa/cypher_utils.py, MIT). See
# docs/inspection_langchain.md for provenance. The patterns handle the
# dominant MATCH shapes agent code emits; full grammar is the AST
# backend's job.
# ---------------------------------------------------------------------------

_PROPERTY_MAP_PATTERN: Final[re.Pattern[str]] = re.compile(r"\{.+?\}")
"""Inline property map ``{name: 'x', age: 30}``. Non-greedy."""

_NODE_PATTERN: Final[re.Pattern[str]] = re.compile(r"\(([^()]*?)\)")
"""Single-level node pattern ``(...)``. Does not match nested parens."""

_REL_PATTERN: Final[re.Pattern[str]] = re.compile(r"\[([^\]]*)\]")
"""Relationship pattern ``[...]``. Captures the brackets' contents."""

_VARLEN_PATTERN: Final[re.Pattern[str]] = re.compile(r"\*[\d.]*")
"""Variable-length relationship suffix ``*1..3``, stripped before
splitting types."""

_PROPERTY_ACCESS_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\b(?P<var>[a-zA-Z_]\w*)\.(?P<prop>[a-zA-Z_]\w*)\b"
)
"""``var.prop`` access. Leading ``[a-zA-Z_]`` avoids numeric literals
like ``1.5``."""

_STRING_LITERAL_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\"(?:[^\"\\]|\\.)*\"|'(?:[^'\\]|\\.)*'"
)
"""Single- and double-quoted strings, with backslash-escaped quotes.
Stripped before structural scans so parens/braces inside strings don't
break balance or label extraction."""


# Cypher clause keywords we will suggest as corrections for near-miss
# uppercase tokens. Conservative list — focuses on clause-level words
# where a typo is high-impact. Lowercase forms are accepted by Neo4j
# but rare in agent output; we only catch uppercase typos.
_CYPHER_KEYWORDS: Final[frozenset[str]] = frozenset(
    {
        "MATCH",
        "OPTIONAL",
        "RETURN",
        "WHERE",
        "WITH",
        "ORDER",
        "LIMIT",
        "SKIP",
        "CREATE",
        "MERGE",
        "DELETE",
        "DETACH",
        "SET",
        "REMOVE",
        "UNWIND",
        "CALL",
        "YIELD",
        "FOREACH",
        "LOAD",
        "CSV",
        "AS",
        "AND",
        "OR",
        "NOT",
        "XOR",
        "NULL",
        "TRUE",
        "FALSE",
        "IS",
        "IN",
        "BY",
        "DISTINCT",
        "EXISTS",
        "UNION",
        "ALL",
        "ASC",
        "DESC",
    }
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _strip_string_literals(query: str) -> str:
    """Replace quoted string contents with same-length space runs.

    Preserves character positions so line/column reports off the
    stripped query map back to the original input cleanly.
    """

    def replace(match: re.Match[str]) -> str:
        return " " * (match.end() - match.start())

    return _STRING_LITERAL_PATTERN.sub(replace, query)


def _offset_to_line_col(text: str, offset: int) -> tuple[int, int]:
    """Convert a 0-based character offset to 1-based (line, column)."""
    if offset < 0:
        offset = 0
    if offset > len(text):
        offset = len(text)
    line = text.count("\n", 0, offset) + 1
    last_newline = text.rfind("\n", 0, offset)
    column = offset - last_newline if last_newline >= 0 else offset + 1
    return line, column


def _snippet_around(text: str, offset: int, *, radius: int = 30) -> str:
    """Return a small context snippet centred on offset, single-line."""
    start = max(0, offset - radius)
    end = min(len(text), offset + radius)
    fragment = text[start:end].replace("\n", " ")
    return fragment.strip()


# ---------------------------------------------------------------------------
# BuiltinValidator
# ---------------------------------------------------------------------------


class BuiltinValidator:
    """Pure-Python regex-based validator.

    Args:
        schema: The loaded Schema to validate against.
        strict_property_existence: When False, skip property-name
            lookups (matches CyVer's ``strict=False`` mode and our
            ``ValidatorConfig.strict_property_existence`` toggle).
        did_you_mean_max_suggestions: Cap on returned suggestion count.
        did_you_mean_cutoff: ``difflib.get_close_matches`` similarity
            cutoff. Defaults to ``0.6`` which catches single-edit
            typos like ``Smaple -> Sample`` without producing spurious
            matches for unrelated tokens.
    """

    def __init__(
        self,
        schema: Schema,
        *,
        strict_property_existence: bool = True,
        did_you_mean_max_suggestions: int = 5,
        did_you_mean_cutoff: float = 0.6,
    ) -> None:
        self._schema = schema
        self._strict_property_existence = strict_property_existence
        self._max_suggestions = did_you_mean_max_suggestions
        self._cutoff = did_you_mean_cutoff

        # Pre-compute lookup sets and per-label property maps.
        self._label_names: set[str] = {nl.name for nl in schema.labels}
        self._rel_names: set[str] = {rt.name for rt in schema.relationship_types}
        self._props_by_label: dict[str, set[str]] = {
            name: {p.name for p in props} for name, props in schema.properties_by_label.items()
        }
        self._props_by_rel: dict[str, set[str]] = {
            name: {p.name for p in props} for name, props in schema.properties_by_rel_type.items()
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(self, query: str) -> StructuralValidatorResult:
        """Run parse and schema-reference stages and return the result."""
        parse_err = self._check_parse(query)
        if parse_err is not None:
            return StructuralValidatorResult(
                passed=False,
                failed_stage="parse",
                error_payload=parse_err,
            )

        schema_err = self._check_schema_references(query)
        if schema_err is not None:
            return StructuralValidatorResult(
                passed=False,
                failed_stage="schema",
                error_payload=schema_err,
            )

        # Property type-mismatch and constraint checks are skipped in
        # this backend; they require expression analysis the regex
        # layer cannot provide. The chain's downstream backends pick
        # them up.
        return StructuralValidatorResult(passed=True, failed_stage="none")

    # ------------------------------------------------------------------
    # Parse stage
    # ------------------------------------------------------------------

    def _check_parse(self, query: str) -> ParseError | None:
        balance_err = self._check_balanced(query)
        if balance_err is not None:
            return balance_err
        return self._check_clause_typos(query)

    def _check_balanced(self, query: str) -> ParseError | None:
        """Return a ParseError if parens/brackets/braces are unbalanced."""
        stripped = _strip_string_literals(query)
        stack: list[tuple[str, int]] = []  # (opener, offset)
        pairs: dict[str, str] = {")": "(", "]": "[", "}": "{"}
        for offset, ch in enumerate(stripped):
            if ch in "([{":
                stack.append((ch, offset))
            elif ch in ")]}":
                if not stack:
                    line, column = _offset_to_line_col(query, offset)
                    return ParseError(
                        message=(f"Unbalanced {ch!r}: closing bracket without a matching opener."),
                        line=line,
                        column=column,
                        snippet=_snippet_around(query, offset),
                    )
                opener, _ = stack.pop()
                if pairs[ch] != opener:
                    line, column = _offset_to_line_col(query, offset)
                    return ParseError(
                        message=(
                            f"Mismatched bracket: opened with {opener!r}, closed with {ch!r}."
                        ),
                        line=line,
                        column=column,
                        snippet=_snippet_around(query, offset),
                    )
        if stack:
            opener, offset = stack[-1]
            line, column = _offset_to_line_col(query, offset)
            return ParseError(
                message=f"Unclosed bracket {opener!r}.",
                line=line,
                column=column,
                snippet=_snippet_around(query, offset),
            )
        return None

    def _check_clause_typos(self, query: str) -> ParseError | None:
        """Flag uppercase tokens that look like typo'd Cypher keywords.

        Skips tokens that are known keywords, declared schema labels, or
        declared relationship types (they're legitimate uses).
        """
        stripped = _strip_string_literals(query)
        for match in re.finditer(r"\b[A-Z][A-Z_]+\b", stripped):
            word = match.group()
            if word in _CYPHER_KEYWORDS:
                continue
            if word in self._label_names or word in self._rel_names:
                continue
            suggestions = difflib.get_close_matches(word, _CYPHER_KEYWORDS, n=1, cutoff=0.75)
            if not suggestions:
                continue
            line, column = _offset_to_line_col(query, match.start())
            return ParseError(
                message=(f"Unknown clause keyword {word!r}; did you mean {suggestions[0]!r}?"),
                line=line,
                column=column,
                snippet=_snippet_around(query, match.start()),
            )
        return None

    # ------------------------------------------------------------------
    # Schema-reference stage
    # ------------------------------------------------------------------

    def _check_schema_references(self, query: str) -> SchemaError | None:
        """Check labels, relationship types, and property accesses.

        Returns ``None`` on pass or a ``SchemaError`` for the first
        offending reference.
        """
        stripped = _strip_string_literals(query)

        # 1. Labels in node patterns.
        bindings: dict[str, set[str]] = {}
        for match in _NODE_PATTERN.finditer(stripped):
            contents = match.group(1)
            var, labels = _parse_node_pattern(contents)
            if var:
                bindings.setdefault(var, set()).update(labels)
            for label in labels:
                if label not in self._label_names:
                    return self._unknown_schema_error(
                        unknown=label,
                        kind="label",
                        query=query,
                        offset=match.start(),
                        vocabulary=self._label_names,
                    )

        # 2. Relationship types in rel patterns.
        for match in _REL_PATTERN.finditer(stripped):
            for rel_type in _parse_rel_pattern(match.group(1)):
                if rel_type not in self._rel_names:
                    return self._unknown_schema_error(
                        unknown=rel_type,
                        kind="relationship",
                        query=query,
                        offset=match.start(),
                        vocabulary=self._rel_names,
                    )

        # 3. Property accesses ``var.prop`` against variable bindings.
        if not self._strict_property_existence:
            return None
        for match in _PROPERTY_ACCESS_PATTERN.finditer(stripped):
            var = match.group("var")
            prop = match.group("prop")
            if var not in bindings:
                # Variable not bound by a node pattern — skip.
                # Matches CyVer's strict=False behaviour for unbound vars.
                continue
            var_labels = bindings[var]
            if not var_labels:
                # Variable declared without a label — no schema info to
                # check against. Skip.
                continue
            allowed: set[str] = set()
            for label in var_labels:
                allowed.update(self._props_by_label.get(label, set()))
            if prop in allowed:
                continue
            return self._unknown_schema_error(
                unknown=prop,
                kind="property",
                query=query,
                offset=match.start("prop"),
                vocabulary=allowed,
            )
        return None

    # ------------------------------------------------------------------
    # Error construction
    # ------------------------------------------------------------------

    def _unknown_schema_error(
        self,
        *,
        unknown: str,
        kind: str,
        query: str,
        offset: int,
        vocabulary: Iterable[str],
    ) -> SchemaError:
        vocab_list = list(vocabulary)
        suggestions = difflib.get_close_matches(
            unknown,
            vocab_list,
            n=self._max_suggestions,
            cutoff=self._cutoff,
        )
        # SchemaError.reference_kind accepts {"label", "relationship",
        # "property"}; the calling sites pass exactly one of those, but
        # we narrow at the type level via cast through Literal Mapping.
        if kind not in ("label", "relationship", "property"):
            raise ValueError(f"Unsupported SchemaError reference_kind: {kind!r}")
        in_scope, truncated = bound_available_in_scope(vocab_list)
        return SchemaError(
            unknown_reference=unknown,
            reference_kind=kind,  # type: ignore[arg-type]
            did_you_mean=list(suggestions),
            query_context=_snippet_around(query, offset),
            available_in_scope=in_scope,
            available_in_scope_truncated=truncated,
        )


# ---------------------------------------------------------------------------
# Pure helpers — extract structure from a single node or rel pattern
# ---------------------------------------------------------------------------


def _parse_node_pattern(contents: str) -> tuple[str, list[str]]:
    """Parse a node-pattern body into ``(variable_name, labels)``.

    ``contents`` is the substring inside the outermost parens (without
    them). Handles forms: ``""``, ``"s"``, ``":Person"``, ``"s:Person"``,
    ``"s:Person:Author"``, with an optional trailing ``{...}`` property
    map (stripped here).
    """
    without_props = _PROPERTY_MAP_PATTERN.sub("", contents).strip()
    if ":" not in without_props:
        return without_props, []
    parts = [p.strip() for p in without_props.split(":")]
    var = parts[0]
    labels = [p for p in parts[1:] if p]
    return var, labels


def _parse_rel_pattern(contents: str) -> list[str]:
    """Parse a rel-pattern body into a list of relationship type names.

    Returns an empty list when the pattern is type-free (e.g. ``[r]``,
    ``[*1..3]``) or only contains a variable-length suffix.
    """
    without_props = _PROPERTY_MAP_PATTERN.sub("", contents).strip()
    without_varlen = _VARLEN_PATTERN.sub("", without_props).strip()
    if ":" not in without_varlen:
        return []
    type_part = without_varlen.split(":", 1)[1].strip()
    if not type_part:
        return []
    return [t.strip() for t in type_part.split("|") if t.strip()]
