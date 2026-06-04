# Copyright (c) 2026 Nikodem Tomczak, Thulge Labs. All rights reserved.

"""AST-based validator backend, on top of ``antlr4-cypher``.

The wrapper choice and rationale are recorded in
``docs/ast_wrapper_evaluation.md``.

Stage coverage:

- **Parse:** the ANTLR4 parser invoked via ``CypherParser.script()``.
  Failures surface through a custom ``ErrorListener``; we convert the
  first reported error into a :class:`cygnet.models.ParseError` carrying
  the line, column, message, and a snippet of the offending region.
- **Schema-reference:** a single listener walk collects every
  ``NodeLabelsContext``, ``RelationshipTypesContext``, and
  ``PropertyExpressionContext`` in the tree. The label-test form
  ``WHERE n:Label`` is caught for free here because the AST reuses
  ``NodeLabelsContext`` for both pattern labels and WHERE predicates —
  this is the case the regex-only builtin backend cannot reach.
- **Property type-mismatch:** for ``WHERE`` comparisons of the form
  ``var.prop <op> <literal>`` whose literal type the AST exposes, we
  compare the literal's type to the property's declared schema type.
  Failures produce :class:`PropertyError` with ``declared_type`` and
  ``used_type`` populated; ``did_you_mean`` is empty for type
  mismatches (it's reserved for the unknown-property branch).
- **Constraint:** for ``CREATE`` patterns with explicit labels, we
  check the schema's existence constraints — every required property
  (per ``NODE_PROPERTY_EXISTENCE``-style constraints, or any property
  with ``optional=False``) must appear in the node pattern's property
  map. Failures produce :class:`ConstraintError`.

Known limitations (documented for ``docs/borrowing_decisions.md``):

- ``CALL { ... }`` subqueries (Neo4j 4.0+) are not supported by the
  bundled Cypher 9 grammar. We detect this case before invoking the
  parser and surface a clear ``ParseError`` mentioning subqueries so
  users recognise the limitation rather than the cryptic ANTLR error.
- **Positive integer literals masquerade as identifiers** in the
  grammar — ``WHERE n.year > 42`` parses ``42`` through ``SymbolContext``
  rather than ``NumLitContext``. Negative integers (``-5``), floats
  (``3.14``), strings (``'x'``), booleans, and ``null`` all parse to
  proper literal contexts. Type-mismatch detection therefore catches
  string-vs-integer mismatches (``s.year > "2020"``) but not integer-
  vs-string (``s.name = 42``).
- Uniqueness constraint violations require runtime data the schema does
  not carry; we only check existence constraints statically. A
  ``CREATE`` that would conflict on a unique key passes the AST
  backend; the EXPLAIN backend catches it.
"""

from __future__ import annotations

import difflib
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final

from antlr4 import (  # type: ignore[import-untyped]
    CommonTokenStream,
    InputStream,
    ParseTreeWalker,
)
from antlr4.error.ErrorListener import ErrorListener  # type: ignore[import-untyped]
from antlr4_cypher import (  # type: ignore[import-untyped]
    CypherLexer,
    CypherParser,
    CypherParserListener,
)

from cygnet._format import bound_available_in_scope, excerpt_with_caret
from cygnet.models import (
    ConstraintError,
    ParseError,
    PropertyError,
    Schema,
    SchemaError,
    StructuralValidatorResult,
)

if TYPE_CHECKING:
    from antlr4 import ParserRuleContext

__all__ = ["ASTValidator"]


# ---------------------------------------------------------------------------
# Subquery pre-scan
# ---------------------------------------------------------------------------

_CALL_SUBQUERY_PATTERN: Final[re.Pattern[str]] = re.compile(r"\bCALL\s*\{", re.IGNORECASE)
"""Detects ``CALL { ... }`` subquery syntax (Neo4j 4.0+). The bundled
grammar predates this; we prefer a friendly error over the cryptic
ANTLR ``mismatched input '{' expecting ...`` message."""

_SUBQUERY_ERROR_MESSAGE: Final[str] = (
    "CALL subqueries (Neo4j 4.0+) are not supported by the bundled grammar; "
    "this is a known AST backend limitation."
)
"""Public error contract — once shipped, this exact string is part of
the AST backend's stable surface. Tests grep for substring 'CALL
subqueries'."""


# ---------------------------------------------------------------------------
# Literal-type mapping
# ---------------------------------------------------------------------------

_LITERAL_CONTEXT_TO_TYPE: Final[dict[str, str]] = {
    "StringLitContext": "STRING",
    "CharLitContext": "STRING",
    "NumLitContext": "FLOAT",
    "BoolLitContext": "BOOLEAN",
}
"""Map from ANTLR context-class names to the type-vocabulary string we
write into ``PropertyError.used_type``. ``NumLitContext`` is reported
as ``FLOAT`` because the bundled grammar routes it only through
floating-point literals — positive integers are routed through
``SymbolContext`` and so are not detectable here."""


# ---------------------------------------------------------------------------
# Schema-vocabulary normalisation
# ---------------------------------------------------------------------------

_TYPE_FAMILIES: Final[dict[str, frozenset[str]]] = {
    "INTEGER": frozenset({"INTEGER", "INT", "LONG", "NUMERIC"}),
    "FLOAT": frozenset({"FLOAT", "DOUBLE", "NUMERIC"}),
    "STRING": frozenset({"STRING", "VARCHAR", "TEXT", "CHAR"}),
    "BOOLEAN": frozenset({"BOOLEAN", "BOOL"}),
    "DATE": frozenset({"DATE"}),
    "DATETIME": frozenset({"DATETIME", "LOCAL_DATETIME", "TIMESTAMP"}),
}
"""Loose equivalence classes for type comparison. Lets a property
declared as ``LONG`` match an ``INTEGER`` literal. Type names that
don't appear in any family are compared as exact strings."""


def _types_match(declared: str, used: str) -> bool:
    """Permissive equality: exact match or shared family."""
    if declared.upper() == used.upper():
        return True
    for family in _TYPE_FAMILIES.values():
        if declared.upper() in family and used.upper() in family:
            return True
    return False


# ---------------------------------------------------------------------------
# Error listener
# ---------------------------------------------------------------------------


class _CapturingErrorListener(ErrorListener):  # type: ignore[misc]
    """Captures ANTLR parse errors with line/column for later wrapping
    into a :class:`ParseError`."""

    def __init__(self) -> None:
        super().__init__()
        self.errors: list[tuple[int, int, str]] = []

    def syntaxError(  # noqa: N802 (overrides ANTLR's camelCase API)
        self,
        recognizer: Any,
        offending_symbol: Any,
        line: int,
        column: int,
        msg: str,
        e: Any,
    ) -> None:
        self.errors.append((line, column, msg))


# ---------------------------------------------------------------------------
# Single-pass collector
# ---------------------------------------------------------------------------


@dataclass
class _Comparison:
    """A ``var.prop <op> <literal>`` (or symmetric) expression we may
    type-check. ``literal_type`` is ``None`` when we couldn't identify
    a literal in the comparison's operands."""

    var: str
    prop: str
    literal_type: str | None
    line: int
    col: int


@dataclass
class _CreatePattern:
    """A ``CREATE (s:Label {...})`` pattern, ready for constraint check."""

    labels: list[str]
    properties_set: set[str]
    line: int
    col: int


@dataclass
class _PathSegment:
    """A single (left-node, relationship, right-node) triple inside a
    path pattern. The pattern ``(a:X)-[:R1|R2]->(b:Y)-[:S]->(c:Z)``
    produces two segments: (X, R1|R2, ->, Y) and (Y, S, ->, Z).

    Used by the rel-endpoint compatibility check. ``left_labels`` and
    ``right_labels`` are the declared labels on each side (empty set
    means no label was declared, in which case the segment is skipped).
    ``direction`` is one of ``"right"`` (``->``), ``"left"`` (``<-``),
    or ``"undirected"`` (``-``).
    """

    left_labels: set[str]
    rel_types: list[str]
    direction: str
    right_labels: set[str]
    line: int
    col: int


@dataclass
class _CollectedAST:
    """Everything one pass over the AST collects. The fields are
    consumed by the per-stage checks in :meth:`ASTValidator.validate`."""

    bindings: dict[str, set[str]] = field(default_factory=dict)
    label_refs: list[tuple[str, int, int]] = field(default_factory=list)
    rel_type_refs: list[tuple[str, int, int]] = field(default_factory=list)
    property_refs: list[tuple[str, str, int, int]] = field(default_factory=list)
    """Property accesses ``var.prop``: list of (var, prop, line, col)."""
    comparisons: list[_Comparison] = field(default_factory=list)
    create_patterns: list[_CreatePattern] = field(default_factory=list)
    path_segments: list[_PathSegment] = field(default_factory=list)
    """One per (node, rel, node) triple in any path pattern. Used by
    the rel-endpoint compatibility check."""


class _Collector(CypherParserListener):  # type: ignore[misc]
    """ANTLR listener that fills a :class:`_CollectedAST` in a single
    parse-tree walk."""

    def __init__(self) -> None:
        self.info = _CollectedAST()
        self._in_create_depth = 0

    # -- helpers -------------------------------------------------------

    @staticmethod
    def _children_of_type(ctx: ParserRuleContext, type_name: str) -> list[Any]:
        out = []
        for ch in getattr(ctx, "children", None) or ():
            if type(ch).__name__ == type_name:
                out.append(ch)
        return out

    @staticmethod
    def _first_descendant_of_type(ctx: ParserRuleContext, type_names: Iterable[str]) -> Any | None:
        target = set(type_names)
        stack: list[Any] = [ctx]
        while stack:
            node = stack.pop()
            if type(node).__name__ in target:
                return node
            for ch in getattr(node, "children", None) or ():
                stack.append(ch)
        return None

    @staticmethod
    def _start_loc(ctx: ParserRuleContext) -> tuple[int, int]:
        tok = getattr(ctx, "start", None)
        if tok is None:
            return 1, 1
        return tok.line, tok.column

    # -- CREATE entry/exit (so node patterns inside CREATE are flagged) --

    def enterCreateSt(self, ctx: ParserRuleContext) -> None:  # noqa: N802
        self._in_create_depth += 1

    def exitCreateSt(self, ctx: ParserRuleContext) -> None:  # noqa: N802
        self._in_create_depth -= 1

    # -- NodePattern: variable + labels + property map -----------------

    def enterNodePattern(self, ctx: ParserRuleContext) -> None:  # noqa: N802
        var: str | None = None
        labels: list[str] = []
        prop_keys: set[str] = set()

        for ch in getattr(ctx, "children", None) or ():
            name = type(ch).__name__
            if name == "SymbolContext":
                var = ch.getText()
            elif name == "NodeLabelsContext":
                for sub in self._children_of_type(ch, "NameContext"):
                    labels.append(sub.getText())
            elif name == "PropertiesContext":
                map_lit = self._first_descendant_of_type(ch, {"MapLitContext"})
                if map_lit is not None:
                    for pair in self._children_of_type(map_lit, "MapPairContext"):
                        # First NameContext child of the pair is the key.
                        name_ctxs = self._children_of_type(pair, "NameContext")
                        if name_ctxs:
                            prop_keys.add(name_ctxs[0].getText())

        # Bindings: always record, even with empty label set, so
        # downstream property-check logic can distinguish "bound without
        # label" from "completely unbound".
        if var is not None:
            self.info.bindings.setdefault(var, set()).update(labels)

        line, col = self._start_loc(ctx)
        for label in labels:
            self.info.label_refs.append((label, line, col))

        if self._in_create_depth > 0 and labels:
            self.info.create_patterns.append(
                _CreatePattern(
                    labels=labels,
                    properties_set=prop_keys,
                    line=line,
                    col=col,
                )
            )

    # -- NodeLabelsContext outside a NodePattern (WHERE n:Label form) --

    def enterPropertyOrLabelExpression(  # noqa: N802
        self, ctx: ParserRuleContext
    ) -> None:
        # Form: 'n:Label[:Label2]' inside an expression context (not a
        # node pattern). The NodeLabelsContext child carries the labels;
        # NodePattern walking already covers the (var:Label) form, but
        # this catches `WHERE n:Label`.
        labels_ctx = self._first_descendant_of_type(ctx, {"NodeLabelsContext"})
        if labels_ctx is None:
            return
        # Only record if this PropertyOrLabelExpression is NOT inside a
        # NodePattern (those are handled above). The parent chain
        # already passed through Expression nodes if so; checking the
        # immediate parent for NodePatternContext is sufficient.
        parent = ctx.parentCtx
        while parent is not None:
            if type(parent).__name__ == "NodePatternContext":
                return
            parent = getattr(parent, "parentCtx", None)

        line, col = self._start_loc(labels_ctx)
        for sub in self._children_of_type(labels_ctx, "NameContext"):
            self.info.label_refs.append((sub.getText(), line, col))

    # -- Relationship types --------------------------------------------

    def enterRelationshipTypes(self, ctx: ParserRuleContext) -> None:  # noqa: N802
        for sub in self._children_of_type(ctx, "NameContext"):
            line, col = self._start_loc(sub)
            self.info.rel_type_refs.append((sub.getText(), line, col))

    # -- Path segments (node-rel-node triples) --------------------------

    def enterPatternElem(self, ctx: ParserRuleContext) -> None:  # noqa: N802
        """Walk a ``PatternElem`` and emit one :class:`_PathSegment` per
        (left-node, rel, right-node) triple. Layout:

            PatternElem
              NodePattern (left)
              PatternElemChain*  (zero or more rel + right-node pairs)
                RelationshipPattern
                  TerminalNodeImpl '<' or '-'  (left of bracket)
                  RelationDetail   (optional; absent means '-[]-')
                    RelationshipTypes (optional)
                  TerminalNodeImpl '-' or '>'  (right of bracket)
                NodePattern (right)

        Multi-hop chains produce multiple segments.
        """
        children = list(getattr(ctx, "children", None) or ())
        # Find the leading NodePattern.
        left_node = None
        idx = 0
        while idx < len(children) and type(children[idx]).__name__ != "NodePatternContext":
            idx += 1
        if idx >= len(children):
            return
        left_node = children[idx]
        idx += 1

        # Walk PatternElemChain children in order.
        while idx < len(children):
            ch = children[idx]
            if type(ch).__name__ != "PatternElemChainContext":
                idx += 1
                continue
            # Extract RelationshipPattern + NodePattern from this chain.
            rel_pat = None
            right_node = None
            for sub in getattr(ch, "children", None) or ():
                sn = type(sub).__name__
                if sn == "RelationshipPatternContext" and rel_pat is None:
                    rel_pat = sub
                elif sn == "NodePatternContext" and right_node is None:
                    right_node = sub
            if rel_pat is None or right_node is None:
                idx += 1
                continue
            segment = self._build_path_segment(left_node, rel_pat, right_node)
            if segment is not None:
                self.info.path_segments.append(segment)
            # The right node of this segment is the left node of the next.
            left_node = right_node
            idx += 1

    @classmethod
    def _node_labels(cls, node_ctx: Any) -> set[str]:
        """Return the declared labels on a ``NodePatternContext`` (may
        be empty)."""
        labels: set[str] = set()
        for ch in getattr(node_ctx, "children", None) or ():
            if type(ch).__name__ != "NodeLabelsContext":
                continue
            for sub in cls._children_of_type(ch, "NameContext"):
                labels.add(sub.getText())
        return labels

    @classmethod
    def _rel_segment_details(
        cls, rel_pat_ctx: Any
    ) -> tuple[list[str], str]:
        """Return (rel_types, direction) for a ``RelationshipPatternContext``.
        ``direction`` is one of ``"right"`` / ``"left"`` / ``"undirected"``.
        ``rel_types`` is the list inside ``-[:R1|R2|...]->``; empty when
        the pattern is ``-[]-`` or ``--``."""
        # Direction: scan the ordered children. A '<' before the bracket
        # means left-pointing; a '>' after the bracket means right-pointing.
        has_left_arrow = False
        has_right_arrow = False
        bracket_seen = False
        rel_types: list[str] = []

        for ch in getattr(rel_pat_ctx, "children", None) or ():
            n = type(ch).__name__
            if n == "TerminalNodeImpl":
                txt = ch.getText()
                if txt == "<" and not bracket_seen:
                    has_left_arrow = True
                elif txt == ">" and bracket_seen:
                    has_right_arrow = True
                elif txt == "[":
                    bracket_seen = True
            elif n == "RelationDetailContext":
                bracket_seen = True
                # Find the RelationshipTypes child (optional).
                for sub in getattr(ch, "children", None) or ():
                    if type(sub).__name__ != "RelationshipTypesContext":
                        continue
                    for name_ctx in cls._children_of_type(sub, "NameContext"):
                        rel_types.append(name_ctx.getText())

        if has_right_arrow and not has_left_arrow:
            direction = "right"
        elif has_left_arrow and not has_right_arrow:
            direction = "left"
        else:
            direction = "undirected"
        return rel_types, direction

    @classmethod
    def _build_path_segment(
        cls, left_node: Any, rel_pat: Any, right_node: Any
    ) -> _PathSegment | None:
        left_labels = cls._node_labels(left_node)
        right_labels = cls._node_labels(right_node)
        rel_types, direction = cls._rel_segment_details(rel_pat)
        line, col = cls._start_loc(rel_pat)
        return _PathSegment(
            left_labels=left_labels,
            rel_types=rel_types,
            direction=direction,
            right_labels=right_labels,
            line=line,
            col=col,
        )

    # -- Property accesses ---------------------------------------------

    def enterPropertyExpression(self, ctx: ParserRuleContext) -> None:  # noqa: N802
        # Children typically: AtomContext (variable wrapper), '.',
        # NameContext (property). Skip cases where the AtomContext
        # holds a literal rather than a variable (e.g. '42.0' parses
        # as AtomContext('42') -> NameContext('0')).
        atom_ctx = None
        name_ctxs: list[Any] = []
        for ch in getattr(ctx, "children", None) or ():
            n = type(ch).__name__
            if n == "AtomContext" and atom_ctx is None:
                atom_ctx = ch
            elif n == "NameContext":
                name_ctxs.append(ch)
        if atom_ctx is None or not name_ctxs:
            return

        var_sym = self._first_descendant_of_type(atom_ctx, {"SymbolContext"})
        if var_sym is None:
            return
        var = var_sym.getText()
        # Skip numeric "variables" (the integer-as-symbol grammar quirk).
        if var.isdigit():
            return

        prop_ctx = name_ctxs[-1]
        prop = prop_ctx.getText()
        line, col = self._start_loc(prop_ctx)
        self.info.property_refs.append((var, prop, line, col))

    # -- Comparison expressions (for type-mismatch) --------------------

    def enterComparisonExpression(self, ctx: ParserRuleContext) -> None:  # noqa: N802
        # Only count comparisons that have a comparison-signs operator;
        # without one, this is just an expression pass-through.
        children = list(getattr(ctx, "children", None) or ())
        sign_index = next(
            (i for i, c in enumerate(children) if type(c).__name__ == "ComparisonSignsContext"),
            -1,
        )
        if sign_index < 0:
            return
        left_children = children[:sign_index]
        right_children = children[sign_index + 1 :]

        # Walk each side independently. We want EXACTLY ONE side to
        # contain a "real" property access (atom + .prop) and the OTHER
        # to contain a literal whose type the grammar surfaces. The
        # grammar wraps bare literals in PropertyExpression too, so we
        # filter for property expressions that actually have a name
        # child.
        left_prop = self._real_property_expression(left_children)
        right_prop = self._real_property_expression(right_children)
        left_lit = self._first_literal_in(left_children)
        right_lit = self._first_literal_in(right_children)

        if left_prop is not None and right_lit is not None and right_prop is None:
            prop_ctx, lit_ctx = left_prop, right_lit
        elif right_prop is not None and left_lit is not None and left_prop is None:
            prop_ctx, lit_ctx = right_prop, left_lit
        else:
            return

        var, prop = self._var_and_prop(prop_ctx)
        if var is None or prop is None or var.isdigit():
            return

        literal_type = _LITERAL_CONTEXT_TO_TYPE[type(lit_ctx).__name__]
        line, col = self._start_loc(prop_ctx)
        self.info.comparisons.append(
            _Comparison(var=var, prop=prop, literal_type=literal_type, line=line, col=col)
        )

    # -- helpers used by enterComparisonExpression ---------------------

    @classmethod
    def _real_property_expression(cls, subtrees: list[Any]) -> Any | None:
        """Return the first PropertyExpression in ``subtrees`` that
        actually represents ``var.prop`` (Atom + NameContext), not a
        bare-literal wrap."""
        for root in subtrees:
            stack = [root]
            while stack:
                node = stack.pop()
                if type(node).__name__ == "PropertyExpressionContext":
                    var, prop = cls._var_and_prop(node)
                    if var is not None and prop is not None:
                        return node
                for ch in getattr(node, "children", None) or ():
                    stack.append(ch)
        return None

    @staticmethod
    def _first_literal_in(subtrees: list[Any]) -> Any | None:
        for root in subtrees:
            stack = [root]
            while stack:
                node = stack.pop()
                if type(node).__name__ in _LITERAL_CONTEXT_TO_TYPE:
                    return node
                for ch in getattr(node, "children", None) or ():
                    stack.append(ch)
        return None

    @classmethod
    def _var_and_prop(cls, prop_ctx: Any) -> tuple[str | None, str | None]:
        """Extract (var, prop) from a PropertyExpressionContext that
        truly represents a ``var.prop`` access. Returns (None, None)
        when the context is a bare-literal wrap (no NameContext child)."""
        var = None
        prop = None
        for ch in getattr(prop_ctx, "children", None) or ():
            n = type(ch).__name__
            if n == "AtomContext":
                sym = cls._first_descendant_of_type(ch, {"SymbolContext"})
                if sym is not None:
                    text = sym.getText()
                    if not text.isdigit():
                        var = text
            elif n == "NameContext":
                prop = ch.getText()
        return var, prop


# ---------------------------------------------------------------------------
# Public validator
# ---------------------------------------------------------------------------


class ASTValidator:
    """AST-based validator. Sits behind ``ValidatorChain`` as the
    second-tier authoritative-ish checker after the builtin fast filter.

    Args:
        schema: the loaded Schema to validate against.
        strict_property_existence: when False, skip property-name
            lookups (matches ``ValidatorConfig.strict_property_existence``
            and the builtin backend's behaviour).
        did_you_mean_max_suggestions: cap on suggestion list length.
        did_you_mean_cutoff: ``difflib.get_close_matches`` similarity
            cutoff. Matches the builtin's default 0.6.
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

        self._label_names: set[str] = {nl.name for nl in schema.labels}
        self._rel_names: set[str] = {rt.name for rt in schema.relationship_types}
        self._props_by_label: dict[str, list[tuple[str, str]]] = {
            name: [(p.name, p.type) for p in props]
            for name, props in schema.properties_by_label.items()
        }
        # Pre-compute label-keyed maps for quick property type lookup.
        self._prop_type_by_label: dict[str, dict[str, str]] = {
            name: {p.name: p.type for p in props}
            for name, props in schema.properties_by_label.items()
        }
        self._required_props_by_label: dict[str, set[str]] = {
            name: {p.name for p in props if not p.optional}
            for name, props in schema.properties_by_label.items()
        }
        # Constraints by (label, kind). We use the type-string substring
        # match because architecture.md keeps Neo4j-style names like
        # 'NODE_PROPERTY_EXISTENCE' rather than a closed enum.
        self._existence_constraints: list[tuple[str, str, str]] = [
            (c.label_or_rel, c.property, c.identifier)
            for c in schema.constraints
            if "EXISTENCE" in c.type.upper() and c.property is not None
        ]
        # Rel endpoint lookup: name -> (source_label, target_label). Used
        # by the rel-endpoint compatibility check to verify that a query
        # like (a:X)-[:R]->(b:Y) declares X and Y as the source/target of
        # R in the schema. Direction-aware.
        self._rel_endpoints: dict[str, tuple[str, str]] = {
            rt.name: (rt.source_label, rt.target_label)
            for rt in schema.relationship_types
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(self, query: str) -> StructuralValidatorResult:
        # 1. Subquery pre-scan.
        subquery_err = self._check_subquery(query)
        if subquery_err is not None:
            return StructuralValidatorResult(
                passed=False, failed_stage="parse", error_payload=subquery_err
            )

        # 2. Parse and collect in a single tree walk.
        tree, errors = self._parse(query)
        if errors:
            line, col_zero, msg = errors[0]
            # ANTLR reports columns 0-based; ParseError requires >= 1.
            col = max(1, col_zero + 1)
            return StructuralValidatorResult(
                passed=False,
                failed_stage="parse",
                error_payload=ParseError(
                    message=msg,
                    line=line,
                    column=col,
                    snippet=_snippet_at(query, line, col_zero),
                    excerpt_with_caret=excerpt_with_caret(query, line, col),
                ),
            )
        collector = _Collector()
        ParseTreeWalker().walk(collector, tree)
        info = collector.info

        # 3. Schema-reference checks: labels, rel types, property names.
        schema_err = self._check_schema_refs(info, query)
        if schema_err is not None:
            return StructuralValidatorResult(
                passed=False, failed_stage="schema", error_payload=schema_err
            )

        # 3b. Rel-endpoint compatibility: for each (a:X)-[:R]->(b:Y) path
        # segment, verify that R is declared in the schema to connect X
        # to Y (direction-aware). Catches plausible-mistake rel-swaps and
        # label-swaps in path patterns that the bare existence checks
        # above accept.
        rel_err = self._check_rel_endpoints(info, query)
        if rel_err is not None:
            return StructuralValidatorResult(
                passed=False, failed_stage="schema", error_payload=rel_err
            )

        # 4. Property type-mismatch checks (WHERE comparisons).
        property_err = self._check_property_types(info, query)
        if property_err is not None:
            return StructuralValidatorResult(
                passed=False, failed_stage="property", error_payload=property_err
            )

        # 5. Constraint checks (CREATE patterns + existence).
        constraint_err = self._check_constraints(info)
        if constraint_err is not None:
            return StructuralValidatorResult(
                passed=False, failed_stage="constraint", error_payload=constraint_err
            )

        return StructuralValidatorResult(passed=True, failed_stage="none")

    # ------------------------------------------------------------------
    # Stage 0: subquery pre-scan
    # ------------------------------------------------------------------

    def _check_subquery(self, query: str) -> ParseError | None:
        match = _CALL_SUBQUERY_PATTERN.search(query)
        if match is None:
            return None
        line, column = _offset_to_line_col(query, match.start())
        return ParseError(
            message=_SUBQUERY_ERROR_MESSAGE,
            line=line,
            column=column,
            snippet=_snippet_at(query, line, column),
            excerpt_with_caret=excerpt_with_caret(query, line, column),
        )

    # ------------------------------------------------------------------
    # Stage 1: parse
    # ------------------------------------------------------------------

    @staticmethod
    def _parse(query: str) -> tuple[Any, list[tuple[int, int, str]]]:
        stream = InputStream(query)
        lexer = CypherLexer(stream)
        lexer.removeErrorListeners()
        captor = _CapturingErrorListener()
        lexer.addErrorListener(captor)
        parser = CypherParser(CommonTokenStream(lexer))
        parser.removeErrorListeners()
        parser.addErrorListener(captor)
        tree = parser.script()
        return tree, captor.errors

    # ------------------------------------------------------------------
    # Stage 2: schema-reference (labels, rel types, properties)
    # ------------------------------------------------------------------

    def _check_schema_refs(self, info: _CollectedAST, query: str) -> SchemaError | None:
        for label, line, col in info.label_refs:
            if label in self._label_names:
                continue
            return _unknown_schema(
                unknown=label,
                kind="label",
                vocabulary=self._label_names,
                query=query,
                line=line,
                col=col,
                max_suggestions=self._max_suggestions,
                cutoff=self._cutoff,
            )

        for rel_type, line, col in info.rel_type_refs:
            if rel_type in self._rel_names:
                continue
            return _unknown_schema(
                unknown=rel_type,
                kind="relationship",
                vocabulary=self._rel_names,
                query=query,
                line=line,
                col=col,
                max_suggestions=self._max_suggestions,
                cutoff=self._cutoff,
            )

        if not self._strict_property_existence:
            return None
        for var, prop, line, col in info.property_refs:
            labels = info.bindings.get(var)
            if labels is None or not labels:
                # Variable unbound or bound without a label — skip
                # property checking (CyVer's lenient-mode semantics).
                continue
            allowed: set[str] = set()
            for label in labels:
                allowed.update(name for name, _ in self._props_by_label.get(label, []))
            if prop in allowed:
                continue
            return _unknown_schema(
                unknown=prop,
                kind="property",
                vocabulary=allowed,
                query=query,
                line=line,
                col=col,
                max_suggestions=self._max_suggestions,
                cutoff=self._cutoff,
            )
        return None

    # ------------------------------------------------------------------
    # Stage 2b: rel-endpoint compatibility
    # ------------------------------------------------------------------

    def _check_rel_endpoints(self, info: _CollectedAST, query: str) -> SchemaError | None:
        """For each path segment ``(a:X)-[:R]->(b:Y)``, verify that R is
        declared in the schema to connect X to Y.

        Skip rules (return ``None`` for the segment, i.e. accept):

        - Either end has no declared label. We can't say which schema
          endpoint is meant; the structural check requires both ends.
        - No rel types in the segment (``-[]-`` or ``--``). Any rel
          type could apply; structural check not meaningful here.
        - One or more rel types are not declared in the schema. Those
          would have already been caught by :meth:`_check_schema_refs`
          (which runs before this method); if we reach here, the rel
          types are declared, but we still defensively skip unknowns.

        Direction semantics:

        - ``"right"`` (``->``): R must declare source=X, target=Y.
        - ``"left"`` (``<-``): R must declare source=Y, target=X.
        - ``"undirected"`` (``-``): either of the above is acceptable.

        Multi-type ``:R1|R2`` is treated as a disjunction — the segment
        is accepted if AT LEAST ONE rel type in the list is compatible
        with the labels and direction.

        Multi-label nodes ``(a:X:Z)`` are treated as conjunctions (the
        node binds to BOTH labels), so the segment is acceptable if the
        rel connects any one of {X, Z} on one side and any one of the
        other side's labels in the correct direction. This matches
        Cypher's bind-to-all-labels semantics.
        """
        for seg in info.path_segments:
            if not seg.left_labels or not seg.right_labels:
                continue
            if not seg.rel_types:
                continue
            # Filter to declared rel types.
            declared_in_segment = [r for r in seg.rel_types if r in self._rel_endpoints]
            if not declared_in_segment:
                continue
            # Check whether ANY declared rel type connects the labels in
            # at least one acceptable direction.
            any_match = False
            for rt in declared_in_segment:
                src, tgt = self._rel_endpoints[rt]
                if seg.direction in {"right", "undirected"}:
                    if src in seg.left_labels and tgt in seg.right_labels:
                        any_match = True
                        break
                if seg.direction in {"left", "undirected"}:
                    if src in seg.right_labels and tgt in seg.left_labels:
                        any_match = True
                        break
            if any_match:
                continue
            # No declared rel type matches. Report on the first one.
            first_rt = declared_in_segment[0]
            src, tgt = self._rel_endpoints[first_rt]
            # Build a human-readable available_in_scope: which rel types
            # *do* connect (any of) the left labels to (any of) the right
            # labels in any direction.
            available: set[str] = set()
            for cand_name, (cand_src, cand_tgt) in self._rel_endpoints.items():
                if ((cand_src in seg.left_labels and cand_tgt in seg.right_labels)
                        or (cand_src in seg.right_labels and cand_tgt in seg.left_labels)):
                    available.add(cand_name)
            # Direction-aware suggestion text encoded into ``did_you_mean``:
            # for now, suggest the rel-types that do connect these labels.
            return SchemaError(
                category="schema",
                unknown_reference=first_rt,
                reference_kind="relationship",
                did_you_mean=sorted(available)[: self._max_suggestions],
                query_context=_snippet_at(query, seg.line, seg.col),
                available_in_scope=sorted(self._rel_names),
                available_in_scope_truncated=False,
            )
        return None

    # ------------------------------------------------------------------
    # Stage 3: property type-mismatch (WHERE comparisons)
    # ------------------------------------------------------------------

    def _check_property_types(self, info: _CollectedAST, query: str) -> PropertyError | None:
        for cmp in info.comparisons:
            if cmp.literal_type is None:
                continue
            labels = info.bindings.get(cmp.var)
            if not labels:
                continue
            # Find the declared type — first label that declares the
            # property wins. Multi-label union semantics: if any of the
            # variable's labels has this property declared, that's the
            # type to compare against.
            declared_type: str | None = None
            for label in labels:
                type_map = self._prop_type_by_label.get(label, {})
                if cmp.prop in type_map:
                    declared_type = type_map[cmp.prop]
                    break
            if declared_type is None:
                # Property name didn't resolve — schema-ref stage would
                # have caught this; skip type check defensively.
                continue
            if _types_match(declared_type, cmp.literal_type):
                continue
            return PropertyError(
                property_name=cmp.prop,
                declared_type=declared_type,
                used_type=cmp.literal_type,
                query_context=_snippet_at(query, cmp.line, cmp.col),
                did_you_mean=[],
            )
        return None

    # ------------------------------------------------------------------
    # Stage 4: constraints (existence on CREATE patterns)
    # ------------------------------------------------------------------

    def _check_constraints(self, info: _CollectedAST) -> ConstraintError | None:
        for pattern in info.create_patterns:
            for label in pattern.labels:
                # Constraint table first — these carry an explicit
                # constraint identifier we can report.
                for c_label, c_prop, c_id in self._existence_constraints:
                    if c_label != label:
                        continue
                    if c_prop in pattern.properties_set:
                        continue
                    return ConstraintError(
                        constraint_id=c_id,
                        constraint_kind="existence",
                        property_name=c_prop,
                    )
                # Fall back to the per-property `optional=False` flag
                # on the schema model when no explicit constraint
                # mentions this property. Synthesises an identifier
                # since none exists for these implicit constraints.
                required = self._required_props_by_label.get(label, set())
                missing = required - pattern.properties_set
                # Exclude properties already covered by the explicit
                # constraint loop above (those would have been reported
                # already by now).
                explicit = {
                    c_prop for c_label, c_prop, _ in self._existence_constraints if c_label == label
                }
                missing = missing - explicit
                if missing:
                    prop = sorted(missing)[0]
                    return ConstraintError(
                        constraint_id=f"_implicit_{label}_{prop}_required",
                        constraint_kind="existence",
                        property_name=prop,
                    )
        return None


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _unknown_schema(
    *,
    unknown: str,
    kind: str,
    vocabulary: Iterable[str],
    query: str,
    line: int,
    col: int,
    max_suggestions: int,
    cutoff: float,
) -> SchemaError:
    if kind not in ("label", "relationship", "property"):
        raise ValueError(f"Unsupported reference_kind: {kind!r}")
    vocab_list = list(vocabulary)
    suggestions = difflib.get_close_matches(unknown, vocab_list, n=max_suggestions, cutoff=cutoff)
    in_scope, truncated = bound_available_in_scope(vocab_list)
    return SchemaError(
        unknown_reference=unknown,
        reference_kind=kind,  # type: ignore[arg-type]
        did_you_mean=list(suggestions),
        query_context=_snippet_at(query, line, col),
        available_in_scope=in_scope,
        available_in_scope_truncated=truncated,
    )


def _offset_to_line_col(text: str, offset: int) -> tuple[int, int]:
    if offset < 0:
        offset = 0
    if offset > len(text):
        offset = len(text)
    line = text.count("\n", 0, offset) + 1
    last_newline = text.rfind("\n", 0, offset)
    column = offset - last_newline if last_newline >= 0 else offset + 1
    return line, column


def _snippet_at(query: str, line: int, col: int, *, radius: int = 30) -> str:
    """Return a single-line context fragment centred on (line, col).

    Line and column are 1-based; col is 0-based in ANTLR's
    ``ctx.start.column``, so callers normalise as needed before passing
    in. We accept both and just clamp.
    """
    lines = query.split("\n")
    if line < 1 or line > len(lines):
        return query[:60]
    target_line = lines[line - 1]
    start = max(0, col - radius)
    end = min(len(target_line), col + radius)
    return target_line[start:end].strip() or target_line.strip()
