# Copyright (c) 2026 Nikodem Tomczak, Thulge Labs. All rights reserved.

"""Pydantic models for the public CYGNET API surface.

This module defines every model that crosses a public module boundary:
schema representation, structural validator results, cost gate results,
the discriminated error vocabulary, and the corrector result.

Field names declared here are part of the public API (per the immutable
artifacts section of the project brief). Adding fields is non-breaking;
renaming or removing is a major-version change.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "ConstraintError",
    "CorrectorResult",
    "CostError",
    "CostGateResult",
    "EmptyResultError",
    "ExplainOperator",
    "ExplainPlan",
    "GateError",
    "GateResult",
    "Index",
    "MirrorBuildReport",
    "NodeLabel",
    "OperatorCost",
    "ParseError",
    "Property",
    "PropertyError",
    "RelationshipType",
    "Schema",
    "SchemaConstraint",
    "SchemaError",
    "StructuralValidatorResult",
]


_AVAILABLE_IN_SCOPE_CAP: int = 50
"""Maximum entries surfaced in ``SchemaError.available_in_scope``. When
the relevant vocabulary exceeds this, the first 50 alphabetically are
returned and ``available_in_scope_truncated`` is set to ``True`` so
correctors know to expect more options exist beyond the surfaced list."""


# ---------------------------------------------------------------------------
# Schema models
# ---------------------------------------------------------------------------


class Property(BaseModel):
    """A single property on a node label or relationship type."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Property name as it appears in Cypher queries.")
    type: str = Field(
        ...,
        description=(
            "Declared property type (e.g. 'STRING', 'INTEGER', 'BOOLEAN', 'DATETIME'). "
            "Free-form to accommodate Neo4j type names; not enforced as an enum."
        ),
    )
    optional: bool = Field(
        default=True,
        description="Whether the property is allowed to be absent on a node/relationship.",
    )
    sparse: bool = Field(
        default=False,
        description=(
            "Whether the property is observed on a small fraction of instances. Drives "
            "query-planning hints; computed at schema introspection time."
        ),
    )


class NodeLabel(BaseModel):
    """A node label in the schema. Properties are tracked separately on `Schema`."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Label name as it appears in Cypher patterns.")
    sparsity_threshold: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description=(
            "Fraction below which a property on this label is flagged 'sparse'. "
            "Compared against observed presence during introspection."
        ),
    )


class RelationshipType(BaseModel):
    """A relationship type in the schema, with directionality endpoints."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Relationship type name (Cypher convention: ALL_CAPS).")
    source_label: str = Field(
        ...,
        description="Label of the source/start node for this relationship type.",
    )
    target_label: str = Field(
        ...,
        description="Label of the target/end node for this relationship type.",
    )


class SchemaConstraint(BaseModel):
    """A Neo4j constraint on a label or relationship type.

    Named ``SchemaConstraint`` (not ``Constraint``) to avoid collision with
    ``pydantic.Constraint`` and Python's typing imports. The error-side type
    is ``ConstraintError``; the two are distinct: this describes the
    constraint definition, that describes a violation.
    """

    model_config = ConfigDict(extra="forbid")

    type: str = Field(
        ...,
        description=(
            "Constraint kind: typically 'UNIQUENESS', 'NODE_PROPERTY_EXISTENCE', "
            "'RELATIONSHIP_PROPERTY_EXISTENCE', or 'NODE_KEY'. Free-form string to "
            "match Neo4j's introspection output."
        ),
    )
    label_or_rel: str = Field(
        ...,
        description="Label or relationship type the constraint applies to.",
    )
    property: str | None = Field(
        default=None,
        description="Property name the constraint references (None for composite keys).",
    )
    identifier: str = Field(
        ...,
        description="Constraint name/identifier returned by Neo4j (e.g. 'constraint_abc123').",
    )


class Index(BaseModel):
    """A Neo4j index definition on a label or relationship type."""

    model_config = ConfigDict(extra="forbid")

    type: str = Field(
        ...,
        description="Index kind ('BTREE', 'RANGE', 'TEXT', 'POINT', 'LOOKUP', ...).",
    )
    label_or_rel: str = Field(
        ...,
        description="Label or relationship type the index applies to.",
    )
    properties: list[str] = Field(
        ...,
        description="Properties covered by the index, in order.",
    )


class Schema(BaseModel):
    """Loaded schema view used by validator backends and exposed to callers.

    Two ingestion paths produce this same representation: Neo4j introspection
    and YAML/JSON spec. Field names here are public API.
    """

    model_config = ConfigDict(extra="forbid")

    labels: list[NodeLabel] = Field(
        default_factory=list,
        description="Node labels present in the schema.",
    )
    relationship_types: list[RelationshipType] = Field(
        default_factory=list,
        description="Relationship types present in the schema.",
    )
    properties_by_label: dict[str, list[Property]] = Field(
        default_factory=dict,
        description="Mapping of label name to its declared properties.",
    )
    properties_by_rel_type: dict[str, list[Property]] = Field(
        default_factory=dict,
        description="Mapping of relationship type name to its declared properties.",
    )
    constraints: list[SchemaConstraint] = Field(
        default_factory=list,
        description="Constraints (uniqueness, existence, etc.) defined on the graph.",
    )
    indexes: list[Index] = Field(
        default_factory=list,
        description="Indexes defined on the graph.",
    )


# ---------------------------------------------------------------------------
# Error vocabulary (discriminated union by `category`)
# ---------------------------------------------------------------------------


class ParseError(BaseModel):
    """Cypher syntax error: query does not parse."""

    model_config = ConfigDict(extra="forbid")

    category: Literal["parse"] = Field(
        default="parse",
        description="Discriminator value: always 'parse' for this payload.",
    )
    message: str = Field(..., description="Parser-emitted error message.")
    line: int = Field(..., ge=1, description="1-based line number of the offending region.")
    column: int = Field(..., ge=1, description="1-based column number of the offending region.")
    snippet: str = Field(..., description="Source snippet around the offending region.")
    excerpt_with_caret: str | None = Field(
        default=None,
        description=(
            "IDE-style two-line excerpt: the offending source line, a newline, "
            "then spaces followed by ``^`` under the offending column. ``None`` "
            "when the backend could not produce a reliable (line, column) "
            "position. Format matches what users see in editor error messages."
        ),
    )


class SchemaError(BaseModel):
    """Query references a label, relationship type, or property not in the schema."""

    model_config = ConfigDict(extra="forbid")

    category: Literal["schema"] = Field(
        default="schema",
        description="Discriminator value: always 'schema' for this payload.",
    )
    unknown_reference: str = Field(..., description="The unknown identifier as it appears.")
    reference_kind: Literal["label", "relationship", "property"] = Field(
        ...,
        description="Whether the unknown reference is a label, relationship type, or property.",
    )
    did_you_mean: list[str] = Field(
        default_factory=list,
        description=(
            "Suggested replacements ranked by edit distance over the schema vocabulary. "
            "Empty if no candidates within the configured distance bound."
        ),
    )
    query_context: str = Field(
        ..., description="Surrounding query fragment for the agent's reference."
    )
    available_in_scope: list[str] = Field(
        default_factory=list,
        description=(
            "Full option space available where the unknown reference was used: "
            "all labels for a label miss, all relationship types for a rel miss, "
            "all properties declared on the bound variable's label(s) for a "
            "property miss. Capped at 50 entries alphabetically; "
            "``available_in_scope_truncated`` flags when more exist."
        ),
    )
    available_in_scope_truncated: bool = Field(
        default=False,
        description=(
            "True when the vocabulary at the failure site contained more than "
            "the 50-entry cap and ``available_in_scope`` was truncated. The "
            "corrector should treat the list as a prefix, not a complete set."
        ),
    )


class PropertyError(BaseModel):
    """Property is referenced with a type incompatible with its declaration.

    ``did_you_mean`` is populated when
    the failure is a reference-not-found — the property name does not
    match any declared property on the bound variable's labels — rather
    than a type mismatch. For pure type-mismatch failures, the list
    stays empty.
    """

    model_config = ConfigDict(extra="forbid")

    category: Literal["property"] = Field(
        default="property",
        description="Discriminator value: always 'property' for this payload.",
    )
    property_name: str = Field(..., description="Property whose usage is invalid.")
    declared_type: str = Field(..., description="Type declared for the property in the schema.")
    used_type: str = Field(
        ..., description="Type the query used (inferred from the literal/expression)."
    )
    query_context: str = Field(
        ..., description="Surrounding query fragment for the agent's reference."
    )
    did_you_mean: list[str] = Field(
        default_factory=list,
        description=(
            "Edit-distance suggestions for the property name when the failure is "
            "reference-not-found rather than type-mismatch. Empty for type-mismatch failures."
        ),
    )


class ConstraintError(BaseModel):
    """CREATE/MERGE pattern would violate a uniqueness, existence, or type constraint."""

    model_config = ConfigDict(extra="forbid")

    category: Literal["constraint"] = Field(
        default="constraint",
        description="Discriminator value: always 'constraint' for this payload.",
    )
    constraint_id: str = Field(
        ..., description="Identifier of the constraint that would be violated."
    )
    constraint_kind: Literal["uniqueness", "existence", "type"] = Field(
        ...,
        description="Constraint family violated by the query.",
    )
    property_name: str | None = Field(
        default=None,
        description="Property involved in the violation; None for composite constraints.",
    )
    violating_value: str | None = Field(
        default=None,
        description=(
            "Literal value the violating ``CREATE``/``MERGE`` pattern would write, "
            "when statically detectable (e.g. a literal ``id: 'X'`` that collides "
            "with a uniqueness constraint). ``None`` when the violation can only "
            "be confirmed at runtime against live data."
        ),
    )


class OperatorCost(BaseModel):
    """A single line item in :attr:`CostError.estimated_cost_breakdown`.

    Mirrors :class:`ExplainOperator` but is scoped to the corrector's
    use case (top contributors only, no per-operator argument map). The
    cost gate populates the top 3-5 operators by ``estimated_rows`` so
    the corrector can reason about *which* operators are expensive,
    not just the single ``cost_driver`` summary.
    """

    model_config = ConfigDict(extra="forbid")

    operator: str = Field(
        ...,
        description=(
            "Neo4j operator name with the ``@neo4j`` suffix stripped "
            "(same vocabulary as :attr:`ExplainOperator.name`)."
        ),
    )
    identifiers: list[str] = Field(
        default_factory=list,
        description=(
            "Variable / alias names this operator binds or carries through. "
            "Same vocabulary as :attr:`ExplainOperator.identifiers`."
        ),
    )
    estimated_rows: int = Field(
        ..., ge=0, description="Cardinality estimate the planner attached to this operator."
    )
    estimated_dbhits: int = Field(
        ...,
        ge=0,
        description=(
            "Per-operator db-hit estimate. EXPLAIN proxies this as rows; "
            "see :mod:`cygnet.cost.explain_parser` for the rationale."
        ),
    )


class CostError(BaseModel):
    """Query parses and references valid schema but exceeds the cost threshold."""

    model_config = ConfigDict(extra="forbid")

    category: Literal["cost"] = Field(
        default="cost",
        description="Discriminator value: always 'cost' for this payload.",
    )
    estimated_rows: int = Field(..., ge=0, description="EXPLAIN-estimated row count.")
    estimated_dbhits: int = Field(..., ge=0, description="EXPLAIN-estimated db-hit count.")
    threshold_used: int = Field(
        ..., ge=0, description="Configured threshold the estimate exceeded."
    )
    cost_driver: str = Field(
        ...,
        description=(
            "Plan operator (and associated query fragment) responsible for the high estimate, "
            "e.g. 'AllNodesScan over Movie'."
        ),
    )
    suggested_mitigations: list[str] = Field(
        default_factory=list,
        description=(
            "Actionable refinement suggestions, e.g. 'add LIMIT', 'add index hint on Movie.title', "
            "'decompose into separate queries'."
        ),
    )
    estimated_cost_breakdown: list[OperatorCost] = Field(
        default_factory=list,
        description=(
            "Top contributors to the rejection, ranked by ``estimated_rows`` "
            "descending. Capped at 5 entries by the cost gate. Gives the "
            "corrector concrete per-operator signal beyond the single "
            "``cost_driver`` summary string."
        ),
    )


class EmptyResultError(BaseModel):
    """Query executed against production but returned zero or suspiciously many rows.

    Optional gate, off by default. Post-execution feedback rather than
    pre-execution gating; lives in the vocabulary because callers will want
    to compose it into the agent loop.
    """

    model_config = ConfigDict(extra="forbid")

    category: Literal["empty"] = Field(
        default="empty",
        description="Discriminator value: always 'empty' for this payload.",
    )
    query: str = Field(..., description="The executed query that produced the suspicious result.")
    # expected_range stays a tuple per the architecture spec. JSON round-trips
    # will surface it as a 2-element list and Pydantic will validate it back to
    # a tuple on parse; revisit if YAML/env loading needs special handling.
    expected_range: tuple[int, int] | None = Field(
        default=None,
        description="Optional (min_rows, max_rows) range configured by the caller.",
    )


GateErrorPayload = Annotated[
    ParseError | SchemaError | PropertyError | ConstraintError | CostError | EmptyResultError,
    Field(discriminator="category"),
]


class GateError(BaseModel):
    """Single discriminated error returned by the gate.

    The `category` field on the wrapper mirrors the payload's `category` for
    convenient dispatch by agent code; both are validated to match.
    """

    model_config = ConfigDict(extra="forbid")

    category: Literal["parse", "schema", "property", "constraint", "cost", "empty"] = Field(
        ...,
        description="Category of the error; mirrors `payload.category`.",
    )
    payload: GateErrorPayload = Field(
        ...,
        description="Typed payload carrying the per-category fields.",
    )

    @model_validator(mode="after")
    def _category_matches_payload(self) -> GateError:
        if self.category != self.payload.category:
            raise ValueError(
                f"GateError.category={self.category!r} does not match "
                f"payload.category={self.payload.category!r}"
            )
        return self


# ---------------------------------------------------------------------------
# Gate result types
# ---------------------------------------------------------------------------


StructuralErrorPayload = Annotated[
    ParseError | SchemaError | PropertyError | ConstraintError,
    Field(discriminator="category"),
]


class StructuralValidatorResult(BaseModel):
    """Outcome of the structural validator (parse → schema → property → constraint).

    In short-circuit mode (the default), the chain stops at the first
    failing backend and ``error_payload`` carries that backend's
    payload; ``all_errors`` then contains the same single entry.

    In ``collect_all`` mode every backend that ran contributes its
    payload; ``all_errors`` is a flat list ordered per the public
    contract in ``docs/architecture.md`` (parse-category first, then
    backend authority descending: explain > mirror_execute > ast >
    builtin). ``error_payload`` always equals ``all_errors[0]`` for
    backwards compatibility with callers that consume the
    single-error surface.
    """

    model_config = ConfigDict(extra="forbid")

    passed: bool = Field(..., description="True if every stage passed.")
    failed_stage: Literal["parse", "schema", "property", "constraint", "none"] = Field(
        ...,
        description="Stage that failed, or 'none' if validation passed.",
    )
    error_payload: StructuralErrorPayload | None = Field(
        default=None,
        description=(
            "The first failing backend's typed payload, or None on success. "
            "In ``collect_all`` mode this mirrors ``all_errors[0]``; in "
            "``short_circuit`` mode it carries the single short-circuited "
            "error. Kept for backwards compatibility with the single-error "
            "surface."
        ),
    )
    all_errors: list[StructuralErrorPayload] = Field(
        default_factory=list,
        description=(
            "Every error every backend that ran produced, ordered by the "
            "public contract: parse-category first, then by backend "
            "authority descending (explain > mirror_execute > ast > "
            "builtin). In ``short_circuit`` mode this contains the single "
            "short-circuited payload (or is empty on pass). In "
            "``collect_all`` mode it may contain up to one payload per "
            "configured backend. ``error_payload`` always equals "
            "``all_errors[0]`` when the list is non-empty."
        ),
    )
    collection_mode_used: Literal["short_circuit", "collect_all"] = Field(
        default="short_circuit",
        description=(
            "Which collection mode produced this result. Consumers that "
            "care about whether ``all_errors`` is the full diversity "
            "(``collect_all``) or just the short-circuited single error "
            "(``short_circuit``) can branch on this."
        ),
    )

    @model_validator(mode="after")
    def _payload_consistent_with_stage(self) -> StructuralValidatorResult:
        if self.passed:
            if self.failed_stage != "none" or self.error_payload is not None:
                raise ValueError("passed=True requires failed_stage='none' and error_payload=None")
            if self.all_errors:
                raise ValueError("passed=True requires all_errors to be empty")
        else:
            if self.failed_stage == "none":
                raise ValueError("passed=False requires a specific failed_stage")
            if self.error_payload is None:
                raise ValueError("passed=False requires an error_payload")
            if self.error_payload.category != self.failed_stage:
                raise ValueError(
                    f"failed_stage={self.failed_stage!r} does not match "
                    f"error_payload.category={self.error_payload.category!r}"
                )
            # Auto-populate ``all_errors`` from ``error_payload`` when the
            # caller built the result with only the single-error fields.
            # A single ``error_payload`` always corresponds to
            # ``all_errors=[payload]``.
            if not self.all_errors:
                object.__setattr__(self, "all_errors", [self.error_payload])
            if self.all_errors[0] != self.error_payload:
                raise ValueError(
                    "error_payload must equal all_errors[0]; this invariant "
                    "is the backwards-compatibility contract between the "
                    "single-error and multi-error surfaces."
                )
        return self


class CostGateResult(BaseModel):
    """Outcome of the cost gate. `passed=True` means the query is below threshold."""

    model_config = ConfigDict(extra="forbid")

    passed: bool = Field(..., description="True if the estimated cost is below the threshold.")
    estimated_rows: int = Field(..., ge=0, description="EXPLAIN-estimated row count.")
    estimated_dbhits: int = Field(..., ge=0, description="EXPLAIN-estimated db-hit count.")
    threshold_used: int = Field(..., ge=0, description="Threshold the result was compared against.")
    cost_driver: str | None = Field(
        default=None,
        description=(
            "Operator responsible for the high estimate, formatted as "
            "``<OperatorName> on [<identifiers>]``. None when the gate "
            "passed; ``'explain_failed'`` when the EXPLAIN call itself "
            "failed (typically because the query is not structurally valid)."
        ),
    )
    suggested_mitigations: list[str] = Field(
        default_factory=list,
        description="Suggested refinements; empty when the gate passed.",
    )
    estimated_cost_breakdown: list[OperatorCost] = Field(
        default_factory=list,
        description=(
            "Top contributors to the cost, ranked by ``estimated_rows`` "
            "descending. Capped at 5 entries. Populated whenever a plan "
            "was parsed (pass or fail); empty when EXPLAIN itself failed."
        ),
    )


class GateResult(BaseModel):
    """Combined output of structural + cost gates.

    Cost is None when structural validation failed and the cost gate was
    skipped. `errors` is a flat list for callers that prefer to iterate
    rather than branch on per-stage results.
    """

    model_config = ConfigDict(extra="forbid")

    passed: bool = Field(..., description="True only if every applicable gate stage passed.")
    structural: StructuralValidatorResult = Field(
        ...,
        description="Structural validation outcome.",
    )
    cost: CostGateResult | None = Field(
        default=None,
        description="Cost gate outcome, or None if skipped (e.g. structural failed first).",
    )
    errors: list[GateError] = Field(
        default_factory=list,
        description="Flat list of all gate errors, in the order they were produced.",
    )


# ---------------------------------------------------------------------------
# Corrector result
# ---------------------------------------------------------------------------


class ExplainOperator(BaseModel):
    """A single operator in a parsed Neo4j EXPLAIN plan.

    Produced by ``cygnet.cost.explain_parser.parse_explain_plan`` and
    consumed by the cost gate's cost-driver heuristic. Field names here
    are public API.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        ...,
        description=(
            "Neo4j operator name with the ``@neo4j`` provider suffix "
            "stripped (e.g. ``AllNodesScan``, ``VarLengthExpand(All)``, "
            "``CartesianProduct``)."
        ),
    )
    estimated_rows: int = Field(
        ...,
        ge=0,
        description="Rounded cardinality estimate the planner attached to this operator.",
    )
    estimated_dbhits: int = Field(
        ...,
        ge=0,
        description=(
            "Per-operator db-hit estimate. Neo4j EXPLAIN does not surface "
            "real db-hit estimates (only PROFILE does); the EXPLAIN-based "
            "parser proxies this as the operator's ``estimated_rows`` so "
            "the field is populated."
        ),
    )
    identifiers: list[str] = Field(
        default_factory=list,
        description="Variable / alias names this operator binds or carries through.",
    )
    arguments: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Serialised key/value pairs from the operator's argument map "
            "(e.g. ``Details``, ``EstimatedRows``, ``Id``). Values are "
            "stringified to keep the model JSON-friendly."
        ),
    )


class ExplainPlan(BaseModel):
    """A flattened Neo4j EXPLAIN plan.

    ``operators`` is in depth-first post-order — children appear before
    their parent — so the first entry is the deepest node in the plan
    tree (closest to the data source). The cost-gate cost-driver
    heuristic exploits this ordering for tie-breaking.
    """

    model_config = ConfigDict(extra="forbid")

    operators: list[ExplainOperator] = Field(
        default_factory=list,
        description="Flattened plan operators in depth-first post-order.",
    )
    total_estimated_rows: int = Field(
        ...,
        ge=0,
        description="Final row-count estimate (typically the root operator's value).",
    )
    total_estimated_dbhits: int = Field(
        ...,
        ge=0,
        description=(
            "Sum of per-operator row estimates, used as the db-hit proxy "
            "because EXPLAIN does not surface real db-hits."
        ),
    )


class MirrorBuildReport(BaseModel):
    """Outcome of ``MirrorGraphBuilder.build_from_schema``.

    The mirror's job is structural — one node per declared label,
    one relationship per declared type, declared properties populated
    with type-correct dummy values. The report tells the caller
    exactly what landed and what was skipped so the build is auditable
    without re-introspecting the mirror Neo4j.
    """

    model_config = ConfigDict(extra="forbid")

    nodes_created: int = Field(
        ...,
        ge=0,
        description="Mirror nodes inserted by this build.",
    )
    relationships_created: int = Field(
        ...,
        ge=0,
        description="Mirror relationships inserted by this build.",
    )
    constraints_applied: list[str] = Field(
        default_factory=list,
        description="Identifiers of constraints applied to the mirror Neo4j.",
    )
    constraints_skipped: list[str] = Field(
        default_factory=list,
        description=(
            "Identifiers of constraints that could not be applied, each "
            "annotated with the reason (e.g. ``'compound_formula_required "
            "(Community Edition rejects existence constraints)'``)."
        ),
    )
    indexes_applied: list[str] = Field(
        default_factory=list,
        description="Identifiers of indexes applied to the mirror Neo4j.",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description=(
            "Human-readable warnings emitted during the build. Includes "
            "''Unknown'' endpoint skips, unsupported property types, and "
            "idempotency hits (build called when the mirror already exists)."
        ),
    )


class CorrectorResult(BaseModel):
    """Result of invoking the configured corrector on a failed gate.

    When the model echoes the input verbatim, the corrector returns
    ``action="abort"`` with ``reason="model_echoed_input"`` and
    ``refined_query`` populated with the echoed text so callers can
    record what the model declined to change. Other abort reasons
    (``protocol_failure``, ``empty_cypher_after_retry``,
    ``budget_exceeded``, ``exception``) leave ``refined_query`` at
    ``None``.
    """

    model_config = ConfigDict(extra="forbid")

    action: Literal["refined", "abort", "escalate", "decompose"] = Field(
        ...,
        description=(
            "Corrector's decision: 'refined' (try `refined_query`), 'abort' (give up), "
            "'escalate' (hand to a human/supervisor), or 'decompose' (break the query into parts)."
        ),
    )
    refined_query: str | None = Field(
        default=None,
        description=(
            "The refined Cypher to retry; populated when ``action='refined'`` "
            "OR when ``action='abort'`` with ``reason='model_echoed_input'`` "
            "(in which case the echoed cypher is preserved for record-keeping)."
        ),
    )
    reasoning: str = Field(
        ...,
        description="Human-readable rationale; surfaced to logs and to the agent loop.",
    )
    attempts_used: int = Field(
        ...,
        ge=0,
        description="How many corrector attempts have been spent on this query so far.",
    )
    reason: str | None = Field(
        default=None,
        description=(
            "Structured abort reason. Populated when ``action='abort'``. "
            "Values: ``protocol_failure``, ``empty_cypher_after_retry``, "
            "``model_echoed_input``, ``budget_exceeded``, ``exception``. "
            "``None`` on ``action='refined'``. Analysis can branch on this "
            "to distinguish transport failures from semantic ones."
        ),
    )
    protocol_attempts: int = Field(
        default=0,
        ge=0,
        description=(
            "How many LLM calls the corrector made for this outcome "
            "(including protocol retries). Stays at 0 on the NullCorrector "
            "abort path. A value of 3 means the corrector exhausted its "
            "inner-retry budget."
        ),
    )
    used_high_temp_retry: bool = Field(
        default=False,
        description=(
            "True when the corrector invoked its high-temperature retry "
            "path (after an initial ``ProtocolEmpty`` outcome). Analysis "
            "can use this to compute how often the creative-tier retry "
            "actually produces something."
        ),
    )

    @model_validator(mode="after")
    def _refined_query_invariant(self) -> CorrectorResult:
        if self.action == "refined" and self.refined_query is None:
            raise ValueError("action='refined' requires refined_query to be set")
        if self.action == "abort":
            # ``refined_query`` populated only on the model_echoed_input
            # path; every other abort reason must leave it at None.
            if self.refined_query is not None and self.reason != "model_echoed_input":
                raise ValueError(
                    "action='abort' with refined_query set requires reason='model_echoed_input'"
                )
        elif self.action != "refined" and self.refined_query is not None:
            raise ValueError(f"refined_query must be None when action={self.action!r}")
        return self
