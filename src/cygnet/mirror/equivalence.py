# Copyright (c) 2026 Nikodem Tomczak, Thulge Labs. All rights reserved.

"""Mirror equivalence verification — empirically test the library's
central claim that a mirror behaves the same as the source database
for validation purposes.

For each query in a sample set, validation is run against three
targets:

1. The reference database (the source of truth).
2. A YAML-built mirror (``MirrorGraphBuilder`` from a YAML schema).
3. An introspection-built mirror (introspect the reference, then
   ``MirrorGraphBuilder`` from the introspected ``Schema``).

The three validation outcomes are compared on three dimensions:

- **verdict** — pass or fail.
- **category** — when failed, the validator stage that rejected
  (``parse`` / ``schema`` / ``property`` / ``constraint``).
- **unknown_reference** — when the failure is a schema-category
  one, the specific unknown identifier name.

Out of scope (explicitly):

- EXPLAIN plan shape and operator order.
- Row estimates and cost numbers from EXPLAIN.
- Wall-clock differences.

These quantities differ by design between a tiny constructed mirror
and a real database, and that divergence is not what the equivalence
claim is about. The claim is that *validation outcomes* are the
same.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Final, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from neo4j import Driver

    from cygnet.models import Schema

__all__ = [
    "DEFAULT_SAMPLE_TEMPLATES",
    "EquivalenceDivergence",
    "EquivalenceOutcome",
    "EquivalenceReport",
    "SchemaIntrospectFn",
    "ValidationCallable",
    "verify_mirror_equivalence",
]

SchemaIntrospectFn = Callable[["Driver"], "Schema"]
"""Type alias for the schema-introspection hook used to substitute
identifiers into the default sample-set templates."""

DivergingField = Literal["verdict", "category", "unknown_reference"]


# ---------------------------------------------------------------------------
# Outcome shape
# ---------------------------------------------------------------------------


class EquivalenceOutcome(BaseModel):
    """A validation outcome captured against one target. Three of
    these are compared per query (one per target).

    The shape is deliberately narrow: only the three dimensions that
    define equivalence (verdict, category, unknown_reference) are
    tracked. Everything else the validator returns is dropped.
    """

    model_config = ConfigDict(extra="forbid")

    verdict: Literal["passed", "failed"] = Field(
        ..., description="``passed`` iff the validator chain returned no error."
    )
    category: Literal["parse", "schema", "property", "constraint", "none"] = Field(
        default="none",
        description=(
            "The validator stage that rejected the query, or ``none`` if ``verdict == 'passed'``."
        ),
    )
    unknown_reference: str | None = Field(
        default=None,
        description=(
            "For ``category == 'schema'`` failures, the specific identifier "
            "(label / relationship type / property) that was rejected. "
            "``None`` for other categories."
        ),
    )


# ---------------------------------------------------------------------------
# Divergence + report shape
# ---------------------------------------------------------------------------


class EquivalenceDivergence(BaseModel):
    """One per-query divergence captured by :func:`verify_mirror_equivalence`."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., description="The query that produced the divergence.")
    yaml_mirror_outcome: EquivalenceOutcome
    introspection_mirror_outcome: EquivalenceOutcome
    reference_outcome: EquivalenceOutcome
    diverging_field: DivergingField = Field(
        ...,
        description=(
            "Which dimension diverged. ``verdict`` takes priority when "
            "multiple dimensions differ, since a passed-vs-failed split is "
            "the strongest signal; otherwise ``category``; otherwise "
            "``unknown_reference``."
        ),
    )
    description: str = Field(
        ...,
        description="Human-readable summary of the divergence for the report.",
    )


class EquivalenceReport(BaseModel):
    """Diagnostic report from an equivalence-verification pass."""

    model_config = ConfigDict(extra="forbid")

    timestamp: datetime
    sample_size: int = Field(..., ge=0)
    yaml_mirror_uri: str
    introspection_mirror_uri: str
    reference_uri: str

    queries_checked: int = Field(..., ge=0)
    queries_equivalent: int = Field(..., ge=0)
    skipped_queries: list[str] = Field(
        default_factory=list,
        description=(
            "Template-driven queries that could not be instantiated against "
            "the reference's schema (e.g. no relationships available) and "
            "were therefore skipped. Empty when ``sample_queries`` is "
            "user-supplied."
        ),
    )
    divergences: list[EquivalenceDivergence] = Field(default_factory=list)
    equivalent: bool = Field(..., description="``True`` iff ``divergences`` is empty.")
    fallback_used: bool = Field(
        default=False,
        description=(
            "``True`` when the default sample-set substitution path could "
            "not introspect the reference and fell back to the hard-coded "
            "Movie/Person/ACTED_IN canonical identifiers. Always ``False`` "
            "when the caller supplied ``sample_queries`` explicitly."
        ),
    )

    def summary(self) -> str:
        """One-paragraph human-readable summary."""
        if self.equivalent:
            return (
                f"All {self.queries_checked} sample queries produced the "
                f"same validation outcome across the YAML-built mirror, "
                f"introspection-built mirror, and reference database. "
                f"Equivalent."
            )
        return (
            f"{len(self.divergences)} of {self.queries_checked} queries "
            f"diverged across targets "
            f"({self._diverging_counts()}). Not equivalent."
        )

    def _diverging_counts(self) -> str:
        counts: dict[str, int] = {}
        for d in self.divergences:
            counts[d.diverging_field] = counts.get(d.diverging_field, 0) + 1
        return ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "none"


# ---------------------------------------------------------------------------
# Validation callable + protocol
# ---------------------------------------------------------------------------


class ValidationCallable(Protocol):
    """Signature for the per-target validation hook used by
    :func:`verify_mirror_equivalence`.

    Called once per (target_driver, query) pair. Returns the captured
    :class:`EquivalenceOutcome`. Default implementation
    (:func:`default_validate_via_gate`) constructs a real ``Gate``
    around the driver; tests typically pass a stub.
    """

    def __call__(self, driver: Driver, query: str) -> EquivalenceOutcome: ...


def default_validate_via_gate(driver: Driver, query: str) -> EquivalenceOutcome:
    """Real-validator hook. Introspects the driver to build a
    :class:`Schema`, constructs a validator chain (``builtin → ast →
    explain``) against the driver, runs ``chain.validate(query)``,
    and returns the trimmed equivalence outcome.

    Skips the full :class:`Gate` construction (which requires
    cost-gate + corrector wiring we don't need here) in favour of
    talking to :func:`cygnet.validator.build_chain` directly.

    Notes:

    - Schema introspection happens per call. For amortisation across
      many calls against the same driver, pass a custom ``validate_fn``
      that caches the chain (or the schema).
    - Errors surface verbatim; the caller decides what to do.
    """
    from cygnet.config import ValidatorConfig
    from cygnet.schema.introspect import introspect_schema
    from cygnet.validator import build_chain

    schema = introspect_schema(driver)
    validator_config = ValidatorConfig(backends=["builtin", "ast", "explain"])
    chain = build_chain(validator_config, schema, driver=driver)
    result = chain.validate(query)
    return _outcome_from_validator_result(result)


def _outcome_from_validator_result(result: object) -> EquivalenceOutcome:
    """Translate a :class:`StructuralValidatorResult` into the trimmed
    equivalence outcome. Accepts ``object`` and reads attributes
    duck-typed-ly so the equivalence layer doesn't have a hard import
    dependency on :mod:`cygnet.models` at module-load time."""
    passed = bool(getattr(result, "passed", False))
    failed_stage = str(getattr(result, "failed_stage", "none"))
    category: Literal["parse", "schema", "property", "constraint", "none"] = (
        "none" if passed else failed_stage  # type: ignore[assignment]
    )
    unknown_reference: str | None = None
    payload = getattr(result, "error_payload", None)
    if payload is not None and getattr(payload, "category", None) == "schema":
        ref = getattr(payload, "unknown_reference", None)
        if ref is not None:
            unknown_reference = str(ref)
    return EquivalenceOutcome(
        verdict="passed" if passed else "failed",
        category=category,
        unknown_reference=unknown_reference,
    )


def _driver_uri(driver: Driver) -> str:
    """Best-effort URI extraction for the report. Same logic as
    :func:`cygnet.schema.mirror_check._driver_uri`."""
    addresses = getattr(driver, "addresses", None)
    if addresses:
        first = addresses[0]
        host = getattr(first, "host", None) or getattr(first, "address", None)
        port = getattr(first, "port", None)
        if host and port:
            return f"bolt://{host}:{port}"
        return str(first)
    pool = getattr(driver, "_pool", None)
    address = getattr(pool, "address", None) if pool else None
    if address is not None:
        return f"bolt://{address}"
    return "<unknown>"


# ---------------------------------------------------------------------------
# Default sample-set templates
# ---------------------------------------------------------------------------


DEFAULT_SAMPLE_TEMPLATES: Final[list[dict[str, str]]] = [
    # Success-shape patterns (queries that should pass validation).
    {
        "id": "success_single_node",
        "template": "MATCH (n:{label}) RETURN n LIMIT 1",
    },
    {
        "id": "success_property_predicate",
        "template": "MATCH (n:{label} {{{prop_str}: {literal_str}}}) RETURN n",
    },
    {
        "id": "success_one_hop_path",
        "template": "MATCH (a:{label})-[r:{rel}]->(b) RETURN a, b LIMIT 1",
    },
    {
        "id": "success_path_with_where",
        "template": (
            "MATCH (a:{label})-[:{rel}]->(b) WHERE a.{prop_str} = {literal_str} RETURN b LIMIT 5"
        ),
    },
    {
        "id": "success_var_length_path",
        "template": "MATCH (a:{label})-[:{rel}*1..3]->(b) RETURN b LIMIT 1",
    },
    {
        "id": "success_count_aggregation",
        "template": "MATCH (n:{label}) RETURN count(n) AS total",
    },
    {
        "id": "success_group_by",
        "template": (
            "MATCH (n:{label}) RETURN n.{prop_str}, count(*) AS c ORDER BY c DESC LIMIT 5"
        ),
    },
    {
        "id": "success_order_by_limit",
        "template": "MATCH (n:{label}) RETURN n ORDER BY n.{prop_str} LIMIT 10",
    },
    {
        "id": "success_optional_match",
        "template": ("MATCH (a:{label}) OPTIONAL MATCH (a)-[:{rel}]->(b) RETURN a, b LIMIT 5"),
    },
    # Failure-shape patterns: each should fail with a specific category.
    {
        "id": "fail_unknown_label",
        "template": "MATCH (n:__cygnet_nonexistent_xyz__) RETURN n",
    },
    {
        "id": "fail_unknown_relationship_type",
        "template": "MATCH (a:{label})-[:__cygnet_nonexistent_rel__]->(b) RETURN b",
    },
    {
        "id": "fail_unknown_property",
        "template": ("MATCH (n:{label}) WHERE n.__cygnet_nonexistent_prop__ = 1 RETURN n"),
    },
    {
        "id": "fail_parse_missing_paren",
        "template": "MATCH (n:{label} RETURN n",
    },
    {
        "id": "fail_parse_missing_keyword",
        "template": "MATCH n:{label} RETURN n",
    },
    # Edge-case patterns: should pass.
    {
        "id": "edge_self_loop",
        "template": "MATCH (a:{label})-[:{rel}]->(a) RETURN a LIMIT 1",
    },
    {
        "id": "edge_pattern_comprehension",
        "template": (
            "MATCH (a:{label}) RETURN [(a)-[:{rel}]->(b) | b.{prop_str}] AS items LIMIT 1"
        ),
    },
    {
        "id": "edge_case_when",
        "template": (
            "MATCH (n:{label}) "
            "RETURN CASE WHEN n.{prop_str} IS NULL THEN 'unknown' "
            "ELSE 'known' END AS state LIMIT 5"
        ),
    },
]
"""Default 20-query sample set as `(id, template)` pairs.

Templates use Python str-format with named slots `{label}`, `{rel}`,
`{prop_str}`, `{literal_str}`. :func:`_instantiate_sample_set` reads
the reference's schema and substitutes appropriate values; templates
referencing unavailable substitutions are skipped.
"""


def _instantiate_sample_set(
    *,
    reference_driver: Driver | None,
    schema_introspect_fn: SchemaIntrospectFn | None,
) -> tuple[list[str], list[str], bool]:
    """Substitute reference-schema identifiers into the default
    templates. Returns ``(realised_queries, skipped_ids, fallback_used)``.

    Falls back to a hard-coded canonical Movie/Person/ACTED_IN schema
    when ``reference_driver`` is ``None`` or when introspection raises.
    """
    label: str | None = None
    rel: str | None = None
    prop_str: str | None = None
    literal_str: str | None = None

    if reference_driver is not None and schema_introspect_fn is not None:
        try:
            schema = schema_introspect_fn(reference_driver)
            label, rel, prop_str, literal_str = _pick_template_slots(schema)
        except Exception:
            # Introspection failed; fall back to canonical defaults so
            # the operator still gets *something* to inspect.
            pass

    fallback_used = label is None
    label = label or "Movie"
    rel = rel or "ACTED_IN"
    prop_str = prop_str or "title"
    literal_str = literal_str or '"_cygnet_canary_value_"'

    realised: list[str] = []
    skipped: list[str] = []
    for entry in DEFAULT_SAMPLE_TEMPLATES:
        try:
            realised.append(
                entry["template"].format(
                    label=label,
                    rel=rel,
                    prop_str=prop_str,
                    literal_str=literal_str,
                )
            )
        except (KeyError, IndexError):
            skipped.append(entry["id"])
    return realised, skipped, fallback_used


def _pick_template_slots(schema: Schema) -> tuple[str, str, str, str]:
    """Pick a label, relationship type, string-typed property, and
    a typed literal from a :class:`Schema` for template substitution.

    Picks deterministically (sorted names) so the realised sample
    is stable across runs against the same schema.
    """
    labels = sorted(node_label.name for node_label in schema.labels)
    if not labels:
        raise ValueError("schema has no labels")
    label_name = labels[0]
    rels = sorted(r.name for r in schema.relationship_types)
    rel = rels[0] if rels else "_CYGNET_NO_REL_"
    props = schema.properties_by_label.get(label_name, [])
    string_prop = next(
        (p.name for p in props if p.type.upper().startswith("STRING")),
        None,
    )
    if string_prop is None and props:
        string_prop = props[0].name
    prop_str = string_prop or "name"
    literal_str = '"_cygnet_canary_value_"'
    return label_name, rel, prop_str, literal_str


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def verify_mirror_equivalence(
    yaml_mirror_driver: Driver,
    introspection_mirror_driver: Driver,
    reference_driver: Driver,
    *,
    sample_queries: list[str] | None = None,
    validate_fn: ValidationCallable | None = None,
    schema_introspect_fn: SchemaIntrospectFn | None = None,
) -> EquivalenceReport:
    """Verify that three validation targets agree on every sample query.

    Args:
        yaml_mirror_driver: Driver pointing at a mirror built from a
            YAML schema spec via ``MirrorGraphBuilder``.
        introspection_mirror_driver: Driver pointing at a mirror built
            from an introspected schema (introspect the reference,
            then ``MirrorGraphBuilder`` from that schema).
        reference_driver: Driver pointing at the source database.
        sample_queries: Optional explicit list of queries to run.
            When ``None``, :data:`DEFAULT_SAMPLE_TEMPLATES` is
            substituted against the reference's schema.
        validate_fn: Per-(driver, query) validation hook. Defaults to
            :func:`default_validate_via_gate`. Tests typically pass a
            stub.
        schema_introspect_fn: Hook used by the default-sample-set
            substitution path. Defaults to
            :func:`cygnet.schema.introspect.introspect_schema`. Tests
            can inject a mock.

    Returns:
        :class:`EquivalenceReport` enumerating any divergences. The
        report's ``equivalent`` flag is ``True`` iff all queries
        produced identical outcomes across all three targets.
    """
    validate = validate_fn or default_validate_via_gate

    if schema_introspect_fn is None:
        # Lazy import — keeps the module import-light and lets tests
        # supply a stub without importing the real introspect.
        from cygnet.schema.introspect import introspect_schema as _introspect

        schema_introspect_fn = _introspect

    if sample_queries is None:
        queries, skipped, fallback_used = _instantiate_sample_set(
            reference_driver=reference_driver,
            schema_introspect_fn=schema_introspect_fn,
        )
    else:
        queries = list(sample_queries)
        skipped = []
        fallback_used = False

    divergences: list[EquivalenceDivergence] = []
    equivalent_count = 0

    for query in queries:
        yaml_outcome = validate(yaml_mirror_driver, query)
        intro_outcome = validate(introspection_mirror_driver, query)
        ref_outcome = validate(reference_driver, query)
        divergence = _compare_outcomes(query, yaml_outcome, intro_outcome, ref_outcome)
        if divergence is None:
            equivalent_count += 1
        else:
            divergences.append(divergence)

    return EquivalenceReport(
        timestamp=datetime.now(UTC),
        sample_size=len(queries) + len(skipped),
        yaml_mirror_uri=_driver_uri(yaml_mirror_driver),
        introspection_mirror_uri=_driver_uri(introspection_mirror_driver),
        reference_uri=_driver_uri(reference_driver),
        queries_checked=len(queries),
        queries_equivalent=equivalent_count,
        skipped_queries=skipped,
        divergences=divergences,
        equivalent=not divergences,
        fallback_used=fallback_used,
    )


def _compare_outcomes(
    query: str,
    yaml_outcome: EquivalenceOutcome,
    introspection_outcome: EquivalenceOutcome,
    reference_outcome: EquivalenceOutcome,
) -> EquivalenceDivergence | None:
    """Compare three outcomes on verdict, category, unknown_reference.
    Return a :class:`EquivalenceDivergence` if any dimension diverges;
    ``None`` if all three agree."""
    verdicts = {yaml_outcome.verdict, introspection_outcome.verdict, reference_outcome.verdict}
    if len(verdicts) > 1:
        return EquivalenceDivergence(
            query=query,
            yaml_mirror_outcome=yaml_outcome,
            introspection_mirror_outcome=introspection_outcome,
            reference_outcome=reference_outcome,
            diverging_field="verdict",
            description=(
                "verdict diverges: "
                f"yaml={yaml_outcome.verdict} "
                f"intro={introspection_outcome.verdict} "
                f"reference={reference_outcome.verdict}"
            ),
        )
    categories = {
        yaml_outcome.category,
        introspection_outcome.category,
        reference_outcome.category,
    }
    if len(categories) > 1:
        return EquivalenceDivergence(
            query=query,
            yaml_mirror_outcome=yaml_outcome,
            introspection_mirror_outcome=introspection_outcome,
            reference_outcome=reference_outcome,
            diverging_field="category",
            description=(
                "category diverges: "
                f"yaml={yaml_outcome.category} "
                f"intro={introspection_outcome.category} "
                f"reference={reference_outcome.category}"
            ),
        )
    refs = {
        yaml_outcome.unknown_reference,
        introspection_outcome.unknown_reference,
        reference_outcome.unknown_reference,
    }
    if len(refs) > 1:
        return EquivalenceDivergence(
            query=query,
            yaml_mirror_outcome=yaml_outcome,
            introspection_mirror_outcome=introspection_outcome,
            reference_outcome=reference_outcome,
            diverging_field="unknown_reference",
            description=(
                "unknown_reference diverges: "
                f"yaml={yaml_outcome.unknown_reference!r} "
                f"intro={introspection_outcome.unknown_reference!r} "
                f"reference={reference_outcome.unknown_reference!r}"
            ),
        )
    return None
