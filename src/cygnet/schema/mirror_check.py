# Copyright (c) 2026 Nikodem Tomczak, Thulge Labs. All rights reserved.

"""Mirror schema validation — compare a running Neo4j mirror against
its declared :class:`cygnet.Schema`.

Diagnostic only. Reports every divergence between the declared schema
(typically loaded from a YAML spec via :func:`load_schema_spec`) and
the live mirror's introspectable state. Does not fix anything; the
caller decides what to do with the report.

Divergence categories:

- Label declared in spec but absent from the mirror.
- Label present in the mirror but not declared in spec.
- Relationship type declared but absent.
- Relationship type present but not declared.
- Property declared on a label or relationship type but never observed
  in a per-entity property-key sample (``LIMIT 50``).
- Property observed on instances but not declared.
- Constraint declared but absent from ``SHOW CONSTRAINTS``.
- Constraint present in ``SHOW CONSTRAINTS`` but not declared.
- Index declared but absent from ``SHOW INDEXES`` (LOOKUP and
  constraint-owned indexes are excluded — these are auto-managed by
  Neo4j and are not part of the user-authored schema).
- Index present in the mirror but not declared.

Sparsity edge case: a YAML-declared ``sparse=True`` property whose
sample of size :data:`_PROPERTY_SAMPLE_LIMIT` finds zero instances is
still reported as missing. The whole point of this module is to flag
*declared but never written* properties. The seed loader is expected
to populate at least one instance of each declared property; a real
"too sparse to observe in 50 samples" property should bump the sample
size locally rather than be exempted globally.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Final, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from neo4j import Driver

    from cygnet.models import Schema


class _ObservedConstraint(TypedDict):
    """Shape of an item returned by :func:`_observed_constraints`."""

    name: str
    type: str
    label_or_rel: str
    property: str | None


class _ObservedIndex(TypedDict):
    """Shape of an item returned by :func:`_observed_indexes`."""

    name: str
    type: str
    label_or_rel: str
    properties: list[str]


__all__ = [
    "MirrorValidationReport",
    "MissingItem",
    "UnexpectedItem",
    "validate_mirror",
]

ItemKind = Literal["label", "relationship_type", "property", "constraint", "index"]

_PROPERTY_SAMPLE_LIMIT: Final[int] = 50
"""How many node/relationship instances to scan when collecting the
observed property-key set per entity. Trades off accuracy against
introspection cost on very large mirrors. Tunable via
:func:`validate_mirror`'s ``property_sample_limit`` argument."""


# ---------------------------------------------------------------------------
# Report models
# ---------------------------------------------------------------------------


class MissingItem(BaseModel):
    """A schema element declared in the spec but absent from the mirror."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Element identifier (label name, property name, etc).")
    kind: ItemKind = Field(..., description="What kind of element this is.")
    parent: str | None = Field(
        default=None,
        description=(
            "Owning entity for properties (the label or relationship type the "
            "property is declared on). ``None`` for top-level elements."
        ),
    )
    detail: str | None = Field(
        default=None,
        description="Optional free-text detail (e.g. constraint type + property tuple).",
    )


class UnexpectedItem(BaseModel):
    """A schema element present in the mirror but not declared in the spec."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Element identifier.")
    kind: ItemKind = Field(..., description="What kind of element this is.")
    parent: str | None = Field(
        default=None,
        description="Owning entity for properties. ``None`` for top-level elements.",
    )
    detail: str | None = Field(default=None, description="Optional free-text detail.")


class MirrorValidationReport(BaseModel):
    """Diagnostic report from a mirror-vs-spec validation pass."""

    model_config = ConfigDict(extra="forbid")

    timestamp: datetime = Field(
        ...,
        description="When the validation pass ran (UTC).",
    )
    schema_source: str = Field(
        ...,
        description=(
            "Path to the YAML/JSON spec (or an in-memory marker like "
            "``'<in-memory>'`` when the caller passed a programmatically "
            "constructed Schema)."
        ),
    )
    mirror_uri: str = Field(
        ...,
        description="Bolt URI of the validated mirror.",
    )
    database: str = Field(
        ...,
        description="Neo4j database name (typically ``'neo4j'``).",
    )
    declared_labels: int = Field(..., ge=0)
    observed_labels: int = Field(..., ge=0)
    declared_relationship_types: int = Field(..., ge=0)
    observed_relationship_types: int = Field(..., ge=0)
    property_sample_limit: int = Field(
        ...,
        ge=1,
        description="Per-entity ``LIMIT`` used when sampling property keys.",
    )
    missing: list[MissingItem] = Field(default_factory=list)
    unexpected: list[UnexpectedItem] = Field(default_factory=list)
    ok: bool = Field(
        ...,
        description="True iff both ``missing`` and ``unexpected`` are empty.",
    )

    def summary(self) -> str:
        """One-paragraph human-readable summary."""
        if self.ok:
            return (
                f"Mirror at {self.mirror_uri} matches the declared schema "
                f"({self.declared_labels} labels, "
                f"{self.declared_relationship_types} relationship types). "
                "No divergences."
            )
        kinds_missing = _counts_by_kind(self.missing)
        kinds_unexpected = _counts_by_kind(self.unexpected)
        return (
            f"Mirror at {self.mirror_uri} diverges from the declared schema. "
            f"{len(self.missing)} missing items ({_fmt_kind_counts(kinds_missing)}); "
            f"{len(self.unexpected)} unexpected items "
            f"({_fmt_kind_counts(kinds_unexpected)})."
        )


def _counts_by_kind(items: list[MissingItem] | list[UnexpectedItem]) -> dict[str, int]:
    out: dict[str, int] = {}
    for item in items:
        out[item.kind] = out.get(item.kind, 0) + 1
    return out


def _fmt_kind_counts(counts: dict[str, int]) -> str:
    if not counts:
        return "none"
    return ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def validate_mirror(
    schema: Schema,
    driver: Driver,
    *,
    database: str = "neo4j",
    schema_source: str = "<in-memory>",
    property_sample_limit: int = _PROPERTY_SAMPLE_LIMIT,
) -> MirrorValidationReport:
    """Compare ``schema`` against the live mirror reachable via ``driver``.

    The comparison is purely diagnostic — no mutations are issued to
    the mirror. The function performs read-only introspection queries
    (``CALL db.labels()``, ``CALL db.relationshipTypes()``,
    per-entity ``keys()`` samples, ``SHOW CONSTRAINTS``,
    ``SHOW INDEXES``) and returns a :class:`MirrorValidationReport`.

    Args:
        schema: the declared schema, typically loaded from YAML via
            :func:`cygnet.schema.load_schema_spec`.
        driver: an already-connected ``neo4j.Driver``. Caller owns
            lifecycle.
        database: the Neo4j database to introspect. Defaults to
            ``"neo4j"``.
        schema_source: a label for the report identifying where the
            schema was loaded from (e.g. a YAML file path). Defaults
            to ``"<in-memory>"`` for programmatic callers.
        property_sample_limit: ``LIMIT`` on the per-entity property-key
            sample. Defaults to :data:`_PROPERTY_SAMPLE_LIMIT`.

    Returns:
        A :class:`MirrorValidationReport` enumerating every divergence.
    """
    observed_labels = _observed_labels(driver, database)
    observed_rel_types = _observed_relationship_types(driver, database)
    observed_props_by_label = _sample_properties_by_label(
        driver, database, observed_labels, property_sample_limit
    )
    observed_props_by_rel = _sample_properties_by_rel_type(
        driver, database, observed_rel_types, property_sample_limit
    )
    observed_constraints = _observed_constraints(driver, database)
    observed_indexes = _observed_indexes(driver, database)

    missing: list[MissingItem] = []
    unexpected: list[UnexpectedItem] = []

    declared_label_names = {label.name for label in schema.labels}
    declared_rel_names = {rel.name for rel in schema.relationship_types}

    # Labels.
    for name in declared_label_names - observed_labels:
        missing.append(MissingItem(name=name, kind="label"))
    for name in observed_labels - declared_label_names:
        unexpected.append(UnexpectedItem(name=name, kind="label"))

    # Relationship types.
    for name in declared_rel_names - observed_rel_types:
        missing.append(MissingItem(name=name, kind="relationship_type"))
    for name in observed_rel_types - declared_rel_names:
        unexpected.append(UnexpectedItem(name=name, kind="relationship_type"))

    # Properties on labels — only compare for labels that exist on
    # both sides; missing labels are already reported above.
    for label_name in declared_label_names & observed_labels:
        declared_props = {p.name for p in schema.properties_by_label.get(label_name, [])}
        observed_props = observed_props_by_label.get(label_name, set())
        for prop in declared_props - observed_props:
            missing.append(MissingItem(name=prop, kind="property", parent=label_name))
        for prop in observed_props - declared_props:
            unexpected.append(UnexpectedItem(name=prop, kind="property", parent=label_name))

    # Properties on relationship types.
    for rel_name in declared_rel_names & observed_rel_types:
        declared_props = {p.name for p in schema.properties_by_rel_type.get(rel_name, [])}
        observed_props = observed_props_by_rel.get(rel_name, set())
        for prop in declared_props - observed_props:
            missing.append(MissingItem(name=prop, kind="property", parent=rel_name))
        for prop in observed_props - declared_props:
            unexpected.append(UnexpectedItem(name=prop, kind="property", parent=rel_name))

    # Constraints — compare by (type, label_or_rel, property) tuple so
    # the report is independent of constraint identifiers (which Neo4j
    # auto-generates when the spec doesn't name them).
    declared_constraint_keys = {
        _constraint_key(c.type, c.label_or_rel, c.property): c for c in schema.constraints
    }
    observed_constraint_keys = {
        _constraint_key(c["type"], c["label_or_rel"], c["property"]): c
        for c in observed_constraints
    }
    for key, decl in declared_constraint_keys.items():
        if key not in observed_constraint_keys:
            missing.append(
                MissingItem(
                    name=decl.identifier,
                    kind="constraint",
                    parent=decl.label_or_rel,
                    detail=_constraint_detail(decl.type, decl.property),
                )
            )
    for key, obs in observed_constraint_keys.items():
        if key not in declared_constraint_keys:
            unexpected.append(
                UnexpectedItem(
                    name=obs["name"],
                    kind="constraint",
                    parent=obs["label_or_rel"],
                    detail=_constraint_detail(obs["type"], obs["property"]),
                )
            )

    # Indexes — compare by (type, label_or_rel, tuple(properties)).
    declared_index_keys = {
        _index_key(ix.type, ix.label_or_rel, ix.properties): ix for ix in schema.indexes
    }
    observed_index_keys = {
        _index_key(ix["type"], ix["label_or_rel"], ix["properties"]): ix for ix in observed_indexes
    }
    for ix_key, decl_ix in declared_index_keys.items():
        if ix_key not in observed_index_keys:
            missing.append(
                MissingItem(
                    name=f"{decl_ix.type}({','.join(decl_ix.properties)})",
                    kind="index",
                    parent=decl_ix.label_or_rel,
                    detail=f"type={decl_ix.type} properties={decl_ix.properties}",
                )
            )
    for ix_key, obs_ix in observed_index_keys.items():
        if ix_key not in declared_index_keys:
            unexpected.append(
                UnexpectedItem(
                    name=obs_ix["name"],
                    kind="index",
                    parent=obs_ix["label_or_rel"],
                    detail=f"type={obs_ix['type']} properties={obs_ix['properties']}",
                )
            )

    return MirrorValidationReport(
        timestamp=datetime.now(UTC),
        schema_source=schema_source,
        mirror_uri=_driver_uri(driver),
        database=database,
        declared_labels=len(declared_label_names),
        observed_labels=len(observed_labels),
        declared_relationship_types=len(declared_rel_names),
        observed_relationship_types=len(observed_rel_types),
        property_sample_limit=property_sample_limit,
        missing=missing,
        unexpected=unexpected,
        ok=not missing and not unexpected,
    )


# ---------------------------------------------------------------------------
# Introspection helpers
# ---------------------------------------------------------------------------


def _observed_labels(driver: Driver, database: str) -> set[str]:
    with driver.session(database=database) as session:
        rows = session.run("CALL db.labels() YIELD label RETURN label")
        return {row["label"] for row in rows}


def _observed_relationship_types(driver: Driver, database: str) -> set[str]:
    with driver.session(database=database) as session:
        rows = session.run(
            "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType"
        )
        return {row["relationshipType"] for row in rows}


def _sample_properties_by_label(
    driver: Driver,
    database: str,
    labels: set[str],
    limit: int,
) -> dict[str, set[str]]:
    """Aggregate the union of property-key sets observed on up to
    ``limit`` instances of each label. Labels with zero instances get
    an empty set."""
    out: dict[str, set[str]] = {}
    with driver.session(database=database) as session:
        for label in labels:
            # Backtick the label to handle reserved words / unusual chars.
            cypher = f"MATCH (n:`{label}`) RETURN keys(n) AS props LIMIT {int(limit)}"
            keys: set[str] = set()
            for row in session.run(cypher):
                for key in row["props"] or []:
                    keys.add(key)
            out[label] = keys
    return out


def _sample_properties_by_rel_type(
    driver: Driver,
    database: str,
    rel_types: set[str],
    limit: int,
) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    with driver.session(database=database) as session:
        for rel in rel_types:
            cypher = f"MATCH ()-[r:`{rel}`]->() RETURN keys(r) AS props LIMIT {int(limit)}"
            keys: set[str] = set()
            for row in session.run(cypher):
                for key in row["props"] or []:
                    keys.add(key)
            out[rel] = keys
    return out


def _observed_constraints(driver: Driver, database: str) -> list[_ObservedConstraint]:
    """Read ``SHOW CONSTRAINTS`` and flatten to a comparable shape.

    Composite-key constraints (multiple properties) are flattened into
    one entry per property — same convention :func:`_read_constraints`
    in :mod:`cygnet.schema.introspect` uses (the first property only),
    so this stays consistent with the rest of the system. Constraints
    with no labels/types are dropped.
    """
    out: list[_ObservedConstraint] = []
    with driver.session(database=database) as session:
        rows = list(session.run("SHOW CONSTRAINTS"))
    for row in rows:
        labels_or_types = row.get("labelsOrTypes") or []
        properties = row.get("properties") or []
        if not labels_or_types:
            continue
        first_prop = properties[0] if properties else None
        out.append(
            _ObservedConstraint(
                name=str(row["name"]),
                type=str(row["type"]),
                label_or_rel=str(labels_or_types[0]),
                property=str(first_prop) if first_prop is not None else None,
            )
        )
    return out


def _observed_indexes(driver: Driver, database: str) -> list[_ObservedIndex]:
    """Read ``SHOW INDEXES`` excluding LOOKUP and constraint-owned
    indexes (those are auto-managed by Neo4j and not part of the
    user-authored schema)."""
    out: list[_ObservedIndex] = []
    with driver.session(database=database) as session:
        rows = list(session.run("SHOW INDEXES WHERE type <> 'LOOKUP' AND owningConstraint IS NULL"))
    for row in rows:
        labels_or_types = row.get("labelsOrTypes") or []
        properties = row.get("properties") or []
        if not labels_or_types or not properties:
            continue
        out.append(
            _ObservedIndex(
                name=str(row["name"]),
                type=str(row["type"]),
                label_or_rel=str(labels_or_types[0]),
                properties=[str(p) for p in properties],
            )
        )
    return out


# ---------------------------------------------------------------------------
# Key normalisation
# ---------------------------------------------------------------------------


def _constraint_key(type_: str, label_or_rel: str, property_: str | None) -> tuple[str, str, str]:
    """Identity tuple for constraint comparison. ``None`` properties
    become an empty string so the tuple stays hashable."""
    return (type_.upper(), label_or_rel, property_ or "")


def _constraint_detail(type_: str, property_: str | None) -> str:
    return f"type={type_} property={property_!r}"


_INDEX_TYPE_ALIASES: Final[dict[str, str]] = {
    # Neo4j 5 renamed BTREE to RANGE (same B-tree-backed implementation).
    # Specs authored against either vocabulary compare equal.
    "BTREE": "RANGE",
    "RANGE": "RANGE",
}


def _index_key(
    type_: str, label_or_rel: str, properties: list[str]
) -> tuple[str, str, tuple[str, ...]]:
    """Identity tuple for index comparison. Normalises the Neo4j 4
    name ``BTREE`` to its Neo4j 5 equivalent ``RANGE`` so YAML specs
    written against either version round-trip cleanly."""
    type_upper = type_.upper()
    normalised = _INDEX_TYPE_ALIASES.get(type_upper, type_upper)
    return (normalised, label_or_rel, tuple(properties))


def _driver_uri(driver: Driver) -> str:
    """Best-effort extraction of the driver's connection URI for the
    report. The neo4j-python driver exposes this differently across
    versions; fall back to ``'<unknown>'`` if not reachable."""
    for attr in ("addresses", "_pool"):
        value = getattr(driver, attr, None)
        if value is None:
            continue
        if attr == "addresses" and value:
            first = value[0]
            host = getattr(first, "host", None) or getattr(first, "address", None)
            port = getattr(first, "port", None)
            if host and port:
                return f"bolt://{host}:{port}"
            return str(first)
    # Last resort: neo4j 5.x exposes the original URI on the pool's
    # address attribute.
    pool = getattr(driver, "_pool", None)
    address = getattr(pool, "address", None) if pool else None
    if address is not None:
        return f"bolt://{address}"
    return "<unknown>"
