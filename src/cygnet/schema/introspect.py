# Copyright (c) 2026 Nikodem Tomczak, Thulge Labs. All rights reserved.

"""Path A schema introspection — build a ``Schema`` from a live Neo4j.

Two strategies are implemented; both produce the same ``Schema``
Pydantic model that ``load_schema_spec`` produces from a YAML/JSON
file. Path A and Path B are fully orthogonal: downstream code does
not care which source produced the Schema.

**APOC strategy (default).** Single ``CALL apoc.meta.schema()``
round-trip. Preferred when available because it's one server call.

**Vanilla strategy.** Falls back when APOC is absent. Uses the stock
Neo4j 5 procedures (``CALL db.labels()``, ``CALL db.relationshipTypes()``,
``CALL db.schema.nodeTypeProperties()``, ``CALL db.schema.relTypeProperties()``,
``SHOW CONSTRAINTS``, ``SHOW INDEXES``).

Both strategies emit a follow-up frequency query per label and per
relationship type to populate ``Property.sparse``; the threshold is
parameterised (default 0.5, matching the brief).

Documented quirks:

- **Property-type vocabulary normalisation.** APOC and vanilla report
  type names differently (``DATE_TIME`` vs ``DateTime``). Both are
  normalised to the canonical names used by ``docs/schema_spec_format_proposals.md``
  (``STRING``, ``INTEGER``, ``FLOAT``, ``BOOLEAN``, ``DATE``,
  ``DATETIME``, ...). Unrecognised type names pass through verbatim.
- **Vanilla property-type ambiguity.** ``db.schema.nodeTypeProperties``
  returns ``propertyTypes`` as a list because a property's observed
  type can vary across instances (e.g. ``[String, Long]``). Per the
  brief, the union is stored as a comma-separated string; a real
  deployment hitting this often probably has data-quality issues.
- **Multi-label relationships.** A relationship type may appear with
  multiple ``(source, target)`` label pairs (e.g. when source nodes
  carry marker labels). Our ``RelationshipType`` model allows a
  single source / target, so we pick the highest-count pair via a
  ``count(*)`` query. The choice is recorded in the relationship's
  source/target labels; alternate combinations are silently dropped
  (a future schema-model change could carry multiple).
- **LOOKUP and constraint-owned indexes are skipped.** Neo4j creates
  these automatically; they're not part of the user-authored schema.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Final

from neo4j.exceptions import ClientError

from cygnet.models import (
    Index,
    NodeLabel,
    Property,
    RelationshipType,
    Schema,
    SchemaConstraint,
)

if TYPE_CHECKING:
    from neo4j import Driver

__all__ = ["SchemaIntrospectionError", "introspect_schema"]

_log = logging.getLogger(__name__)


class SchemaIntrospectionError(Exception):
    """Raised when schema introspection fails unrecoverably (database
    unreachable, both strategies fail, malformed response)."""


# ---------------------------------------------------------------------------
# Type-vocabulary normalisation
# ---------------------------------------------------------------------------

_TYPE_NORMALIZE: Final[dict[str, str]] = {
    # Vanilla (db.schema.nodeTypeProperties returns PascalCase).
    "String": "STRING",
    "Long": "INTEGER",
    "Double": "FLOAT",
    "Float": "FLOAT",
    "Boolean": "BOOLEAN",
    "Date": "DATE",
    "Time": "TIME",
    "DateTime": "DATETIME",
    "LocalDateTime": "LOCAL_DATETIME",
    "LocalTime": "TIME",
    "Duration": "DURATION",
    "Point": "POINT",
    "StringArray": "LIST",
    "LongArray": "LIST",
    "DoubleArray": "LIST",
    # APOC (uppercase with underscores).
    "STRING": "STRING",
    "INTEGER": "INTEGER",
    "FLOAT": "FLOAT",
    "BOOLEAN": "BOOLEAN",
    "DATE": "DATE",
    "TIME": "TIME",
    "DATE_TIME": "DATETIME",
    "LOCAL_DATE_TIME": "LOCAL_DATETIME",
    "LOCAL_TIME": "TIME",
    "DURATION": "DURATION",
    "POINT": "POINT",
    "LIST": "LIST",
    "MAP": "MAP",
}


def _normalize_type(name: str) -> str:
    """Map Neo4j's type vocabulary to the canonical CYGNET vocabulary.
    Unrecognised names pass through verbatim (permissive stance)."""
    return _TYPE_NORMALIZE.get(name, name)


def _normalize_type_list(names: list[str] | None) -> str:
    """Join multi-typed vanilla output as a comma-separated string."""
    if not names:
        return "UNKNOWN"
    normalised = [_normalize_type(n) for n in names]
    # Deduplicate while preserving order.
    seen: list[str] = []
    for t in normalised:
        if t not in seen:
            seen.append(t)
    return ",".join(seen)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def introspect_schema(
    driver: Driver,
    database: str = "neo4j",
    *,
    use_apoc: bool = True,
    sparsity_threshold: float = 0.5,
) -> Schema:
    """Build a ``Schema`` from a live Neo4j instance.

    Args:
        driver: an already-connected ``neo4j.Driver``. Caller owns
            lifecycle (typically ``Gate`` owns it).
        database: Neo4j database name. Defaults to ``"neo4j"``.
        use_apoc: when ``True`` (default), try ``CALL apoc.meta.schema()``
            first and fall back to the vanilla strategy with a logged
            warning if APOC is missing. When ``False``, skip APOC.
        sparsity_threshold: a property whose observed frequency on its
            label is below this is marked ``sparse=True``. Default 0.5.

    Raises:
        SchemaIntrospectionError: when the driver is unreachable or
            both strategies fail.
    """
    if use_apoc:
        try:
            return _apoc_introspect(driver, database, sparsity_threshold)
        except ClientError as exc:
            if _is_procedure_not_found(exc):
                _log.warning(
                    "APOC procedure apoc.meta.schema is not available; "
                    "falling back to vanilla introspection. Install APOC "
                    "or pass use_apoc=False to silence this warning."
                )
            else:
                raise SchemaIntrospectionError(
                    f"APOC introspection failed with an unrecognised error: {exc}"
                ) from exc
        except Exception as exc:
            raise SchemaIntrospectionError(f"APOC introspection failed: {exc}") from exc
    try:
        return _vanilla_introspect(driver, database, sparsity_threshold)
    except Exception as exc:
        raise SchemaIntrospectionError(f"Vanilla introspection failed: {exc}") from exc


def _is_procedure_not_found(exc: ClientError) -> bool:
    code = getattr(exc, "code", "") or ""
    if "ProcedureNotFound" in code:
        return True
    message = str(getattr(exc, "message", None) or exc)
    return "apoc.meta.schema" in message and "no procedure" in message.lower()


# ---------------------------------------------------------------------------
# APOC strategy
# ---------------------------------------------------------------------------


def _apoc_introspect(driver: Driver, database: str, sparsity_threshold: float) -> Schema:
    with driver.session(database=database) as session:
        record = session.run("CALL apoc.meta.schema() YIELD value RETURN value").single()
    if record is None:
        raise SchemaIntrospectionError(
            "apoc.meta.schema() returned no rows; database is empty or APOC is misconfigured."
        )
    data: dict[str, Any] = record["value"]

    node_entries: dict[str, dict[str, Any]] = {
        name: entry for name, entry in data.items() if entry.get("type") == "node"
    }
    rel_entries: dict[str, dict[str, Any]] = {
        name: entry for name, entry in data.items() if entry.get("type") == "relationship"
    }

    labels = [NodeLabel(name=name) for name in node_entries]
    properties_by_label: dict[str, list[Property]] = {}
    for label_name, entry in node_entries.items():
        properties_by_label[label_name] = _properties_from_apoc(entry.get("properties", {}))

    # Relationship endpoints: prefer the highest-count (src, tgt) pair.
    rel_types: list[RelationshipType] = []
    properties_by_rel_type: dict[str, list[Property]] = {}
    for rel_name in rel_entries:
        src, tgt = _resolve_rel_endpoints(driver, database, rel_name)
        rel_types.append(
            RelationshipType(
                name=rel_name,
                source_label=src,
                target_label=tgt,
            )
        )
        properties_by_rel_type[rel_name] = _properties_from_apoc(
            rel_entries[rel_name].get("properties", {})
        )

    # Sparsity: query frequency per label/property and per rel/property.
    _apply_sparsity(
        driver,
        database,
        properties_by_label,
        is_relationship=False,
        threshold=sparsity_threshold,
    )
    _apply_sparsity(
        driver,
        database,
        properties_by_rel_type,
        is_relationship=True,
        threshold=sparsity_threshold,
    )

    constraints = _read_constraints(driver, database)
    indexes = _read_indexes(driver, database)

    return Schema(
        labels=labels,
        relationship_types=rel_types,
        properties_by_label=properties_by_label,
        properties_by_rel_type=properties_by_rel_type,
        constraints=constraints,
        indexes=indexes,
    )


def _properties_from_apoc(props: dict[str, Any]) -> list[Property]:
    """Translate APOC's properties dict into our Property list."""
    out: list[Property] = []
    for prop_name, info in props.items():
        type_name = _normalize_type(info.get("type", "UNKNOWN"))
        # APOC reports `existence: true` when there's an existence
        # constraint enforcing the property. We use the schema-driven
        # optional/sparse interpretation: optional defaults True
        # (Community Edition can't even install existence
        # constraints), and sparsity comes from the per-property
        # frequency query.
        out.append(
            Property(
                name=prop_name,
                type=type_name,
                optional=not bool(info.get("existence", False)),
            )
        )
    return out


def _resolve_rel_endpoints(driver: Driver, database: str, rel_name: str) -> tuple[str, str]:
    """For a given relationship type, return the (source_label, target_label)
    pair that appears most frequently. Picks the first label of each end
    node — multi-label endpoints (e.g. nodes carrying a marker label
    alongside the primary) are resolved deterministically by the order
    Neo4j returns ``labels(...)``."""
    cypher = (
        f"MATCH (a)-[r:`{rel_name}`]->(b) "
        "RETURN labels(a) AS src, labels(b) AS tgt, count(*) AS c "
        "ORDER BY c DESC LIMIT 1"
    )
    with driver.session(database=database) as session:
        record = session.run(cypher).single()
    if record is None or not record["src"] or not record["tgt"]:
        # No instances of this rel type yet, or endpoints are unlabeled.
        # Fall back to a sentinel; the caller will surface this as an
        # introspection limitation rather than crash.
        return ("Unknown", "Unknown")
    return str(record["src"][0]), str(record["tgt"][0])


# ---------------------------------------------------------------------------
# Vanilla strategy
# ---------------------------------------------------------------------------


def _vanilla_introspect(driver: Driver, database: str, sparsity_threshold: float) -> Schema:
    with driver.session(database=database) as session:
        label_names = [r["label"] for r in session.run("CALL db.labels() YIELD label RETURN label")]
        rel_names = [
            r["relationshipType"]
            for r in session.run(
                "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType"
            )
        ]
        node_prop_rows = list(
            session.run(
                "CALL db.schema.nodeTypeProperties() YIELD nodeLabels, "
                "propertyName, propertyTypes, mandatory "
                "RETURN nodeLabels, propertyName, propertyTypes, mandatory"
            )
        )
        rel_prop_rows = list(
            session.run(
                "CALL db.schema.relTypeProperties() YIELD relType, "
                "propertyName, propertyTypes, mandatory "
                "RETURN relType, propertyName, propertyTypes, mandatory"
            )
        )

    labels = [NodeLabel(name=name) for name in label_names]

    # nodeTypeProperties returns one row per (label-combination, property).
    # Flatten to per-label, preferring the per-single-label row (those
    # are most informative for our flat-label Schema). When a property
    # only appears in multi-label rows, fall back to the multi-label
    # entry's types and mark non-mandatory.
    properties_by_label: dict[str, list[Property]] = {name: [] for name in label_names}
    _seen: dict[tuple[str, str], Property] = {}
    for row in node_prop_rows:
        prop_name = row["propertyName"]
        if prop_name is None:
            continue
        prop_types = row["propertyTypes"]
        mandatory = bool(row["mandatory"])
        for label in row["nodeLabels"]:
            if label not in properties_by_label:
                continue
            key = (label, prop_name)
            if key in _seen:
                # Already recorded under this label.
                continue
            prop = Property(
                name=prop_name,
                type=_normalize_type_list(prop_types),
                optional=not mandatory,
            )
            properties_by_label[label].append(prop)
            _seen[key] = prop

    # Relationship properties: relType comes back as ":`MEASURED_BY`";
    # strip backticks and the leading colon.
    properties_by_rel_type: dict[str, list[Property]] = {name: [] for name in rel_names}
    for row in rel_prop_rows:
        rel_label = row["relType"].strip(":` ")
        prop_name = row["propertyName"]
        if prop_name is None or rel_label not in properties_by_rel_type:
            continue
        properties_by_rel_type[rel_label].append(
            Property(
                name=prop_name,
                type=_normalize_type_list(row["propertyTypes"]),
                optional=not bool(row["mandatory"]),
            )
        )

    # Resolve relationship endpoints via the same count-based query.
    rel_types: list[RelationshipType] = []
    for rel_name in rel_names:
        src, tgt = _resolve_rel_endpoints(driver, database, rel_name)
        rel_types.append(RelationshipType(name=rel_name, source_label=src, target_label=tgt))

    _apply_sparsity(
        driver,
        database,
        properties_by_label,
        is_relationship=False,
        threshold=sparsity_threshold,
    )
    _apply_sparsity(
        driver,
        database,
        properties_by_rel_type,
        is_relationship=True,
        threshold=sparsity_threshold,
    )

    constraints = _read_constraints(driver, database)
    indexes = _read_indexes(driver, database)

    return Schema(
        labels=labels,
        relationship_types=rel_types,
        properties_by_label=properties_by_label,
        properties_by_rel_type=properties_by_rel_type,
        constraints=constraints,
        indexes=indexes,
    )


# ---------------------------------------------------------------------------
# Sparsity follow-up
# ---------------------------------------------------------------------------


def _apply_sparsity(
    driver: Driver,
    database: str,
    properties_by_entity: dict[str, list[Property]],
    *,
    is_relationship: bool,
    threshold: float,
) -> None:
    """For each (entity, property), compute observed frequency on the
    entity in the live database and set ``sparse=True`` when below
    ``threshold``. Mutates the Property objects in place."""
    for entity_name, props in properties_by_entity.items():
        if not props:
            continue
        if is_relationship:
            count_clause = f"MATCH ()-[r:`{entity_name}`]->()"
            entity_var = "r"
        else:
            count_clause = f"MATCH (r:`{entity_name}`)"
            entity_var = "r"
        prop_counts = ", ".join(
            f"count({entity_var}.`{p.name}`) AS `cnt_{i}`" for i, p in enumerate(props)
        )
        cypher = f"{count_clause} RETURN count({entity_var}) AS total, {prop_counts}"
        with driver.session(database=database) as session:
            record = session.run(cypher).single()
        if record is None or record["total"] == 0:
            continue
        total = record["total"]
        for i, prop in enumerate(props):
            observed = record[f"cnt_{i}"]
            frequency = observed / total if total else 0.0
            if frequency < threshold:
                # Mutate via model_copy because Pydantic v2 models are
                # frozen-ish by default for Field(default=...) values.
                replacement = prop.model_copy(update={"sparse": True})
                # Replace the entry in the list to keep object identity
                # consistent with the dict's contents.
                idx = props.index(prop)
                props[idx] = replacement


# ---------------------------------------------------------------------------
# Constraints and indexes (shared between strategies)
# ---------------------------------------------------------------------------


def _read_constraints(driver: Driver, database: str) -> list[SchemaConstraint]:
    with driver.session(database=database) as session:
        rows = list(session.run("SHOW CONSTRAINTS"))
    out: list[SchemaConstraint] = []
    for row in rows:
        labels_or_types = row.get("labelsOrTypes") or []
        properties = row.get("properties") or []
        # Composite-key constraints have multiple properties; our
        # SchemaConstraint model carries one. Take the first and note
        # the limitation in the module docstring.
        first_prop = properties[0] if properties else None
        if not labels_or_types:
            continue
        out.append(
            SchemaConstraint(
                type=row["type"],
                label_or_rel=labels_or_types[0],
                property=first_prop,
                identifier=row["name"],
            )
        )
    return out


def _read_indexes(driver: Driver, database: str) -> list[Index]:
    """Read user-created indexes (excluding LOOKUP and constraint-owned)."""
    with driver.session(database=database) as session:
        rows = list(session.run("SHOW INDEXES WHERE type <> 'LOOKUP' AND owningConstraint IS NULL"))
    out: list[Index] = []
    for row in rows:
        labels_or_types = row.get("labelsOrTypes") or []
        properties = row.get("properties") or []
        if not labels_or_types or not properties:
            continue
        out.append(
            Index(
                type=row["type"],
                label_or_rel=labels_or_types[0],
                properties=list(properties),
            )
        )
    return out
