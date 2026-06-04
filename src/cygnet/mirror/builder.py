# Copyright (c) 2026 Nikodem Tomczak, Thulge Labs. All rights reserved.

"""Mirror graph builder: populate a Neo4j instance from a ``Schema``.

The mirror is a structural surrogate for production: one node per
declared label, one relationship per declared type, declared
properties populated with type-correct dummy values. It is the
canonical EXPLAIN / cost-gate target for environments where running
those calls against production is undesirable.

The mirror is *not* meant to be data-realistic. Per the brief, its
job is "does Neo4j's planner think this query is structurally valid
against this schema." Carrying one node per label with dummy
properties is enough for the planner to produce a plan; cost
estimates against a mirror reflect schema shape, not production
volume.

Type-to-dummy-value mapping (treat as part of the public contract;
documented here and in the module's public docstring):

    STRING          -> "sample_string"
    INTEGER, LONG   -> 0
    FLOAT, DOUBLE   -> 0.0
    BOOLEAN         -> False
    DATETIME        -> datetime(2026, 1, 1)
    DATE            -> date(2026, 1, 1)
    POINT           -> point({"x": 0.0, "y": 0.0, "crs": "cartesian"})
    LIST            -> []
    other / unknown -> "sample_value"  (emits a warning in the report)

Distinction from ``tests/integration/conftest.py::loaded_schema_fixture``:
that helper applies constraints and indexes only — a test-side DDL
helper. ``MirrorGraphBuilder`` applies the *full* mirror: DDL plus
mirror nodes and relationships with dummy properties, all tagged with
a ``mirror_id`` carrying a configurable prefix so the mirror is
auditable from outside the library.

Multi-label co-occurrence (e.g. Sample + Validated on a single node)
is intentionally not modelled. The current ``Schema`` model has no
co-occurrence concept (``RelationshipType`` endpoints are single
strings, ``labels`` is flat). The builder therefore emits one
node per declared label; co-occurrence is recorded as a deferred
schema-extension item.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import TYPE_CHECKING, Any, Final

from neo4j.exceptions import DatabaseError

from cygnet.models import (
    Index,
    MirrorBuildReport,
    Property,
    RelationshipType,
    Schema,
    SchemaConstraint,
)

if TYPE_CHECKING:
    from neo4j import Driver, Session

__all__ = ["MirrorGraphBuilder"]


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public type→dummy mapping. Constants live at module level so users can
# import and inspect; the values themselves are part of the public
# contract per the brief.
# ---------------------------------------------------------------------------

_DEFAULT_DATETIME: Final[datetime] = datetime(2026, 1, 1)
_DEFAULT_DATE: Final[date] = date(2026, 1, 1)

DUMMY_VALUES: Final[dict[str, object]] = {
    "STRING": "sample_string",
    "INTEGER": 0,
    "LONG": 0,
    "FLOAT": 0.0,
    "DOUBLE": 0.0,
    "BOOLEAN": False,
    "DATETIME": _DEFAULT_DATETIME,
    "DATE": _DEFAULT_DATE,
    "POINT": {"x": 0.0, "y": 0.0, "crs": "cartesian"},
    "LIST": [],
}
"""Type-to-dummy-value lookup. Keyed by upper-cased type name."""

_FALLBACK_DUMMY: Final[str] = "sample_value"

# Sentinel value used by introspection when relationship endpoints
# could not be resolved from data. Mirrored from the introspection
# module and kept in lock-step here so the builder can recognise it.
_UNKNOWN_ENDPOINT: Final[str] = "Unknown"


class MirrorGraphBuilder:
    """Build and tear down a Neo4j mirror graph from a :class:`Schema`.

    Args:
        driver: an already-connected ``neo4j.Driver`` for the mirror
            Neo4j. The builder does not own the driver; closing it is
            the caller's responsibility (when wired into ``Gate``, the
            mirror driver is closed by ``Gate.close``).
        database: Neo4j database name. Defaults to ``"neo4j"``.
        prefix: applied to every mirror node's ``mirror_id`` property
            so mirror nodes are distinguishable from any other data in
            the database. Users with a dedicated mirror instance can
            keep the default; users overlaying onto a shared instance
            should pick a project-specific prefix. The default value
            ``"_cygnet_mirror_"`` is part of the user-facing
            convention.

    The same prefix is used to identify the mirror's own constraints
    and indexes for the ``teardown`` path. Constraints/indexes in the
    schema that pre-exist on the database under a different name
    are left untouched.
    """

    DEFAULT_PREFIX: Final[str] = "_cygnet_mirror_"

    def __init__(
        self,
        driver: Driver,
        database: str = "neo4j",
        prefix: str = "_cygnet_mirror_",
    ) -> None:
        self._driver = driver
        self._database = database
        self._prefix = prefix

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_built(self) -> bool:
        """Return True if any mirror node already exists in the
        database. Useful for idempotency checks before calling
        :meth:`build_from_schema`."""
        with self._session() as session:
            record = session.run(
                "MATCH (n) WHERE n.mirror_id STARTS WITH $prefix RETURN count(n) AS c LIMIT 1",
                prefix=self._prefix,
            ).single()
        if record is None:
            return False
        return int(record["c"]) > 0

    def build_from_schema(
        self,
        schema: Schema,
        *,
        strict: bool = False,
    ) -> MirrorBuildReport:
        """Populate the mirror Neo4j from the loaded schema.

        Per the brief, the build is idempotent within a single mirror
        instance: if the mirror already contains nodes with the
        configured prefix, the call short-circuits and returns an
        empty report with an idempotency warning rather than
        duplicating the structure.

        Args:
            schema: the schema to mirror.
            strict: if True, refuse to silently skip constraints that
                Community Edition rejects (existence, node-key) and
                raise ``RuntimeError`` instead. Default False, matching
                the permissive behaviour of the
                ``loaded_schema_fixture`` test helper.
        """
        warnings: list[str] = []
        if self.is_built():
            warnings.append(
                "build_from_schema called on a mirror that already contains "
                f"nodes with prefix {self._prefix!r}; build skipped."
            )
            return MirrorBuildReport(
                nodes_created=0,
                relationships_created=0,
                constraints_applied=[],
                constraints_skipped=[],
                indexes_applied=[],
                warnings=warnings,
            )

        with self._session() as session:
            nodes_created = self._create_nodes(session, schema, warnings)
            relationships_created = self._create_relationships(session, schema, warnings)
            constraints_applied, constraints_skipped = self._apply_constraints(
                session, schema.constraints, strict
            )
            indexes_applied = self._apply_indexes(session, schema.indexes)

        return MirrorBuildReport(
            nodes_created=nodes_created,
            relationships_created=relationships_created,
            constraints_applied=constraints_applied,
            constraints_skipped=constraints_skipped,
            indexes_applied=indexes_applied,
            warnings=warnings,
        )

    def teardown(self) -> int:
        """Remove every mirror node (and its relationships) plus every
        mirror-owned constraint and index. Returns the number of nodes
        removed; safe to call when the mirror is already empty."""
        with self._session() as session:
            self._drop_constraints_and_indexes(session)
            record = session.run(
                "MATCH (n) WHERE n.mirror_id STARTS WITH $prefix "
                "WITH n, count(*) AS _ "
                "DETACH DELETE n "
                "RETURN count(_) AS removed",
                prefix=self._prefix,
            ).single()
        if record is None:
            return 0
        return int(record["removed"])

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _session(self) -> Session:
        return self._driver.session(database=self._database)

    def _create_nodes(
        self,
        session: Session,
        schema: Schema,
        warnings: list[str],
    ) -> int:
        created = 0
        for label in schema.labels:
            props = self._dummy_props_for_label(label.name, schema, warnings)
            props["mirror_id"] = f"{self._prefix}{label.name}"
            session.run(
                f"CREATE (n:`{label.name}`) SET n = $props",
                props=props,
            )
            created += 1
        return created

    def _create_relationships(
        self,
        session: Session,
        schema: Schema,
        warnings: list[str],
    ) -> int:
        created = 0
        for rel in schema.relationship_types:
            if rel.source_label == _UNKNOWN_ENDPOINT or rel.target_label == _UNKNOWN_ENDPOINT:
                warnings.append(
                    f"Skipped relationship {rel.name!r}: endpoint sentinel "
                    f"'Unknown' (source={rel.source_label!r}, "
                    f"target={rel.target_label!r}). Introspection could not "
                    "resolve the endpoint pair; the mirror has no node to "
                    "anchor this relationship to."
                )
                continue
            if not _label_in_schema(schema, rel.source_label) or not _label_in_schema(
                schema, rel.target_label
            ):
                warnings.append(
                    f"Skipped relationship {rel.name!r}: endpoint label "
                    f"{rel.source_label!r} or {rel.target_label!r} not in "
                    "schema.labels. The mirror has no node to anchor this "
                    "relationship to."
                )
                continue
            rel_props = self._dummy_props_for_rel(rel, schema, warnings)
            session.run(
                f"MATCH (s:`{rel.source_label}` {{mirror_id: $sid}}), "
                f"(t:`{rel.target_label}` {{mirror_id: $tid}}) "
                f"CREATE (s)-[r:`{rel.name}`]->(t) SET r = $props",
                sid=f"{self._prefix}{rel.source_label}",
                tid=f"{self._prefix}{rel.target_label}",
                props=rel_props,
            )
            created += 1
        return created

    def _dummy_props_for_label(
        self,
        label_name: str,
        schema: Schema,
        warnings: list[str],
    ) -> dict[str, Any]:
        return self._dummy_props(
            schema.properties_by_label.get(label_name, []),
            owner=f"label {label_name!r}",
            warnings=warnings,
        )

    def _dummy_props_for_rel(
        self,
        rel: RelationshipType,
        schema: Schema,
        warnings: list[str],
    ) -> dict[str, Any]:
        return self._dummy_props(
            schema.properties_by_rel_type.get(rel.name, []),
            owner=f"relationship {rel.name!r}",
            warnings=warnings,
        )

    @staticmethod
    def _dummy_props(
        props: list[Property],
        *,
        owner: str,
        warnings: list[str],
    ) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for p in props:
            type_key = p.type.upper()
            if type_key in DUMMY_VALUES:
                out[p.name] = DUMMY_VALUES[type_key]
            else:
                warnings.append(
                    f"Unknown property type {p.type!r} on {owner} property "
                    f"{p.name!r}; using fallback dummy value {_FALLBACK_DUMMY!r}."
                )
                out[p.name] = _FALLBACK_DUMMY
        return out

    def _apply_constraints(
        self,
        session: Session,
        constraints: list[SchemaConstraint],
        strict: bool,
    ) -> tuple[list[str], list[str]]:
        applied: list[str] = []
        skipped: list[str] = []
        for c in constraints:
            if not c.property:
                # Composite constraints are not in the SchemaConstraint
                # surface; skip silently.
                continue
            stmt = _constraint_ddl(c, prefix=self._prefix)
            if stmt is None:
                continue
            prefixed = f"{self._prefix}{c.identifier}"
            try:
                session.run(stmt)
            except DatabaseError as exc:
                msg = str(exc)
                if "Enterprise Edition" in msg or "is not supported in community" in msg.lower():
                    reason = (
                        f"{c.identifier} ({c.type}: Community Edition rejects "
                        "existence/node-key constraints)"
                    )
                    if strict:
                        raise RuntimeError(
                            "MirrorGraphBuilder(strict=True) refused to skip "
                            f"constraint {c.identifier!r}: {msg}"
                        ) from exc
                    logger.warning("mirror: %s", reason)
                    skipped.append(reason)
                    continue
                raise
            applied.append(prefixed)
        return applied, skipped

    def _apply_indexes(
        self,
        session: Session,
        indexes: list[Index],
    ) -> list[str]:
        applied: list[str] = []
        for i, ix in enumerate(indexes):
            stmt, identifier = _index_ddl(ix, index_number=i, prefix=self._prefix)
            if stmt is None:
                continue
            session.run(stmt)
            applied.append(identifier)
        return applied

    def _drop_constraints_and_indexes(self, session: Session) -> None:
        """Drop every constraint and index whose name starts with the
        configured prefix. Names that don't match are left alone.

        We list constraints/indexes via ``SHOW`` rather than tracking
        the names in memory because ``teardown`` may be invoked by a
        different process from the one that ran ``build_from_schema``.
        """
        for record in list(session.run("SHOW CONSTRAINTS")):
            name = record["name"]
            if isinstance(name, str) and name.startswith(self._prefix):
                session.run(f"DROP CONSTRAINT `{name}`")
        for record in list(session.run("SHOW INDEXES WHERE type <> 'LOOKUP'")):
            name = record["name"]
            if isinstance(name, str) and name.startswith(self._prefix):
                session.run(f"DROP INDEX `{name}`")


# ---------------------------------------------------------------------------
# Free functions: DDL generation. Kept module-private to make the
# builder's __init__ surface easy to mock.
# ---------------------------------------------------------------------------


def _constraint_ddl(c: SchemaConstraint, *, prefix: str) -> str | None:
    if not c.property:
        return None
    name = f"{prefix}{c.identifier}"
    type_upper = c.type.upper()
    if "UNIQUENESS" in type_upper or "UNIQUE" in type_upper:
        return (
            f"CREATE CONSTRAINT `{name}` IF NOT EXISTS "
            f"FOR (n:`{c.label_or_rel}`) REQUIRE n.`{c.property}` IS UNIQUE"
        )
    if "EXISTENCE" in type_upper or "NOT NULL" in type_upper:
        return (
            f"CREATE CONSTRAINT `{name}` IF NOT EXISTS "
            f"FOR (n:`{c.label_or_rel}`) REQUIRE n.`{c.property}` IS NOT NULL"
        )
    if "NODE_KEY" in type_upper or "NODE KEY" in type_upper:
        return (
            f"CREATE CONSTRAINT `{name}` IF NOT EXISTS "
            f"FOR (n:`{c.label_or_rel}`) REQUIRE n.`{c.property}` IS NODE KEY"
        )
    return None


def _index_ddl(ix: Index, *, index_number: int, prefix: str) -> tuple[str | None, str]:
    name = f"{prefix}idx_{index_number}"
    type_upper = ix.type.upper()
    if type_upper not in {"BTREE", "RANGE"}:
        return (None, name)
    prop_list = ", ".join(f"n.`{p}`" for p in ix.properties)
    # Emit ``CREATE RANGE INDEX`` explicitly so
    # the index type that lands in ``SHOW INDEXES`` is deterministic
    # across Neo4j versions. Neo4j 5 removed BTREE; RANGE is the
    # current name for the same B-tree-backed implementation. Without
    # an explicit type, ``CREATE INDEX`` defaults to RANGE on Neo4j 5
    # but a YAML schema declaring BTREE would round-trip through
    # introspection as RANGE — surfacing a spurious divergence in
    # :func:`validate_mirror`. Picking RANGE explicitly is the
    # version-portable encoding.
    stmt = f"CREATE RANGE INDEX `{name}` IF NOT EXISTS FOR (n:`{ix.label_or_rel}`) ON ({prop_list})"
    return (stmt, name)


def _label_in_schema(schema: Schema, label_name: str) -> bool:
    return any(nl.name == label_name for nl in schema.labels)
