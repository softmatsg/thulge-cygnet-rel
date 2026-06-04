# Copyright (c) 2026 Nikodem Tomczak, Thulge Labs. All rights reserved.

"""YAML/JSON schema spec loader (Path B).

Loads a Proposal-B-shaped spec file (sectioned, name-as-key) into a
`cygnet.Schema`. Format reference: ``docs/schema_spec_format_proposals.md``.

Spec-to-model renames done here:
- `relationship_types[name].source`  -> `RelationshipType.source_label`
- `relationship_types[name].target`  -> `RelationshipType.target_label`
- `constraints[i].on`                -> `SchemaConstraint.label_or_rel`
- `indexes[i].on`                    -> `Index.label_or_rel`

Validation rules enforced beyond Pydantic are numbered 1..10 below and
match the rule list in the proposals doc (rule 7 is parser-enforced
through map-key uniqueness; rules 11 and 12 cover ``$ref`` and apply
only to Proposal C, which is not implemented).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import yaml
from pydantic import ValidationError

from cygnet.models import (
    Index,
    NodeLabel,
    Property,
    RelationshipType,
    Schema,
    SchemaConstraint,
)

if TYPE_CHECKING:
    pass

__all__ = ["SUPPORTED_VERSION", "SchemaSpecError", "load_schema_spec"]


SUPPORTED_VERSION: Final[str] = "0.1"
"""The only spec version this loader accepts. Future versions land via an
in-loader compatibility shim; the public API does not change."""


class _CygnetYAMLLoader(yaml.SafeLoader):
    """SafeLoader variant using YAML 1.2 boolean semantics.

    PyYAML defaults to YAML 1.1, which treats ``on``/``off``/``yes``/``no``
    as booleans. Our spec format uses ``on:`` as the constraint/index
    scope key (immutable artifact #7), so authors expect it to parse as
    a string. This loader keeps only ``true``/``false`` as bool literals
    (YAML 1.2), making ``on: Sample`` parse as ``{"on": "Sample"}``.
    """


# Rebuild the implicit-resolver map without the YAML 1.1 bool patterns,
# then add back a YAML-1.2-style resolver that only matches true/false.
_CygnetYAMLLoader.yaml_implicit_resolvers = {
    first_char: [(tag, regexp) for tag, regexp in resolvers if tag != "tag:yaml.org,2002:bool"]
    for first_char, resolvers in _CygnetYAMLLoader.yaml_implicit_resolvers.items()
}
_CygnetYAMLLoader.add_implicit_resolver(  # type: ignore[no-untyped-call]
    "tag:yaml.org,2002:bool",
    re.compile(r"^(?:true|True|TRUE|false|False|FALSE)$"),
    list("tTfF"),
)


# Recommended property types — informational, not enforced.
_RECOMMENDED_PROPERTY_TYPES: Final[frozenset[str]] = frozenset(
    {
        "STRING",
        "INTEGER",
        "FLOAT",
        "BOOLEAN",
        "DATE",
        "TIME",
        "DATETIME",
        "LOCAL_DATETIME",
        "DURATION",
        "POINT",
        "LIST",
        "MAP",
    }
)


class SchemaSpecError(Exception):
    """Raised when a schema spec fails to load or validate.

    Carries optional 1-based ``line`` and ``column`` attributes pointing
    into the source file (or string) when the failure is locatable.
    PyYAML supplies these on parse errors; JSON errors give a character
    offset which is converted to (line, column) by the loader.
    """

    def __init__(
        self,
        message: str,
        *,
        line: int | None = None,
        column: int | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.line = line
        self.column = column

    def __str__(self) -> str:
        if self.line is not None and self.column is not None:
            return f"{self.message} (line {self.line}, column {self.column})"
        if self.line is not None:
            return f"{self.message} (line {self.line})"
        return self.message


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def load_schema_spec(source: Path | str | dict[str, Any]) -> Schema:
    """Load a YAML/JSON schema spec into a ``Schema`` Pydantic model.

    Args:
        source: One of:
            - ``Path``: filesystem path; format dispatched by extension
              (``.yaml``/``.yml`` for YAML, ``.json`` for JSON).
            - ``str``: raw spec content. JSON is attempted first; on
              failure YAML is attempted.
            - ``dict``: pre-parsed mapping; used as-is.

    Returns:
        The validated ``Schema`` model.

    Raises:
        SchemaSpecError: when parsing fails, when an unknown version is
            encountered, or when any of the structural validation rules
            (1..10 in ``docs/schema_spec_format_proposals.md``) is
            violated. The exception's ``line`` and ``column`` attributes
            point into the source when the failure is locatable.
    """
    if isinstance(source, Path):
        data = _load_from_path(source)
    elif isinstance(source, str):
        data = _parse_string(source)
    elif isinstance(source, dict):
        data = source
    else:
        raise SchemaSpecError(
            f"Unsupported source type for load_schema_spec: {type(source).__name__}"
        )
    return _build_schema(data)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _load_from_path(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix not in {".yaml", ".yml", ".json"}:
        raise SchemaSpecError(
            f"Unsupported file extension {suffix!r}; expected .yaml, .yml, or .json."
        )
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SchemaSpecError(f"Could not read spec file {path}: {exc}") from exc

    if suffix == ".json":
        return _load_json(text, source=str(path))
    return _load_yaml(text, source=str(path))


def _parse_string(s: str) -> dict[str, Any]:
    if not s.strip():
        raise SchemaSpecError("Empty schema spec source.")
    try:
        data = json.loads(s)
    except json.JSONDecodeError:
        try:
            data = yaml.load(s, Loader=_CygnetYAMLLoader)
        except yaml.YAMLError as exc:
            line, column = _yaml_error_location(exc)
            raise SchemaSpecError(
                f"Failed to parse spec as JSON or YAML: {exc}",
                line=line,
                column=column,
            ) from exc
    if not isinstance(data, dict):
        raise SchemaSpecError(
            f"Spec root must be a mapping; got "
            f"{type(data).__name__ if data is not None else 'null'}."
        )
    return data


def _load_yaml(text: str, *, source: str) -> dict[str, Any]:
    try:
        data = yaml.load(text, Loader=_CygnetYAMLLoader)
    except yaml.YAMLError as exc:
        line, column = _yaml_error_location(exc)
        raise SchemaSpecError(
            f"YAML parse error in {source}: {exc}",
            line=line,
            column=column,
        ) from exc
    if not isinstance(data, dict):
        raise SchemaSpecError(
            f"Spec root in {source} must be a mapping; got "
            f"{type(data).__name__ if data is not None else 'null'}."
        )
    return data


def _load_json(text: str, *, source: str) -> dict[str, Any]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        line, column = _offset_to_line_col(text, exc.pos)
        raise SchemaSpecError(
            f"JSON parse error in {source}: {exc.msg}",
            line=line,
            column=column,
        ) from exc
    if not isinstance(data, dict):
        raise SchemaSpecError(
            f"Spec root in {source} must be a mapping; got "
            f"{type(data).__name__ if data is not None else 'null'}."
        )
    return data


def _yaml_error_location(exc: yaml.YAMLError) -> tuple[int | None, int | None]:
    mark = getattr(exc, "problem_mark", None)
    if mark is None:
        return None, None
    return mark.line + 1, mark.column + 1


def _offset_to_line_col(text: str, offset: int) -> tuple[int, int]:
    """Convert a 0-based character offset into 1-based (line, column)."""
    if offset < 0:
        offset = 0
    if offset > len(text):
        offset = len(text)
    line = text.count("\n", 0, offset) + 1
    last_newline = text.rfind("\n", 0, offset)
    column = offset - last_newline if last_newline >= 0 else offset + 1
    return line, column


# ---------------------------------------------------------------------------
# Build pipeline
# ---------------------------------------------------------------------------


def _build_schema(data: dict[str, Any]) -> Schema:
    _validate_version(data)

    node_labels = _require_mapping(data, "node_labels", default={})
    relationship_types = _require_mapping(data, "relationship_types", default={})
    constraints = _require_list(data, "constraints", default=[])
    indexes = _require_list(data, "indexes", default=[])

    labels, properties_by_label = _build_node_labels(node_labels)
    rels, properties_by_rel_type = _build_relationship_types(relationship_types)
    schema_constraints = _build_constraints(constraints)
    schema_indexes = _build_indexes(indexes)

    label_names = {nl.name for nl in labels}
    rel_names = {rt.name for rt in rels}

    _validate_relationship_endpoints(rels, label_names)
    _validate_constraint_scope_and_property(
        schema_constraints,
        label_names,
        rel_names,
        properties_by_label,
        properties_by_rel_type,
    )
    _validate_index_scope_and_properties(
        schema_indexes,
        label_names,
        rel_names,
        properties_by_label,
        properties_by_rel_type,
    )
    _validate_unique_constraint_identifiers(schema_constraints)

    return Schema(
        labels=labels,
        relationship_types=rels,
        properties_by_label=properties_by_label,
        properties_by_rel_type=properties_by_rel_type,
        constraints=schema_constraints,
        indexes=schema_indexes,
    )


def _require_mapping(data: dict[str, Any], key: str, *, default: dict[str, Any]) -> dict[str, Any]:
    value = data.get(key, default)
    if value is None:
        return default
    if not isinstance(value, dict):
        raise SchemaSpecError(f"Top-level {key!r} must be a mapping; got {type(value).__name__}.")
    return value


def _require_list(data: dict[str, Any], key: str, *, default: list[Any]) -> list[Any]:
    value = data.get(key, default)
    if value is None:
        return default
    if not isinstance(value, list):
        raise SchemaSpecError(f"Top-level {key!r} must be a list; got {type(value).__name__}.")
    return value


def _build_node_labels(
    node_labels: dict[str, Any],
) -> tuple[list[NodeLabel], dict[str, list[Property]]]:
    labels: list[NodeLabel] = []
    properties_by_label: dict[str, list[Property]] = {}
    for label_name, label_body in node_labels.items():
        scope = f"node_labels[{label_name!r}]"
        body = label_body if label_body is not None else {}
        if not isinstance(body, dict):
            raise SchemaSpecError(f"{scope} must be a mapping; got {type(body).__name__}.")
        kwargs: dict[str, Any] = {"name": label_name}
        if "sparsity_threshold" in body:
            kwargs["sparsity_threshold"] = body["sparsity_threshold"]
        try:
            labels.append(NodeLabel(**kwargs))
        except ValidationError as exc:
            raise SchemaSpecError(f"Invalid {scope}: {exc}") from exc
        props = body.get("properties", {})
        if props is None:
            props = {}
        if not isinstance(props, dict):
            raise SchemaSpecError(
                f"{scope}.properties must be a mapping; got {type(props).__name__}."
            )
        properties_by_label[label_name] = _build_properties(props, scope=scope)
    return labels, properties_by_label


def _build_relationship_types(
    relationship_types: dict[str, Any],
) -> tuple[list[RelationshipType], dict[str, list[Property]]]:
    rels: list[RelationshipType] = []
    properties_by_rel_type: dict[str, list[Property]] = {}
    for rel_name, rel_body in relationship_types.items():
        scope = f"relationship_types[{rel_name!r}]"
        body = rel_body if rel_body is not None else {}
        if not isinstance(body, dict):
            raise SchemaSpecError(f"{scope} must be a mapping; got {type(body).__name__}.")
        if "source" not in body:
            raise SchemaSpecError(f"{scope} missing required 'source' field.")
        if "target" not in body:
            raise SchemaSpecError(f"{scope} missing required 'target' field.")
        try:
            rels.append(
                RelationshipType(
                    name=rel_name,
                    source_label=body["source"],
                    target_label=body["target"],
                )
            )
        except ValidationError as exc:
            raise SchemaSpecError(f"Invalid {scope}: {exc}") from exc
        props = body.get("properties", {})
        if props is None:
            props = {}
        if not isinstance(props, dict):
            raise SchemaSpecError(
                f"{scope}.properties must be a mapping; got {type(props).__name__}."
            )
        properties_by_rel_type[rel_name] = _build_properties(props, scope=scope)
    return rels, properties_by_rel_type


def _build_properties(props_dict: dict[str, Any], *, scope: str) -> list[Property]:
    """Build a list of ``Property`` from a name-keyed mapping.

    Enforces rule 9 (``type`` is a non-empty string) and rule 10
    (``sparse=True`` requires ``optional=True``; hard-reject otherwise).
    """
    out: list[Property] = []
    for pname, pbody in props_dict.items():
        prop_scope = f"{scope}.properties[{pname!r}]"
        body = pbody if pbody is not None else {}
        if not isinstance(body, dict):
            raise SchemaSpecError(f"{prop_scope} must be a mapping; got {type(body).__name__}.")
        if "type" not in body:
            raise SchemaSpecError(f"{prop_scope} missing required 'type' field.")
        type_value = body["type"]
        if not isinstance(type_value, str) or not type_value:
            raise SchemaSpecError(
                f"{prop_scope}.type must be a non-empty string; got {type_value!r}."
            )
        optional = body.get("optional", True)
        sparse = body.get("sparse", False)
        if sparse and not optional:
            raise SchemaSpecError(
                f"{prop_scope} has sparse=true with optional=false; "
                "a sparse property must be optional (rule 10)."
            )
        try:
            out.append(Property(name=pname, type=type_value, optional=optional, sparse=sparse))
        except ValidationError as exc:
            raise SchemaSpecError(f"Invalid {prop_scope}: {exc}") from exc
    return out


def _build_constraints(constraints: list[Any]) -> list[SchemaConstraint]:
    out: list[SchemaConstraint] = []
    for i, c in enumerate(constraints):
        scope = f"constraints[{i}]"
        if not isinstance(c, dict):
            raise SchemaSpecError(f"{scope} must be a mapping; got {type(c).__name__}.")
        for required in ("identifier", "type", "on"):
            if required not in c:
                raise SchemaSpecError(f"{scope} missing required {required!r} field.")
        try:
            out.append(
                SchemaConstraint(
                    type=c["type"],
                    label_or_rel=c["on"],
                    property=c.get("property"),
                    identifier=c["identifier"],
                )
            )
        except ValidationError as exc:
            raise SchemaSpecError(f"Invalid {scope}: {exc}") from exc
    return out


def _build_indexes(indexes: list[Any]) -> list[Index]:
    out: list[Index] = []
    for i, ix in enumerate(indexes):
        scope = f"indexes[{i}]"
        if not isinstance(ix, dict):
            raise SchemaSpecError(f"{scope} must be a mapping; got {type(ix).__name__}.")
        for required in ("type", "on"):
            if required not in ix:
                raise SchemaSpecError(f"{scope} missing required {required!r} field.")
        properties = ix.get("properties", [])
        if not isinstance(properties, list):
            raise SchemaSpecError(
                f"{scope}.properties must be a list; got {type(properties).__name__}."
            )
        try:
            out.append(
                Index(
                    type=ix["type"],
                    label_or_rel=ix["on"],
                    properties=list(properties),
                )
            )
        except ValidationError as exc:
            raise SchemaSpecError(f"Invalid {scope}: {exc}") from exc
    return out


# ---------------------------------------------------------------------------
# Cross-cutting validation rules
# ---------------------------------------------------------------------------


def _validate_version(data: dict[str, Any]) -> None:
    """Rule 1: ``version`` is required and must equal ``SUPPORTED_VERSION``."""
    if "version" not in data:
        raise SchemaSpecError(
            f"Spec missing required 'version' field (expected {SUPPORTED_VERSION!r})."
        )
    version = data["version"]
    if version != SUPPORTED_VERSION:
        raise SchemaSpecError(
            f"Unsupported spec version {version!r}; this loader accepts "
            f"{SUPPORTED_VERSION!r}. Future versions will be handled by an "
            "in-loader compatibility shim."
        )


def _validate_relationship_endpoints(rels: list[RelationshipType], label_names: set[str]) -> None:
    """Rule 2: every relationship's source/target references a declared label."""
    for r in rels:
        if r.source_label not in label_names:
            raise SchemaSpecError(
                f"relationship_types[{r.name!r}].source references undeclared "
                f"label {r.source_label!r}."
            )
        if r.target_label not in label_names:
            raise SchemaSpecError(
                f"relationship_types[{r.name!r}].target references undeclared "
                f"label {r.target_label!r}."
            )


def _validate_constraint_scope_and_property(
    constraints: list[SchemaConstraint],
    label_names: set[str],
    rel_names: set[str],
    properties_by_label: dict[str, list[Property]],
    properties_by_rel_type: dict[str, list[Property]],
) -> None:
    """Rules 3 and 4: constraint scope and property exist on the named entity."""
    for c in constraints:
        if c.label_or_rel in label_names:
            allowed_props = {p.name for p in properties_by_label[c.label_or_rel]}
        elif c.label_or_rel in rel_names:
            allowed_props = {p.name for p in properties_by_rel_type[c.label_or_rel]}
        else:
            raise SchemaSpecError(
                f"constraint {c.identifier!r} references undeclared "
                f"label/relationship {c.label_or_rel!r}."
            )
        if c.property is None:
            continue
        if c.property not in allowed_props:
            raise SchemaSpecError(
                f"constraint {c.identifier!r} references property "
                f"{c.property!r} not declared on {c.label_or_rel!r}."
            )


def _validate_index_scope_and_properties(
    indexes: list[Index],
    label_names: set[str],
    rel_names: set[str],
    properties_by_label: dict[str, list[Property]],
    properties_by_rel_type: dict[str, list[Property]],
) -> None:
    """Rules 5 and 6: index scope and listed properties exist on the named entity."""
    for i, ix in enumerate(indexes):
        if ix.label_or_rel in label_names:
            allowed_props = {p.name for p in properties_by_label[ix.label_or_rel]}
        elif ix.label_or_rel in rel_names:
            allowed_props = {p.name for p in properties_by_rel_type[ix.label_or_rel]}
        else:
            raise SchemaSpecError(
                f"indexes[{i}] references undeclared label/relationship {ix.label_or_rel!r}."
            )
        for pname in ix.properties:
            if pname not in allowed_props:
                raise SchemaSpecError(
                    f"indexes[{i}] references property {pname!r} not declared "
                    f"on {ix.label_or_rel!r}."
                )


def _validate_unique_constraint_identifiers(
    constraints: list[SchemaConstraint],
) -> None:
    """Rule 8: constraint identifiers are unique across the spec."""
    seen: set[str] = set()
    for c in constraints:
        if c.identifier in seen:
            raise SchemaSpecError(f"Duplicate constraint identifier {c.identifier!r}.")
        seen.add(c.identifier)
