# Copyright (c) 2026 Nikodem Tomczak, Thulge Labs. All rights reserved.

"""Schema ingestion and exposure.

Loads the schema the validator works against, from either a YAML/JSON
spec file (Path B) or Neo4j introspection (Path A). Both paths
produce the same ``Schema`` Pydantic model; downstream validator code
does not care which source produced it.

Modules:

- ``spec``: YAML/JSON spec loader (Path B).
- ``introspect``: Neo4j introspection (Path A).
- ``drift``: drift audit comparing spec vs. introspected schema (deferred).
"""

from cygnet.schema.introspect import SchemaIntrospectionError, introspect_schema
from cygnet.schema.mirror_check import (
    MirrorValidationReport,
    MissingItem,
    UnexpectedItem,
    validate_mirror,
)
from cygnet.schema.spec import SUPPORTED_VERSION, SchemaSpecError, load_schema_spec

__all__ = [
    "SUPPORTED_VERSION",
    "MirrorValidationReport",
    "MissingItem",
    "SchemaIntrospectionError",
    "SchemaSpecError",
    "UnexpectedItem",
    "introspect_schema",
    "load_schema_spec",
    "validate_mirror",
]
