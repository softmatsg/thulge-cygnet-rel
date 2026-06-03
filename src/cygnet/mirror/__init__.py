# Copyright (c) 2026 Nikodem Tomczak, Thulge Labs. All rights reserved.

"""Mirror graph: a small Neo4j instance that mirrors the production schema.

Built from the loaded schema spec deterministically (sampling-based
construction is deferred). Used by the EXPLAIN validator backend and
by the cost gate so neither needs to touch production Neo4j directly.

The library does not manage Neo4j containers; users supply the Bolt URI
for a running mirror instance.

Public surface:

- :class:`MirrorGraphBuilder` -- ``build_from_schema``,
  ``teardown``, ``is_built``.
- :class:`MirrorBuildReport` -- typed return value of
  ``build_from_schema``; re-exported from :mod:`cygnet.models`.

Distinction from ``tests/integration/conftest.py::loaded_schema_fixture``:
that helper applies constraints and indexes only. The mirror builder
populates the full mirror (nodes + relationships + DDL).
"""

from __future__ import annotations

from cygnet.mirror.builder import MirrorGraphBuilder
from cygnet.mirror.equivalence import (
    EquivalenceDivergence,
    EquivalenceOutcome,
    EquivalenceReport,
    verify_mirror_equivalence,
)
from cygnet.models import MirrorBuildReport

__all__ = [
    "EquivalenceDivergence",
    "EquivalenceOutcome",
    "EquivalenceReport",
    "MirrorBuildReport",
    "MirrorGraphBuilder",
    "verify_mirror_equivalence",
]
