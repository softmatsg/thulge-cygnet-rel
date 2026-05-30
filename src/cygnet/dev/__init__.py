# Copyright (c) 2026 Nikodem Tomczak, Thulge Labs. All rights reserved.

"""Dev-time scaffolding for CYGNET.

This subpackage is **not** part of the production API surface. It
provides developer utilities — Docker container lifecycle helpers,
test fixtures, scratch tools — that are convenient when developing
against the library but not intended for production deployment.

Modules under :mod:`cygnet.dev`:

- :mod:`cygnet.dev.neo4j_lifecycle` — manages a Neo4j 5 + APOC
  container via ``docker compose``. Used by the benchmark suite,
  by the ``library_tests/scripts/`` reference-dataset checks, and
  by demo notebooks that bring up a local Neo4j.

Docker is a lazy import inside each consumer that needs it; the
``cygnet.dev`` package imports cleanly without Docker installed.
The actual ``docker`` CLI is only invoked when a function in
:mod:`cygnet.dev.neo4j_lifecycle` is called.
"""

from __future__ import annotations

__all__: list[str] = []
