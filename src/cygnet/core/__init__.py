# Copyright (c) 2026 Nikodem Tomczak, Thulge Labs. All rights reserved.

"""Core orchestration: the `Gate` class that wires validator, cost, and corrector.

The `Gate` does not loop, retry, or execute queries against production
itself; it provides the per-call gate operations and exposes the schema.
Callers compose the agent loop on top.

Modules planned (per docs/architecture.md):
- `gate`: the Gate orchestration object (`from_config`, `validate`,
  `estimate_cost`, `gate`, `correct`, `get_schema`, `refresh_schema`).

This slice (bootstrap) ships a placeholder Gate at the package root
(`cygnet.Gate`); the real implementation lands here in a later slice.
"""
