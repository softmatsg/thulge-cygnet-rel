# Copyright (c) 2026 Nikodem Tomczak, Thulge Labs. All rights reserved.

"""Core orchestration: the `Gate` class that wires validator, cost, and corrector.

The `Gate` does not loop, retry, or execute queries against production
itself; it provides the per-call gate operations and exposes the schema.
Callers compose the agent loop on top.

Re-exports the package-root Gate construction surface.
"""
