# Copyright (c) 2026 Nikodem Tomczak, Thulge Labs. All rights reserved.

"""CLI entry points.

CLI commands are thin wrappers around the library API; everything callable
via CLI is first available as `cygnet.<function>(...)`. Notebooks are the
primary exploration/demo surface; the CLI is for scheduled jobs and
deployment integration.

Modules planned (per docs/architecture.md):
- `audit`: drift audit comparing schema spec vs introspected schema.
- `serve`: transport launcher (MCP, HTTP).

This slice (bootstrap) ships the package directory only.
"""
