# Copyright (c) 2026 Nikodem Tomczak, Thulge Labs. All rights reserved.

"""Shared formatting helpers used by validator backends and the cost gate.

Two utilities live here so all three backends (`builtin`, `ast`,
`explain`) produce error payloads with the same shape:

- :func:`excerpt_with_caret` builds the IDE-style two-line excerpt
  for ``ParseError.excerpt_with_caret``.
- :func:`bound_available_in_scope` caps the vocabulary list for
  ``SchemaError.available_in_scope`` at the per-payload limit and
  returns the matching ``available_in_scope_truncated`` flag.
"""

from __future__ import annotations

from collections.abc import Iterable

__all__ = ["bound_available_in_scope", "excerpt_with_caret"]


AVAILABLE_IN_SCOPE_CAP: int = 50
"""Maximum entries returned by :func:`bound_available_in_scope`. Matches
the documented cap on ``SchemaError.available_in_scope``."""


def excerpt_with_caret(query: str, line: int, column: int) -> str | None:
    """Return an IDE-style ``"<line>\\n<spaces>^"`` excerpt or ``None``.

    Args:
        query: the source Cypher.
        line: 1-based line number of the offending region.
        column: 1-based column number of the offending region.

    Returns ``None`` when the position is non-positive, points past the
    end of the query, or lands on an empty line.
    """
    if line < 1 or column < 1:
        return None
    lines = query.split("\n")
    if line > len(lines):
        return None
    target = lines[line - 1]
    if not target:
        return None
    column_clamped = min(column, len(target) + 1)
    caret = " " * (column_clamped - 1) + "^"
    return f"{target}\n{caret}"


def bound_available_in_scope(vocabulary: Iterable[str]) -> tuple[list[str], bool]:
    """Sort, deduplicate, and cap a vocabulary for the corrector.

    Returns ``(entries, truncated)``: the first :data:`AVAILABLE_IN_SCOPE_CAP`
    entries alphabetically, plus a bool that is True when the input
    contained more entries than the cap.
    """
    items = sorted({s for s in vocabulary if s})
    truncated = len(items) > AVAILABLE_IN_SCOPE_CAP
    return items[:AVAILABLE_IN_SCOPE_CAP], truncated
