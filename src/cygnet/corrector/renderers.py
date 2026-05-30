# Copyright (c) 2026 Nikodem Tomczak, Thulge Labs. All rights reserved.

"""Shared prompt-block renderers for (query, error) pairs.

Both the corrector intent block (the "current attempt" framing) and
the prior-attempt history blocks (the "what was tried and how it
failed" framing) render the same body: a fenced query block, the
error's category, and the full validator payload as JSON. Only the
header differs. The :func:`render_gate_error_block` function takes a
``header_kind`` literal and produces the right framing.

The corrector's multi-error "fix-all-at-once" intent block (collect-
all mode) has a meaningfully different body (per-error subsections,
no single category line) and lives in :func:`render_multi_error_intent`
to keep the main signature clean. Two functions, one body shape each.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from cygnet.models import GateError

__all__ = ["render_gate_error_block", "render_multi_error_intent"]


HeaderKind = Literal["intent", "prior_attempt"]


def render_gate_error_block(
    *,
    query: str,
    error: GateError,
    header_kind: HeaderKind,
    attempt_index: int | None = None,
) -> str:
    """Render one (query, error) pair as a prompt block.

    Args:
        query: the Cypher query the error is about.
        error: the validator's structured failure.
        header_kind: ``"intent"`` for the current-attempt framing
            (``# Refinement intent (attempt N)``) or
            ``"prior_attempt"`` for history-feedback framing
            (``# Prior attempt N``).
        attempt_index: required when ``header_kind`` is ``"intent"``
            (the attempt number) or ``"prior_attempt"`` (the zero-
            based position in the history). Pass-through to the
            header; not interpreted.

    Returns the rendered markdown block. Pure: no side effects, no IO.
    """
    if attempt_index is None:
        raise ValueError(
            "render_gate_error_block: attempt_index is required for both header_kind values"
        )

    payload_json = json.dumps(
        error.payload.model_dump(mode="json"),
        indent=2,
        default=str,
    )

    if header_kind == "intent":
        return (
            f"# Refinement intent (attempt {attempt_index})\n\n"
            "## Failing query\n\n"
            "```cypher\n"
            f"{query}\n"
            "```\n\n"
            f"## Failure category: `{error.category}`\n\n"
            "## Error payload (JSON)\n\n"
            "```json\n"
            f"{payload_json}\n"
            "```\n"
        )

    # header_kind == "prior_attempt"
    return (
        f"# Prior attempt {attempt_index}\n\n"
        "## Query\n\n"
        "```cypher\n"
        f"{query}\n"
        "```\n\n"
        f"## Failed with category `{error.category}`\n\n"
        "Payload:\n\n"
        "```json\n"
        f"{payload_json}\n"
        "```\n"
    )


def render_multi_error_intent(
    *,
    query: str,
    errors: list[GateError],
    attempt_number: int,
) -> str:
    """Render the corrector's collect-all-mode intent block.

    The chain ran in ``collect_all`` and produced N errors that need
    fixing in one shot. The framing line shifts to "fix all the
    errors below" and each error gets its own subsection.

    Pre-condition: ``len(errors) >= 2``. Single-error callers should
    use :func:`render_gate_error_block` with ``header_kind="intent"``
    — that path has a more compact header and a single category line.
    """
    if len(errors) < 2:
        raise ValueError(
            "render_multi_error_intent: expected >=2 errors; for single-"
            "error callers use render_gate_error_block(header_kind='intent')"
        )
    parts = [
        f"# Refinement intent (attempt {attempt_number})\n",
        "## Failing query\n",
        "```cypher",
        query,
        "```\n",
        f"## Fix all {len(errors)} errors below in one refined query\n",
        (
            "The chain ran in `collect_all` mode and reported every "
            "issue every backend found. Produce one refined query that "
            "addresses every error. The errors are listed in priority "
            "order (parse-shape problems first, then by backend "
            "authority).\n"
        ),
    ]
    for i, error in enumerate(errors, start=1):
        payload_json = json.dumps(
            error.payload.model_dump(mode="json"),
            indent=2,
            default=str,
        )
        parts.append(f"### Error {i} — category `{error.category}`\n")
        parts.append("```json")
        parts.append(payload_json)
        parts.append("```\n")
    return "\n".join(parts)
