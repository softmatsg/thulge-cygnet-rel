# Copyright (c) 2026 Nikodem Tomczak, Thulge Labs. All rights reserved.

"""Shared outcome → :class:`CorrectorResult` helper (
v0.0.42).

The four parser outcomes (``ProtocolOK`` / ``ProtocolEchoed`` /
``ProtocolEmpty`` / ``ProtocolMalformed``) map to a :class:`CorrectorResult`
in the same way regardless of which corrector produced them. Pre-
v0.0.42 the mapping was hand-coded in both :class:`RampartCorrector`
and :class:`TemplateCorrector`; v0.0.42 factors it into this free
function. Per the seam-map Decision 4, this is intentionally
NOT a mixin or base-class method — keeping the correctors loosely
coupled was the rationale.

The function preserves the v0.0.30 outcome reasoning strings
verbatim (per-corrector format strings) so a behaviour-preservation
test can pin equivalence vs the pre-Phase-B implementations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from cygnet.corrector.response_parser import (
    ProtocolEchoed,
    ProtocolEmpty,
    ProtocolMalformed,
    ProtocolOK,
)
from cygnet.models import CorrectorResult

if TYPE_CHECKING:
    from cygnet.corrector.response_parser import ProtocolOutcome

__all__ = ["outcome_to_result"]


def outcome_to_result(
    outcome: ProtocolOutcome,
    *,
    attempt_number: int,
    corrector_name: str,
    protocol_attempts: int = 1,
    used_high_temp_retry: bool = False,
) -> CorrectorResult:
    """Map a parser outcome to a :class:`CorrectorResult`.

    Args:
        outcome: the parser's verdict on the LLM response.
        attempt_number: outer-loop ``CorrectorContext.attempt_number``
            value to propagate into the result.
        corrector_name: human-friendly class name used in the
            ``reasoning`` string (e.g. ``"RampartCorrector"`` or
            ``"RawCorrector"``). Matches the pre-v0.0.42 wording.
        protocol_attempts: how many LLM calls this corrector made
            for this single ``correct`` invocation. Default 1 —
            the unified protocol is single-shot. Decorators
            (:class:`ProtocolRetryingCorrector` /
            :class:`EmptyRetryingCorrector`) overwrite this when
            wrapping.
        used_high_temp_retry: True when the result came from an
            :class:`EmptyRetryingCorrector`'s high-temperature
            retry. Threaded through so consumers can attribute
            behaviour.

    Returns:
        A :class:`CorrectorResult` with the appropriate ``action``,
        ``refined_query``, ``reason``, and ``reasoning`` fields.

    The four-branch dispatch:

    - :class:`ProtocolOK` → ``action="refined"`` with the cypher.
    - :class:`ProtocolEchoed` → ``action="abort",
      reason="model_echoed_input"``; the echoed cypher is preserved
      on ``refined_query`` for inspection.
    - :class:`ProtocolEmpty` → ``action="abort",
      reason="empty_cypher"`` (the corrector saw the model decline;
      :class:`EmptyRetryingCorrector` may still retry above this).
    - :class:`ProtocolMalformed` → ``action="abort",
      reason="protocol_failure"``.
    """
    if isinstance(outcome, ProtocolOK):
        suffix = " (high-temp retry)" if used_high_temp_retry else ""
        return CorrectorResult(
            action="refined",
            refined_query=outcome.cypher,
            reasoning=(
                f"Refined via {corrector_name}; protocol_attempts={protocol_attempts}{suffix}"
            ),
            attempts_used=attempt_number,
            reason=None,
            protocol_attempts=protocol_attempts,
            used_high_temp_retry=used_high_temp_retry,
        )
    if isinstance(outcome, ProtocolEchoed):
        return CorrectorResult(
            action="abort",
            refined_query=outcome.cypher,
            reasoning=(
                f"{corrector_name}: model returned the input query "
                "unchanged after whitespace + keyword-case "
                "normalisation (declined to refine)."
            ),
            attempts_used=attempt_number,
            reason="model_echoed_input",
            protocol_attempts=protocol_attempts,
            used_high_temp_retry=used_high_temp_retry,
        )
    if isinstance(outcome, ProtocolEmpty):
        return CorrectorResult(
            action="abort",
            refined_query=None,
            reasoning=f"{corrector_name}: model returned an empty cypher field.",
            attempts_used=attempt_number,
            reason="empty_cypher",
            protocol_attempts=protocol_attempts,
            used_high_temp_retry=used_high_temp_retry,
        )
    if isinstance(outcome, ProtocolMalformed):
        return CorrectorResult(
            action="abort",
            refined_query=None,
            reasoning=(f"{corrector_name} aborted: protocol_failure; reason: {outcome.reason}"),
            attempts_used=attempt_number,
            reason="protocol_failure",
            protocol_attempts=protocol_attempts,
            used_high_temp_retry=used_high_temp_retry,
        )
    raise AssertionError(
        f"outcome_to_result: unexpected ProtocolOutcome type "
        f"{type(outcome).__name__!r}; protocol parser added a fifth outcome?"
    )
