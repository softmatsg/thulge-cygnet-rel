# Copyright (c) 2026 Nikodem Tomczak, Thulge Labs. All rights reserved.

"""Null corrector: the safe default when no real corrector is wired in.

Always returns ``action="abort"``. Lets ``Gate.correct`` return a
typed :class:`CorrectorResult` without requiring an LLM client or
network access; users who want actual refinement plug in their own
corrector (the RAMPART-backed default arrives in a later slice).

This is not a no-op — the result carries a clear ``reasoning``
string explaining to the agent loop that no refinement was attempted.
That signal is the corrector's stable surface: callers who pattern-
match on ``action`` see ``"abort"`` and stop the loop cleanly rather
than mis-interpreting silent ``None`` returns.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from cygnet.models import CorrectorResult

if TYPE_CHECKING:
    from cygnet.corrector.interface import CorrectorContext
    from cygnet.corrector.telemetry import ObservationCallback
    from cygnet.models import GateError

__all__ = ["NullCorrector"]


_REASONING: str = (
    "NullCorrector does not refine queries. Configure a real corrector "
    "by setting CorrectorConfig.corrector to a factory returning a "
    "Corrector instance, or pass corrector= to Gate.from_config."
)


class NullCorrector:
    """Reference Corrector implementation. Returns ``action='abort'``.

    Accepts ``on_observation`` (per the v0.0.42 unified protocol) and
    ignores it — :class:`NullCorrector` makes no LLM calls and has
    nothing to emit."""

    def correct(
        self,
        query: str,
        error: GateError,
        context: CorrectorContext,
        *,
        on_observation: ObservationCallback | None = None,
    ) -> CorrectorResult:
        return CorrectorResult(
            action="abort",
            refined_query=None,
            reasoning=_REASONING,
            attempts_used=context.attempt_number,
        )
