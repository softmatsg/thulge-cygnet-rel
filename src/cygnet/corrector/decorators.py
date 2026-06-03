# Copyright (c) 2026 Nikodem Tomczak, Thulge Labs. All rights reserved.

"""Protocol-level and Empty-retry corrector decorators.

These decorators compose retry behaviour on top of single-shot
:class:`Corrector` implementations:

- :class:`ProtocolRetryingCorrector` — on
  :class:`ProtocolMalformed` from the inner corrector, retry with
  an incremented ``_protocol_attempt`` metadata hint. The inner
  corrector (currently only :class:`RampartCorrector`) interprets
  the hint to escalate its system prompt with a retry preamble;
  other correctors ignore the hint and produce the same response.
  Default ``retries=2`` gives ``protocol_retries=2`` (3 total LLM
  calls per :meth:`correct`).

- :class:`EmptyRetryingCorrector` — on :class:`ProtocolEmpty` from
  the inner corrector, retry once at a provider-appropriate high
  temperature. ``_temperature`` and ``_protocol_attempt`` metadata
  hints are set so the inner can pick up the new temperature and
  emit the on-disk ``+100`` marker convention.

Composition for the production wrapping::

    ProtocolRetryingCorrector(
        EmptyRetryingCorrector(
            RampartCorrector(...),
            high_temp_by_provider=...,
            provider=...,
        ),
        retries=2,
    )

This is the wrapping that :func:`cygnet.run_correction` applies
automatically. Direct callers of :class:`RampartCorrector` get
single-shot behaviour; wrap explicitly to reproduce the production
configuration.

Both decorators are transparent to non-LLM-backed correctors
(:class:`NullCorrector` produces a single abort regardless of
retry attempts). The decorators are safe to apply to any
:class:`Corrector`.

The three retry layers remain composable and distinct:

- Transport retry: :class:`cygnet.corrector.llm.ResilientLLMClient`
  / :class:`cygnet.corrector.llm.RetryPolicy` (HTTP 429 / 503 /
  timeout).
- Protocol retry: :class:`ProtocolRetryingCorrector` / :class:`EmptyRetryingCorrector`
  (malformed JSON / declined-to-refine).
- Refinement retry: :class:`cygnet.corrector.refinement_loop.RefinementLoop`
  (validator-rejected refinement).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Final

from cygnet.models import CorrectorResult

if TYPE_CHECKING:
    from cygnet.corrector.interface import Corrector, CorrectorContext
    from cygnet.corrector.telemetry import ObservationCallback
    from cygnet.models import GateError

__all__ = ["EmptyRetryingCorrector", "ProtocolRetryingCorrector"]


_logger = logging.getLogger("cygnet.corrector.decorators")


# Default high-temperature values per provider. The decorator owns
# its own defaults rather than reaching into the corrector it wraps.
_DEFAULT_HIGH_TEMP_BY_PROVIDER: Final[dict[str, float]] = {
    "anthropic": 0.7,
    "openai": 0.7,
    "gemini": 0.9,
    "google": 0.9,
    "ollama": 0.7,
    "local": 0.7,
    # Fallback applied when the provider is unknown or the model
    # identifier doesn't carry a recognisable provider prefix.
    "__default__": 0.7,
}


class ProtocolRetryingCorrector:
    """Decorator that retries the inner corrector on
    :class:`ProtocolMalformed` aborts.

    On each retry, sets ``context.metadata["_protocol_attempt"]`` to
    the 1-based attempt number. The inner corrector reads this hint
    to escalate its prompt (RAMPART prepends a retry preamble on
    attempts > 1; the template correctors are hint-blind and produce
    the same response).

    Args:
        inner: any :class:`Corrector` whose ``correct`` follows the
            unified protocol signature.
        retries: how many ADDITIONAL attempts beyond the first to
            make on :class:`ProtocolMalformed` (``reason="protocol_failure"``)
            results. Default ``2`` (``protocol_retries=2``): one
            first attempt + 2 retries = 3 total LLM calls.

    The returned :class:`CorrectorResult` has
    ``protocol_attempts`` aggregated across all inner calls. The
    on-disk observations (one per call, emitted via
    ``on_observation``) carry the per-call
    :data:`LLMCallObservation.protocol_attempt` value the inner
    chose for that call.
    """

    def __init__(self, inner: Corrector, *, retries: int = 2) -> None:
        if retries < 0:
            raise ValueError("ProtocolRetryingCorrector: retries must be >= 0")
        self._inner = inner
        self._retries = retries

    @property
    def inner(self) -> Corrector:
        """The wrapped corrector, exposed for introspection / tests."""
        return self._inner

    def correct(
        self,
        query: str,
        error: GateError,
        context: CorrectorContext,
        *,
        on_observation: ObservationCallback | None = None,
    ) -> CorrectorResult:
        max_attempts = self._retries + 1
        # The starting protocol attempt is whatever the outer decorator
        # already supplied (a nested EmptyRetryingCorrector may have
        # set it to a high-temp marker like 101). Default to "1" when
        # the metadata is empty.
        base_attempt = int(context.metadata.get("_protocol_attempt", "1"))
        # If a nested decorator already set the metadata (e.g. base
        # is 101 meaning high-temp marker on attempt 1), preserve
        # that pattern when iterating: the iteration mod-100 ladder
        # is "1, 2, 3" but the actual stored value carries the
        # high-temp offset.
        high_temp_offset = (base_attempt // 100) * 100
        last_result: CorrectorResult | None = None
        total_protocol_calls = 0

        for attempt in range(1, max_attempts + 1):
            stored_attempt = attempt + high_temp_offset
            new_metadata = {
                **context.metadata,
                "_protocol_attempt": str(stored_attempt),
            }
            new_context = context.model_copy(update={"metadata": new_metadata})
            result = self._inner.correct(query, error, new_context, on_observation=on_observation)
            total_protocol_calls += max(1, result.protocol_attempts)

            is_protocol_failure = result.action == "abort" and result.reason == "protocol_failure"
            if not is_protocol_failure:
                # OK / Echoed / Empty / other-abort: terminal. Aggregate
                # protocol_attempts across all inner calls and forward.
                return result.model_copy(update={"protocol_attempts": total_protocol_calls})

            last_result = result
            if attempt < max_attempts:
                _logger.info(
                    "ProtocolRetryingCorrector: malformed on attempt %d/%d (%s); retrying",
                    attempt,
                    max_attempts,
                    result.reasoning,
                )

        # Exhausted: last_result is the final protocol_failure abort.
        # Aggregate the protocol_attempts count and return it.
        assert last_result is not None
        return last_result.model_copy(update={"protocol_attempts": total_protocol_calls})


class EmptyRetryingCorrector:
    """Decorator that retries the inner corrector once at high
    temperature on :class:`ProtocolEmpty` aborts.

    On the retry, sets two metadata hints:

    - ``_temperature`` (default ``0.7``-``0.9`` depending on
      provider; configurable via ``high_temp_by_provider``).
    - ``_protocol_attempt`` set to ``(current + 100)``, the on-disk
      marker for "this is the high-temperature retry".

    Args:
        inner: any :class:`Corrector` following the unified protocol.
        high_temp_by_provider: per-provider temperature for the
            retry. Falls back to ``__default__`` (``0.7``) for
            unknown providers.
        provider: provider identifier used to look up the
            temperature value. ``None`` is treated as "unknown" and
            uses the ``__default__`` value.

    The returned :class:`CorrectorResult` has
    ``used_high_temp_retry=True`` when the retry fired; ``False``
    when the inner returned non-Empty on the first call. The
    ``protocol_attempts`` field is aggregated across inner +
    retry.
    """

    def __init__(
        self,
        inner: Corrector,
        *,
        high_temp_by_provider: dict[str, float] | None = None,
        provider: str | None = None,
    ) -> None:
        self._inner = inner
        self._high_temp_by_provider: dict[str, float] = {
            **_DEFAULT_HIGH_TEMP_BY_PROVIDER,
            **(high_temp_by_provider or {}),
        }
        self._provider = provider

    @property
    def inner(self) -> Corrector:
        """The wrapped corrector, exposed for introspection / tests."""
        return self._inner

    def correct(
        self,
        query: str,
        error: GateError,
        context: CorrectorContext,
        *,
        on_observation: ObservationCallback | None = None,
    ) -> CorrectorResult:
        first_result = self._inner.correct(query, error, context, on_observation=on_observation)

        # Non-Empty: terminal. Forward as-is.
        if not (first_result.action == "abort" and first_result.reason == "empty_cypher"):
            return first_result

        # Empty: one high-temperature retry.
        high_temp = self._high_temp_by_provider.get(
            self._provider or "", self._high_temp_by_provider["__default__"]
        )
        current_attempt = int(context.metadata.get("_protocol_attempt", "1"))
        # Add +100 so the on-disk observation marker identifies this
        # call as the high-temperature retry. If a nested
        # ProtocolRetryingCorrector is wrapping this and already
        # added an offset, preserve the logical-attempt mod 100; just
        # bump the marker by 100.
        new_metadata = {
            **context.metadata,
            "_protocol_attempt": str(current_attempt + 100),
            "_temperature": str(high_temp),
        }
        new_context = context.model_copy(update={"metadata": new_metadata})

        _logger.info(
            "EmptyRetryingCorrector: empty on first call; retrying at temperature=%.2f",
            high_temp,
        )
        retry_result = self._inner.correct(query, error, new_context, on_observation=on_observation)

        total_attempts = first_result.protocol_attempts + retry_result.protocol_attempts

        # Retry also returned Empty: terminal "both attempts empty".
        if retry_result.action == "abort" and retry_result.reason == "empty_cypher":
            return CorrectorResult(
                action="abort",
                refined_query=None,
                reasoning=(
                    f"{type(self._inner).__name__}: model returned empty cypher "
                    "field at both low and high temperature."
                ),
                attempts_used=context.attempt_number,
                reason="empty_cypher_after_retry",
                protocol_attempts=total_attempts,
                used_high_temp_retry=True,
            )

        # Other outcomes (refined / echoed / protocol_failure /
        # exception): forward the retry result with adjusted fields.
        return retry_result.model_copy(
            update={
                "protocol_attempts": total_attempts,
                "used_high_temp_retry": True,
            }
        )
