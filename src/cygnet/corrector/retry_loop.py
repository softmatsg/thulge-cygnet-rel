# Copyright (c) 2026 Nikodem Tomczak, Thulge Labs. All rights reserved.

"""Generic retry loop.

A library class that retries a callable up to ``max_attempts`` times
and stops when an injected ``stop_condition`` says so. It is not
correction-specific and can drive any task where the right shape is
"call something repeatedly until one of N attempts satisfies a
caller-defined success test."

The corrector world uses it via
:class:`cygnet.corrector.refinement_loop.RefinementLoop`, which
supplies a correction-flavoured work callable + an
:class:`AcceptanceCriteria`-shaped stop condition. The transport
layer (:class:`cygnet.corrector.llm.ResilientLLMClient`) is a
DIFFERENT retry layer (HTTP 429/503/timeout); it does not use this
loop. The protocol decorators (:class:`ProtocolRetryingCorrector`,
:class:`EmptyRetryingCorrector`) are yet a THIRD retry layer
(malformed JSON / declined-to-refine); they don't use this loop
either. See the package docstring of :mod:`cygnet.corrector` for the
layering picture.

Design rules pinned by the seam-map:

- The loop does not know what "done" means. The stop condition does.
- The loop does not know what the work returns. It carries each
  attempt's return value forward unmodified.
- Between-attempt sleep is a generic backoff knob with default 0.0;
  the correction path doesn't need it because protocol-level retry
  has its own backoff inside the protocol decorators, and transport
  retry has its own in :class:`RetryPolicy`. The knob exists for
  callers who use the loop for non-correction purposes.

A non-correction example (also exercised by the test suite to keep
the genericity honest)::

    loop = RetryLoop(max_attempts=10)
    secret = 7
    guesses: list[int] = []

    def guess(attempt: int) -> int:
        n = attempt * 2  # deterministic; just for illustration
        guesses.append(n)
        return n

    def correct(value: int, attempt: int) -> bool:
        return value == secret

    outcome = loop.run(guess, correct)
    # outcome.stopped_by == "exhausted" because 7 isn't even.

"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ["RetryLoop", "RetryLoopOutcome", "StopCondition"]


# The stop-condition contract. Implementations decide when the loop
# should terminate. Receives the latest work result and the 1-based
# attempt number. Returns ``True`` to stop, ``False`` to continue.
# Type alias rather than a ``Protocol`` so any callable with the
# right shape works without subclassing. :class:`AcceptanceCriteria`
# satisfies this contract via its :meth:`should_stop` method
# (Phase C reshape).
StopCondition = "Callable[[Any, int], bool]"


@dataclass(frozen=True)
class RetryLoopOutcome:
    """One :class:`RetryLoop.run` result.

    Carries every attempt's work result in chronological order, plus
    metadata about why the loop terminated. The work-result type is
    deliberately ``Any`` — the loop doesn't introspect it.

    Attributes:
        attempts: per-attempt work results in chronological order.
        total_attempts: ``len(attempts)``. Convenience for callers
            that don't want to take a length.
        stopped_by: ``"stop_condition"`` when the injected stop
            condition returned ``True`` on one of the attempts;
            ``"exhausted"`` when the loop hit ``max_attempts``
            without any attempt satisfying the stop condition.
        last_result: the most recent attempt's work result. ``None``
            only when the loop ran zero iterations (impossible given
            ``max_attempts >= 1`` is validated at construction).
    """

    attempts: list[Any] = field(default_factory=list)
    total_attempts: int = 0
    stopped_by: Literal["stop_condition", "exhausted"] = "exhausted"
    last_result: Any = None


class RetryLoop:
    """Generic retry loop.

    Args:
        max_attempts: hard cap on the number of times ``run`` calls
            the work callable. Default 3 matches the correction
            world's default refinement budget. Must be >= 1.
        between_attempts_sleep_seconds: real-time sleep between
            attempts when ``> 0``. Default 0.0 (no sleep). This is
            the generic backoff knob; the correction path doesn't
            use it. See the module docstring for why.
    """

    def __init__(
        self,
        *,
        max_attempts: int = 3,
        between_attempts_sleep_seconds: float = 0.0,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("RetryLoop: max_attempts must be >= 1")
        if between_attempts_sleep_seconds < 0:
            raise ValueError("RetryLoop: between_attempts_sleep_seconds must be >= 0")
        self._max_attempts = max_attempts
        self._sleep_seconds = between_attempts_sleep_seconds

    @property
    def max_attempts(self) -> int:
        return self._max_attempts

    def run(
        self,
        work: Callable[[int], Any],
        stop_condition: Callable[[Any, int], bool],
    ) -> RetryLoopOutcome:
        """Run the loop.

        Args:
            work: callable invoked with the 1-based attempt number.
                Returns the attempt's result. The loop does not
                introspect the return; whatever the caller produced
                lands in :attr:`RetryLoopOutcome.attempts`.
            stop_condition: callable invoked with ``(result,
                attempt_number)`` AFTER each ``work`` call. Returns
                ``True`` to stop the loop, ``False`` to continue.

        Returns:
            A :class:`RetryLoopOutcome` recording each attempt's
            result, the total attempt count, why the loop stopped,
            and the last result.
        """
        attempts: list[Any] = []
        for attempt_number in range(1, self._max_attempts + 1):
            if attempt_number > 1 and self._sleep_seconds > 0:
                time.sleep(self._sleep_seconds)
            result = work(attempt_number)
            attempts.append(result)
            if stop_condition(result, attempt_number):
                return RetryLoopOutcome(
                    attempts=attempts,
                    total_attempts=attempt_number,
                    stopped_by="stop_condition",
                    last_result=result,
                )
        return RetryLoopOutcome(
            attempts=attempts,
            total_attempts=self._max_attempts,
            stopped_by="exhausted",
            last_result=attempts[-1] if attempts else None,
        )
