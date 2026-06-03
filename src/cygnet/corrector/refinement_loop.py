# Copyright (c) 2026 Nikodem Tomczak, Thulge Labs. All rights reserved.

"""Outer refinement loop.

The corrector's :meth:`Corrector.correct` returns a single refined
Cypher (or an abort) for one ``(query, error)`` pair. It does NOT
run the validator chain on the result; it does NOT decide whether
to retry with a new error if the result still fails. Those are
*outer-loop* concerns, handled by :class:`RefinementLoop`.

End-to-end refinement, in one call:

1. Initialise an empty ``prior_attempts`` list.
2. For each refinement attempt up to ``max_attempts``:

   a. Construct a fresh :class:`CorrectorContext` with a snapshot
      of ``prior_attempts``.
   b. Call the corrector. Collect the per-LLM-call
      :class:`LLMCallObservation` records it emits (one per
      protocol attempt).
   c. Apply :class:`AcceptanceCriteria` — does this result count
      as a refinement, or do we treat the corrector's "refined"
      action as a soft-abort (e.g. echoed input when
      ``require_distinct_from_input=True``)?
   d. If acceptable, run the validator chain on the refined
      cypher.
   e. Wrap each observation in :class:`LLMCallRecord` with
      outer-loop context (query_id / condition / refinement_attempt
      / validator_outcome) and forward to telemetry.
   f. If validator passed (or acceptance says don't validate),
      return :class:`RefinementResult` with ``action="refined"``.
   g. If corrector aborted, return ``action="abort"`` with the
      corrector's reason.
   h. Otherwise: append ``(refined_query, new_error)`` to
      ``prior_attempts`` and loop.

3. If we exhaust ``max_attempts``: return abort with the
   last attempt's cypher preserved in ``refined_query`` (for
   debugging / inspection).

Layering: this module knows about the corrector protocol, the
validator chain, and the telemetry hook. It does NOT know about
the Gate, GateConfig, or any wider library context. The Gate
constructs a ``RefinementLoop`` in :meth:`Gate.__init__` from
its existing corrector + chain and delegates :meth:`Gate.correct`
to it.

The corrector's internal :class:`CorrectorResult` is kept as the
corrector's return type. The loop wraps its sequence of
:class:`CorrectorResult` outputs into a single
:class:`RefinementResult` that callers consume.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field

from cygnet.corrector.interface import CorrectorContext, PriorAttempt
from cygnet.corrector.retry_loop import RetryLoop
from cygnet.corrector.telemetry import (
    CorrectorTelemetry,
    LLMCallObservation,
    LLMCallRecord,
    NullTelemetry,
    ObservationCallback,
)
from cygnet.models import CorrectorResult, GateError

if TYPE_CHECKING:
    from cygnet.corrector.interface import Corrector
    from cygnet.models import Schema
    from cygnet.validator.chain import ValidatorChain

__all__ = [
    "AcceptanceCriteria",
    "RefinementAttempt",
    "RefinementLoop",
    "RefinementResult",
]


_logger = logging.getLogger("cygnet.corrector.refinement_loop")


# ---------------------------------------------------------------------------
# Result / configuration types
# ---------------------------------------------------------------------------


class AcceptanceCriteria(BaseModel):
    """Defines what counts as a successful refinement.

    The loop applies these rules to the corrector's per-attempt
    output to decide whether to terminate (success) or continue
    iterating (try again with the new validator error).

    Default policy is the strictest: the refinement must pass
    validation AND be distinct from the input.
    """

    model_config = ConfigDict(extra="forbid")

    require_validates: bool = Field(
        default=True,
        description=(
            "If True, the refined cypher must pass the validator "
            "chain before being accepted. If False, the loop "
            "returns whatever cypher the corrector produced, "
            "validated or not. False is useful when the caller "
            "wants to inspect refinements that the validator "
            "rejects."
        ),
    )
    require_distinct_from_input: bool = Field(
        default=True,
        description=(
            "If True, an echoed refinement (model returned the input "
            "query unchanged after whitespace + keyword-case "
            "normalisation) is treated as an abort. The corrector "
            "marks these as ``action='abort', reason='model_echoed_input'``; "
            "callers that explicitly want to keep echoed cypher can "
            "set this False."
        ),
    )
    backends: list[str] | None = Field(
        default=None,
        description=(
            "Optional override of which validator backends to run "
            "when ``require_validates=True``. ``None`` means use "
            "the gate's configured chain unchanged. A list of "
            "backend names (matching ``ValidatorChainConfig.backends``) "
            "narrows the check to just that subset — handy when the "
            "caller wants to accept refinements that pass the cheap "
            "checks even if EXPLAIN says no."
        ),
    )


class RefinementAttempt(BaseModel):
    """One refinement-attempt record. ``RefinementResult.attempts``
    is a list of these in chronological order — one per refinement
    attempt the loop made before terminating."""

    model_config = ConfigDict(extra="forbid")

    attempt_number: int = Field(..., ge=1)
    refined_query: str | None = Field(
        default=None,
        description=(
            "Cypher the corrector returned for this attempt. ``None`` "
            "when the corrector aborted before producing a refinement "
            "(e.g. a protocol failure that exhausted all retries)."
        ),
    )
    protocol_outcome: Literal["ok", "echoed", "empty", "malformed", "exception"] = Field(
        ...,
        description=(
            "What the parser said about the corrector's last LLM "
            "response on this attempt. ``ok`` = refined; ``echoed`` "
            "= model returned the input; ``empty`` = explicit "
            "'I cannot refine'; ``malformed`` = retries exhausted "
            "without parseable JSON; ``exception`` = SDK raised."
        ),
    )
    protocol_attempts: int = Field(
        ...,
        ge=0,
        description=(
            "How many LLM calls the corrector made for this refinement "
            "attempt. ``0`` is valid — e.g. :class:`NullCorrector` "
            "aborts without making any LLM call."
        ),
    )
    validator_outcome: Literal["passed", "failed", "not_run"] = Field(
        ...,
        description=(
            "``passed`` when the validator chain accepted the refined "
            "cypher; ``failed`` when it rejected it; ``not_run`` when "
            "the loop didn't run the chain (corrector aborted, "
            "echoed, or ``acceptance.require_validates=False``)."
        ),
    )
    validator_error: GateError | None = Field(
        default=None,
        description=(
            "Error returned by the validator chain when "
            "``validator_outcome='failed'``. ``None`` otherwise."
        ),
    )
    elapsed_seconds: float = Field(
        default=0.0,
        ge=0.0,
        description="Sum of wall-clock time across all LLM calls in this attempt.",
    )
    input_tokens: int = Field(
        default=0,
        ge=0,
        description=(
            "Sum of prompt tokens across every LLM call this attempt "
            "made — including the corrector's internal protocol "
            "retries and the high-temperature retry. Aggregated by "
            "the loop from each :class:`LLMCallObservation` so "
            "consumers don't need a side-band token-counting fold."
        ),
    )
    output_tokens: int = Field(
        default=0,
        ge=0,
        description=(
            "Sum of output tokens across every LLM call this attempt "
            "made (includes any reasoning tokens reported by thinking "
            "models). Aggregated by the loop."
        ),
    )


class RefinementResult(BaseModel):
    """The outer-loop output of :meth:`RefinementLoop.refine` (and
    hence of :meth:`Gate.correct`).

    Carries the final verdict, the refined cypher (if any), and a
    full chronological history of attempts. Token counts are
    available per-attempt via the telemetry hook; total cost is a
    consumer concern.

    .. important::
       ``refined_query`` is NOT a success signal. The field may
       carry text on ``action="abort"`` whenever the loop produced
       *some* cypher before deciding to abort — for example, on
       ``reason="model_echoed_input"`` the echoed cypher is
       preserved, and on ``reason="max_attempts_exhausted"`` the
       last attempt's cypher is preserved for inspection. Treat
       ``action == "refined"`` as the success condition; treat
       ``refined_query`` as "the last thing the loop produced,
       valid or not".
    """

    model_config = ConfigDict(extra="forbid")

    action: Literal["refined", "abort"] = Field(
        ...,
        description=(
            "``refined`` when at least one attempt produced a "
            "validated cypher (or an acceptable cypher under the "
            "configured acceptance criteria). ``abort`` otherwise."
        ),
    )
    refined_query: str | None = Field(
        default=None,
        description=(
            "On ``action='refined'``: the validated / accepted cypher. "
            "On ``action='abort'``: the last cypher the loop produced "
            "(if any) — useful for inspection. ``None`` when no "
            "attempt produced parseable cypher."
        ),
    )
    reason: str | None = Field(
        default=None,
        description=(
            "Structured abort reason: ``protocol_failure`` / "
            "``model_echoed_input`` / ``empty_cypher_after_retry`` / "
            "``exception`` / ``max_attempts_exhausted``. ``None`` on "
            "``action='refined'``."
        ),
    )
    attempts: list[RefinementAttempt] = Field(
        default_factory=list,
        description="Chronological per-attempt history.",
    )
    total_protocol_attempts: int = Field(
        default=0,
        ge=0,
        description=(
            "Sum of ``RefinementAttempt.protocol_attempts`` across "
            "every attempt — i.e. total LLM calls the loop made."
        ),
    )
    used_high_temp_retry: bool = Field(
        default=False,
        description=(
            "True when at least one attempt invoked the corrector's "
            "high-temperature retry path (triggered by a "
            "``ProtocolEmpty`` outcome)."
        ),
    )


# ---------------------------------------------------------------------------
# Loop
# ---------------------------------------------------------------------------


class RefinementLoop:
    """Outer refinement loop.

    Constructed with a corrector + validator chain. Calls
    :meth:`refine(query, error)` to run end-to-end refinement
    (Loop A inside, Loop B around it).

    Args:
        corrector: any :class:`Corrector` implementation. For
            CYGNET's production path this is a
            :class:`RampartCorrector` configured with the prior-
            attempts renderer matching the deployment.
        validator_chain: the chain to run on each refined cypher.
        schema: the active gate schema, used to build per-call
            :class:`CorrectorContext` objects. Required because
            the corrector needs it.
        max_attempts: hard cap on outer-loop attempts (default 3).
        acceptance: success criteria; defaults to "validates and
            distinct from input".
        telemetry: hook receiving one :class:`LLMCallRecord` per
            LLM call (including protocol-retry attempts inside
            the corrector). Defaults to :class:`NullTelemetry`.
    """

    def __init__(
        self,
        corrector: Corrector,
        validator_chain: ValidatorChain,
        schema: Schema,
        *,
        max_attempts: int = 3,
        acceptance: AcceptanceCriteria | None = None,
        telemetry: CorrectorTelemetry | None = None,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("RefinementLoop: max_attempts must be >= 1")
        self._corrector = corrector
        self._chain = validator_chain
        self._schema = schema
        self._max_attempts = max_attempts
        self._acceptance = acceptance or AcceptanceCriteria()
        self._telemetry: CorrectorTelemetry = telemetry or NullTelemetry()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def refine(
        self,
        broken_query: str,
        error: GateError,
        *,
        query_id: str | None = None,
        condition: str | None = None,
        all_errors: list[GateError] | None = None,
        conversation_history: list[str] | None = None,
    ) -> RefinementResult:
        """Run the outer refinement loop for one ``(query, error)``.

        The iteration mechanics live in
        :class:`cygnet.corrector.retry_loop.RetryLoop`. This method
        owns the correction-specific state (the closure over
        ``prior_attempts`` / ``current_query`` / ``current_error``
        / token aggregation) and the correction-specific stop
        condition (combines :class:`AcceptanceCriteria` with
        intrinsic-abort logic), then delegates the iteration to
        :class:`RetryLoop`.
        """
        # State accumulated across iterations. The work closure
        # mutates these; the stop condition reads the most recent
        # attempt outcome.
        prior_attempts: list[PriorAttempt] = []
        total_protocol = 0
        used_high_temp = False
        current_query = broken_query
        current_error = error
        cached_all_errors = list(all_errors) if all_errors else []
        cached_conv = list(conversation_history) if conversation_history else []

        def work(attempt_number: int) -> _RefinementAttemptOutcome:
            nonlocal current_query, current_error, total_protocol, used_high_temp

            context = CorrectorContext(
                schema_=self._schema,
                attempt_number=attempt_number,
                prior_attempts=list(prior_attempts),
                conversation_history=cached_conv,
                max_attempts=self._max_attempts,
                metadata={},
                all_errors=cached_all_errors,
            )

            observations: list[LLMCallObservation] = []
            corrector_result = self._call_corrector(
                current_query,
                current_error,
                context,
                observations.append,
            )

            total_protocol += corrector_result.protocol_attempts
            used_high_temp = used_high_temp or corrector_result.used_high_temp_retry

            elapsed = sum(o.elapsed_seconds for o in observations)
            input_tokens_total = sum(o.input_tokens for o in observations)
            output_tokens_total = sum(o.output_tokens for o in observations)
            protocol_outcome = _protocol_outcome_from_observations(
                observations, corrector_result.action, corrector_result.reason
            )

            # Branch 1: corrector aborted → terminate with the abort.
            if corrector_result.action == "abort":
                attempt = RefinementAttempt(
                    attempt_number=attempt_number,
                    refined_query=corrector_result.refined_query,
                    protocol_outcome=protocol_outcome,
                    protocol_attempts=corrector_result.protocol_attempts,
                    validator_outcome="not_run",
                    validator_error=None,
                    elapsed_seconds=elapsed,
                    input_tokens=input_tokens_total,
                    output_tokens=output_tokens_total,
                )
                self._emit_records(
                    observations,
                    query_id=query_id,
                    condition=condition,
                    refinement_attempt=attempt_number,
                    validator_outcome="not_run",
                )
                return _RefinementAttemptOutcome(
                    attempt=attempt,
                    terminate_action="abort",
                    terminate_reason=corrector_result.reason,
                    refined_query_for_result=corrector_result.refined_query,
                )

            # action == "refined". Apply acceptance criteria.
            refined_query = corrector_result.refined_query or ""

            # Echo defence — corrector usually catches this itself, but
            # if a future corrector returns an unchanged refinement as
            # action="refined", honour require_distinct_from_input.
            if (
                self._acceptance.require_distinct_from_input
                and refined_query.strip() == current_query.strip()
            ):
                attempt = RefinementAttempt(
                    attempt_number=attempt_number,
                    refined_query=refined_query,
                    protocol_outcome=protocol_outcome,
                    protocol_attempts=corrector_result.protocol_attempts,
                    validator_outcome="not_run",
                    validator_error=None,
                    elapsed_seconds=elapsed,
                    input_tokens=input_tokens_total,
                    output_tokens=output_tokens_total,
                )
                self._emit_records(
                    observations,
                    query_id=query_id,
                    condition=condition,
                    refinement_attempt=attempt_number,
                    validator_outcome="not_run",
                )
                return _RefinementAttemptOutcome(
                    attempt=attempt,
                    terminate_action="abort",
                    terminate_reason="model_echoed_input",
                    refined_query_for_result=refined_query,
                )

            if not self._acceptance.require_validates:
                # Trust the corrector; skip validation.
                attempt = RefinementAttempt(
                    attempt_number=attempt_number,
                    refined_query=refined_query,
                    protocol_outcome=protocol_outcome,
                    protocol_attempts=corrector_result.protocol_attempts,
                    validator_outcome="not_run",
                    validator_error=None,
                    elapsed_seconds=elapsed,
                    input_tokens=input_tokens_total,
                    output_tokens=output_tokens_total,
                )
                self._emit_records(
                    observations,
                    query_id=query_id,
                    condition=condition,
                    refinement_attempt=attempt_number,
                    validator_outcome="not_run",
                )
                return _RefinementAttemptOutcome(
                    attempt=attempt,
                    terminate_action="refined",
                    terminate_reason=None,
                    refined_query_for_result=refined_query,
                )

            # Validate. The chain may raise; treat any exception as
            # a validator-failure rather than letting it propagate.
            from cygnet.models import StructuralValidatorResult

            new_error: GateError | None = None
            validator_passed = False
            result: StructuralValidatorResult | None
            try:
                result = self._chain.validate(refined_query)
            except Exception as exc:
                _logger.warning(
                    "RefinementLoop: validator chain raised on attempt %d: %s",
                    attempt_number,
                    exc,
                )
                result = None
            if result is not None:
                if result.passed:
                    validator_passed = True
                elif result.error_payload is not None:
                    new_error = GateError(
                        category=result.failed_stage,  # type: ignore[arg-type]
                        payload=result.error_payload,
                    )

            validator_outcome_lit: Literal["passed", "failed", "not_run"] = (
                "passed" if validator_passed else "failed"
            )
            attempt = RefinementAttempt(
                attempt_number=attempt_number,
                refined_query=refined_query,
                protocol_outcome=protocol_outcome,
                protocol_attempts=corrector_result.protocol_attempts,
                validator_outcome=validator_outcome_lit,
                validator_error=new_error,
                elapsed_seconds=elapsed,
                input_tokens=input_tokens_total,
                output_tokens=output_tokens_total,
            )
            self._emit_records(
                observations,
                query_id=query_id,
                condition=condition,
                refinement_attempt=attempt_number,
                validator_outcome=validator_outcome_lit,
            )

            if validator_passed:
                return _RefinementAttemptOutcome(
                    attempt=attempt,
                    terminate_action="refined",
                    terminate_reason=None,
                    refined_query_for_result=refined_query,
                )

            # Validator failed → feed the new error into the next
            # attempt with the refined cypher as input.
            if new_error is not None:
                prior_attempts.append(PriorAttempt(query=refined_query, error=new_error))
                current_query = refined_query
                current_error = new_error
            else:
                # Validator failed but produced no structured error —
                # rare degenerate case. Keep current_error; record
                # the prior with no error.
                prior_attempts.append(PriorAttempt(query=refined_query, error=None))

            return _RefinementAttemptOutcome(
                attempt=attempt,
                terminate_action=None,  # continue iterating
                terminate_reason=None,
                refined_query_for_result=None,
            )

        def stop_condition(outcome: _RefinementAttemptOutcome, _attempt_number: int) -> bool:
            """Refinement-specific stop condition. Stops when the work
            closure signals a terminal outcome (corrector aborted,
            echo, validator skipped, or validator passed). The
            ``AcceptanceCriteria`` checks are inside the closure
            because they need to peek at multiple per-attempt
            fields; the stop condition just reads the terminal
            signal the closure already produced."""
            return outcome.terminate_action is not None

        retry_loop = RetryLoop(max_attempts=self._max_attempts)
        loop_outcome = retry_loop.run(work, stop_condition)
        all_attempts = [o.attempt for o in loop_outcome.attempts]

        if loop_outcome.stopped_by == "stop_condition":
            last = loop_outcome.last_result
            assert last is not None  # set when stopped_by is "stop_condition"
            assert last.terminate_action is not None
            return RefinementResult(
                action=last.terminate_action,
                refined_query=last.refined_query_for_result,
                reason=last.terminate_reason,
                attempts=all_attempts,
                total_protocol_attempts=total_protocol,
                used_high_temp_retry=used_high_temp,
            )

        # Loop exhausted without acceptance.
        last_attempt = all_attempts[-1] if all_attempts else None
        return RefinementResult(
            action="abort",
            refined_query=last_attempt.refined_query if last_attempt else None,
            reason="max_attempts_exhausted",
            attempts=all_attempts,
            total_protocol_attempts=total_protocol,
            used_high_temp_retry=used_high_temp,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _call_corrector(
        self,
        query: str,
        error: GateError,
        context: CorrectorContext,
        on_observation: ObservationCallback,
    ) -> CorrectorResult:
        """Call the corrector, threading the observation callback."""
        return self._corrector.correct(
            query,
            error,
            context,
            on_observation=on_observation,
        )

    def _emit_records(
        self,
        observations: list[LLMCallObservation],
        *,
        query_id: str | None,
        condition: str | None,
        refinement_attempt: int,
        validator_outcome: Literal["passed", "failed", "not_run"],
    ) -> None:
        """Wrap each observation in :class:`LLMCallRecord` and
        forward to the telemetry hook. Telemetry exceptions are
        caught — telemetry must not break the refinement path."""
        for obs in observations:
            record = LLMCallRecord(
                observation=obs,
                query_id=query_id,
                condition=condition,
                refinement_attempt=refinement_attempt,
                validator_outcome=validator_outcome,
            )
            try:
                self._telemetry.on_llm_call(record)
            except Exception:
                _logger.warning(
                    "RefinementLoop: telemetry.on_llm_call raised; "
                    "suppressing to keep the refinement path alive",
                    exc_info=True,
                )


@dataclass(frozen=True)
class _RefinementAttemptOutcome:
    """Per-attempt outcome the refinement-loop work closure returns.

    Internal type used to bridge :class:`RetryLoop`'s generic
    iteration semantics with refinement-specific termination logic.
    Not part of the public surface.

    Attributes:
        attempt: the on-disk :class:`RefinementAttempt` record for
            this iteration.
        terminate_action: ``"refined"`` or ``"abort"`` when this
            attempt should terminate the loop; ``None`` when the
            loop should continue iterating (validator failed, more
            attempts available).
        terminate_reason: the ``reason`` field for the resulting
            :class:`RefinementResult` when ``terminate_action`` is
            set. ``None`` on successful termination.
        refined_query_for_result: what to put in
            :attr:`RefinementResult.refined_query` on termination.
            May be ``None`` (corrector aborted with no refined
            cypher) or a non-None string (e.g. echoed cypher, or
            the successfully-refined cypher).
    """

    attempt: RefinementAttempt
    terminate_action: Literal["refined", "abort"] | None
    terminate_reason: str | None
    refined_query_for_result: str | None


def _protocol_outcome_from_observations(
    observations: list[LLMCallObservation],
    action: str,
    reason: str | None,
) -> Literal["ok", "echoed", "empty", "malformed", "exception"]:
    """Derive the attempt-level protocol outcome from the corrector's
    last observation (or from the corrector's abort reason when no
    observations were recorded — e.g. exception before the LLM call)."""
    if observations:
        last = observations[-1].parser_outcome
        if last in {"ok", "echoed", "empty", "malformed", "exception"}:
            return last  # type: ignore[return-value]
    if action == "abort" and reason in {"model_echoed_input"}:
        return "echoed"
    if action == "abort" and reason == "empty_cypher_after_retry":
        return "empty"
    if action == "abort" and reason == "exception":
        return "exception"
    return "malformed"
