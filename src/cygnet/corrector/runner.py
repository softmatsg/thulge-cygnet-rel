# Copyright (c) 2026 Nikodem Tomczak, Thulge Labs. All rights reserved.

"""Convenience runner — the single entry point for the correction
path.

:func:`run_correction` wires a corrector, the generic
:class:`RetryLoop`, and an :class:`AcceptanceCriteria` together
with defaults that reproduce the production configuration.

User-facing surface::

    from cygnet import run_correction
    from cygnet.corrector import RampartCorrector, make_llm_client

    corrector = RampartCorrector(make_llm_client("gemini", model="..."))
    result = run_correction(
        corrector=corrector,
        query="MATCH (n:Smaple) RETURN n",
        error_context=gate_error,
        validator_chain=gate.chain,
        schema=gate.get_schema(),
    )
    # result.action == "refined" on success.

Defaults:

- ``loop=True`` — iterate up to ``loop_options.max_attempts`` (3 by
  default). Set ``loop=False`` for a single corrector call.
- ``apply_default_wrapping=True`` — wrap the supplied corrector
  with :class:`EmptyRetryingCorrector` and
  :class:`ProtocolRetryingCorrector` per the always-wrap policy.
  For the four template correctors the wrapping is a no-op; for
  :class:`RampartCorrector` it adds the protocol retry behaviour.
  Set ``False`` to opt out (single-shot corrector, caller manages
  retry).
- ``acceptance=AcceptanceCriteria()`` — the default
  (``require_validates=True, require_distinct_from_input=True``).
- ``telemetry=NullTelemetry()`` — discard observations. Pass a
  :class:`FileTelemetry` to write per-call JSON.

The bring-your-own-loop path is two-fold:

- ``loop=False`` calls the corrector exactly once and returns a
  :class:`RefinementResult` with a single-attempt history. Use
  when the caller wants to drive iteration themselves.
- Calling a corrector's ``correct`` directly is also fine — the
  unified protocol is single-shot, so callers don't have to use
  :func:`run_correction` at all. :func:`run_correction` is the
  convenience for the common case.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field

from cygnet.corrector.decorators import (
    EmptyRetryingCorrector,
    ProtocolRetryingCorrector,
)
from cygnet.corrector.interface import CorrectorContext
from cygnet.corrector.refinement_loop import (
    AcceptanceCriteria,
    RefinementAttempt,
    RefinementLoop,
    RefinementResult,
    _protocol_outcome_from_observations,
)
from cygnet.corrector.telemetry import (
    CorrectorTelemetry,
    LLMCallObservation,
    LLMCallRecord,
    NullTelemetry,
)
from cygnet.models import GateError

if TYPE_CHECKING:
    from cygnet.corrector.interface import Corrector
    from cygnet.models import Schema
    from cygnet.validator.chain import ValidatorChain

__all__ = ["LoopOptions", "apply_default_wrapping", "run_correction"]


class LoopOptions(BaseModel):
    """Per-call loop knobs for :func:`run_correction`.

    ``max_attempts`` matches the default of 3 (1 first
    attempt + 2 refinement retries after validator failure).
    ``between_attempts_sleep_seconds`` is the generic
    :class:`RetryLoop` backoff knob; defaults to 0.0 because the
    correction path doesn't need inter-attempt sleep (protocol
    retry inside the decorators has its own backoff; transport
    retry inside ``ResilientLLMClient`` has its own).
    """

    model_config = ConfigDict(extra="forbid")

    max_attempts: int = Field(default=3, ge=1)
    between_attempts_sleep_seconds: float = Field(default=0.0, ge=0.0)


def apply_default_wrapping(
    corrector: Corrector,
    *,
    provider: str | None = None,
    high_temp_by_provider: dict[str, float] | None = None,
    protocol_retries: int = 2,
) -> Corrector:
    """Wrap a corrector with the production retry decorators.

    Returns ``ProtocolRetryingCorrector(EmptyRetryingCorrector(corrector,
    high_temp_by_provider=..., provider=...), retries=protocol_retries)``.

    The wrapping is unconditional. For :class:`RampartCorrector`
    the wrapped form adds ``protocol_retries=2`` behaviour. For
    the four template correctors (Raw / Verbal / NaiveFull /
    NaiveTruncated), the decorators are hint-emitters the
    correctors ignore, so the wrapping is a no-op behaviourally —
    one unconditional code path is simpler than type-sniffing
    whether the corrector is or contains a RAMPART.

    Args:
        corrector: any :class:`Corrector` to wrap. Returned as-is
            inside the two decorators.
        provider: forwarded to :class:`EmptyRetryingCorrector` so
            it picks the right high-temperature value for the
            high-temp Empty retry.
        high_temp_by_provider: optional override of the per-
            provider high-temperature table. Falls through to the
            :class:`EmptyRetryingCorrector` defaults when ``None``.
        protocol_retries: ``retries`` argument for
            :class:`ProtocolRetryingCorrector`. Default 2.

    Returns:
        The wrapped corrector. Conforms to :class:`Corrector`.
    """
    inner = EmptyRetryingCorrector(
        corrector,
        high_temp_by_provider=high_temp_by_provider,
        provider=provider,
    )
    return ProtocolRetryingCorrector(inner, retries=protocol_retries)


# Module-level alias so :func:`run_correction` can accept a kwarg
# named ``apply_default_wrapping`` (bare, no trailing underscore)
# without shadowing the helper above. The kwarg is the public-facing
# name on every entry point — :func:`run_correction` and
# :meth:`Gate.correct` — so the helper reference inside the function
# body goes through this private alias to avoid the collision.
_apply_default_wrapping = apply_default_wrapping


def run_correction(
    *,
    corrector: Corrector,
    query: str,
    error_context: GateError,
    validator_chain: ValidatorChain,
    schema: Schema,
    loop: bool = True,
    loop_options: LoopOptions | None = None,
    acceptance: AcceptanceCriteria | None = None,
    telemetry: CorrectorTelemetry | None = None,
    apply_default_wrapping: bool = True,
    provider: str | None = None,
    high_temp_by_provider: dict[str, float] | None = None,
    query_id: str | None = None,
    condition: str | None = None,
    all_errors: list[GateError] | None = None,
    conversation_history: list[str] | None = None,
) -> RefinementResult:
    """Run a correction with the published-configuration defaults.

    Args:
        corrector: any :class:`Corrector`. Wrapped with the
            production decorators when ``apply_default_wrapping`` is
            True (the default). Pass a bare corrector here; let
            ``run_correction`` apply the wrapping policy.
        query: the broken Cypher to refine.
        error_context: the :class:`GateError` that caused the gate
            to reject ``query``.
        validator_chain: the :class:`ValidatorChain` used to verify
            refined cyphers when ``acceptance.require_validates``
            (the default). Typically ``gate.chain``.
        schema: the active :class:`Schema`. Typically
            ``gate.get_schema()``.
        loop: when True (default), iterate up to
            ``loop_options.max_attempts`` until acceptance or
            exhaustion. When False, call the corrector exactly once
            and return a single-attempt :class:`RefinementResult`.
            The loop-off path still wraps with the default
            decorators when ``apply_default_wrapping`` is True —
            "loop" refers to refinement retry, not protocol retry.
        loop_options: :class:`LoopOptions` instance. Defaults to
            its built-in defaults.
        acceptance: :class:`AcceptanceCriteria` instance. Defaults
            to ``require_validates=True, require_distinct_from_input=True``.
        telemetry: :class:`CorrectorTelemetry`. Defaults to
            :class:`NullTelemetry` (no per-call records emitted).
        apply_default_wrapping: when True (default), wrap the
            supplied corrector via :func:`apply_default_wrapping`.
            Set ``False`` for bring-your-own-wrapping or when the
            caller has already wrapped. Same kwarg name as on
            :meth:`Gate.correct` so users only learn it once.
        provider: forwarded to :func:`apply_default_wrapping`.
        high_temp_by_provider: forwarded to
            :func:`apply_default_wrapping`.
        query_id, condition, all_errors, conversation_history:
            forwarded to :class:`RefinementLoop.refine` /
            single-shot context construction. See
            :class:`RefinementLoop.refine`'s docstring for
            semantics.

    Returns:
        :class:`RefinementResult`. ``action == "refined"`` is the
        success condition.
    """
    effective_options = loop_options or LoopOptions()
    effective_acceptance = acceptance or AcceptanceCriteria()
    effective_telemetry: CorrectorTelemetry = telemetry or NullTelemetry()

    wrapped = (
        _apply_default_wrapping(
            corrector,
            provider=provider,
            high_temp_by_provider=high_temp_by_provider,
        )
        if apply_default_wrapping
        else corrector
    )

    if not loop:
        return _run_single_shot(
            wrapped,
            query=query,
            error_context=error_context,
            validator_chain=validator_chain,
            schema=schema,
            acceptance=effective_acceptance,
            telemetry=effective_telemetry,
            query_id=query_id,
            condition=condition,
            all_errors=all_errors,
            conversation_history=conversation_history,
        )

    refinement_loop = RefinementLoop(
        wrapped,
        validator_chain,
        schema,
        max_attempts=effective_options.max_attempts,
        acceptance=effective_acceptance,
        telemetry=effective_telemetry,
    )
    return refinement_loop.refine(
        query,
        error_context,
        query_id=query_id,
        condition=condition,
        all_errors=all_errors,
        conversation_history=conversation_history,
    )


def _run_single_shot(
    corrector: Corrector,
    *,
    query: str,
    error_context: GateError,
    validator_chain: ValidatorChain,
    schema: Schema,
    acceptance: AcceptanceCriteria,
    telemetry: CorrectorTelemetry,
    query_id: str | None,
    condition: str | None,
    all_errors: list[GateError] | None,
    conversation_history: list[str] | None,
) -> RefinementResult:
    """Loop-off path. One corrector call → one attempt → return.

    Used when ``run_correction(loop=False)``. Builds a single-
    attempt :class:`RefinementResult` so the caller gets a
    consistent return shape regardless of ``loop=`` value. The
    corrector is called inside whatever wrapping
    :func:`run_correction` applied (i.e. protocol-retry still
    fires; only the validator-feedback iteration is suppressed).
    """
    context = CorrectorContext(
        schema_=schema,
        attempt_number=1,
        prior_attempts=[],
        conversation_history=list(conversation_history) if conversation_history else [],
        max_attempts=1,
        metadata={},
        all_errors=list(all_errors) if all_errors else [],
    )

    observations: list[LLMCallObservation] = []
    corrector_result = corrector.correct(
        query,
        error_context,
        context,
        on_observation=observations.append,
    )

    elapsed = sum(o.elapsed_seconds for o in observations)
    input_tokens_total = sum(o.input_tokens for o in observations)
    output_tokens_total = sum(o.output_tokens for o in observations)
    protocol_outcome = _protocol_outcome_from_observations(
        observations, corrector_result.action, corrector_result.reason
    )

    refined_query = corrector_result.refined_query or ""

    # Decide terminal action (mirrors RefinementLoop's per-attempt
    # branches but without the validator-feedback continuation).
    terminate_action: Literal["refined", "abort"]
    terminate_reason: str | None
    validator_outcome_lit: Literal["passed", "failed", "not_run"]
    validator_error: GateError | None = None

    if corrector_result.action == "abort":
        terminate_action = "abort"
        terminate_reason = corrector_result.reason
        validator_outcome_lit = "not_run"
    elif acceptance.require_distinct_from_input and refined_query.strip() == query.strip():
        terminate_action = "abort"
        terminate_reason = "model_echoed_input"
        validator_outcome_lit = "not_run"
    elif not acceptance.require_validates:
        terminate_action = "refined"
        terminate_reason = None
        validator_outcome_lit = "not_run"
    else:
        # Run the validator chain on the refined query.
        from cygnet.models import StructuralValidatorResult

        result: StructuralValidatorResult | None
        try:
            result = validator_chain.validate(refined_query)
        except Exception:
            result = None
        validator_passed = bool(result and result.passed)
        if result is not None and not result.passed and result.error_payload is not None:
            validator_error = GateError(
                category=result.failed_stage,  # type: ignore[arg-type]
                payload=result.error_payload,
            )
        if validator_passed:
            terminate_action = "refined"
            terminate_reason = None
            validator_outcome_lit = "passed"
        else:
            # Single-shot path: validator failed and we don't loop.
            # The caller gets an abort with the failure surfaced.
            terminate_action = "abort"
            terminate_reason = "validator_rejected_refinement"
            validator_outcome_lit = "failed"

    attempt = RefinementAttempt(
        attempt_number=1,
        refined_query=corrector_result.refined_query
        if terminate_action == "abort"
        else refined_query,
        protocol_outcome=protocol_outcome,
        protocol_attempts=corrector_result.protocol_attempts,
        validator_outcome=validator_outcome_lit,
        validator_error=validator_error,
        elapsed_seconds=elapsed,
        input_tokens=input_tokens_total,
        output_tokens=output_tokens_total,
    )

    import contextlib

    for obs in observations:
        record = LLMCallRecord(
            observation=obs,
            query_id=query_id,
            condition=condition,
            refinement_attempt=1,
            validator_outcome=validator_outcome_lit,
        )
        with contextlib.suppress(Exception):
            telemetry.on_llm_call(record)

    return RefinementResult(
        action=terminate_action,
        refined_query=(
            corrector_result.refined_query if terminate_action == "abort" else refined_query
        ),
        reason=terminate_reason,
        attempts=[attempt],
        total_protocol_attempts=corrector_result.protocol_attempts,
        used_high_temp_retry=corrector_result.used_high_temp_retry,
    )
