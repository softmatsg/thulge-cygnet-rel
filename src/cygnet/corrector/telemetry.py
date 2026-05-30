# Copyright (c) 2026 Nikodem Tomczak, Thulge Labs. All rights reserved.

"""Per-LLM-call telemetry hook for the refinement loop.

Two records, layered. The corrector emits :class:`LLMCallObservation`
— inner-loop scope, one per LLM call (including protocol retries).
:class:`RefinementLoop` wraps each observation in
:class:`LLMCallRecord` with outer-loop context (query_id, condition,
refinement-attempt number, validator outcome) and forwards the full
record to the configured :class:`CorrectorTelemetry`.

Two implementations:

- :class:`NullTelemetry` — does nothing. Library default. Zero
  overhead when no telemetry is configured.
- :class:`FileTelemetry` — writes one JSON file per call into a
  configured directory. Optional ``compute_extras`` hook lets the
  caller augment the record with downstream-computed fields
  (e.g. the bench adds ``cost_usd`` from its price table).

The on-disk JSON shape from v0.0.30 (``query_id``, ``condition``,
``attempt_number`` etc.) is a subset of :class:`LLMCallRecord`'s
serialised form, so analysis notebooks reading old keys continue
to work. Filename pattern is unchanged:
``{query_id}_{condition}_a{refinement_attempt}_p{protocol_attempt}.json``.

This module replaces the v0.0.30 ``CallObservation`` /
``CallObserver`` pair in :mod:`cygnet.corrector.rampart_backed`,
which was removed in v0.0.31.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "CorrectorTelemetry",
    "FileTelemetry",
    "LLMCallObservation",
    "LLMCallRecord",
    "NullTelemetry",
    "ObservationCallback",
]


_logger = logging.getLogger("cygnet.corrector.telemetry")


# v0.0.42: callback the corrector invokes once per
# LLM call to surface a per-call observation. Previously defined in
# :mod:`cygnet.corrector.rampart_backed`; relocated here when the
# :class:`Corrector` protocol gained ``on_observation`` as a
# first-class kwarg. Forward-referenced by ``cygnet.corrector.interface``
# to avoid a circular import.
ObservationCallback = Callable[["LLMCallObservation"], None]


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


class LLMCallObservation(BaseModel):
    """One LLM-call observation emitted by the corrector.

    Inner-loop scope: the corrector knows which prompts it sent
    and which response came back, plus the parser's verdict, plus
    the LLM client's token counts and wall-clock latency. It does
    NOT know which refinement attempt this is, which query, or
    which experimental condition — that context is added by
    :class:`RefinementLoop` when the observation is wrapped in
    :class:`LLMCallRecord`.

    One observation per LLM call. A refinement attempt may produce
    multiple observations when protocol-retries fire (the corrector
    retries on ``ProtocolMalformed`` and again at high temperature
    on ``ProtocolEmpty``).
    """

    model_config = ConfigDict(extra="forbid")

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When the call completed (UTC).",
    )
    model: str = Field(..., description="Model identifier.")
    provider: str = Field(..., description="Provider name.")
    temperature: float = Field(..., description="Sampling temperature used for this call.")
    system_prompt: str = Field(..., description="System prompt sent to the model.")
    user_prompt: str = Field(..., description="Compiled user prompt sent to the model.")
    raw_response: str = Field(..., description="Raw text returned by the model.")
    input_tokens: int = Field(..., ge=0, description="Prompt tokens billed by the provider.")
    output_tokens: int = Field(
        ..., ge=0, description="Output tokens (includes any reasoning tokens)."
    )
    parser_outcome: str = Field(
        ...,
        description="Parser verdict: ``ok`` / ``echoed`` / ``empty`` / ``malformed`` / ``exception``.",
    )
    parser_reason: str | None = Field(
        default=None,
        description="Parser's textual reason on malformed/exception outcomes; None otherwise.",
    )
    extracted_cypher: str | None = Field(
        default=None,
        description="Cypher extracted from the response, when the parser succeeded.",
    )
    protocol_attempt: int = Field(
        ...,
        ge=1,
        description=(
            "Which protocol-retry attempt this is within the corrector's "
            "internal loop. ``1`` for the first call; higher values for "
            "retries after ``ProtocolMalformed``. The high-temperature "
            "``ProtocolEmpty`` retry uses ``protocol_attempt+100`` as a "
            "discriminator marker."
        ),
    )
    elapsed_seconds: float = Field(
        ..., ge=0.0, description="Wall-clock time spent in the LLM call."
    )


class LLMCallRecord(BaseModel):
    """One LLM-call record emitted by :class:`RefinementLoop` to the
    telemetry hook. Wraps an :class:`LLMCallObservation` with outer-
    loop context the corrector does not see.

    The serialised form (Pydantic ``model_dump`` / ``model_dump_json``)
    is what :class:`FileTelemetry` writes. The v0.0.30 per-call JSON
    file shape is a subset of this record's serialised shape;
    analysis notebooks reading the old keys continue to work.
    """

    model_config = ConfigDict(extra="forbid")

    observation: LLMCallObservation
    query_id: str | None = Field(
        default=None,
        description="Caller-supplied query identifier, when the caller is doing per-query accounting (the bench is).",
    )
    condition: str | None = Field(
        default=None,
        description="Caller-supplied condition tag, when the caller is sweeping conditions (the bench is).",
    )
    refinement_attempt: int = Field(
        ...,
        ge=1,
        description="Which outer-loop refinement attempt this call belongs to (1-indexed).",
    )
    validator_outcome: str | None = Field(
        default=None,
        description=(
            "Result of running the validator chain on the extracted cypher: "
            "``passed`` / ``failed`` / ``not_run``. Populated by the loop "
            "after the validator chain runs (or stays None when no chain "
            "was run, e.g. on a protocol abort)."
        ),
    )


# ---------------------------------------------------------------------------
# Protocol + implementations
# ---------------------------------------------------------------------------


@runtime_checkable
class CorrectorTelemetry(Protocol):
    """Per-LLM-call telemetry hook bound to :class:`RefinementLoop`.

    Implementations receive one :class:`LLMCallRecord` per LLM call.
    Exceptions raised from ``on_llm_call`` are caught and logged by
    the loop — telemetry must never break the refinement path.

    Library default is :class:`NullTelemetry` (no-op).
    """

    def on_llm_call(self, record: LLMCallRecord) -> None: ...


class NullTelemetry:
    """Zero-overhead default. Library callers that don't configure
    telemetry get this implicitly."""

    def on_llm_call(self, record: LLMCallRecord) -> None:
        return None


ComputeExtras = Callable[[LLMCallRecord], dict[str, Any]]
"""Hook for adding caller-computed fields to the on-disk JSON. The
bench uses this to add a ``cost_usd`` field computed from the
record's token counts via the bench's price table — preserving the
v0.0.30 per-call file shape exactly."""


FilenameTemplate = Callable[[LLMCallRecord], str]
"""Hook for customising the on-disk filename per record. Defaults
to :func:`_default_filename`. Useful when running multiple models
against the same corpus — see :func:`_default_filename`'s docstring."""


def _sanitise_filename_segment(value: str) -> str:
    """Replace filesystem-unsafe characters with ``_``.

    Ollama model identifiers like ``qwen3:14b-q4_K_M`` contain
    colons, which are legal on POSIX but reserved on Windows (and
    interpreted as ADS separators). Slashes are reserved
    everywhere. We rewrite both to underscores so a per-call JSON
    file written on Linux can still be opened on a different
    filesystem during analysis. Trailing dots / spaces are also
    Windows-hostile; strip them.
    """
    if not value:
        return ""
    cleaned = value
    for ch in (":", "/", "\\", "*", "?", '"', "<", ">", "|"):
        cleaned = cleaned.replace(ch, "_")
    return cleaned.rstrip(" .")


def _default_filename(record: LLMCallRecord) -> str:
    """Default on-disk filename pattern:
    ``{model}_{query_id}_{condition}_a{refinement_attempt}_p{protocol_attempt}.json``

    v0.0.33: includes ``model`` so multiple
    models in a single lineup don't collide on
    (query_id, condition, attempt) cells. Sanitises filesystem-
    unsafe characters (colon, slash, etc.) so Ollama-style
    identifiers like ``qwen3:14b-q4_K_M`` round-trip across
    filesystems.

    When any of the contributing fields is empty/None, falls back
    to a stable placeholder ("m" / "q" / "_") so the filename
    always parses.
    """
    model = _sanitise_filename_segment(record.observation.model) or "m"
    query_id = _sanitise_filename_segment(record.query_id or "") or "q"
    condition = _sanitise_filename_segment(record.condition or "") or "_"
    return (
        f"{model}_{query_id}_{condition}"
        f"_a{record.refinement_attempt}"
        f"_p{record.observation.protocol_attempt}.json"
    )


class FileTelemetry:
    """Write one JSON file per LLM call into ``directory``.

    Default filename pattern (v0.0.33+):
    ``{model}_{query_id}_{condition}_a{refinement_attempt}_p{protocol_attempt}.json``

    v0.0.32 used ``{query_id}_{condition}_a{refinement_attempt}_p{protocol_attempt}.json``
    without a model segment, which caused collisions when running
    multiple models against the same corpus.
    The ``filename_template`` parameter lets a caller restore the
    old pattern or supply something custom.

    When ``query_id`` / ``condition`` are ``None`` or empty, segments
    fall back to ``"q"`` and ``"_"`` so the filename always parses.
    Model identifiers are sanitised (colons → underscores) for
    Windows-filesystem safety.

    ``compute_extras`` is an optional callable that receives the
    record and returns a dict of extra top-level fields to merge
    into the JSON output. The bench passes a hook that adds a
    ``cost_usd`` field. Library users typically pass ``None``.
    """

    def __init__(
        self,
        directory: Path,
        *,
        compute_extras: ComputeExtras | None = None,
        filename_template: FilenameTemplate | None = None,
    ) -> None:
        self._directory = Path(directory)
        self._directory.mkdir(parents=True, exist_ok=True)
        self._compute_extras = compute_extras
        self._filename_template: FilenameTemplate = filename_template or _default_filename

    def on_llm_call(self, record: LLMCallRecord) -> None:
        filename = self._filename_template(record)
        path = self._directory / filename
        # ``mode="json"`` so datetime fields serialise to ISO strings.
        payload: dict[str, Any] = record.model_dump(mode="json")
        if self._compute_extras is not None:
            try:
                extras = self._compute_extras(record)
            except Exception as exc:
                _logger.warning(
                    "FileTelemetry: compute_extras raised; skipping extras for %s: %s",
                    path.name,
                    exc,
                )
                extras = {}
            if extras:
                payload.update(extras)
        try:
            path.write_text(
                json.dumps(payload, indent=2, default=str),
                encoding="utf-8",
            )
        except OSError as exc:
            _logger.warning("FileTelemetry: failed to write %s: %s", path, exc)
