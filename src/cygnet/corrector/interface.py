# Copyright (c) 2026 Nikodem Tomczak, Thulge Labs. All rights reserved.

"""Corrector protocol and supporting context models.

A *corrector* refines a query that failed gating. The interface is
deliberately duck-typed: any object with a ``correct(query, error,
context) -> CorrectorResult`` method satisfies it. The RAMPART-backed
LLM corrector arrives in a later slice; until then, users wire in
their own implementation or accept the :class:`NullCorrector`
default.

The :class:`CorrectorContext` is the corrector's only window into the
gate's state. It carries the active schema, optional conversation
history, prior attempts (so iterative correctors can avoid known-bad
refinements), and a metadata bag for user-defined extensions. The
``attempt_number``/``max_attempts`` fields exist for refinement loops
that callers implement themselves — this library does not loop.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from cygnet.models import GateError, Schema

if TYPE_CHECKING:
    from cygnet.corrector.telemetry import ObservationCallback
    from cygnet.models import CorrectorResult

__all__ = ["Corrector", "CorrectorContext", "PriorAttempt"]


class PriorAttempt(BaseModel):
    """A previously-attempted query and its gate verdict.

    A corrector inspects this list to avoid re-emitting refinements
    that already failed for the same query. ``error`` is ``None`` when
    the prior attempt passed (rare in a refinement loop but possible
    when the loop continues for non-gating reasons).
    """

    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., description="The Cypher attempted.")
    error: GateError | None = Field(
        default=None,
        description="The gate's verdict; ``None`` when the attempt passed gating.",
    )


class CorrectorContext(BaseModel):
    """Everything a corrector needs beyond the current ``(query, error)``.

    Constructed by :meth:`cygnet.Gate.correct` from the Gate's state
    and per-call kwargs. Most fields default to empty / 1, so simple
    one-shot corrections need not pass anything beyond ``schema_``.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_: Schema = Field(
        ...,
        alias="schema",
        validation_alias=AliasChoices("schema", "schema_"),
        description=(
            "The active gate schema. Python field name is ``schema_`` to "
            "avoid shadowing ``pydantic.BaseModel.schema``; YAML/JSON key "
            "is ``schema``."
        ),
    )
    conversation_history: list[str] = Field(
        default_factory=list,
        description=(
            "Raw conversational turns leading up to the failed query. "
            "The corrector decides how to use this; many implementations "
            "feed it into the LLM prompt verbatim."
        ),
    )
    attempt_number: int = Field(
        default=1,
        ge=1,
        description="1-based attempt index for the current refinement loop.",
    )
    prior_attempts: list[PriorAttempt] = Field(
        default_factory=list,
        description="Earlier attempts in the refinement loop, oldest first.",
    )
    max_attempts: int = Field(
        default=3,
        ge=1,
        description=(
            "Soft cap the corrector may use to decide between 'refined' "
            "and 'abort'. Honoured by convention; no slice in the library "
            "enforces it (the refinement loop is user-built)."
        ),
    )
    metadata: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Free-form extension point for callers. Anthropic-managed "
            "fields would use a namespace prefix; user fields are "
            "untouched by the library."
        ),
    )
    all_errors: list[GateError] = Field(
        default_factory=list,
        description=(
            "All errors the chain found, when ``collect_all`` mode was "
            "active. Empty list otherwise. The primary ``error`` "
            "parameter on :meth:`Corrector.correct` is the first of "
            "these for backwards compatibility; multi-error-aware "
            "correctors (e.g. :class:`RampartCorrector`) consume the "
            "full list when non-empty to build fix-all-at-once "
            "prompts. Single-error callers and short-circuit-mode "
            "callers leave this empty."
        ),
    )


@runtime_checkable
class Corrector(Protocol):
    """Duck-typed corrector interface.

    Any object exposing a ``correct(query, error, context)`` method
    that returns a :class:`CorrectorResult` satisfies this protocol.
    Marked ``runtime_checkable`` so users can verify conformance via
    ``isinstance(my_corrector, Corrector)``.

    The ``on_observation`` callback is a first-class kwarg of the
    protocol. Correctors that don't emit observations (notably
    :class:`NullCorrector`) accept the kwarg and ignore it.

    The protocol is **single-shot**: one ``correct`` call returns
    one :class:`CorrectorResult` for one ``(query, error)`` pair. No
    loop logic in any corrector. The :class:`ProtocolRetryingCorrector`
    and :class:`EmptyRetryingCorrector` decorators compose retry
    behaviour on top of single-shot correctors; the
    :class:`RefinementLoop` composes refinement-level retry on top of
    them.
    """

    def correct(
        self,
        query: str,
        error: GateError,
        context: CorrectorContext,
        *,
        on_observation: ObservationCallback | None = None,
    ) -> CorrectorResult: ...
