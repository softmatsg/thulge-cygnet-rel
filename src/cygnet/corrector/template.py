# Copyright (c) 2026 Nikodem Tomczak, Thulge Labs. All rights reserved.

"""Template-based correctors.

Four single-shot :class:`Corrector` implementations that build their
prompts via string concatenation rather than RAMPART block compilation:

- :class:`RawCorrector` — Neo4j-style raw error string + flat schema
  summary.
- :class:`VerbalCorrector` — prose translation of the structured
  error payload + flat schema summary (MAC-SQL-style baseline).
- :class:`NaiveFullCorrector` — full schema dump + full JSON error
  payload. The unbounded strawman.
- :class:`NaiveTruncatedCorrector` — same as NaiveFull but the
  schema section is alphabetically clipped to fit under
  ``token_budget``.

Relocated from ``benchmarks/benchmarks/infrastructure/template_corrector.py``
into the library in v0.0.42 They are real
shippable correctors implementing the unified single-shot
:class:`Corrector` protocol, not bench-only artefacts. Whether
NaiveFull and NaiveTruncated are competitive with RAMPART is a
measurement question, not a labelling one.

All four are **strictly single-shot** — one LLM call per
:meth:`correct` invocation, no internal retry. Protocol-level
retry on malformed JSON and Empty-cypher high-temperature retry
are composed externally via
:class:`cygnet.corrector.decorators.ProtocolRetryingCorrector` and
:class:`cygnet.corrector.decorators.EmptyRetryingCorrector`
respectively. This matches the bare behaviour of these correctors
under (they had no internal retry there either).

The shared outcome → :class:`CorrectorResult` dispatch is factored
into :func:`cygnet.corrector._outcome_helpers.outcome_to_result`
(v0.0.42); this module imports the helper rather than carrying its
own copy of the four-branch logic.
"""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Final

from cygnet.corrector._outcome_helpers import outcome_to_result
from cygnet.corrector.interface import Corrector, CorrectorContext, PriorAttempt
from cygnet.corrector.response_parser import parse_corrector_response_v2
from cygnet.corrector.telemetry import LLMCallObservation, ObservationCallback
from cygnet.models import CorrectorResult, GateError, Schema

if TYPE_CHECKING:
    from cygnet.corrector.llm import LLMClient

__all__ = [
    "NaiveFullCorrector",
    "NaiveTruncatedCorrector",
    "RawCorrector",
    "TemplateCorrector",
    "VerbalCorrector",
]


_logger = logging.getLogger("cygnet.corrector.template")


# Shared JSON-output framing the subclasses append to their user
# prompts. Matches the system prompt's contract; redundant but cheap
# insurance against models that under-weight system instructions.
_JSON_OUTPUT_TAIL: Final[str] = 'Return a JSON object: `{"cypher": "<corrected query>"}`.'


# Shared system prompt for the direct-call template correctors. Same
# wording as the pre-v0.0.42 bench-side ``_SHARED_SYSTEM_PROMPT``. The
# RAMPART corrector uses its own system prompt (v0.0.41
# :data:`cygnet.corrector.rampart_backed.V4_SYSTEM_PROMPT` via the
# ``DEFAULT_PROMPT_BY_MODEL`` registry).
_SHARED_SYSTEM_PROMPT: Final[str] = (
    "You are an expert Cypher query writer working against a Neo4j "
    "knowledge graph. The user will give you a broken Cypher query "
    "and an error description. Return a single JSON object with "
    "exactly two fields: `cypher` (required string, the corrected "
    "query) and `explanation` (optional string). Return nothing "
    "outside the JSON object — no prose, no code-fence wrapping. "
    'If you cannot fix the query, return `{"cypher": ""}`.\n'
)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class TemplateCorrector(Corrector, ABC):
    """Abstract base for the four single-shot template correctors.

    Subclasses implement two methods:

    - :meth:`_build_initial_prompt(query, error, schema)` — the
      corrector's user prompt when no prior attempts exist.
    - :meth:`_render_prior_attempt(query, error, attempt_index)`
      — render one entry of the prior-attempts section in the
      corrector's style.

    The base class handles:

    - Prompt assembly: ``_build_initial_prompt`` first, then an
      optional ``## Prior attempts`` section composed from
      ``CorrectorContext.prior_attempts`` via
      ``_render_prior_attempt``.
    - LLM call via ``self._llm.complete(json_object=...)``.
    - Parsing via :func:`parse_corrector_response_v2`.
    - Observation emission to the ``on_observation`` callback.
    - :class:`CorrectorResult` assembly via the shared
      :func:`outcome_to_result` helper.

    Args:
        llm_client: any :class:`LLMClient`.
        schema: the active gate schema; passed to
            ``_build_initial_prompt``.
        system_prompt: the system prompt sent on every call.
            Defaults to the v0.0.42 :data:`_SHARED_SYSTEM_PROMPT`.
        user_prompt_tail: the closing instruction line appended to
            the subclass's initial prompt. Defaults to
            :data:`_JSON_OUTPUT_TAIL`.
        request_json_object: whether to pass ``json_object=True``
            to :meth:`LLMClient.complete`. Defaults to True (the
            v0.0.30 contract). False is for the V5
            raw-Cypher experiment; the standard parser then marks
            non-JSON responses ``ProtocolMalformed``.
        provider: provider name; populates
            :class:`LLMCallObservation.provider` when the LLM
            client doesn't expose its own ``provider`` attribute.
        model: model identifier; same role as ``provider``.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        schema: Schema,
        *,
        system_prompt: str = _SHARED_SYSTEM_PROMPT,
        user_prompt_tail: str = _JSON_OUTPUT_TAIL,
        request_json_object: bool = True,
        provider: str = "",
        model: str = "",
    ) -> None:
        self._llm = llm_client
        self._schema = schema
        self._system_prompt = system_prompt
        self._user_prompt_tail = user_prompt_tail
        self._request_json_object = request_json_object
        self._provider_override = provider
        self._model_override = model

    # ------------------------------------------------------------------
    # Subclass override points
    # ------------------------------------------------------------------

    @abstractmethod
    def _build_initial_prompt(
        self,
        query: str,
        error: GateError,
        schema: Schema,
    ) -> str:
        """Corrector-specific first-attempt prompt body."""

    @abstractmethod
    def _render_prior_attempt(
        self,
        query: str,
        error: GateError,
        attempt_index: int,
    ) -> str:
        """Corrector-specific prior-attempt block. ``attempt_index``
        is 1-based, matching v0.0.30's ``Attempt N:`` format."""

    # ------------------------------------------------------------------
    # Corrector protocol implementation
    # ------------------------------------------------------------------

    def correct(
        self,
        query: str,
        error: GateError,
        context: CorrectorContext,
        *,
        on_observation: ObservationCallback | None = None,
    ) -> CorrectorResult:
        """One LLM call per :meth:`correct` invocation; no internal
        retry. The unified single-shot protocol.

        Parser outcomes propagate to :class:`CorrectorResult` via the
        :func:`outcome_to_result` helper:

        - ``ProtocolOK`` → ``action="refined"``.
        - ``ProtocolEchoed`` → ``action="abort",
          reason="model_echoed_input"``; the echoed cypher is
          preserved on ``refined_query``.
        - ``ProtocolEmpty`` → ``action="abort",
          reason="empty_cypher"``. The :class:`EmptyRetryingCorrector`
          decorator can wrap this corrector to add a high-
          temperature retry on Empty.
        - ``ProtocolMalformed`` → ``action="abort",
          reason="protocol_failure"``. The
          :class:`ProtocolRetryingCorrector` decorator can wrap
          this corrector to add malformed-JSON re-asks.

        Exceptions from the LLM client propagate as
        ``action="abort", reason="exception"``.
        """
        user_prompt = self._assemble_user_prompt(query, error, context.prior_attempts)
        started = time.perf_counter()
        try:
            response = self._llm.complete(
                system_prompt=self._system_prompt,
                user_prompt=user_prompt,
                json_object=self._request_json_object,
            )
        except Exception as exc:
            elapsed = time.perf_counter() - started
            self._emit_observation(
                on_observation,
                system_prompt=self._system_prompt,
                user_prompt=user_prompt,
                raw_response="",
                parser_outcome="exception",
                parser_reason=f"{type(exc).__name__}: {exc}",
                extracted_cypher=None,
                temperature=0.1,
                input_tokens=0,
                output_tokens=0,
                elapsed_seconds=elapsed,
            )
            return CorrectorResult(
                action="abort",
                refined_query=None,
                reasoning=f"{type(self).__name__} aborted: {type(exc).__name__}: {exc}",
                attempts_used=context.attempt_number,
                reason="exception",
                protocol_attempts=1,
                used_high_temp_retry=False,
            )

        outcome = parse_corrector_response_v2(response.text, input_cypher=query)
        observed_model = response.model or self._model_override
        observed_provider = response.provider or self._provider_override
        self._emit_observation(
            on_observation,
            system_prompt=self._system_prompt,
            user_prompt=user_prompt,
            raw_response=response.text,
            parser_outcome=outcome.outcome,
            parser_reason=getattr(outcome, "reason", None),
            extracted_cypher=getattr(outcome, "cypher", None),
            temperature=0.1,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            elapsed_seconds=response.elapsed_seconds,
            model=observed_model,
            provider=observed_provider,
        )

        return outcome_to_result(
            outcome,
            attempt_number=context.attempt_number,
            corrector_name=type(self).__name__,
            protocol_attempts=1,
            used_high_temp_retry=False,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _assemble_user_prompt(
        self,
        query: str,
        error: GateError,
        prior_attempts: list[PriorAttempt],
    ) -> str:
        """Initial prompt + optional ``## Prior attempts`` section."""
        body = self._build_initial_prompt(query, error, self._schema)
        if not prior_attempts:
            return body
        history_chunks: list[str] = ["\n\n## Prior attempts\n"]
        for i, pa in enumerate(prior_attempts, start=1):
            if pa.error is not None:
                history_chunks.append(
                    self._render_prior_attempt(pa.query, pa.error, attempt_index=i)
                )
            else:
                history_chunks.append(
                    f"\nAttempt {i}: ```cypher\n{pa.query}\n```\n"
                    "(Passed gating; included for context.)\n"
                )
        return body + "".join(history_chunks)

    @staticmethod
    def _emit_observation(
        on_observation: ObservationCallback | None,
        *,
        system_prompt: str,
        user_prompt: str,
        raw_response: str,
        parser_outcome: str,
        parser_reason: str | None,
        extracted_cypher: str | None,
        temperature: float,
        input_tokens: int,
        output_tokens: int,
        elapsed_seconds: float,
        model: str = "",
        provider: str = "",
    ) -> None:
        if on_observation is None:
            return
        try:
            on_observation(
                LLMCallObservation(
                    model=model,
                    provider=provider,
                    temperature=temperature,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    raw_response=raw_response,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    parser_outcome=parser_outcome,
                    parser_reason=parser_reason,
                    extracted_cypher=extracted_cypher,
                    protocol_attempt=1,
                    elapsed_seconds=elapsed_seconds,
                )
            )
        except Exception:
            _logger.warning(
                "on_observation callback raised; suppressing to keep the corrector path alive",
                exc_info=True,
            )


# ---------------------------------------------------------------------------
# Shared rendering helpers
# ---------------------------------------------------------------------------


def _construct_raw_error_string(error: GateError) -> str:
    """Render the error payload as a Neo4j-style raw error string.

    For errors CYGNET catches before Neo4j sees them (parse / schema
    / property / constraint / cost / empty), the rendering is
    synthesised to look like what Neo4j's parser or planner would
    emit. Documented as a methodological caveat in the bench's
    implications doc.
    """
    payload = error.payload
    category = error.category
    if category == "parse":
        line = getattr(payload, "line", 1) or 1
        col = getattr(payload, "column", 1) or 1
        msg = getattr(payload, "message", "syntax error") or "syntax error"
        return f"{msg} (line {line}, column {col})"
    if category == "schema":
        unknown = getattr(payload, "unknown_reference", "?") or "?"
        kind = getattr(payload, "reference_kind", "label")
        return f"The {kind} `{unknown}` does not exist."
    if category == "property":
        prop = getattr(payload, "property_name", "?") or "?"
        return (
            f"The property `{prop}` referenced in the query is "
            "not declared on the label it was used with."
        )
    if category == "constraint":
        identifier = getattr(payload, "constraint_id", "?") or "?"
        return f"Constraint `{identifier}` would be violated by this query."
    if category == "cost":
        rows = getattr(payload, "estimated_rows", 0) or 0
        return f"Query plan estimates {rows} rows; refusing to execute."
    return f"Query rejected with category `{category}`."


def _construct_verbal_error_string(error: GateError) -> str:
    """Render the error payload as natural-language prose.

    Carries the same factual content as the structured JSON payload
    — ``did_you_mean`` candidates, ``available_in_scope``
    suggestions, property types — but rendered as sentences.
    """
    payload = error.payload
    category = error.category
    if category == "parse":
        line = getattr(payload, "line", 1) or 1
        col = getattr(payload, "column", 1) or 1
        msg = getattr(payload, "message", "") or ""
        excerpt = getattr(payload, "excerpt_with_caret", None)
        out = f"The query has a syntax error at line {line}, column {col}: {msg}."
        if excerpt:
            out += f"\n\nExcerpt:\n```\n{excerpt}\n```"
        return out
    if category == "schema":
        unknown = getattr(payload, "unknown_reference", "?") or "?"
        kind = getattr(payload, "reference_kind", "label")
        suggestions = getattr(payload, "did_you_mean", []) or []
        in_scope = getattr(payload, "available_in_scope", []) or []
        sentence = f"The {kind} `{unknown}` referenced in the query is not declared in the schema."
        if suggestions:
            sentence += f" Did you mean one of: {', '.join(f'`{s}`' for s in suggestions)}?"
        if in_scope:
            sentence += (
                f" Other valid {kind}s in scope include: "
                f"{', '.join(f'`{s}`' for s in in_scope[:8])}."
            )
        return sentence
    if category == "property":
        prop = getattr(payload, "property_name", "?") or "?"
        declared_type = getattr(payload, "declared_type", None)
        used_type = getattr(payload, "used_type", None)
        suggestions = getattr(payload, "did_you_mean", []) or []
        if declared_type and used_type and declared_type != used_type:
            sentence = (
                f"The property `{prop}` is declared as type "
                f"`{declared_type}` in the schema, but the query "
                f"uses it as type `{used_type}`."
            )
        else:
            sentence = (
                f"The property `{prop}` referenced in the query is "
                "not declared on the label it was used with."
            )
        if suggestions:
            sentence += f" Did you mean one of: {', '.join(f'`{s}`' for s in suggestions)}?"
        return sentence
    if category == "constraint":
        identifier = getattr(payload, "constraint_id", "?") or "?"
        kind = getattr(payload, "constraint_kind", "existence") or "existence"
        property_name = getattr(payload, "property_name", None)
        if property_name:
            return (
                f"The query would violate the {kind} constraint "
                f"`{identifier}` involving property `{property_name}`."
            )
        return (
            f"The query would violate the {kind} constraint `{identifier}` declared on the schema."
        )
    if category == "cost":
        rows = getattr(payload, "estimated_rows", 0) or 0
        return (
            f"The query is estimated to scan approximately {rows} "
            "rows, which exceeds the cost gate's threshold."
        )
    return f"The query was rejected with error category `{category}`."


def _format_schema_summary(schema: Schema) -> str:
    """Compact prose schema summary used by the Raw and Verbal
    correctors."""
    label_names = [label.name for label in schema.labels]
    rel_lines = [
        f"  - (:{rel.source_label})-[:{rel.name}]->(:{rel.target_label})"
        for rel in schema.relationship_types
    ]
    return f"Labels: {', '.join(label_names)}\nRelationships:\n" + "\n".join(rel_lines)


def _format_full_schema(schema: Schema) -> str:
    """Render the full schema as a single concatenated text block.

    Sectioned by labels then relationships then constraints,
    alphabetically ordered within each section so the
    :func:`_truncate_schema_block` clip in
    :class:`NaiveTruncatedCorrector` produces a stable subset.
    """
    lines: list[str] = ["## Schema"]
    labels = sorted(schema.labels, key=lambda lab: lab.name)
    rels = sorted(schema.relationship_types, key=lambda r: r.name)
    constraints = sorted(schema.constraints, key=lambda c: c.identifier)

    lines.append("\n### Labels")
    for label in labels:
        props = schema.properties_by_label.get(label.name, [])
        prop_strs = ", ".join(
            f"{p.name}:{p.type}{'?' if p.optional else ''}"
            for p in sorted(props, key=lambda p: p.name)
        )
        lines.append(
            f"- {label.name}: {prop_strs}" if prop_strs else f"- {label.name}: (no properties)"
        )

    lines.append("\n### Relationships")
    for rel in rels:
        props = schema.properties_by_rel_type.get(rel.name, [])
        prop_strs = ", ".join(
            f"{p.name}:{p.type}{'?' if p.optional else ''}"
            for p in sorted(props, key=lambda p: p.name)
        )
        suffix = f" props=[{prop_strs}]" if prop_strs else ""
        lines.append(f"- (:{rel.source_label})-[:{rel.name}]->(:{rel.target_label}){suffix}")

    if constraints:
        lines.append("\n### Constraints")
        for c in constraints:
            lines.append(f"- {c.identifier}: {c.type} on :{c.label_or_rel}({c.property})")

    return "\n".join(lines)


def _truncate_schema_block(
    schema_block: str,
    budget_tokens: int,
    fixed_overhead_chars: int,
) -> str:
    """Clip ``schema_block`` alphabetically by line so the total
    prompt fits inside ``budget_tokens``. Character-based estimate
    (``chars / 4``) — conservative; lands under budget more often
    than over.
    """
    char_budget = max(0, budget_tokens * 4 - fixed_overhead_chars)
    if len(schema_block) <= char_budget:
        return schema_block

    lines = schema_block.split("\n")
    kept: list[str] = []
    running_chars = 0
    for line in lines:
        if running_chars + len(line) + 1 > char_budget:
            kept.append("…(schema truncated)…")
            break
        kept.append(line)
        running_chars += len(line) + 1
    return "\n".join(kept)


# ---------------------------------------------------------------------------
# Concrete correctors
# ---------------------------------------------------------------------------


class RawCorrector(TemplateCorrector):
    """Neo4j-style raw error string + flat schema summary + broken
    query. Minimal-information baseline.

 bench04 ``"raw"`` condition. Single-shot per the unified
    v0.0.42 protocol — no internal retry."""

    def _build_initial_prompt(
        self,
        query: str,
        error: GateError,
        schema: Schema,
    ) -> str:
        raw_error = _construct_raw_error_string(error)
        schema_summary = _format_schema_summary(schema)
        return (
            "## Schema\n\n"
            f"{schema_summary}\n\n"
            "## Broken query\n\n"
            "```cypher\n"
            f"{query}\n"
            "```\n\n"
            "## Error from Neo4j\n\n"
            f"```\n{raw_error}\n```\n\n"
            f"{self._user_prompt_tail}"
        )

    def _render_prior_attempt(
        self,
        query: str,
        error: GateError,
        attempt_index: int,
    ) -> str:
        raw_error = _construct_raw_error_string(error)
        return (
            f"\nAttempt {attempt_index}: ```cypher\n"
            f"{query}\n"
            "```\n"
            f"Error from Neo4j: ```\n{raw_error}\n```\n"
        )


class VerbalCorrector(TemplateCorrector):
    """Prose translation of the structured error + flat schema
    summary. MAC-SQL-style baseline.

 bench04 ``"verbal"`` condition. Same factual content as
    RAMPART's structured payload, sentence-rendered. Single-shot per
    the unified v0.0.42 protocol."""

    def _build_initial_prompt(
        self,
        query: str,
        error: GateError,
        schema: Schema,
    ) -> str:
        explanation = _construct_verbal_error_string(error)
        schema_summary = _format_schema_summary(schema)
        return (
            "## Schema\n\n"
            f"{schema_summary}\n\n"
            "## Broken query\n\n"
            "```cypher\n"
            f"{query}\n"
            "```\n\n"
            "## Why the query is broken\n\n"
            f"{explanation}\n\n"
            f"{self._user_prompt_tail}"
        )

    def _render_prior_attempt(
        self,
        query: str,
        error: GateError,
        attempt_index: int,
    ) -> str:
        explanation = _construct_verbal_error_string(error)
        return f"\nAttempt {attempt_index}: ```cypher\n{query}\n```\nWhy it failed: {explanation}\n"


class NaiveFullCorrector(TemplateCorrector):
    """Full schema dump + full JSON error payload + broken query.

 bench05 ``"naive_full"`` condition. The unbounded
    strawman — no relevance gating, no priority, no truncation.
    Single-shot per the unified v0.0.42 protocol."""

    def _build_initial_prompt(
        self,
        query: str,
        error: GateError,
        schema: Schema,
    ) -> str:
        schema_block = _format_full_schema(schema)
        error_json = json.dumps(error.model_dump(mode="json"), indent=2, default=str)
        return (
            f"{schema_block}\n\n"
            "## Failing query\n\n"
            "```cypher\n"
            f"{query}\n"
            "```\n\n"
            "## Error payload (JSON)\n\n"
            "```json\n"
            f"{error_json}\n"
            "```\n\n"
            f"{self._user_prompt_tail}"
        )

    def _render_prior_attempt(
        self,
        query: str,
        error: GateError,
        attempt_index: int,
    ) -> str:
        # Full-payload conditions render prior attempts with the
        # shared library renderer so the on-disk shape matches the
        # corrector-mediated conditions.
        from cygnet.corrector.renderers import render_gate_error_block

        return render_gate_error_block(
            query=query,
            error=error,
            header_kind="prior_attempt",
            attempt_index=attempt_index,
        )


class NaiveTruncatedCorrector(TemplateCorrector):
    """Same as :class:`NaiveFullCorrector` but the schema section is
    alphabetically clipped to fit under ``token_budget``.

 bench05 ``"naive_truncated"`` condition. Prior-attempts
    are rendered AFTER the truncated schema and are NOT counted
    against the truncation budget. Single-shot per the unified
    v0.0.42 protocol."""

    def __init__(
        self,
        llm_client: LLMClient,
        schema: Schema,
        *,
        token_budget: int,
        system_prompt: str = _SHARED_SYSTEM_PROMPT,
        user_prompt_tail: str = _JSON_OUTPUT_TAIL,
        request_json_object: bool = True,
        provider: str = "",
        model: str = "",
    ) -> None:
        super().__init__(
            llm_client,
            schema,
            system_prompt=system_prompt,
            user_prompt_tail=user_prompt_tail,
            request_json_object=request_json_object,
            provider=provider,
            model=model,
        )
        self._token_budget = token_budget

    def _build_initial_prompt(
        self,
        query: str,
        error: GateError,
        schema: Schema,
    ) -> str:
        error_json = json.dumps(error.model_dump(mode="json"), indent=2, default=str)
        fixed_sections = (
            "## Failing query\n\n"
            "```cypher\n"
            f"{query}\n"
            "```\n\n"
            "## Error payload (JSON)\n\n"
            "```json\n"
            f"{error_json}\n"
            "```\n\n"
            f"{self._user_prompt_tail}"
        )
        schema_block = _format_full_schema(schema)
        schema_block_truncated = _truncate_schema_block(
            schema_block=schema_block,
            budget_tokens=self._token_budget,
            fixed_overhead_chars=len(self._system_prompt) + len(fixed_sections) + 8,
        )
        return f"{schema_block_truncated}\n\n{fixed_sections}"

    def _render_prior_attempt(
        self,
        query: str,
        error: GateError,
        attempt_index: int,
    ) -> str:
        # Prior-attempts deliberately bypass the truncation budget
        # — they're high-signal, low-volume, and the v0.0.30
        # experimental design pinned this semantics.
        from cygnet.corrector.renderers import render_gate_error_block

        return render_gate_error_block(
            query=query,
            error=error,
            header_kind="prior_attempt",
            attempt_index=attempt_index,
        )
