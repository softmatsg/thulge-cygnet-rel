# Copyright (c) 2026 Nikodem Tomczak, Thulge Labs. All rights reserved.

"""RAMPART-backed LLM corrector.

The :class:`RampartCorrector` is the production corrector. On each
``correct(query, error, context)`` call it:

1. Rebuilds a fresh :class:`rampart.BlockRegistry` from a
   per-call subset of the shipped block library. Three subset
   rules (v0.0.22, post-Brief-#27 prompt audit):

   - ``system.md`` is **not** loaded as a seed block; its
     content already reaches the LLM via the ``system_prompt``
     argument to ``LLMClient.complete``. The v0.0.11 belt-and-
     suspenders pattern that included it in both places was
     measured against and removed — it doubled
     ~400 tokens of identical content per call.
   - Only **one** error-vocab block — the one matching
     ``error.category`` — is loaded. The other five
     (``parse``, ``schema``, ``property``, ``constraint``,
     ``cost``, ``empty``, minus the matching one) used to ride
     along uselessly; they're now filtered at registration
     time. The matching block lands at position 0 directly,
     so no per-call promotion is needed.
   - Schema label/relationship blocks are filtered to the
     entities the failing query actually references, plus any
     candidates the error's ``did_you_mean`` or
     ``available_in_scope`` lists carry, plus any relationships
     whose endpoints are in the included label set.
     :func:`_relevant_schema_entities` is the helper; the
     fallback case (a query that parses to no specific
     references) includes the full schema.

2. Writes per-call agent blocks: an intent block carrying the
   failing query + the structured error payload, schema blocks
   restricted per (3), and prior-attempt blocks from
   ``context.prior_attempts``.
3. Compiles within ``token_budget`` and dispatches the result to
   the configured :class:`LLMClient`.
4. Extracts the refined query from the LLM's response (must be a
   single ``cypher`` fenced code block). Empty fenced block ->
   abort signal from the LLM; missing block -> abort with parse-
   failure reasoning.

Any unrecoverable failure (LLM exception, RAMPART compile failure,
missing block library) returns ``action="abort"`` with a
descriptive ``reasoning``. The corrector never raises.

RAMPART API translation (priority/position): the brief writes
priorities in a 1-10 range; RAMPART uses ``float`` in ``[0.0, 1.0]``.
Translation: brief "priority-10 non-evictable" -> ``priority=1.0,
evictable=False`` (``from_files`` already pins seed blocks as
non-evictable); brief "priority 5" -> ``priority=0.5``. Block
ordering in the compiled prompt is determined by **position**, not
priority — priority is only the eviction-scoring weight.
"""

from __future__ import annotations

import contextlib
import logging
import re
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING, Final

from cygnet.corrector._outcome_helpers import outcome_to_result
from cygnet.corrector.renderers import (
    render_gate_error_block,
    render_multi_error_intent,
)
from cygnet.corrector.response_parser import (
    ProtocolEchoed,
    ProtocolEmpty,
    ProtocolMalformed,
    ProtocolOK,
    parse_corrector_response_v2,
)
from cygnet.corrector.telemetry import LLMCallObservation
from cygnet.models import CorrectorResult

if TYPE_CHECKING:
    from rampart import BlockRegistry

    from cygnet.corrector.interface import CorrectorContext
    from cygnet.corrector.llm import LLMClient
    from cygnet.models import GateError, Schema

# v0.0.42: ``ObservationCallback`` moved to
# :mod:`cygnet.corrector.telemetry` so the unified
# :class:`Corrector` protocol can reference it without importing
# from ``rampart_backed``. The name is re-exported below for
# backwards compatibility with any pre-v0.0.42 caller that imported
# it from this module.
from cygnet.corrector.telemetry import ObservationCallback

__all__ = ["ObservationCallback", "RampartCorrector"]


def _module_logger() -> logging.Logger:
    """Lazy logger getter so import order doesn't pin a handler-less
    logger at class-load time."""
    return logging.getLogger("cygnet.corrector.rampart_backed")


# ---------------------------------------------------------------------------
# Block layout
# ---------------------------------------------------------------------------

_DEFAULT_TOKEN_BUDGET: Final[int] = 4000
_DEFAULT_MODEL: Final[str] = "claude-sonnet-4-5"

# Position constants. After the v0.0.22 prompt-density fix the matching
# error-vocab block lands at position 0 directly (no promotion). Intent
# goes immediately after at position 1. Schema/prior blocks append.
_INTENT_POSITION: Final[int] = 1

# Priority translation from the brief's 1-10 to RAMPART's [0.0, 1.0].
_PRIORITY_INTENT: Final[float] = 1.0  # brief "priority 10"
_PRIORITY_SCHEMA: Final[float] = 0.5  # brief "priority 5"
_PRIORITY_PRIOR: Final[float] = 0.4

# Schema-entity extraction. Cypher labels and relationship types both
# follow ``:Name`` syntax; we resolve which is which by checking the
# schema's declared vocabulary rather than parsing the surrounding
# bracket shape.
_SCHEMA_NAME_REF: Final[re.Pattern[str]] = re.compile(r":([A-Z_][A-Za-z0-9_]*)")

# NOTE: the v0.0.26 ``_CYPHER_BLOCK_PATTERN`` / ``_ANY_CODE_BLOCK_PATTERN``
# constants and their ``_extract_*_block`` helpers were removed in
# v0.0.28. Response parsing now lives in
# :mod:`cygnet.corrector.response_parser` as ``parse_corrector_response``,
# which separates protocol compliance from refinement quality. Callers
# that imported the old constants should use the new parser instead.


# ---------------------------------------------------------------------------
# Protocol-retry preamble + call-observation types (v0.0.28)
# ---------------------------------------------------------------------------


_PROTOCOL_RETRY_PREAMBLE: Final[str] = (
    "# Response shape — your last response violated the protocol\n\n"
    "The previous response was rejected by the JSON parser before it "
    "ever reached the validator. The parser enforces three rules: the "
    'response is a single JSON object of the shape `{"cypher": "<query>", '
    '"explanation"?: "<note>"}`; the `cypher` field is a string; no '
    "prose or markdown wrapping appears outside the JSON object. "
    "Re-read the response-format section below and produce a single "
    "JSON object now.\n\n"
    "---\n\n"
)
"""Prepended to the system prompt on a protocol-retry attempt. Keeps
the original system prompt intact (so error-payload guidance, schema,
and the rest of the standard contract still arrive) while giving the
model a stricter cue about what the parser rejected the previous time.
v0.0.30: updated for the JSON contract."""


# v0.0.31 note: the v0.0.28 ``CallObservation`` / ``CallObserver``
# pair was removed. Per-LLM-call telemetry now flows through
# :class:`cygnet.corrector.telemetry.LLMCallObservation` and the
# ``on_observation`` callback parameter on :meth:`correct`. The
# refinement loop wraps each observation in an outer
# ``LLMCallRecord`` and forwards to the configured telemetry sink.


# v0.0.30: per-provider high-temperature retry values. Applied on a
# single retry after ``ProtocolEmpty`` — the model said "I cannot
# refine" at low temperature; we try once at a creative-tier temp
# before accepting the abort. Keys match the provider names
# ``make_llm_client`` accepts. Override at the call site with
# ``RampartCorrector(high_temp_by_provider={...})``.
_DEFAULT_HIGH_TEMP_BY_PROVIDER: Final[dict[str, float]] = {
    "anthropic": 0.7,
    "openai": 0.7,
    "gemini": 0.9,  # Gemini accepts higher temperatures cleanly
    "google": 0.9,  # alias
    "ollama": 0.7,
    "local": 0.7,
    # Fallback applied when the provider is unknown or the model
    # identifier doesn't carry a recognisable provider prefix.
    "__default__": 0.7,
}


# ---------------------------------------------------------------------------
# V4 promotion (v0.0.41)
# ---------------------------------------------------------------------------

V4_SYSTEM_PROMPT: Final[str] = (
    "You are an expert Cypher query writer working against a Neo4j "
    "knowledge graph. The user will give you a broken Cypher query "
    "and an error description. Return a single JSON object with one "
    "field: `cypher` (the corrected query as a string).\n"
    "Do not include explanations or apologies in your response.\n"
    "Do not include triple backticks or any text outside the JSON "
    "object.\n"
    "\n"
    "Example:\n"
    "Broken: MATCH (m:Movie) WHERE m.tagline IS NOT NULL RETURN m.tagline LIMIT 1\n"
    "Error: schema error: property `tagline` does not exist on `Movie`\n"
    'Output: {"cypher":"MATCH (m:Movie) WHERE m.title IS NOT NULL RETURN m.title LIMIT 1"}\n'
)
""" V4 — the promoted production system prompt (v0.0.41).

Selected by the sweep across Phase 1 (n=19, 5 variants),
Phase 2 (n=28 confirmation on Flash 3), and Step 2 (n=28 clean
Gemma re-run after v0.0.40's 429 long-backoff removed rate-limit
contamination). V4 lifts the worst-performing Google-family model
(Gemini 3 Flash) by +26.1 pp prompt-following / +22.6 pp overall
vs the prior baseline, while not regressing any of Gemini 2.5
Flash Lite, Gemma 4 26B, or Gemma 4 31B.

The promotion includes a paired change to the LLM client default
``response_schema`` (now :data:`cygnet.corrector.llm._CYPHER_ONLY_RESPONSE_SCHEMA`)
so the SDK's structured-output mode enforces the cypher-only
shape. Without that pairing, Flash 3 still spends its
``max_tokens`` budget on a stray ``explanation`` field which
truncates the JSON mid-string.

The text is keyed in :data:`DEFAULT_PROMPT_BY_MODEL` under the
``"default"`` key, so every model gets V4 unless explicitly
overridden via ``RampartCorrector(prompt_by_model={...})`` with a
model-specific entry."""


DEFAULT_PROMPT_BY_MODEL: Final[dict[str, str]] = {
    "default": V4_SYSTEM_PROMPT,
}
"""Per-model system-prompt registry, shipped as
:class:`RampartCorrector`'s default. The ``"default"`` key applies
to every model that doesn't have its own entry. Today every Google-
family model uses V4; the dict shape exists so future per-model
divergence (e.g. an Anthropic-specific prompt) is a config change,
not a code change."""


# ---------------------------------------------------------------------------
# Corrector
# ---------------------------------------------------------------------------


class RampartCorrector:
    """LLM-backed corrector that uses RAMPART to assemble its prompt.

    Args:
        llm_client: a :class:`LLMClient` instance. Constructed via
            :func:`cygnet.corrector.llm.make_llm_client` or supplied
            directly by the caller (handy for mocking).
        token_budget: RAMPART compile budget in tokens. The shipped
            blocks plus typical per-call agent content fit in ~4000
            tokens; raise this for richer schemas or longer
            prior-attempt histories.
        model: LLM model identifier. Forwarded to the LLM client only
            if the client honours per-call overrides (the shipped
            clients pin the model at construction time, so this is
            currently informational).
        system_prompt_override: legacy single-prompt override. When
            set, applied to every call regardless of model. Takes
            precedence over ``prompt_by_model``. Kept for
            reproducibility of pre-v0.0.41 callers; new code should
            prefer ``prompt_by_model`` even for single-model setups.
        prompt_by_model: per-model system-prompt registry (v0.0.41,
            ). Keys are model identifiers (matching
            the ``model=`` constructor argument); the ``"default"``
            key applies when the model isn't explicitly listed.
            ``None`` (the default) resolves to
            :data:`DEFAULT_PROMPT_BY_MODEL`, which maps every model
            to V4 — the promoted production prompt. Pass an empty
            dict to opt out and fall through to ``system_prompt_override``
            or the bundled ``system.md`` (the pre-v0.0.41
            behaviour).
    """

    def __init__(
        self,
        llm_client: LLMClient,
        token_budget: int = _DEFAULT_TOKEN_BUDGET,
        model: str = _DEFAULT_MODEL,
        system_prompt_override: str | None = None,
        protocol_retries: int = 2,
        provider: str | None = None,
        high_temp_by_provider: dict[str, float] | None = None,
        request_json_object: bool = True,
        prompt_by_model: dict[str, str] | None = None,
    ) -> None:
        self._llm = llm_client
        self._token_budget = token_budget
        self._model = model
        self._protocol_retries = max(0, protocol_retries)
        # v0.0.30: provider identifier so the corrector can pick a
        # provider-appropriate high-temperature value for the
        # ProtocolEmpty retry. ``None`` means "unknown provider";
        # the corrector falls back to the ``__default__`` entry.
        self._provider = provider
        self._high_temp_by_provider: dict[str, float] = {
            **_DEFAULT_HIGH_TEMP_BY_PROVIDER,
            **(high_temp_by_provider or {}),
        }
        # v0.0.30: when True (the default), pass ``json_object=True``
        # to ``LLMClient.complete()`` so providers with native
        # structured output enforce the JSON contract at the API
        # level. Older callers / non-conformant LLMClient
        # implementations can set this False; the parser still
        # accepts fence-wrapped or prose-wrapped JSON either way.
        self._request_json_object = request_json_object
        self._blocks_root = _resolve_blocks_root()
        # v0.0.41: the prompt_by_model registry is
        # the new shipped mechanism; system_prompt_override remains
        # as a single-prompt legacy escape hatch.
        self._prompt_by_model: dict[str, str] = (
            DEFAULT_PROMPT_BY_MODEL if prompt_by_model is None else dict(prompt_by_model)
        )
        self._system_prompt_override = system_prompt_override
        self._system_prompt = self._resolve_system_prompt()
        # Eager validation: the seed-block library must exist at
        # construction time. A misshipped wheel surfaces as a clear
        # error here, not as a cryptic FileNotFoundError on first
        # correct() call.
        self._error_vocab_paths = self._discover_error_vocab_paths()
        self._pattern_paths = self._discover_pattern_paths()
        if not self._error_vocab_paths:
            raise FileNotFoundError(
                "RampartCorrector: no error-vocab block files found under "
                f"{self._blocks_root / 'error_vocab'!s}. The package data "
                "is likely missing from your install."
            )

    def _resolve_system_prompt(self) -> str:
        """Resolve the active system prompt at construction time.

        Order (first hit wins):

        1. ``system_prompt_override`` — legacy single-prompt escape
           hatch. Applied to every model.
        2. ``prompt_by_model[self._model]`` — explicit per-model
           entry.
        3. ``prompt_by_model["default"]`` — registry default.
        4. Bundled ``system.md`` — pre-v0.0.41 fallback.
        """
        if self._system_prompt_override is not None:
            return self._system_prompt_override
        if self._model in self._prompt_by_model:
            return self._prompt_by_model[self._model]
        if "default" in self._prompt_by_model:
            return self._prompt_by_model["default"]
        return (self._blocks_root / "system.md").read_text(encoding="utf-8")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def correct(
        self,
        query: str,
        error: GateError,
        context: CorrectorContext,
        *,
        on_observation: ObservationCallback | None = None,
    ) -> CorrectorResult:
        """Single LLM call → parse → :class:`CorrectorResult`.

        v0.0.42: strictly single-shot. The pre-
        v0.0.42 in-class protocol-retry loop (re-prompt on
        :class:`ProtocolMalformed` with an escalated system-prompt
        preamble) and one-shot high-temperature retry on
        :class:`ProtocolEmpty` are factored out into the
        composable :class:`ProtocolRetryingCorrector` and
        :class:`EmptyRetryingCorrector` decorators in
        :mod:`cygnet.corrector.decorators`.

        To preserve the behaviour, the production wrapping
        is::

            ProtocolRetryingCorrector(
                EmptyRetryingCorrector(
                    RampartCorrector(...),
                    high_temp_by_provider=...,
                    provider=...,
                ),
                retries=2,
            )

        The convenience :func:`cygnet.run_correction` (Phase C)
        applies this wrapping automatically by default. Users
        calling :class:`RampartCorrector` directly get bare single-
        shot behaviour; wrap explicitly for the
        configuration.

        Metadata hints the decorators set, which this method reads
        from :attr:`CorrectorContext.metadata`:

        - ``_protocol_attempt`` (``int`` as str, default ``"1"``):
          the on-disk observation marker. Values >= 100 indicate a
          high-temperature retry (the +100 convention from
          v0.0.30); the logical attempt count is
          ``value % 100``. The system-prompt preamble is prepended
          when ``value % 100 > 1`` (i.e. on every non-first
          protocol attempt, including the high-temp retry of a
          non-first protocol attempt).
        - ``_temperature`` (``float`` as str, default ``"0.1"``):
          the temperature override for this call. The
          :class:`EmptyRetryingCorrector` decorator sets it to a
          provider-appropriate high value when retrying after
          :class:`ProtocolEmpty`.

        ``protocol_attempts=1`` and ``used_high_temp_retry=False``
        are returned on the :class:`CorrectorResult`. Decorators
        override these fields when wrapping multiple calls.
        """
        # v0.0.25: when the gate ran in collect_all mode,
        # ``context.all_errors`` carries the full set the LLM should
        # fix in one shot. Deduplicate by category so two backends
        # naming the same problem twice don't double-bill the block
        # budget while preserving both phrasings in the intent block.
        # Single-error callers leave ``all_errors`` empty.
        errors: list[GateError] = list(context.all_errors) if context.all_errors else [error]
        categories: list[str] = []
        seen_categories: set[str] = set()
        for e in errors:
            if e.category not in seen_categories:
                categories.append(e.category)
                seen_categories.add(e.category)

        # Read decorator-supplied hints from context.metadata.
        raw_attempt = int(context.metadata.get("_protocol_attempt", "1"))
        logical_attempt = raw_attempt % 100
        temperature = float(context.metadata.get("_temperature", "0.1"))

        system_prompt = self._system_prompt
        if logical_attempt > 1:
            system_prompt = _PROTOCOL_RETRY_PREAMBLE + system_prompt

        outcome = self._make_one_call(
            query=query,
            errors=errors,
            categories=categories,
            context=context,
            system_prompt=system_prompt,
            temperature=temperature,
            protocol_attempt=raw_attempt,
            on_observation=on_observation,
        )
        if isinstance(outcome, CorrectorResult):
            # An exception happened inside ``_make_one_call``;
            # it already returned a fully-formed abort result.
            return outcome

        return outcome_to_result(
            outcome,
            attempt_number=context.attempt_number,
            corrector_name=f"RampartCorrector ({self._model})",
            protocol_attempts=1,
            used_high_temp_retry=False,
        )

    def _make_one_call(
        self,
        *,
        query: str,
        errors: list[GateError],
        categories: list[str],
        context: CorrectorContext,
        system_prompt: str,
        temperature: float,
        protocol_attempt: int,
        on_observation: ObservationCallback | None,
    ) -> ProtocolOK | ProtocolEchoed | ProtocolEmpty | ProtocolMalformed | CorrectorResult:
        """One LLM call + parse. Returns a parser outcome on success,
        or a fully-formed abort :class:`CorrectorResult` when the SDK
        raises. Centralises the registry compile + observation emit
        so the retry loop in :meth:`correct` stays linear.

        v0.0.30: the temperature is a parameter so the high-temp
        retry path can call this with the creative-tier value.

        v0.0.31: emits :class:`LLMCallObservation` records to the
        ``on_observation`` callback. The callback is supplied by
        :class:`RefinementLoop` (or a direct caller); when ``None``,
        no observations are emitted.
        """
        try:
            registry, vocab_count = self._build_registry(categories)
            self._inject_per_call_blocks(registry, query, errors, context, vocab_count)
            compiled = registry.compile(max_tokens=self._token_budget)
            response = self._llm.complete(
                system_prompt=system_prompt,
                user_prompt=compiled.prompt,
                max_tokens=2000,
                temperature=temperature,
                json_object=self._request_json_object,
            )
        except Exception as exc:
            self._emit_observation(
                on_observation,
                system_prompt=self._system_prompt,
                user_prompt="(prompt not assembled — exception before build)",
                raw_response="",
                parser_outcome="exception",
                parser_reason=f"{type(exc).__name__}: {exc}",
                extracted_cypher=None,
                protocol_attempt=protocol_attempt,
                temperature=temperature,
                input_tokens=0,
                output_tokens=0,
                elapsed_seconds=0.0,
                model=self._model,
                provider=self._provider or "",
            )
            return CorrectorResult(
                action="abort",
                refined_query=None,
                reasoning=(f"RampartCorrector aborted: {type(exc).__name__}: {exc}"),
                attempts_used=context.attempt_number,
                reason="exception",
                protocol_attempts=protocol_attempt,
                used_high_temp_retry=temperature > 0.5,
            )
        finally:
            registry_to_release = locals().get("registry")
            if registry_to_release is not None:
                with contextlib.suppress(Exception):
                    registry_to_release.release()

        outcome = parse_corrector_response_v2(response.text, input_cypher=query)
        self._emit_observation(
            on_observation,
            system_prompt=system_prompt,
            user_prompt=compiled.prompt,
            raw_response=response.text,
            parser_outcome=outcome.outcome,
            parser_reason=getattr(outcome, "reason", None),
            extracted_cypher=getattr(outcome, "cypher", None),
            protocol_attempt=protocol_attempt,
            temperature=temperature,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            elapsed_seconds=response.elapsed_seconds,
            model=response.model,
            provider=response.provider,
        )
        return outcome

    # ------------------------------------------------------------------
    # Per-call observation hook (v0.0.31)
    # ------------------------------------------------------------------

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
        protocol_attempt: int,
        temperature: float,
        input_tokens: int,
        output_tokens: int,
        elapsed_seconds: float,
        model: str,
        provider: str,
    ) -> None:
        """Build an :class:`LLMCallObservation` and forward to the
        caller's callback. The callback may raise (e.g. disk full
        when writing telemetry); we catch and log so the corrector
        path stays alive."""
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
                    protocol_attempt=protocol_attempt,
                    elapsed_seconds=elapsed_seconds,
                )
            )
        except Exception:
            _module_logger().warning(
                "on_observation callback raised; suppressing to keep the corrector path alive",
                exc_info=True,
            )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _discover_error_vocab_paths(self) -> dict[str, Path]:
        """Discover the per-category error-vocab block files.

        Keys are the category names (``"parse"`` / ``"schema"`` /
        ``"property"`` / ``"constraint"`` / ``"cost"`` /
        ``"empty"``), values are the on-disk paths to the matching
        ``.md`` block. Only the matching block gets loaded into the
        registry per call ( fix 2 — the other five used to
        be loaded uselessly).
        """
        out: dict[str, Path] = {}
        error_vocab_dir = self._blocks_root / "error_vocab"
        if not error_vocab_dir.is_dir():
            return out
        for path in sorted(error_vocab_dir.glob("*.md")):
            out[path.stem] = path
        return out

    def _discover_pattern_paths(self) -> list[Path]:
        """Discover the pattern-seed block files (``patterns/*.md``).

        These are loaded into the registry on every call regardless
        of error category — they're the general "here's how typical
        Cypher patterns look" guidance.
        """
        patterns_dir = self._blocks_root / "patterns"
        if not patterns_dir.is_dir():
            return []
        return sorted(patterns_dir.glob("*.md"))

    def _build_registry(self, error_categories: list[str]) -> tuple[BlockRegistry, int]:
        """Build the per-call registry and report the vocab-block count.

        Returns ``(registry, n_vocab_loaded)`` where ``n_vocab_loaded``
        is the number of error-vocab seed blocks that landed at the
        head of the registry. The caller uses this to compute the
        first agent-block insertion position so the intent block sits
        immediately after the vocab seeds and before the pattern
        seeds.
        """
        # Local import so the corrector package imports cleanly even
        # when the ``corrector`` extra hasn't been installed (the
        # NullCorrector path stays available).
        from rampart import BlockRegistry

        vocab_paths: list[Path] = []
        # Load one error-vocab block per distinct failing category
        # (v0.0.25). For single-error callers ``error_categories`` is a
        # one-element list and we load the same one block as v0.0.24;
        # for collect_all callers we load each category's block in the
        # order the chain produced them (parse first, then by backend
        # authority — see ``ValidatorChain._validate_collect_all``).
        # Unknown categories (future CYGNET versions) are silently
        # skipped — the intent block still carries the full payloads.
        for category in error_categories:
            matching = self._error_vocab_paths.get(category)
            if matching is not None:
                vocab_paths.append(matching)
        seed_paths = vocab_paths + self._pattern_paths
        return BlockRegistry.from_files(seed_paths), len(vocab_paths)

    def _inject_per_call_blocks(
        self,
        registry: BlockRegistry,
        query: str,
        errors: list[GateError],
        context: CorrectorContext,
        vocab_block_count: int,
    ) -> None:
        """Lay out per-call blocks ahead of the pattern seeds.

        The compile walk is front-to-back and stops when the budget is
        exhausted. The high-signal content (intent, schema,
        prior-attempts) is therefore inserted immediately after the
        matching error-vocab block(s) so it survives even when the
        token budget can't fit the pattern library at the tail.

        Final layout (v0.0.25, extended from)::

            0..k-1: matching error_vocab blocks (seed; one per distinct
                    failing category in collect_all mode, one in
                    short_circuit mode)
            k:      intent (agent, priority 1.0; renders one error or
                    a multi-error fix-all-at-once payload)
            k+1..:  schema blocks (only for labels/rels the query or
                    any error-payload references) (agent)
            ...:    prior_attempts (agent)
            ...:    conversation_history (agent, optional)
            ...:    pattern seeds
        """
        trajectory = f"attempt_{context.attempt_number}"
        # In v0.0.24 the intent block sat at position 1 immediately
        # after the single error-vocab block. In v0.0.25 collect_all
        # mode there may be more than one vocab block at the head, so
        # the intent position shifts to the first slot after all the
        # vocab seeds. Pattern seeds are loaded after vocab in
        # ``_build_registry`` so they get pushed to the tail as agent
        # blocks fill in here.
        cursor = vocab_block_count

        # 1. Intent block — one for the primary error and a "fix all"
        #    framing when there are multiple errors.
        registry.write_agent_block(
            content=_format_intent(query, errors, context.attempt_number),
            trajectory_id=trajectory,
            semantic_name="cygnet_intent",
            priority=_PRIORITY_INTENT,
            position=cursor,
        )
        cursor += 1

        # 2. Schema blocks: filtered to labels/rels the query or any
        #    error payload in ``errors`` references (or the full schema
        #    in the fallback case).
        cursor = self._write_schema_blocks(
            registry, query, errors, context.schema_, trajectory, cursor
        )

        # 3. Prior attempts.
        for i, attempt in enumerate(context.prior_attempts):
            registry.write_agent_block(
                content=_format_prior_attempt(attempt, i),
                trajectory_id=trajectory,
                semantic_name=f"cygnet_prior_attempt_{i}",
                priority=_PRIORITY_PRIOR,
                position=cursor,
            )
            cursor += 1

        # 4. Optional conversation history.
        if context.conversation_history:
            registry.write_agent_block(
                content=_format_conversation(context.conversation_history),
                trajectory_id=trajectory,
                semantic_name="cygnet_conversation_history",
                priority=_PRIORITY_PRIOR,
                position=cursor,
            )

    @staticmethod
    def _write_schema_blocks(
        registry: BlockRegistry,
        query: str,
        errors: list[GateError],
        schema: Schema,
        trajectory: str,
        start_position: int,
    ) -> int:
        cursor = start_position
        # Union the relevance sets across all errors so a schema block
        # is included if *any* error references its label/rel.
        relevant_labels: set[str] = set()
        relevant_rels: set[str] = set()
        for e in errors:
            ls, rs = _relevant_schema_entities(query, e, schema)
            relevant_labels |= ls
            relevant_rels |= rs
        for label in schema.labels:
            if label.name not in relevant_labels:
                continue
            props = schema.properties_by_label.get(label.name, [])
            registry.write_agent_block(
                content=_format_label_schema(label.name, props),
                trajectory_id=trajectory,
                semantic_name=f"cygnet_schema_label_{label.name}",
                priority=_PRIORITY_SCHEMA,
                position=cursor,
            )
            cursor += 1
        for rel in schema.relationship_types:
            if rel.name not in relevant_rels:
                continue
            props = schema.properties_by_rel_type.get(rel.name, [])
            registry.write_agent_block(
                content=_format_rel_schema(rel, props),
                trajectory_id=trajectory,
                semantic_name=f"cygnet_schema_rel_{rel.name}",
                priority=_PRIORITY_SCHEMA,
                position=cursor,
            )
            cursor += 1
        return cursor


# ---------------------------------------------------------------------------
# Block formatters
# ---------------------------------------------------------------------------


def _format_intent(query: str, errors: list[GateError], attempt_number: int) -> str:
    """Render the intent block.

    Delegates to :mod:`cygnet.corrector.renderers` so the bench
    runners can reuse the same renderer for their structured-style
    prior-attempt blocks. v0.0.30 refactor; behaviour unchanged from
    v0.0.25.
    """
    if len(errors) == 1:
        return render_gate_error_block(
            query=query,
            error=errors[0],
            header_kind="intent",
            attempt_index=attempt_number,
        )
    return render_multi_error_intent(query=query, errors=errors, attempt_number=attempt_number)


def _format_label_schema(label_name: str, props: object) -> str:
    """Render a single label's declared properties for the prompt."""
    lines = [f"# Schema: label `{label_name}`"]
    if not props:
        lines.append("\n(No declared properties.)")
    else:
        lines.append("")
        for p in props:  # type: ignore[attr-defined]
            optional = "optional" if p.optional else "required"
            sparse = ", sparse" if p.sparse else ""
            lines.append(f"- `{p.name}`: {p.type} ({optional}{sparse})")
    return "\n".join(lines)


def _format_rel_schema(rel: object, props: object) -> str:
    """Render a single relationship type's declared endpoints + properties."""
    lines = [
        f"# Schema: relationship `:{rel.name}`",  # type: ignore[attr-defined]
        "",
        f"- source label: `{rel.source_label}`",  # type: ignore[attr-defined]
        f"- target label: `{rel.target_label}`",  # type: ignore[attr-defined]
    ]
    if props:
        lines.append("- properties:")
        for p in props:  # type: ignore[attr-defined]
            optional = "optional" if p.optional else "required"
            lines.append(f"    - `{p.name}`: {p.type} ({optional})")
    return "\n".join(lines)


def _format_prior_attempt(attempt: object, index: int) -> str:
    """Render a single prior-attempt entry for the prompt.

    The error-bearing path delegates to
    :func:`cygnet.corrector.renderers.render_gate_error_block` (shared
    with the bench's structured-style conditions). The "passed
    gating" path stays inline — it's a single conditional line that
    doesn't reuse anywhere else.
    """
    if attempt.error is None:  # type: ignore[attr-defined]
        return (
            f"# Prior attempt {index}\n\n"
            "## Query\n\n"
            "```cypher\n"
            f"{attempt.query}\n"  # type: ignore[attr-defined]
            "```\n\n"
            "(Passed gating; included for context.)"
        )
    return render_gate_error_block(
        query=attempt.query,  # type: ignore[attr-defined]
        error=attempt.error,  # type: ignore[attr-defined]
        header_kind="prior_attempt",
        attempt_index=index,
    )


def _format_conversation(history: list[str]) -> str:
    return "# Conversation history\n\n" + "\n\n".join(history)


# ---------------------------------------------------------------------------
# Query-relevance filter for schema blocks
# ---------------------------------------------------------------------------


def _relevant_schema_entities(
    query: str,
    error: GateError,
    schema: Schema,
) -> tuple[set[str], set[str]]:
    """Return ``(relevant_labels, relevant_rel_types)``: the schema
    entities the failing query — or the error payload's suggestion
    fields — actually references.

    Resolution rules:

    1. Extract every ``:Name`` token from the query. Classify each
       against the schema's declared vocabulary: a name that's
       declared as a label is a label reference, declared as a
       relationship type is a rel reference, anything else is the
       unknown-reference (the broken one) and isn't added.
    2. When the error is a :class:`SchemaError` whose payload carries
       ``did_you_mean`` and ``available_in_scope`` lists, fold those
       candidates into the relevant set under the error's
       ``reference_kind`` (label / relationship / property). For
       property misses, do not add labels — the relevant label is
       the bound variable's label, already caught by step 1.
    3. Include every relationship type whose source or target label
       is in the relevant-label set. This carries connection context
       for labels the LLM is being shown — useful for refinements
       that may need to traverse from those labels. The rel block's
       prose body already names the endpoint labels, so the LLM
       knows what's reachable even without the full schema entry
       for those labels.
    4. **Fallback.** When steps 1-3 produce an empty label set
       (e.g. ``MATCH (n) RETURN n`` references no specific labels and
       no schema error fires), return the full schema. The LLM needs
       *some* schema context; the all-bets-off case is the brief's
       conservative-defaults instruction.

    Note the asymmetry: rels expand from labels (step 3), but
    endpoint labels of those rels do **not** further expand the
    label set. The conservative choice keeps the prompt small and
    relies on the rel block's prose to surface the missing labels'
    names. Symmetric expansion (both directions) cascades too far on
    densely-connected schemas — a query touching one label can pull
    in half the schema via two hops.

    Property errors deliberately don't expand the label set via
    ``available_in_scope`` because that list is property names, not
    labels.
    """
    declared_labels = {label.name for label in schema.labels}
    declared_rels = {rel.name for rel in schema.relationship_types}
    relevant_labels: set[str] = set()
    relevant_rels: set[str] = set()

    # Step 1: query-side reference extraction.
    for match in _SCHEMA_NAME_REF.finditer(query):
        name = match.group(1)
        if name in declared_labels:
            relevant_labels.add(name)
        elif name in declared_rels:
            relevant_rels.add(name)

    # Step 2: error-payload suggestion expansion. ``getattr`` keeps the
    # branch payload-shape-agnostic — the schema payload is the only
    # one with reference_kind + available_in_scope today, but other
    # payloads can add fields without breaking this code.
    if error.category == "schema":
        payload = error.payload
        kind = getattr(payload, "reference_kind", None)
        suggestions: list[str] = list(getattr(payload, "did_you_mean", []) or [])
        in_scope: list[str] = list(getattr(payload, "available_in_scope", []) or [])
        if kind == "label":
            for name in suggestions + in_scope:
                if name in declared_labels:
                    relevant_labels.add(name)
        elif kind == "relationship":
            for name in suggestions + in_scope:
                if name in declared_rels:
                    relevant_rels.add(name)
        # kind == "property": handled by step 1 + the bound label

    # Step 3: include rels touching any relevant label. We do NOT
    # cascade further to add the rel's endpoint labels — the brief's
    # "conservative" intent stops at one hop. The rel block's body
    # already lists the endpoint labels in prose, so the LLM knows
    # what's reachable without paying for full label schemas of
    # everything one hop away.
    for rel in schema.relationship_types:
        if rel.source_label in relevant_labels or rel.target_label in relevant_labels:
            relevant_rels.add(rel.name)

    # Step 4: fallback to full schema when no labels matched.
    if not relevant_labels:
        relevant_labels = set(declared_labels)
        relevant_rels = set(declared_rels)

    return relevant_labels, relevant_rels


# NOTE: response-parser helpers (``_extract_cypher_block`` /
# ``_extract_any_code_block``) were removed in v0.0.28. Use
# :func:`cygnet.corrector.response_parser.parse_corrector_response`
# instead. The new parser settles protocol compliance with a
# three-stage discriminated-union outcome rather than two boolean
# fallbacks; the caller branches on ``outcome`` in {``ok``,
# ``recoverable``, ``unrecoverable``}.


# ---------------------------------------------------------------------------
# Filesystem lookup
# ---------------------------------------------------------------------------


def _resolve_blocks_root() -> Path:
    """Locate the shipped ``rampart_blocks`` directory on disk.

    Uses :mod:`importlib.resources` so the lookup works for both
    editable and wheel installs. Returns a :class:`pathlib.Path` so
    callers can use :meth:`Path.read_text`, :meth:`Path.glob`, and
    pass values straight to :func:`rampart.parse_file`.
    """
    traversable = files("cygnet.corrector").joinpath("rampart_blocks")
    return Path(str(traversable))
