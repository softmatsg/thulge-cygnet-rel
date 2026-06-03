# Copyright (c) 2026 Nikodem Tomczak, Thulge Labs. All rights reserved.

"""LLM client abstraction used by the RAMPART-backed corrector.

A thin Protocol-shaped surface with four concrete implementations
(Anthropic / OpenAI / Gemini / Ollama) that the corrector can call
without committing to any SDK at import time. Each implementation
is tiny — the corrector's complexity belongs in RAMPART prompt
assembly, not in vendor-specific request marshalling.

The :class:`LLMClient` protocol is duck-typed; nothing in this
module requires inheritance. Callers can substitute a mock by
exposing a ``complete`` method with the same signature.

``complete`` returns :class:`LLMResponse`, which carries text
plus token counts, model identifier, provider name, and wall-clock
latency so the corrector can emit an :class:`LLMCallObservation`
and downstream consumers can compute cost from the same single
response.
"""

from __future__ import annotations

import json
import os
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Final, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    pass

__all__ = [
    "AnthropicClient",
    "GoogleGeminiClient",
    "LLMCallTimeoutError",
    "LLMClient",
    "LLMResponse",
    "OllamaClient",
    "OpenAIClient",
    "ResilientLLMClient",
    "RetryPolicy",
    "TransientLLMError",
    "make_llm_client",
]


class LLMResponse(BaseModel):
    """One LLM completion result.

    Concrete clients fill all fields. The corrector reads
    :attr:`text` for parsing and uses the rest to populate a
    per-call observation handed to telemetry. Downstream cost-aware
    wrappers can read :attr:`input_tokens` / :attr:`output_tokens`
    and apply a price table.

    ``elapsed_seconds`` is wall-clock time around the SDK call,
    measured client-side. Useful for telemetry and for spotting
    runaway latency without trusting the SDK's own timing fields.
    """

    model_config = ConfigDict(extra="forbid")

    text: str = Field(..., description="The assistant's response text.")
    model: str = Field(..., description="Model identifier (echo of client's configured model).")
    provider: str = Field(
        ...,
        description=("Provider name: ``anthropic`` / ``openai`` / ``gemini`` / ``ollama``."),
    )
    input_tokens: int = Field(
        ...,
        ge=0,
        description="Prompt token count reported by the provider.",
    )
    output_tokens: int = Field(
        ...,
        ge=0,
        description=(
            "Output token count. For thinking models (e.g. Gemini 3.x preview), "
            "this includes reasoning tokens — the consumer's pricing logic "
            "decides whether to bill them at the output rate."
        ),
    )
    elapsed_seconds: float = Field(
        ...,
        ge=0.0,
        description="Wall-clock time spent in the SDK call (measured client-side).",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When the call completed (UTC).",
    )


@runtime_checkable
class LLMClient(Protocol):
    """Minimal LLM-completion surface the corrector binds against.

    When ``json_object`` is ``True`` the client should use its
    provider-native structured-output mode where available (Gemini
    ``responseSchema``, Anthropic forced-tool, OpenAI
    ``response_format``). Providers without a native mode (Ollama)
    should fall back to prompt-side instruction; the parser
    tolerates fence-wrapping and surrounding prose either way.

    The return type is :class:`LLMResponse` so the corrector and
    downstream consumers can read tokens and latency without poking
    at provider-specific response shapes.
    """

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 2000,
        temperature: float = 0.1,
        *,
        json_object: bool = False,
    ) -> LLMResponse: ...


# The CYGNET corrector's JSON contract. Each provider's structured-
# output mode is configured against this schema where available.
# Kept here (not in response_parser) so the LLM clients are
# self-contained — they need to know the shape but not the parser
# internals. The schema is intentionally permissive on
# ``additionalProperties`` because some models add fields like
# ``confidence`` that we don't need but don't want to reject either
# (the parser drops unknown fields).
_CYPHER_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "cypher": {"type": "string"},
        "explanation": {"type": "string"},
    },
    "required": ["cypher"],
}

_CYPHER_ONLY_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "cypher": {"type": "string"},
    },
    "required": ["cypher"],
}
"""The shipped default ``response_schema`` fallback for every LLM
client. Pairs with the system prompt in
:data:`cygnet.corrector.rampart_backed.V4_SYSTEM_PROMPT`.

The cypher-only variant strips ``explanation`` so the SDK's
structured-output enforcement doesn't suggest it.

The constant intentionally omits ``additionalProperties``. The
Gemini API's structured-output validator rejects unknown JSON-
Schema keys, so adding it breaks every Gemini call. OpenAI's
structured-outputs strict mode REQUIRES
``additionalProperties: false`` — the OpenAI client wrapper
injects the field at call time rather than baking it into this
shared constant; see :class:`OpenAIClient.complete`.

Callers that need the cypher-with-explanation schema opt in via
``response_schema=_CYPHER_RESPONSE_SCHEMA`` on the client
constructor.
"""


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------


class AnthropicClient:
    """Anthropic-backed :class:`LLMClient`.

    Reads ``ANTHROPIC_API_KEY`` from the environment unless an explicit
    ``api_key`` is passed. The SDK import is lazy so the corrector
    package stays importable when the ``corrector`` extra is not
    installed.
    """

    provider: str = "anthropic"

    def __init__(
        self,
        model: str = "claude-sonnet-4-5",
        api_key: str | None = None,
        *,
        response_schema: dict[str, Any] | None = None,
    ) -> None:
        import anthropic

        self._model = model
        self._client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        # Per-client override for the structured-output schema. When
        # ``None``, falls back to :data:`_CYPHER_ONLY_RESPONSE_SCHEMA`.
        self._response_schema = response_schema

    @property
    def model(self) -> str:
        return self._model

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 2000,
        temperature: float = 0.1,
        *,
        json_object: bool = False,
    ) -> LLMResponse:
        started = time.perf_counter()
        if json_object:
            tool: Any = {
                "name": "submit_refinement",
                "description": (
                    "Submit the refined Cypher query (or an empty "
                    "string with an explanation when no refinement "
                    "is possible)."
                ),
                "input_schema": self._response_schema or _CYPHER_ONLY_RESPONSE_SCHEMA,
            }
            tool_choice_arg: Any = {"type": "tool", "name": "submit_refinement"}
            response = self._client.messages.create(
                model=self._model,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                tools=[tool],
                tool_choice=tool_choice_arg,
            )
            text = ""
            for part in response.content:
                if getattr(part, "type", None) == "tool_use":
                    payload = getattr(part, "input", None)
                    if isinstance(payload, dict):
                        text = json.dumps(payload)
                        break
            if not text:
                text = "".join(
                    getattr(part, "text", "")
                    for part in response.content
                    if getattr(part, "text", None)
                )
        else:
            response = self._client.messages.create(
                model=self._model,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            text = "".join(
                getattr(part, "text", "")
                for part in response.content
                if getattr(part, "text", None)
            )
        elapsed = time.perf_counter() - started
        usage = response.usage
        return LLMResponse(
            text=text,
            model=self._model,
            provider=self.provider,
            input_tokens=int(usage.input_tokens),
            output_tokens=int(usage.output_tokens),
            elapsed_seconds=elapsed,
        )


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------


class OpenAIClient:
    """OpenAI-backed :class:`LLMClient`.

    Reads ``OPENAI_API_KEY`` from the environment unless overridden.
    """

    provider: str = "openai"

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        *,
        response_schema: dict[str, Any] | None = None,
    ) -> None:
        import openai

        self._model = model
        self._client = openai.OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        # Per-client schema override (see Anthropic).
        self._response_schema = response_schema

    @property
    def model(self) -> str:
        return self._model

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 2000,
        temperature: float = 0.1,
        *,
        json_object: bool = False,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if json_object:
            strict_schema = dict(self._response_schema or _CYPHER_ONLY_RESPONSE_SCHEMA)
            strict_schema["additionalProperties"] = False
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "cypher_response",
                    "strict": True,
                    "schema": strict_schema,
                },
            }
        started = time.perf_counter()
        response = self._client.chat.completions.create(**kwargs)
        elapsed = time.perf_counter() - started
        text = response.choices[0].message.content or ""
        usage = response.usage
        # OpenAI's usage object names tokens ``prompt_tokens`` /
        # ``completion_tokens`` (not input/output). Normalise to the
        # LLMResponse vocabulary.
        input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0
        output_tokens = int(getattr(usage, "completion_tokens", 0) or 0) if usage else 0
        return LLMResponse(
            text=text,
            model=self._model,
            provider=self.provider,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            elapsed_seconds=elapsed,
        )


# ---------------------------------------------------------------------------
# Google Gemini
# ---------------------------------------------------------------------------


class GoogleGeminiClient:
    """Google Gemini-backed :class:`LLMClient`.

    Uses the unified ``google-genai`` SDK (the post-2024 replacement
    for the older ``google-generativeai`` package). Reads
    ``GOOGLE_API_KEY`` from the environment unless an explicit
    ``api_key`` is passed; ``GEMINI_API_KEY`` is honoured as a
    fallback name since both spellings appear in Google's docs.

    Gemini's content interface differs from the chat-completion shape
    Anthropic / OpenAI use: there is no first-class ``system`` role.
    The system prompt is passed via
    :class:`GenerateContentConfig.system_instruction`, matching
    Google's recommendation for system-instruction equivalents.
    """

    provider: str = "gemini"

    def __init__(
        self,
        model: str = "gemini-2.5-pro",
        api_key: str | None = None,
        *,
        response_schema: dict[str, Any] | None = None,
    ) -> None:
        from google import genai

        self._model = model
        resolved = api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        self._client = genai.Client(api_key=resolved)
        # Per-client schema override (see Anthropic).
        self._response_schema = response_schema

    @property
    def model(self) -> str:
        return self._model

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 2000,
        temperature: float = 0.1,
        *,
        json_object: bool = False,
    ) -> LLMResponse:
        from google.genai import types

        config_kwargs: dict[str, Any] = {
            "system_instruction": system_prompt,
            "max_output_tokens": max_tokens,
            "temperature": temperature,
        }
        if json_object:
            config_kwargs["response_mime_type"] = "application/json"
            config_kwargs["response_schema"] = self._response_schema or _CYPHER_ONLY_RESPONSE_SCHEMA
        config = types.GenerateContentConfig(**config_kwargs)
        started = time.perf_counter()
        response = self._client.models.generate_content(
            model=self._model,
            contents=user_prompt,
            config=config,
        )
        elapsed = time.perf_counter() - started
        text = response.text or ""
        # Read total_token_count and compute output_tokens =
        # total - prompt so thinking-model reasoning tokens are
        # captured. The candidates_token_count field undercounts by
        # ~10x on Gemini 3.x preview models.
        usage = getattr(response, "usage_metadata", None)
        prompt_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
        candidate_tokens = int(getattr(usage, "candidates_token_count", 0) or 0)
        total_tokens = int(getattr(usage, "total_token_count", 0) or 0)
        thoughts_tokens = int(getattr(usage, "thoughts_token_count", 0) or 0)
        if total_tokens >= prompt_tokens + candidate_tokens:
            effective_output = total_tokens - prompt_tokens
        else:
            effective_output = candidate_tokens + thoughts_tokens
        return LLMResponse(
            text=text,
            model=self._model,
            provider=self.provider,
            input_tokens=prompt_tokens,
            output_tokens=effective_output,
            elapsed_seconds=elapsed,
        )


# ---------------------------------------------------------------------------
# Ollama (local)
# ---------------------------------------------------------------------------


class OllamaClient:
    """Ollama-backed :class:`LLMClient` for local model serving.

    Talks to a locally-running ``ollama`` server (default
    ``http://localhost:11434``); the model must already be pulled
    (``ollama pull <model>``). Token counts are reported via
    :attr:`LLMResponse.input_tokens` / :attr:`output_tokens` so
    prompt-shape efficiency can be compared across local and
    remote backends. Pricing is a consumer concern; the local-cost
    detection lives in the consumer's price table, not in this
    client.

    Unlike the cloud clients, ``model`` is **required**: there is
    no sensible default since each user pulls a different set of
    models. The library does not validate that the model is
    available — Ollama returns a clear error from the server when
    it isn't.
    """

    provider: str = "ollama"

    def __init__(
        self,
        model: str,
        host: str | None = None,
        *,
        response_schema: dict[str, Any] | None = None,
    ) -> None:
        import ollama

        self._model = model
        resolved_host = host or os.environ.get("OLLAMA_HOST")
        # ollama.Client accepts host=None (uses its built-in default
        # of http://localhost:11434).
        self._client = ollama.Client(host=resolved_host) if resolved_host else ollama.Client()
        # Per-client schema override (see Anthropic).
        self._response_schema = response_schema

    @property
    def model(self) -> str:
        return self._model

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 2000,
        temperature: float = 0.1,
        *,
        json_object: bool = False,
    ) -> LLMResponse:
        chat_kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        }
        if json_object:
            chat_kwargs["format"] = self._response_schema or _CYPHER_ONLY_RESPONSE_SCHEMA
        started = time.perf_counter()
        response = self._client.chat(**chat_kwargs)
        elapsed = time.perf_counter() - started
        message = (
            response.get("message")
            if isinstance(response, dict)
            else getattr(response, "message", None)
        )
        if message is None:
            text = ""
        elif isinstance(message, dict):
            text = str(message.get("content") or "")
        else:
            text = str(getattr(message, "content", "") or "")
        # Ollama's response carries ``prompt_eval_count`` (input
        # tokens) and ``eval_count`` (output tokens) at the top
        # level on both dict and object shapes.
        input_tokens = int(_field_from(response, "prompt_eval_count") or 0)
        output_tokens = int(_field_from(response, "eval_count") or 0)
        return LLMResponse(
            text=text,
            model=self._model,
            provider=self.provider,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            elapsed_seconds=elapsed,
        )


def _field_from(response: Any, name: str) -> Any:
    """Read ``name`` from an Ollama response that may be dict- or
    object-shaped depending on SDK version."""
    if isinstance(response, dict):
        return response.get(name)
    return getattr(response, name, None)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def make_llm_client(backend: str = "anthropic", **kwargs: Any) -> LLMClient:
    """Construct an :class:`LLMClient` for the named backend.

    Accepts ``"anthropic"``, ``"openai"``, ``"gemini"`` (alias
    ``"google"``), or ``"ollama"`` (alias ``"local"``). Extra
    keyword arguments pass through to the chosen client's
    constructor (e.g. ``model``, ``api_key``, ``host``).
    """
    backend_lower = backend.lower()
    if backend_lower == "anthropic":
        return AnthropicClient(**kwargs)
    if backend_lower == "openai":
        return OpenAIClient(**kwargs)
    if backend_lower in {"gemini", "google"}:
        return GoogleGeminiClient(**kwargs)
    if backend_lower in {"ollama", "local"}:
        return OllamaClient(**kwargs)
    raise ValueError(
        f"Unknown LLM backend {backend!r}. Supported: 'anthropic', 'openai', 'gemini', 'ollama'."
    )


# ---------------------------------------------------------------------------
# Resilience layer
# ---------------------------------------------------------------------------


class LLMCallTimeoutError(Exception):
    """Raised by :class:`ResilientLLMClient` when an LLM call
    exceeds the configured wall-clock timeout. The underlying SDK
    call may still be running in a daemon thread — Python cannot
    preempt blocking native code — but the resilient client has
    abandoned it and decided to retry or surface the failure."""


class TransientLLMError(Exception):
    """Marker base class for SDK errors that the resilient client
    should retry. Provider SDKs raise their own concrete exception
    types (e.g. ``anthropic.APIError``, ``openai.RateLimitError``);
    the :class:`RetryPolicy` lists which of those should be treated
    as transient.

    Users wrapping their own SDK calls can also raise this directly
    if they want the retry layer to handle a custom transient
    condition.
    """


def _provider_transient_exceptions() -> tuple[type[BaseException], ...]:
    """Collect provider-SDK transient-error classes for the default
    :class:`RetryPolicy` retryable set.

    Each provider import is independently guarded so a user without
    one of the SDKs installed can still import this module. Google
    SDK's ``ResourceExhausted`` (429), ``ServiceUnavailable`` (503),
    and Anthropic / OpenAI equivalents retry by default. The SDKs
    themselves respect ``Retry-After`` headers, and
    :class:`RetryPolicy` adds the exponential-backoff layer on top.

    Returns a tuple of every successfully-imported exception class
    plus :class:`TransientLLMError`. Order doesn't matter for
    ``isinstance`` matching.
    """
    classes: list[type[BaseException]] = [TransientLLMError]

    # Google ``google-genai`` 2.x SDK transient classes (used by
    # GoogleGeminiClient). The SDK collapses transients into
    # ``ClientError`` (all 4xx including 429 rate-limit) and
    # ``ServerError`` (all 5xx including 500/502/503/504). We retry
    # both. ``ClientError`` is broader than ideal (catches
    # 400/401/404 too); those will exhaust the retry budget
    # naturally and surface, costing a handful of extra seconds.
    # Worth it to unblock 429 retry on free-tier endpoints.
    try:
        from google.genai import errors as _genai_excs

        classes.extend(
            [
                _genai_excs.ClientError,  # 4xx (incl. 429 ResourceExhausted)
                _genai_excs.ServerError,  # 5xx (incl. 500/502/503/504)
            ]
        )
    except ImportError:
        pass

    # Older ``google.api_core.exceptions`` set (used when the
    # legacy ``google-generativeai`` SDK or other google-cloud
    # client libraries are present). Kept for compatibility even
    # though ``google-genai`` is the corrector's default.
    try:
        from google.api_core import exceptions as _google_excs

        classes.extend(
            [
                _google_excs.ResourceExhausted,
                _google_excs.ServiceUnavailable,
                _google_excs.DeadlineExceeded,
                _google_excs.InternalServerError,
            ]
        )
    except ImportError:
        pass

    # Anthropic SDK transient classes.
    try:
        import anthropic

        classes.extend(
            [
                anthropic.RateLimitError,
                anthropic.APIConnectionError,
                anthropic.InternalServerError,
                anthropic.APITimeoutError,
            ]
        )
    except ImportError:
        pass

    # OpenAI SDK transient classes.
    try:
        import openai

        classes.extend(
            [
                openai.RateLimitError,
                openai.APIConnectionError,
                openai.InternalServerError,
                openai.APITimeoutError,
            ]
        )
    except ImportError:
        pass

    return tuple(classes)


_DEFAULT_RETRYABLE_EXCEPTIONS: Final[tuple[type[BaseException], ...]] = (
    _provider_transient_exceptions()
)


def _detect_rate_limit_hint(exc: BaseException) -> tuple[bool, float | None]:
    """Identify HTTP 429 / rate-limit errors and extract a quota-reset
    hint if the response carries one.

    Returns ``(is_rate_limit, hint_seconds)``:

    - ``(True, float)`` — exc is a rate-limit and carries an explicit
      retry-after / ``retryDelay`` hint. Honour it.
    - ``(True, None)`` — exc is a rate-limit but no hint is present.
      Caller should fall back to :data:`RetryPolicy.rate_limit_backoff_seconds`.
    - ``(False, None)`` — exc is not a rate-limit (e.g. 503 / timeout
      / generic transient). Caller uses the exponential path.

    Detection is conservative: an explicit HTTP 429 status code on the
    exception, an SDK class named ``RateLimitError`` or
    ``ResourceExhausted``, or a ``RESOURCE_EXHAUSTED`` substring in
    the stringified exception all count. Other 4xx codes (400, 401,
    404) are NOT treated as rate limits — they're terminal and
    callers should not retry.

    Hint extraction is best-effort string parsing against the SDK's
    serialised error payload. The Gemini API returns a
    ``RetryInfo.retryDelay`` field (also surfaced in the message as
    ``Please retry in Xs``); Anthropic / OpenAI surface a
    ``Retry-After`` header. The hint is honoured verbatim; if it
    turns out to be a burst-limiter hint that underestimates the
    true quota-window wait, that surfaces as the next retry hitting
    another 429 (and the wall-clock cap protects against unbounded
    loops).
    """
    import re

    msg = str(exc)
    msg_lower = msg.lower()
    code = getattr(exc, "code", None)
    cls_name = type(exc).__name__

    is_rate_limit = False
    if (
        code == 429
        or cls_name in {"ResourceExhausted", "RateLimitError"}
        or "RESOURCE_EXHAUSTED" in msg
        or " 429" in msg
        or "(429" in msg
        or "429 RESOURCE" in msg
        or msg.startswith("429")
    ):
        is_rate_limit = True

    if not is_rate_limit:
        return (False, None)

    # Hint extraction — try Gemini's RetryInfo first, then Retry-After
    # header, then a plain-text "retry in Xs" pattern.
    hint: float | None = None
    # google.genai RetryInfo.retryDelay: '1s' or '1.5s'
    m = re.search(r"['\"]retryDelay['\"]:\s*['\"]?(\d+(?:\.\d+)?)s?['\"]?", msg)
    if m:
        hint = float(m.group(1))
    if hint is None:
        # HTTP Retry-After header (seconds; HTTP-date form not supported)
        m = re.search(r"retry[-_\s]?after[:\s'\"]+(\d+(?:\.\d+)?)", msg_lower)
        if m:
            hint = float(m.group(1))
    if hint is None:
        # Plain-text "Please retry in 1.5s" (Gemini message field)
        m = re.search(r"retry in (\d+(?:\.\d+)?)\s*s", msg_lower)
        if m:
            hint = float(m.group(1))

    return (True, hint)


class RetryPolicy(BaseModel):
    """Configuration for :class:`ResilientLLMClient`'s retry shape.

    Defaults: 3 total attempts, exponential backoff capped at 30s.
    The resilient client surfaces timeouts via
    :class:`LLMCallTimeoutError` but does not itself enforce a
    "skip the model" policy — that's a caller concern.

    The default ``retryable_exceptions`` set includes provider-SDK
    transient classes for Google / Anthropic / OpenAI. SDK imports
    are conditional so the module stays importable when a provider
    SDK isn't installed.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    max_attempts: int = Field(
        default=3,
        ge=1,
        description="Total attempts including the first call.",
    )
    base_delay_seconds: float = Field(
        default=1.0,
        ge=0.0,
        description="Initial backoff delay; doubles on each retry up to ``backoff_cap_seconds``.",
    )
    backoff_cap_seconds: float = Field(
        default=30.0,
        ge=0.0,
        description="Maximum backoff between attempts.",
    )
    retry_on_timeout: bool = Field(
        default=True,
        description=(
            "If True, :class:`LLMCallTimeoutError` is retried per "
            "policy. Set False to make timeouts terminal — useful "
            "when the caller knows the SDK has its own internal "
            "retries and a single library-level timeout means the "
            "call is truly dead."
        ),
    )
    retryable_exceptions: tuple[type[BaseException], ...] = Field(
        default=_DEFAULT_RETRYABLE_EXCEPTIONS,
        description=(
            "Exception types that trigger a retry. The default "
            "includes :class:`TransientLLMError` plus provider-SDK "
            "transient classes (Google ``ResourceExhausted`` / "
            "``ServiceUnavailable`` / ``DeadlineExceeded`` / "
            "``InternalServerError``; Anthropic ``RateLimitError`` / "
            "``APIConnectionError`` / ``InternalServerError`` / "
            "``APITimeoutError``; OpenAI equivalents). The SDKs "
            "respect ``Retry-After`` themselves; this layer adds "
            "exponential backoff on top. Pass a narrower tuple to "
            "opt out."
        ),
    )
    rate_limit_backoff_seconds: float = Field(
        default=60.0,
        ge=0.0,
        description=(
            "Fixed backoff used when a retryable exception is "
            "identified as an HTTP 429 / rate-limit / "
            "RESOURCE_EXHAUSTED. Tuned to ride out a per-minute "
            "token-quota window. The exponential path "
            "(``base_delay_seconds`` / ``backoff_cap_seconds``) "
            "continues to govern non-rate-limit retries (503, "
            "timeouts, generic transient errors)."
        ),
    )
    rate_limit_jitter_seconds: float = Field(
        default=5.0,
        ge=0.0,
        description=(
            "Uniform random jitter added to "
            ":data:`rate_limit_backoff_seconds`. Prevents two "
            "concurrent callers retrying in lock-step against the "
            "same quota window."
        ),
    )
    total_retry_wall_clock_cap_seconds: float = Field(
        default=300.0,
        ge=0.0,
        description=(
            "Hard ceiling on cumulative retry sleep within a single "
            ":meth:`ResilientLLMClient.complete` call. When the next "
            "backoff would push total sleep past this cap, the "
            "policy gives up and re-raises the last exception rather "
            "than continuing indefinitely. Five minutes is enough "
            "for a few full per-minute quota windows; production "
            "callers that can wait longer should raise this. ``0.0`` "
            "disables the cap."
        ),
    )

    @classmethod
    def for_ollama_cold_load(cls) -> RetryPolicy:
        """Convenience constructor for the ``ollama:`` cold-load
        variant — a long first-attempt window so a freshly-loaded
        model isn't killed by the default 90s timeout. Bumps backoff
        to be more patient; the per-call timeout is set separately
        via ``timeout_seconds`` on :class:`ResilientLLMClient`."""
        return cls(
            max_attempts=3,
            base_delay_seconds=2.0,
            backoff_cap_seconds=60.0,
            retry_on_timeout=True,
        )


class ResilientLLMClient:
    """Composable :class:`LLMClient` decorator that adds a wall-
    clock timeout and exponential-backoff retry around any base
    client.

    The timeout is enforced via a :class:`ThreadPoolExecutor`: the
    SDK call runs on a worker thread; the main thread waits up to
    ``timeout_seconds`` and raises :class:`LLMCallTimeoutError` on
    expiry. The worker thread is daemonised and **may keep running
    in the background** — Python cannot preempt blocking native
    code. Long-lived processes wrapping this client should be aware.

    Retry policy is governed by :class:`RetryPolicy`:

    - Retries on :class:`TransientLLMError` (and any other types
      listed in ``retry_policy.retryable_exceptions``).
    - Retries on :class:`LLMCallTimeoutError` when
      ``retry_policy.retry_on_timeout`` is True (the default).
    - Re-raises everything else immediately.

    Cost tracking and budget enforcement are deliberately NOT
    handled here — they're caller concerns. Layer a separate
    cost-aware wrapper on top of this client to record token usage
    or enforce a budget cap.
    """

    def __init__(
        self,
        base_client: LLMClient,
        *,
        timeout_seconds: float = 90.0,
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("ResilientLLMClient: timeout_seconds must be > 0")
        self._base = base_client
        self._timeout_seconds = timeout_seconds
        self._policy = retry_policy or RetryPolicy()

    @property
    def base(self) -> LLMClient:
        """The wrapped client — useful for accessing provider-
        specific attributes (e.g. ``base.model``) without unwrapping
        manually."""
        return self._base

    @property
    def model(self) -> str:
        """Convenience accessor that mirrors the base client's
        ``model`` attribute when present, empty string otherwise."""
        return str(getattr(self._base, "model", "") or "")

    @property
    def provider(self) -> str:
        return str(getattr(self._base, "provider", "") or "")

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 2000,
        temperature: float = 0.1,
        *,
        json_object: bool = False,
    ) -> LLMResponse:
        """Run the base client's ``complete`` with timeout + retry.

        Rate-limit-aware backoff: when a retryable exception is
        identified as an HTTP 429 / quota-exhausted, the policy uses
        :data:`RetryPolicy.rate_limit_backoff_seconds` (default 60s +
        jitter) so the per-minute quota window can refresh. The
        503 / timeout / generic transient path uses exponential
        backoff.

        Cumulative retry sleep is capped at
        :data:`RetryPolicy.total_retry_wall_clock_cap_seconds` (default
        300s); when the next backoff would exceed the cap, the policy
        gives up and re-raises rather than looping forever.
        """
        import logging
        import random

        logger = logging.getLogger("cygnet.corrector.llm.resilient")

        last_exc: BaseException | None = None
        total_sleep: float = 0.0
        for attempt in range(1, self._policy.max_attempts + 1):
            try:
                return self._timed_call(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    json_object=json_object,
                )
            except LLMCallTimeoutError as exc:
                if not self._policy.retry_on_timeout:
                    raise
                last_exc = exc
                if attempt >= self._policy.max_attempts:
                    raise
                delay = self._backoff_delay(attempt)
                if self._exceeds_wall_clock_cap(total_sleep, delay):
                    logger.warning(
                        "ResilientLLMClient: cumulative retry wall-clock cap "
                        "%.1fs reached after attempt %d; surfacing failure",
                        self._policy.total_retry_wall_clock_cap_seconds,
                        attempt,
                    )
                    raise
                logger.warning(
                    "ResilientLLMClient: attempt %d/%d timed out after %.1fs; retrying in %.1fs",
                    attempt,
                    self._policy.max_attempts,
                    self._timeout_seconds,
                    delay,
                )
                time.sleep(delay)
                total_sleep += delay
            except self._policy.retryable_exceptions as exc:
                last_exc = exc
                if attempt >= self._policy.max_attempts:
                    raise
                is_rate_limit, hint = _detect_rate_limit_hint(exc)
                if is_rate_limit:
                    if hint is not None and hint > 0:
                        delay = hint
                        backoff_path = f"rate-limit (hint={hint:.2f}s)"
                    else:
                        delay = self._policy.rate_limit_backoff_seconds + random.uniform(
                            0.0, self._policy.rate_limit_jitter_seconds
                        )
                        backoff_path = f"rate-limit (fixed={delay:.1f}s)"
                else:
                    delay = self._backoff_delay(attempt)
                    backoff_path = f"exponential ({delay:.1f}s)"
                if self._exceeds_wall_clock_cap(total_sleep, delay):
                    logger.warning(
                        "ResilientLLMClient: cumulative retry wall-clock cap "
                        "%.1fs reached after attempt %d (%s, %s); surfacing failure",
                        self._policy.total_retry_wall_clock_cap_seconds,
                        attempt,
                        type(exc).__name__,
                        backoff_path,
                    )
                    raise
                logger.warning(
                    "ResilientLLMClient: attempt %d/%d raised %s (%s); retrying in %s",
                    attempt,
                    self._policy.max_attempts,
                    type(exc).__name__,
                    exc,
                    backoff_path,
                )
                time.sleep(delay)
                total_sleep += delay
        # Unreachable in practice — the loop above either returns
        # or re-raises. Defensive fall-through for static analysers.
        assert last_exc is not None
        raise last_exc

    def _exceeds_wall_clock_cap(self, total_sleep: float, next_delay: float) -> bool:
        """Return True if sleeping ``next_delay`` would push the
        cumulative retry-sleep past the policy's cap. A cap of ``0.0``
        disables the check."""
        cap = self._policy.total_retry_wall_clock_cap_seconds
        if cap <= 0.0:
            return False
        return bool(total_sleep + next_delay > cap)

    def _timed_call(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        temperature: float,
        json_object: bool,
    ) -> LLMResponse:
        """Run one ``complete`` call with a wall-clock timeout.

        Uses :class:`ThreadPoolExecutor` so the SDK's blocking call
        runs on a worker thread. The main thread waits up to
        ``self._timeout_seconds`` and raises
        :class:`LLMCallTimeoutError` if the worker hasn't returned.
        """
        from concurrent.futures import (
            ThreadPoolExecutor,
        )
        from concurrent.futures import (
            TimeoutError as FutTimeoutError,
        )

        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="cygnet-llm-call") as executor:
            future = executor.submit(
                self._base.complete,
                system_prompt,
                user_prompt,
                max_tokens,
                temperature,
                json_object=json_object,
            )
            try:
                return future.result(timeout=self._timeout_seconds)
            except FutTimeoutError as exc:
                # Future is daemonised by the executor's context-
                # exit; the SDK call keeps running in the background
                # but we no longer wait. Cancel returns False for an
                # already-running future on standard threads — best-
                # effort.
                future.cancel()
                raise LLMCallTimeoutError(
                    f"LLM call exceeded {self._timeout_seconds:.0f}s ({type(self._base).__name__})"
                ) from exc

    def _backoff_delay(self, attempt: int) -> float:
        """Exponential backoff: ``base_delay * 2**(attempt-1)``, capped."""
        raw: float = self._policy.base_delay_seconds * (2 ** (attempt - 1))
        return float(min(raw, self._policy.backoff_cap_seconds))
