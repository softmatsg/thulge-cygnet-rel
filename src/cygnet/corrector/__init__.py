# Copyright (c) 2026 Nikodem Tomczak, Thulge Labs. All rights reserved.

"""Corrector: pluggable callable that refines a query against a `GateError`.

Public surface:

- :class:`Corrector` — duck-typed protocol; ``runtime_checkable``.
- :class:`CorrectorContext` — Pydantic context model passed to every
  ``correct(...)`` call.
- :class:`PriorAttempt` — entry in ``CorrectorContext.prior_attempts``.
- :class:`NullCorrector` — safe default, always returns
  ``action="abort"``.
- :class:`RampartCorrector` — production LLM-backed corrector that
  uses RAMPART to assemble its prompt. Requires the ``corrector``
  extra: ``pip install thulge-cygnet[corrector]``.
- :class:`LLMClient`, :class:`AnthropicClient`, :class:`OpenAIClient`,
  :class:`GoogleGeminiClient`, :class:`OllamaClient`,
  :func:`make_llm_client` — LLM abstraction used by
  :class:`RampartCorrector`. Cloud backends require the
  corresponding extra (``corrector`` covers Anthropic, OpenAI,
  Gemini, and Ollama); the Ollama path is for **local** model
  serving and needs a running ``ollama`` daemon plus a pulled
  model.

The RAMPART-backed surface (``RampartCorrector``, the LLM clients) is
imported lazily so this module stays importable even when the
``corrector`` extra is missing — the NullCorrector path continues to
work without any external dependency.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from cygnet.corrector.interface import Corrector, CorrectorContext, PriorAttempt
from cygnet.corrector.null import NullCorrector

if TYPE_CHECKING:  # pragma: no cover - typing only
    from cygnet.corrector.decorators import (
        EmptyRetryingCorrector,
        ProtocolRetryingCorrector,
    )
    from cygnet.corrector.llm import (
        AnthropicClient,
        GoogleGeminiClient,
        LLMClient,
        OllamaClient,
        OpenAIClient,
        make_llm_client,
    )
    from cygnet.corrector.rampart_backed import (
        DEFAULT_PROMPT_BY_MODEL,
        V4_SYSTEM_PROMPT,
        RampartCorrector,
    )
    from cygnet.corrector.refinement_loop import (
        AcceptanceCriteria,
        RefinementAttempt,
        RefinementLoop,
        RefinementResult,
    )
    from cygnet.corrector.retry_loop import RetryLoop, RetryLoopOutcome
    from cygnet.corrector.runner import (
        LoopOptions,
        apply_default_wrapping,
        run_correction,
    )
    from cygnet.corrector.template import (
        NaiveFullCorrector,
        NaiveTruncatedCorrector,
        RawCorrector,
        TemplateCorrector,
        VerbalCorrector,
    )


__all__ = [
    "DEFAULT_PROMPT_BY_MODEL",
    "V4_SYSTEM_PROMPT",
    "AcceptanceCriteria",
    "AnthropicClient",
    "Corrector",
    "CorrectorContext",
    "EmptyRetryingCorrector",
    "GoogleGeminiClient",
    "LLMClient",
    "LoopOptions",
    "NaiveFullCorrector",
    "NaiveTruncatedCorrector",
    "NullCorrector",
    "OllamaClient",
    "OpenAIClient",
    "PriorAttempt",
    "ProtocolRetryingCorrector",
    "RampartCorrector",
    "RawCorrector",
    "RefinementAttempt",
    "RefinementLoop",
    "RefinementResult",
    "RetryLoop",
    "RetryLoopOutcome",
    "TemplateCorrector",
    "VerbalCorrector",
    "apply_default_wrapping",
    "make_llm_client",
    "run_correction",
]


_LLM_LAZY_NAMES = frozenset(
    {
        "AnthropicClient",
        "GoogleGeminiClient",
        "LLMClient",
        "OllamaClient",
        "OpenAIClient",
        "make_llm_client",
    }
)
_RAMPART_LAZY_NAMES = frozenset(
    {
        "DEFAULT_PROMPT_BY_MODEL",
        "RampartCorrector",
        "V4_SYSTEM_PROMPT",
    }
)
_TEMPLATE_LAZY_NAMES = frozenset(
    {
        "NaiveFullCorrector",
        "NaiveTruncatedCorrector",
        "RawCorrector",
        "TemplateCorrector",
        "VerbalCorrector",
    }
)
_DECORATOR_LAZY_NAMES = frozenset(
    {
        "EmptyRetryingCorrector",
        "ProtocolRetryingCorrector",
    }
)
_LOOP_LAZY_NAMES = frozenset(
    {
        "AcceptanceCriteria",
        "RefinementAttempt",
        "RefinementLoop",
        "RefinementResult",
    }
)
_RETRY_LOOP_LAZY_NAMES = frozenset(
    {
        "RetryLoop",
        "RetryLoopOutcome",
    }
)
_RUNNER_LAZY_NAMES = frozenset(
    {
        "LoopOptions",
        "apply_default_wrapping",
        "run_correction",
    }
)


def __getattr__(name: str) -> object:  # pragma: no cover - thin lazy proxy
    """Lazy attribute access for the corrector-extra members.

    Importing :mod:`cygnet.corrector` should not fail when the
    ``corrector`` extra is not installed. The protocol, context,
    null corrector, and prior-attempt models are always available;
    everything else (the LLM clients, the RAMPART-backed corrector
    and its prompt-by-model surface, the template correctors, and
    the retry decorators) is loaded on first access and raises a
    clear ``ImportError`` when the extra is missing.
    """
    if name in _LLM_LAZY_NAMES:
        from cygnet.corrector import llm

        return getattr(llm, name)
    if name in _RAMPART_LAZY_NAMES:
        from cygnet.corrector import rampart_backed

        return getattr(rampart_backed, name)
    if name in _TEMPLATE_LAZY_NAMES:
        from cygnet.corrector import template

        return getattr(template, name)
    if name in _DECORATOR_LAZY_NAMES:
        from cygnet.corrector import decorators

        return getattr(decorators, name)
    if name in _LOOP_LAZY_NAMES:
        from cygnet.corrector import refinement_loop

        return getattr(refinement_loop, name)
    if name in _RETRY_LOOP_LAZY_NAMES:
        from cygnet.corrector import retry_loop

        return getattr(retry_loop, name)
    if name in _RUNNER_LAZY_NAMES:
        from cygnet.corrector import runner

        return getattr(runner, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
