# Copyright (c) 2026 Nikodem Tomczak, Thulge Labs. All rights reserved.

"""Corrector response parser.

The LLM is asked to return ``{"cypher": "...", "explanation"?: "..."}``,
and the parser is just ``json.loads`` + a field check. A failed
validation under the JSON contract is always a Cypher-quality
failure, never an extraction failure.

Four outcomes via a discriminated union:

- :class:`ProtocolOK` — JSON parsed, ``cypher`` non-empty and not
  echoing the input. The cypher is the cleanest extraction.
- :class:`ProtocolEchoed` — JSON parsed, ``cypher`` non-empty but
  matches the input (after whitespace + keyword-case
  normalisation). The model declined to change anything. The
  cypher is preserved so callers can record what the model
  refused to refine.
- :class:`ProtocolEmpty` — JSON parsed, ``cypher`` field present
  but empty / whitespace-only. The model's explicit "I cannot
  refine" signal. The corrector's inner-retry loop bumps the
  temperature once before accepting the abort.
- :class:`ProtocolMalformed` — JSON parse failure, missing
  ``cypher`` field, or wrong type. Retry-able with an escalated
  prompt.

Kept as a thin compatibility shim over the current parser:
``parse_corrector_response`` and the
``ProtocolComplianceOK / Recoverable / Unrecoverable`` outcomes.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Final, Literal

__all__ = [  # noqa: RUF022 — grouped by API generation, not alphabetised
    # Current API
    "ProtocolEchoed",
    "ProtocolEmpty",
    "ProtocolMalformed",
    "ProtocolOK",
    "ProtocolOutcome",
    "parse_corrector_response_v2",
    # Compatibility shim
    "ProtocolCompliance",
    "ProtocolComplianceOK",
    "ProtocolComplianceRecoverable",
    "ProtocolComplianceUnrecoverable",
    "parse_corrector_response",
]


# ---------------------------------------------------------------------------
# Outcome types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProtocolOK:
    """JSON parsed, ``cypher`` non-empty and distinct from input.
    Caller hands the extracted cypher to the validator chain."""

    outcome: Literal["ok"] = "ok"
    cypher: str = ""
    explanation: str | None = None


@dataclass(frozen=True)
class ProtocolEchoed:
    """JSON parsed, ``cypher`` non-empty but matches the input
    (after whitespace + keyword-case normalisation). The model
    declined to refine. Carries the echoed cypher so callers can
    record what the model refused to change."""

    outcome: Literal["echoed"] = "echoed"
    cypher: str = ""
    explanation: str | None = None


@dataclass(frozen=True)
class ProtocolEmpty:
    """JSON parsed, ``cypher`` field present but empty or whitespace-
    only. Explicit "I cannot refine" signal. Corrector's inner-retry
    loop bumps temperature once before accepting the abort."""

    outcome: Literal["empty"] = "empty"
    explanation: str | None = None


@dataclass(frozen=True)
class ProtocolMalformed:
    """JSON parse failure, missing ``cypher`` field, or wrong type.
    Retry-able with an escalated prompt."""

    outcome: Literal["malformed"] = "malformed"
    reason: str = ""


ProtocolOutcome = ProtocolOK | ProtocolEchoed | ProtocolEmpty | ProtocolMalformed
"""Discriminated union of the four parser outcomes. Branch on the
``outcome`` literal."""


# ---------------------------------------------------------------------------
# Prose-extraction helpers
# ---------------------------------------------------------------------------


# Some models still wrap the JSON in a fenced code block or
# surrounding prose despite the contract. Strip a code fence if
# present; otherwise look for the first { ... } object literal in
# the text.
_FENCE_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"```(?:json)?\s*\n(.*?)\n?```",
    re.DOTALL | re.IGNORECASE,
)

_FIRST_JSON_OBJECT: Final[re.Pattern[str]] = re.compile(
    r"\{.*\}",
    re.DOTALL,
)


def _strip_to_json(raw_text: str) -> str:
    """Return the most likely JSON-shaped substring of ``raw_text``.

    Order of attempts:
    1. ``raw_text`` itself, if it starts with ``{``.
    2. Contents of the first ``\\`\\`\\`json`` (or untagged) fence.
    3. The first ``{ ... }`` substring (greedy match) in the text.
    """
    stripped = raw_text.strip()
    if stripped.startswith("{"):
        return stripped
    fence = _FENCE_PATTERN.search(raw_text)
    if fence is not None:
        return fence.group(1).strip()
    obj = _FIRST_JSON_OBJECT.search(raw_text)
    if obj is not None:
        return obj.group(0)
    return stripped


# ---------------------------------------------------------------------------
# Echo detection
# ---------------------------------------------------------------------------


_RUN_OF_WHITESPACE: Final[re.Pattern[str]] = re.compile(r"\s+")


def _normalise_cypher_for_echo(text: str) -> str:
    """Normalise a Cypher string for echo comparison.

    - Strip leading and trailing whitespace from each line.
    - Drop blank lines.
    - Collapse runs of whitespace to a single space.
    - Lower-case the whole string.

    Cypher keywords are conventionally uppercase but the language
    is case-insensitive on them; identifiers are case-sensitive
    but the model returning ``MATCH (n:Sample) RETURN n`` vs
    ``MATCH(n:Sample)RETURN n`` is still an echo. Whitespace +
    case is the right bar.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    flattened = " ".join(lines)
    collapsed = _RUN_OF_WHITESPACE.sub(" ", flattened)
    return collapsed.strip().lower()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_corrector_response_v2(raw_text: str, input_cypher: str) -> ProtocolOutcome:
    """Apply the JSON parser to ``raw_text``.

    Args:
        raw_text: the LLM's response, ideally a JSON object but
            tolerant of code-fence wrapping and surrounding prose.
        input_cypher: the original broken query the LLM was asked
            to refine. Used for echo detection — if the model
            returns identical-after-normalisation cypher, the
            outcome is ``ProtocolEchoed`` rather than ``ProtocolOK``.

    Returns one of the four outcome dataclasses. Pure: no side
    effects, no IO. Caller decides what to do with the outcome
    (the docstrings on each outcome class spell out the contract).
    """
    if not raw_text or not raw_text.strip():
        return ProtocolMalformed(reason="empty response")

    candidate = _strip_to_json(raw_text)
    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError as exc:
        return ProtocolMalformed(reason=f"invalid JSON: {exc.msg} at pos {exc.pos}")

    if not isinstance(obj, dict):
        return ProtocolMalformed(reason=f"JSON root is not an object (got {type(obj).__name__})")

    if "cypher" not in obj:
        return ProtocolMalformed(reason="JSON missing required 'cypher' field")

    cypher_raw = obj["cypher"]
    if not isinstance(cypher_raw, str):
        return ProtocolMalformed(
            reason=f"'cypher' field is not a string (got {type(cypher_raw).__name__})"
        )

    explanation = obj.get("explanation")
    if explanation is not None and not isinstance(explanation, str):
        # Don't fail on a wrong-type explanation; it's optional.
        # Just drop it.
        explanation = None

    cypher = cypher_raw.strip()
    if not cypher:
        return ProtocolEmpty(explanation=explanation)

    # Echo detection. Identifiers are case-sensitive in Cypher, but
    # the keyword-case + whitespace normalisation correctly catches
    # the common patterns where the model "responds" by reformatting
    # the input. Differently-cased *identifiers* (``Sample`` vs
    # ``sample``) would NOT match — that's a real edit, not an echo.
    # We accept the false-negative on keyword-only differences since
    # the cost (one wasted refinement attempt) is small.
    if _normalise_cypher_for_echo(cypher) == _normalise_cypher_for_echo(input_cypher):
        return ProtocolEchoed(cypher=cypher, explanation=explanation)

    return ProtocolOK(cypher=cypher, explanation=explanation)


# ---------------------------------------------------------------------------
# Compatibility shim
# ---------------------------------------------------------------------------
#
# Kept as a thin mapping over the current outcomes so callers that
# haven't migrated still work.


@dataclass(frozen=True)
class ProtocolComplianceOK:
    """Compatibility outcome. Use :class:`ProtocolOK` for new code."""

    outcome: Literal["ok"] = "ok"
    extracted_cypher: str = ""


@dataclass(frozen=True)
class ProtocolComplianceRecoverable:
    """Compatibility outcome. Use :class:`ProtocolOK` (with note in
    caller) or :class:`ProtocolEchoed` / :class:`ProtocolEmpty` for
    new code."""

    outcome: Literal["recoverable"] = "recoverable"
    reason: str = ""
    extracted_cypher: str = ""


@dataclass(frozen=True)
class ProtocolComplianceUnrecoverable:
    """Compatibility outcome. Use :class:`ProtocolMalformed` for new
    code."""

    outcome: Literal["unrecoverable"] = "unrecoverable"
    reason: str = ""


ProtocolCompliance = (
    ProtocolComplianceOK | ProtocolComplianceRecoverable | ProtocolComplianceUnrecoverable
)
"""Compatibility discriminated union."""


# Regexes for fenced-code-block extraction, used by the
# compatibility shim.
_CYPHER_FENCE: Final[re.Pattern[str]] = re.compile(
    r"```cypher\s*\n(.*?)\n?```",
    re.DOTALL | re.IGNORECASE,
)
_ANY_FENCE: Final[re.Pattern[str]] = re.compile(
    r"```([a-zA-Z0-9_+-]*)\s*\n(.*?)\n?```",
    re.DOTALL,
)
_CYPHER_CLAUSE_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\b(MATCH|CREATE|MERGE|CALL|RETURN|WITH|UNWIND|OPTIONAL\s+MATCH|DELETE|DETACH\s+DELETE|REMOVE|SET|SHOW)\b",
    re.IGNORECASE,
)
_REJECT_PREFIXES: Final[tuple[str, ...]] = (
    "{",
    "[",
    "<",
    "SELECT",
    "INSERT",
    "UPDATE",
    "DELETE FROM",
    "DROP",
)
_BALANCE_PAIRS: Final[tuple[tuple[str, str], ...]] = (
    ("(", ")"),
    ("[", "]"),
    ("{", "}"),
)
_RAW_CYPHER_LINE: Final[re.Pattern[str]] = re.compile(
    r"^[ \t]*(?:MATCH|CREATE|MERGE|CALL|RETURN|WITH|UNWIND|OPTIONAL\s+MATCH).*",
    re.IGNORECASE | re.MULTILINE,
)


def parse_corrector_response(raw_text: str) -> ProtocolCompliance:
    """Compatibility parser. Use :func:`parse_corrector_response_v2`
    for new code.

    Three stages (non-empty / code-block extraction / plausible
    Cypher) with a discriminated-union outcome of three variants.
    """
    if not raw_text or not raw_text.strip():
        return ProtocolComplianceUnrecoverable(reason="empty response")

    cypher_matches = _CYPHER_FENCE.findall(raw_text)
    any_matches = _ANY_FENCE.findall(raw_text)

    extracted: str | None = None
    fallback_reason: str | None = None

    if cypher_matches:
        extracted = cypher_matches[0].strip()
        if len(cypher_matches) > 1:
            fallback_reason = "multiple cypher-tagged blocks"
    elif any_matches:
        cypher_shaped_blocks = [
            (tag, body) for tag, body in any_matches if _CYPHER_CLAUSE_PATTERN.search(body or "")
        ]
        if cypher_shaped_blocks:
            tag, body = cypher_shaped_blocks[0]
            extracted = body.strip()
            if len(any_matches) > 1:
                fallback_reason = (
                    f"multiple code blocks; chose first Cypher-shaped block (language tag: {tag!r})"
                )
            else:
                fallback_reason = f"non-cypher language tag: {tag!r}"
        else:
            return ProtocolComplianceUnrecoverable(
                reason="fenced code blocks present but none contain Cypher-shaped clauses"
            )
    else:
        m = _RAW_CYPHER_LINE.search(raw_text)
        if m is None:
            return ProtocolComplianceUnrecoverable(
                reason="no fenced code block and no Cypher-shaped text found"
            )
        extracted = raw_text[m.start() :].strip()
        fallback_reason = "no fenced code block; extracted raw Cypher-shaped text"

    if extracted is None or not extracted.strip():
        return ProtocolComplianceUnrecoverable(reason="extracted Cypher was empty")

    rejection = _check_plausible_cypher(extracted)
    if rejection is not None:
        return ProtocolComplianceUnrecoverable(reason=rejection)

    if fallback_reason is None:
        return ProtocolComplianceOK(extracted_cypher=extracted)
    return ProtocolComplianceRecoverable(reason=fallback_reason, extracted_cypher=extracted)


def _check_plausible_cypher(text: str) -> str | None:
    stripped = text.strip()
    if not stripped:
        return "extracted Cypher was empty"
    upper_prefix = stripped[:32].upper()
    for prefix in _REJECT_PREFIXES:
        if upper_prefix.startswith(prefix):
            return f"extracted text starts with non-Cypher token {prefix!r}"
    if not _CYPHER_CLAUSE_PATTERN.search(stripped):
        return "extracted text contains no Cypher clause keyword"
    if not _outer_brackets_balance(stripped):
        return "extracted text has unbalanced outer brackets"
    return None


def _outer_brackets_balance(text: str) -> bool:
    stack: list[str] = []
    open_to_close = {opener: closer for opener, closer in _BALANCE_PAIRS}
    closers = {closer for _, closer in _BALANCE_PAIRS}
    in_string: str | None = None
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if in_string is not None:
            if ch == "\\" and in_string in {"'", '"'} and i + 1 < n:
                i += 2
                continue
            if ch == in_string:
                in_string = None
            i += 1
            continue
        if ch in {"'", '"', "`"}:
            in_string = ch
            i += 1
            continue
        if ch in open_to_close:
            stack.append(open_to_close[ch])
        elif ch in closers:
            if not stack or stack[-1] != ch:
                return False
            stack.pop()
        i += 1
    return not stack and in_string is None
