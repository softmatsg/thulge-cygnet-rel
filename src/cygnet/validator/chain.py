# Copyright (c) 2026 Nikodem Tomczak, Thulge Labs. All rights reserved.

"""Validator chain orchestration.

Runs a sequence of validator backends in one of two collection modes
(v0.0.25):

- ``short_circuit`` (default): each validator runs in order; the
  first failure short-circuits the chain. Behaviour matches v0.0.24
  byte-for-byte.

- ``collect_all``: every backend that can meaningfully run on the
  input contributes a verdict. Backends downstream of a parse failure
  that cannot operate on parse-broken input (``explain``,
  ``mirror_execute``) are skipped — they need a parseable query to
  produce useful output. The collected payloads are ordered by the
  public contract: parse-category first, then by backend authority
  descending (``explain`` > ``mirror_execute`` > ``ast`` > ``builtin``).
  ``error_payload`` always equals ``all_errors[0]`` so v0.0.24 callers
  consuming the single-error surface continue to work.

The contract — each chain element must expose ``validate(query: str) ->
StructuralValidatorResult`` — is intentionally a structural duck-type
rather than an abstract base class so backends from different modules
can compose without inheriting a shared parent. The validators that
ship today (``BuiltinValidator``, ``ASTValidator``, ``ExplainValidator``,
``MirrorExecuteValidator``) implement this same shape.
"""

from __future__ import annotations

from typing import Literal, Protocol

from cygnet.models import StructuralErrorPayload, StructuralValidatorResult

__all__ = ["CollectionMode", "StructuralValidator", "ValidatorChain"]


CollectionMode = Literal["short_circuit", "collect_all"]
"""Identifier for the chain's collection mode. Mirrors
``ValidatorConfig.collection_mode``."""


# Backend-authority ordering (public API contract). Lower rank == higher
# authority; ties broken by the order the backends appear in the chain.
# Documented in ``docs/architecture.md``; reorderings are a breaking
# change.
_BACKEND_AUTHORITY: dict[str, int] = {
    "ExplainValidator": 0,
    "MirrorExecuteValidator": 1,
    "ASTValidator": 2,
    "BuiltinValidator": 3,
}

# Backends that cannot meaningfully run on parse-broken input.
# ``explain`` round-trips to Neo4j which raises ``CypherSyntaxError``
# on a malformed query (the validator catches that and emits its own
# parse payload — but the payload duplicates whatever AST already
# emitted, so it adds no signal). ``mirror_execute`` cannot begin a
# transaction on an unparseable query at all. We skip both when a
# prior backend in the chain run already emitted a parse-category
# error.
_PARSE_INTOLERANT_BACKENDS: frozenset[str] = frozenset(
    {"ExplainValidator", "MirrorExecuteValidator"}
)


class StructuralValidator(Protocol):
    """Protocol every chain element must satisfy."""

    def validate(self, query: str) -> StructuralValidatorResult: ...


class ValidatorChain:
    """Runs validators in order under the configured collection mode.

    Empty chains are not allowed: a chain with zero validators would
    silently pass every query, which is almost certainly a config bug.
    """

    def __init__(
        self,
        validators: list[StructuralValidator],
        *,
        collection_mode: CollectionMode = "short_circuit",
    ) -> None:
        if not validators:
            raise ValueError("ValidatorChain requires at least one validator")
        self._validators = list(validators)
        self._collection_mode = collection_mode

    @property
    def validators(self) -> list[StructuralValidator]:
        """The configured validators, in chain order."""
        return list(self._validators)

    @property
    def collection_mode(self) -> CollectionMode:
        """The chain's configured collection mode."""
        return self._collection_mode

    def __len__(self) -> int:
        return len(self._validators)

    def validate(self, query: str) -> StructuralValidatorResult:
        """Run the chain under its configured collection mode."""
        if self._collection_mode == "short_circuit":
            return self._validate_short_circuit(query)
        return self._validate_collect_all(query)

    # ------------------------------------------------------------------
    # Mode implementations
    # ------------------------------------------------------------------

    def _validate_short_circuit(self, query: str) -> StructuralValidatorResult:
        """Original v0.0.24 behaviour: stop on the first failure."""
        for validator in self._validators:
            result = validator.validate(query)
            if not result.passed:
                # The backend's own result may have been built before
                # this field landed (unit-test stubs in particular); set
                # the chain-level fields here so the returned object
                # always carries the public contract.
                payload = result.error_payload
                if payload is None:
                    # Defensive — passed=False without a payload is
                    # already rejected by the result model validator,
                    # but if we ever get here we'd rather raise on a
                    # clear branch than land an invalid result.
                    raise ValueError(
                        "Validator reported passed=False but did not "
                        "populate error_payload; cannot assemble chain "
                        "result without a payload."
                    )
                return StructuralValidatorResult(
                    passed=False,
                    failed_stage=result.failed_stage,
                    error_payload=payload,
                    all_errors=[payload],
                    collection_mode_used="short_circuit",
                )
        return StructuralValidatorResult(
            passed=True,
            failed_stage="none",
            error_payload=None,
            all_errors=[],
            collection_mode_used="short_circuit",
        )

    def _validate_collect_all(self, query: str) -> StructuralValidatorResult:
        """Collect-all-where-possible mode (v0.0.25).

        Runs each validator in chain order. If any prior validator in
        this chain run emitted a parse-category error, skips
        ``_PARSE_INTOLERANT_BACKENDS`` for the remaining iterations
        (their output on a parse-broken query duplicates the parse
        error already in hand and would otherwise raise the
        round-trip cost without adding signal). Sorts the collected
        payloads per the public contract.
        """
        # Track (rank_key, payload) tuples so we can sort deterministically.
        # rank_key is a 3-tuple: (parse-first, backend-authority,
        # chain-position-tiebreaker). The chain position is the
        # backend's index in self._validators, used only to break ties
        # when two backends share the same authority rank (which can
        # happen for user-supplied custom backends that fall to the
        # unknown bucket).
        collected: list[tuple[tuple[int, int, int], StructuralErrorPayload]] = []
        parse_emitted = False
        for position, validator in enumerate(self._validators):
            type_name = type(validator).__name__
            if parse_emitted and type_name in _PARSE_INTOLERANT_BACKENDS:
                continue
            result = validator.validate(query)
            if result.passed:
                continue
            payload = result.error_payload
            if payload is None:
                raise ValueError(
                    "Validator reported passed=False but did not "
                    "populate error_payload; cannot assemble chain "
                    "result without a payload."
                )
            authority = _BACKEND_AUTHORITY.get(type_name, len(_BACKEND_AUTHORITY))
            parse_marker = 0 if payload.category == "parse" else 1
            collected.append(((parse_marker, authority, position), payload))
            if payload.category == "parse":
                parse_emitted = True

        if not collected:
            return StructuralValidatorResult(
                passed=True,
                failed_stage="none",
                error_payload=None,
                all_errors=[],
                collection_mode_used="collect_all",
            )

        collected.sort(key=lambda item: item[0])
        ordered = [payload for _, payload in collected]
        primary = ordered[0]
        return StructuralValidatorResult(
            passed=False,
            failed_stage=primary.category,
            error_payload=primary,
            all_errors=ordered,
            collection_mode_used="collect_all",
        )
