# Copyright (c) 2026 Nikodem Tomczak, Thulge Labs. All rights reserved.

"""Configuration models for CYGNET.

`GateConfig` is the top-level entry point. It composes per-area configs
(Neo4j connection, schema source, validator chain, cost gate, corrector,
mirror, transports) and supports loading from YAML files and from
environment variables under the `CYGNET_` prefix.

Field names declared here are public API; see the immutable artifacts
section of the project brief.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = [
    "CorrectorConfig",
    "CostConfig",
    "GateConfig",
    "MirrorConfig",
    "Neo4jConfig",
    "RefinementConfig",
    "SchemaConfig",
    "TransportConfig",
    "ValidatorBackend",
    "ValidatorConfig",
]


ValidatorBackend = Literal["ast", "explain", "builtin", "mirror_execute"]
"""Validator backend identifiers. The first three are part of the
project's immutable artifact #3; ``mirror_execute`` was added in
v0.0.23 to close the runtime-error coverage gap EXPLAIN leaves open
(procedure-not-found, parameter-missing, type-coercion failures,
post-``WITH`` scoping bugs). The addition is additive — older
configurations explicitly pinning ``backends=["builtin", "ast",
"explain"]`` still validate and behave exactly as before."""


def _default_validator_backends() -> list[ValidatorBackend]:
    """Default chain (v0.0.23): builtin → ast → explain → mirror_execute.

    Builtin is pure-Python with no I/O cost and catches obvious errors
    (unbalanced brackets, unknown labels, typo'd property names) in
    sub-millisecond time. AST runs after to catch grammar issues the
    regex layer missed. EXPLAIN is the authoritative Neo4j-backed
    plan-time check. ``mirror_execute`` is the runtime-time backend —
    it runs the query inside a rolled-back transaction so runtime-only
    errors (procedure-not-found, parameter-missing, type-coercion)
    surface without committing anything to the mirror. Each tier
    short-circuits on failure, so the expensive backends only run on
    queries that survive the cheaper ones.

    The ``mirror_execute`` tier was added in v0.0.23. Users who
    explicitly pinned ``backends=["builtin", "ast", "explain"]``
    continue to get exactly that chain (the default expansion is
    additive but not retroactive — only the default factory changed).
    """
    return ["builtin", "ast", "explain", "mirror_execute"]


class Neo4jConfig(BaseModel):
    """Connection details for the production Neo4j instance."""

    model_config = ConfigDict(extra="forbid")

    uri: str = Field(
        ...,
        description="Bolt URI for the Neo4j instance (e.g. 'bolt://host:7687', 'neo4j+s://...').",
    )
    user: str = Field(..., description="Username for Neo4j authentication.")
    password: str = Field(
        ...,
        description=(
            "Password for Neo4j authentication. Never log or echo this value; it is "
            "expected to be resolved from environment or a secret store at construction time."
        ),
    )
    database: str = Field(
        default="neo4j",
        description="Database name within the Neo4j instance (default Neo4j 5 database is 'neo4j').",
    )

    @field_validator("uri")
    @classmethod
    def _uri_has_scheme(cls, v: str) -> str:
        allowed = ("bolt://", "bolt+s://", "bolt+ssc://", "neo4j://", "neo4j+s://", "neo4j+ssc://")
        if not v.startswith(allowed):
            raise ValueError(f"neo4j.uri must start with one of {allowed!r}; got {v!r}")
        return v


class SchemaConfig(BaseModel):
    """Where the schema comes from and how often it is refreshed."""

    model_config = ConfigDict(extra="forbid")

    source: Literal["introspect", "spec_file", "spec_object"] = Field(
        ...,
        description=(
            "Schema ingestion mode: 'introspect' calls Neo4j's `db.schema.*` procedures; "
            "'spec_file' loads YAML/JSON from `spec_path`; 'spec_object' uses an inline dict."
        ),
    )
    spec_path: Path | None = Field(
        default=None,
        description="Path to schema spec file; required when source='spec_file'.",
    )
    spec_object: dict[str, Any] | None = Field(
        default=None,
        description="Inline schema spec; required when source='spec_object'.",
    )
    refresh_interval_seconds: int = Field(
        default=0,
        ge=0,
        description="Re-introspect interval in seconds; 0 disables auto-refresh.",
    )

    @field_validator("spec_path", mode="after")
    @classmethod
    def _spec_path_is_set_when_needed(cls, v: Path | None) -> Path | None:
        return v


class ValidatorConfig(BaseModel):
    """Validator backend chain and per-stage knobs."""

    model_config = ConfigDict(extra="forbid")

    backends: list[ValidatorBackend] = Field(
        default_factory=_default_validator_backends,
        description=(
            "Ordered chain of validator backends. Each runs in turn; the first "
            "to fail short-circuits the chain. If every backend passes, the "
            "chain result is a pass. Default (v0.0.23) is ['builtin', 'ast', "
            "'explain', 'mirror_execute']: builtin is the pure-Python fast "
            "filter, ast adds full Cypher grammar checking via "
            "libcypher-parser, explain is the authoritative Neo4j plan-time "
            "check, and mirror_execute runs the query inside a rolled-back "
            "transaction against the mirror to catch runtime-only errors "
            "(procedure-not-found, parameter-missing, type-coercion). Users "
            "without a mirror Neo4j should drop both 'explain' and "
            "'mirror_execute'; users without libcypher-parser binaries should "
            "drop 'ast'. Users who pinned the v0.0.3 three-backend chain "
            "should continue passing it explicitly — the default expansion is "
            "not retroactive."
        ),
    )
    collection_mode: Literal["short_circuit", "collect_all"] = Field(
        default="short_circuit",
        description=(
            "How the chain assembles results across backends (v0.0.25). "
            "``short_circuit`` (default): run validators in order, return "
            "the first non-pass result; matches v0.0.24 behaviour byte-for-"
            "byte. ``collect_all``: run every backend that can run on the "
            "input (backends downstream of a parse failure that need a "
            "parseable query are skipped) and collect each backend's "
            "payload into ``StructuralValidatorResult.all_errors``, ordered "
            "by the public contract (parse-category first; then explain > "
            "mirror_execute > ast > builtin). collect_all surfaces more "
            "errors per call at the cost of running every backend; "
            "short_circuit fails fast. The design rationale lives in "
            "``docs/chain_collection_mode.md``."
        ),
    )
    strict_property_existence: bool = Field(
        default=True,
        description="If True, queries referencing unknown properties fail validation.",
    )
    mirror_uri: str | None = Field(
        default=None,
        description="Bolt URI for the mirror Neo4j instance; required if 'explain' is in backends.",
    )
    register_failed_queries: bool = Field(
        default=False,
        description="If True, failed queries are written to the broken-query register.",
    )
    register_path: Path | None = Field(
        default=None,
        description="JSONL file for the broken-query register; in-memory if None.",
    )
    register_replay_on_schema_change: bool = Field(
        default=False,
        description="If True, schema refresh replays the register against the new schema.",
    )

    @field_validator("backends")
    @classmethod
    def _backends_non_empty(cls, v: list[ValidatorBackend]) -> list[ValidatorBackend]:
        if not v:
            raise ValueError("ValidatorConfig.backends must contain at least one backend")
        return v


class CostConfig(BaseModel):
    """Cost gate thresholds and toggles."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(
        default=True,
        description="Master switch for the cost gate. If False, only structural validation runs.",
    )
    threshold_rows: int = Field(
        default=100_000,
        ge=0,
        description="Reject queries with EXPLAIN-estimated rows above this value.",
    )
    threshold_dbhits: int = Field(
        default=1_000_000,
        ge=0,
        description="Reject queries with EXPLAIN-estimated db-hits above this value.",
    )


class CorrectorConfig(BaseModel):
    """LLM corrector parameters; consumed by the default RAMPART-backed corrector."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    llm: str = Field(
        default="claude-sonnet-4-5",
        description=(
            "Model identifier passed to the corrector's LLM client. Default is "
            "``'claude-sonnet-4-5'`` (Anthropic). For ``backend='openai'`` set "
            "to a valid OpenAI model name (e.g. ``'gpt-4o'``)."
        ),
    )
    backend: Literal["anthropic", "openai"] = Field(
        default="anthropic",
        description=(
            "LLM backend identifier. ``'anthropic'`` uses ``AnthropicClient`` "
            "(reads ``ANTHROPIC_API_KEY``); ``'openai'`` uses ``OpenAIClient`` "
            "(reads ``OPENAI_API_KEY``). Both clients are constructed by "
            "``cygnet.corrector.llm.make_llm_client``."
        ),
    )
    token_budget: int = Field(
        default=4000,
        ge=256,
        description=(
            "RAMPART compile budget in tokens. The shipped corrector blocks "
            "plus typical per-call agent content fit in ~4000 tokens; raise "
            "for richer schemas or longer prior-attempt histories. The LLM's "
            "own context window must be at least this large plus the system "
            "prompt and response budget."
        ),
    )
    max_attempts: int = Field(
        default=3,
        ge=1,
        description=(
            "Recommended cap on refinement attempts. The library does not loop; this value is "
            "exposed to callers that implement their own retry loop."
        ),
    )
    temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="LLM sampling temperature for the corrector.",
    )
    corrector: Any = Field(
        default=None,
        exclude=True,
        description=(
            "Zero-arg callable returning a Corrector instance (or a Corrector "
            "instance directly). Pluggable override used by Gate.from_config; "
            "``None`` falls back to RampartCorrector when the ``corrector`` "
            "extra is installed and an API key is available, else "
            "NullCorrector. Excluded from serialisation because callables "
            "don't round-trip through YAML/JSON — set programmatically, never "
            "via config files.\n\n"
            "**Resolver gotcha** (documented for future contributors): "
            "``Gate._resolve_corrector`` distinguishes 'factory' from "
            "'instance' via ``inspect.isclass`` plus a callable-without-"
            "``correct``-attribute check. A naive "
            "``isinstance(candidate, Corrector)`` does not work because the "
            "``Corrector`` Protocol is ``@runtime_checkable`` and matches "
            "any object exposing a ``correct`` attribute — including a "
            "*class* whose unbound ``correct`` method satisfies the check. "
            "Without the ``inspect.isclass`` branch, passing a class here "
            "would be treated as an already-constructed instance and the "
            "first ``correct(...)`` call would fail with 'missing positional "
            "argument: context'."
        ),
    )


class RefinementConfig(BaseModel):
    """Outer refinement-loop parameters.

    Consumed by :class:`cygnet.corrector.refinement_loop.RefinementLoop`
    which the Gate constructs in its ``__init__`` when a corrector is
    configured. Separate from :class:`CorrectorConfig` because the
    corrector and the loop have distinct responsibilities — the
    corrector does one (query, error) → cypher protocol exchange,
    the loop iterates with validator feedback.
    """

    model_config = ConfigDict(extra="forbid")

    max_attempts: int = Field(
        default=3,
        ge=1,
        description=(
            "Hard cap on outer-loop refinement attempts. The loop "
            "stops on success, on a corrector abort (echo/empty/"
            "malformed retries exhausted), or when it hits this cap "
            "without producing a validated cypher."
        ),
    )
    require_validates: bool = Field(
        default=True,
        description=(
            "If True, the refined cypher must pass the validator "
            "chain before the loop accepts it as ``refined``. If "
            "False, the loop returns whatever the corrector produced "
            "without running the chain."
        ),
    )
    require_distinct_from_input: bool = Field(
        default=True,
        description=(
            "If True, an echoed refinement (model returned the input "
            "unchanged after whitespace + keyword-case normalisation) "
            "is treated as an abort. The corrector already detects "
            "echoes; this flag controls the loop's response."
        ),
    )
    acceptance_backends: list[str] | None = Field(
        default=None,
        description=(
            "Optional subset of validator backend names "
            "(``builtin`` / ``ast`` / ``explain`` / ``mirror_execute``) "
            "to use for the loop's acceptance check. ``None`` (the "
            "default) means use the gate's configured chain unchanged."
        ),
    )


class MirrorConfig(BaseModel):
    """Mirror graph construction and lifecycle settings."""

    model_config = ConfigDict(extra="forbid")

    construction_source: Literal["spec", "sampling"] = Field(
        default="spec",
        description=(
            "How the mirror is built. 'spec' walks the loaded schema spec deterministically; "
            "'sampling' walks production and samples per label/relationship type."
        ),
    )
    sampling_size_per_label: int = Field(
        default=1,
        ge=1,
        description="Nodes sampled per label when construction_source='sampling'.",
    )
    rebuild_on_schema_change: bool = Field(
        default=True,
        description="If True, schema refresh that detects changes triggers a mirror rebuild.",
    )
    auto_build: bool = Field(
        default=False,
        description=(
            "If True, ``Gate.from_config`` invokes the mirror builder against "
            "the mirror Neo4j at construction time, populating it from the "
            "loaded schema. When False (default), the mirror is assumed to "
            "already exist and the user owns its lifecycle."
        ),
    )


class TransportConfig(BaseModel):
    """Optional transport server settings (MCP, HTTP).

    Empty by default; real transports are added in a later slice. Defined
    now so the `GateConfig` shape stays stable across slices.
    """

    model_config = ConfigDict(extra="forbid")

    mcp_enabled: bool = Field(
        default=False,
        description="If True, the MCP server transport is exposed.",
    )
    http_enabled: bool = Field(
        default=False,
        description="If True, the HTTP/FastAPI transport is exposed.",
    )
    http_port: int = Field(
        default=8080,
        ge=1,
        le=65535,
        description="Port for the HTTP transport when enabled.",
    )
    http_path_prefix: str = Field(
        default="/",
        description="URL prefix mounted by the HTTP transport.",
    )


class GateConfig(BaseSettings):
    """Top-level configuration for a Gate instance.

    Loads from constructor kwargs, a YAML file (`from_yaml`), or environment
    variables (`from_env`, default prefix 'CYGNET_'). Nested fields use '__'
    as the env-var separator: `CYGNET_NEO4J__URI`, `CYGNET_SCHEMA__SOURCE`, etc.
    """

    model_config = SettingsConfigDict(
        env_prefix="CYGNET_",
        env_nested_delimiter="__",
        populate_by_name=True,
        extra="forbid",
    )

    neo4j: Neo4jConfig = Field(..., description="Neo4j connection settings.")
    schema_: SchemaConfig = Field(
        ...,
        alias="schema",
        validation_alias=AliasChoices("schema", "schema_"),
        description=(
            "Schema source and refresh settings. YAML/JSON key: 'schema'. "
            "Env var prefix: 'CYGNET_SCHEMA__' (e.g. CYGNET_SCHEMA__SOURCE)."
        ),
    )
    validator: ValidatorConfig = Field(
        default_factory=ValidatorConfig,
        description="Validator backend chain configuration.",
    )
    cost: CostConfig = Field(
        default_factory=CostConfig,
        description="Cost gate configuration.",
    )
    corrector: CorrectorConfig | None = Field(
        default=None,
        description="Corrector configuration; None disables the corrector entirely.",
    )
    refinement: RefinementConfig = Field(
        default_factory=RefinementConfig,
        description=(
            "Outer refinement-loop parameters (v0.0.31+). Inherited "
            "defaults match the bench's existing 3-attempt cap with "
            "strict 'must validate, must be distinct' acceptance."
        ),
    )
    mirror: MirrorConfig | None = Field(
        default=None,
        description="Mirror graph settings; None when no mirror is in use.",
    )
    transports: TransportConfig | None = Field(
        default=None,
        description="Transport server settings; None when only the Python API is used.",
    )

    @classmethod
    def from_yaml(cls, path: Path | str) -> GateConfig:
        """Load a GateConfig from a YAML file."""
        path = Path(path)
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        if not isinstance(data, dict):
            raise ValueError(
                f"Top-level YAML at {path} must be a mapping; got {type(data).__name__}"
            )
        return cls.model_validate(data)

    @classmethod
    def from_env(cls, prefix: str = "CYGNET_") -> GateConfig:
        """Load a GateConfig from environment variables.

        Env vars are folded into a nested dict using ``__`` as the delimiter
        and then validated. The top-level key for the schema section is
        ``schema`` (mapping to the ``schema_`` field via alias), so e.g.::

            CYGNET_NEO4J__URI=bolt://host:7687
            CYGNET_SCHEMA__SOURCE=introspect
            CYGNET_VALIDATOR__BACKENDS='["ast","explain"]'

        Values that parse as JSON (lists, dicts, booleans, numbers) are
        decoded; otherwise they are passed as strings and coerced by
        Pydantic during validation.

        Note: this is done manually rather than through pydantic-settings'
        env source because pydantic-settings constructs nested env-var
        names from the raw Python field name (``schema_``), which would
        require triple-underscore env var names. The manual approach lets
        callers use the natural ``schema`` segment via the field alias.
        """
        env_data: dict[str, Any] = {}
        for raw_key, raw_value in os.environ.items():
            if not raw_key.startswith(prefix):
                continue
            path = raw_key[len(prefix) :].lower().split("__")
            try:
                value: Any = json.loads(raw_value)
            except (json.JSONDecodeError, ValueError):
                value = raw_value
            cursor: dict[str, Any] = env_data
            for part in path[:-1]:
                existing = cursor.get(part)
                if not isinstance(existing, dict):
                    existing = {}
                    cursor[part] = existing
                cursor = existing
            cursor[path[-1]] = value
        return cls.model_validate(env_data)
