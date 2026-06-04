# Copyright (c) 2026 Nikodem Tomczak, Thulge Labs. All rights reserved.

"""Core orchestration: the ``Gate`` that wires schema, validator chain,
cost, and corrector.

Provides schema spec loading (Path B), schema introspection (Path A),
the builtin and AST validator backends, the validator chain, the cost
gate, and the corrector surface — all reachable through a single
``Gate`` instance constructed via :meth:`Gate.from_config`.
"""

from __future__ import annotations

import inspect
import logging
import os
from typing import TYPE_CHECKING

from neo4j import GraphDatabase

from cygnet.config import GateConfig
from cygnet.corrector import Corrector, NullCorrector
from cygnet.corrector.refinement_loop import RefinementResult
from cygnet.corrector.telemetry import CorrectorTelemetry
from cygnet.cost import CostGate
from cygnet.mirror import MirrorGraphBuilder
from cygnet.models import (
    CostGateResult,
    GateError,
    GateResult,
    Schema,
    StructuralValidatorResult,
)
from cygnet.schema import introspect_schema, load_schema_spec
from cygnet.validator import ValidatorChain, build_chain

if TYPE_CHECKING:
    from neo4j import Driver

    from cygnet.config import ValidatorBackend
    from cygnet.validator.chain import CollectionMode

__all__ = ["Gate"]


logger = logging.getLogger(__name__)


class Gate:
    """Orchestrator: schema + validator chain + (future) cost + corrector.

    Constructed via :meth:`from_config` from a fully-populated
    :class:`cygnet.config.GateConfig`. The constructor itself takes
    already-resolved dependencies so callers can compose differently
    in tests.

    Method status:

    - ``validate``: implemented; runs the validator chain.
    - ``gate``: implemented; runs the validator chain only (cost
      gate is :data:`None`).
    - ``get_schema``: implemented.
    - ``estimate_cost``: implemented (when ``config.cost.enabled``).
    - ``correct``: implemented (delegates to :func:`cygnet.run_correction`).
    - ``refresh_schema``: implemented.
    """

    def __init__(
        self,
        *,
        config: GateConfig,
        schema: Schema,
        chain: ValidatorChain,
        driver: Driver | None = None,
        cost_gate: CostGate | None = None,
        mirror_driver: Driver | None = None,
        corrector: Corrector | None = None,
        telemetry: CorrectorTelemetry | None = None,
    ) -> None:
        self._config = config
        self._schema = schema
        self._chain = chain
        self._driver = driver
        self._cost_gate = cost_gate
        self._mirror_driver = mirror_driver
        self._corrector: Corrector = corrector if corrector is not None else NullCorrector()
        # Outer refinement loop, built once at Gate construction;
        # ``Gate.correct`` delegates to it. The ``telemetry`` kwarg
        # threads the per-LLM-call hook into the loop so downstream
        # consumers can attach a :class:`FileTelemetry` for per-call
        # JSON writes plus a ``compute_extras`` hook for cost
        # accounting.
        from cygnet.corrector.refinement_loop import (
            AcceptanceCriteria,
            RefinementLoop,
        )

        refinement = config.refinement
        self._refinement_acceptance = AcceptanceCriteria(
            require_validates=refinement.require_validates,
            require_distinct_from_input=refinement.require_distinct_from_input,
            backends=refinement.acceptance_backends,
        )
        self._refinement_telemetry = telemetry
        self._refinement_loop_max_attempts = refinement.max_attempts
        # ``self._refinement_loop`` is kept for backwards compatibility
        # with any caller that reached into it directly.
        # :meth:`Gate.correct` delegates to :func:`cygnet.run_correction`
        # instead, which is built with the same components so both
        # paths land in identical :meth:`RefinementLoop.refine`
        # behaviour.
        self._refinement_loop = RefinementLoop(
            self._corrector,
            self._chain,
            self._schema,
            max_attempts=refinement.max_attempts,
            acceptance=self._refinement_acceptance,
            telemetry=telemetry,
        )

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        config: GateConfig,
        *,
        corrector: Corrector | None = None,
        telemetry: CorrectorTelemetry | None = None,
    ) -> Gate:
        """Build a Gate from configuration.

        Loads the schema from ``config.schema_``:

        - ``spec_file`` / ``spec_object``: Path B, no driver needed.
        - ``introspect``: Path A, requires a Neo4j driver against the
          *production* instance.

        Then constructs the validator chain from ``config.validator.backends``.

        **Driver routing.**

        - When ``config.mirror`` is ``None``: the Gate owns at most one
          driver. It is shared by every component that needs Neo4j
          (EXPLAIN backend, introspection, cost gate). Credentials come
          from ``Neo4jConfig``; the URI comes from
          ``ValidatorConfig.mirror_uri`` when set (the validator-level
          escape hatch that pre-dates first-class mirror config) or
          from ``Neo4jConfig.uri`` as a fallback.
        - When ``config.mirror`` is populated: the Gate owns *two*
          drivers. The **mirror driver** is opened against
          ``ValidatorConfig.mirror_uri`` (required in this mode) and
          routes the EXPLAIN backend, the cost gate, and (when
          ``mirror.auto_build`` is True) the
          :class:`MirrorGraphBuilder`. The **production driver** is
          opened against ``Neo4jConfig.uri`` only when
          ``schema.source == "introspect"`` requires reading from
          production. Both drivers are closed by :meth:`close`.

        When ``mirror.auto_build`` is True, ``Gate.from_config``
        invokes :class:`MirrorGraphBuilder` against the mirror driver
        after the schema is loaded. When False, the mirror must
        already be populated and the user is responsible for its
        lifecycle.

        **Corrector resolution.** The corrector is picked in this
        order:

        1. The ``corrector=`` kwarg on this call (an instance).
        2. ``CorrectorConfig.corrector`` — an instance or a zero-arg
           factory returning one.
        3. :class:`RampartCorrector` when ``config.corrector`` is set
           AND the ``corrector`` extra is importable AND the configured
           backend's API key is present (``ANTHROPIC_API_KEY`` for the
           Anthropic backend, ``OPENAI_API_KEY`` for OpenAI). The
           token budget, backend, and model come from the config.
        4. :class:`NullCorrector` (the safe fallback). Used when
           ``config.corrector`` is ``None``, or when the auto-detection
           in step 3 fails (missing extra, missing API key). The fall-
           back path logs a ``debug`` message; the caller can see it
           by enabling ``logging.getLogger('cygnet.core.gate')``.
        """
        prod_driver, mirror_driver = cls._open_drivers(config)
        try:
            schema = cls._load_schema_from_config(config, prod_driver)
            validator_driver = mirror_driver if mirror_driver is not None else prod_driver
            chain = build_chain(config.validator, schema, driver=validator_driver)
            cost_gate = cls._build_cost_gate(config, validator_driver)
            cls._maybe_build_mirror(config, mirror_driver, schema)
            resolved_corrector = cls._resolve_corrector(config, corrector)
        except Exception:
            if prod_driver is not None:
                prod_driver.close()
            if mirror_driver is not None:
                mirror_driver.close()
            raise
        return cls(
            config=config,
            schema=schema,
            chain=chain,
            driver=prod_driver,
            cost_gate=cost_gate,
            mirror_driver=mirror_driver,
            corrector=resolved_corrector,
            telemetry=telemetry,
        )

    @staticmethod
    def _resolve_corrector(
        config: GateConfig,
        override: Corrector | None,
    ) -> Corrector:
        """Pick the corrector instance for this gate.

        Priority order:
        1. The ``corrector=`` kwarg on ``from_config`` (an instance).
        2. ``CorrectorConfig.corrector`` (an instance or a zero-arg
           factory returning one).
        3. :class:`RampartCorrector` when ``config.corrector`` is set
           and (a) the ``corrector`` extra is importable and (b) the
           backend's API key is present in the environment.
        4. :class:`NullCorrector` (the safe default).
        """
        if override is not None:
            return override
        if config.corrector is not None and config.corrector.corrector is not None:
            candidate = config.corrector.corrector
            # A class always needs instantiating (its unbound ``correct``
            # makes it satisfy the Protocol's isinstance check, which is
            # why we can't just rely on isinstance(candidate, Corrector)
            # to skip the call). For plain callables (lambdas, factory
            # functions) with no ``correct`` attribute, call to get the
            # instance. See CorrectorConfig.corrector field description
            # for the full rationale.
            if inspect.isclass(candidate) or (
                callable(candidate) and not hasattr(candidate, "correct")
            ):
                candidate = candidate()
            if not isinstance(candidate, Corrector):
                raise TypeError(
                    "CorrectorConfig.corrector must be a Corrector instance "
                    "or a zero-arg callable returning one; got "
                    f"{type(candidate).__name__}."
                )
            return candidate
        if config.corrector is not None:
            auto = Gate._try_build_rampart_corrector(config.corrector)
            if auto is not None:
                return auto
        return NullCorrector()

    @staticmethod
    def _try_build_rampart_corrector(corrector_cfg: object) -> Corrector | None:
        """Attempt to construct a :class:`RampartCorrector` from
        ``CorrectorConfig``. Returns ``None`` (with a debug log) when
        the ``corrector`` extra is not installed or the backend's API
        key is missing — :meth:`_resolve_corrector` falls back to
        :class:`NullCorrector` in that case.
        """
        try:
            from cygnet.corrector.llm import make_llm_client
            from cygnet.corrector.rampart_backed import RampartCorrector
        except ImportError as exc:
            logger.debug(
                "RampartCorrector unavailable (corrector extra not installed?): %s",
                exc,
            )
            return None

        backend = getattr(corrector_cfg, "backend", "anthropic")
        env_var = "ANTHROPIC_API_KEY" if backend == "anthropic" else "OPENAI_API_KEY"
        if not os.environ.get(env_var):
            logger.debug(
                "RampartCorrector skipped: %s not set; falling back to NullCorrector.",
                env_var,
            )
            return None
        try:
            llm_client = make_llm_client(
                backend,
                model=getattr(corrector_cfg, "llm", "claude-sonnet-4-5"),
            )
            return RampartCorrector(
                llm_client,
                token_budget=getattr(corrector_cfg, "token_budget", 4000),
                model=getattr(corrector_cfg, "llm", "claude-sonnet-4-5"),
            )
        except Exception as exc:
            logger.debug(
                "RampartCorrector construction failed: %s. Falling back to NullCorrector.",
                exc,
            )
            return None

    @staticmethod
    def _open_drivers(config: GateConfig) -> tuple[Driver | None, Driver | None]:
        """Open the production and mirror drivers per the routing rules.

        Returns ``(prod_driver, mirror_driver)``. Either may be ``None``:
        the production driver is only opened when something on the
        production side needs it; the mirror driver is only opened
        when ``config.mirror`` is populated.
        """
        if config.mirror is None:
            needs_driver = (
                "explain" in config.validator.backends
                or "mirror_execute" in config.validator.backends
                or config.schema_.source == "introspect"
                or config.cost.enabled
            )
            if not needs_driver:
                return (None, None)
            prod_uri = config.validator.mirror_uri or config.neo4j.uri
            return (
                GraphDatabase.driver(
                    prod_uri,
                    auth=(config.neo4j.user, config.neo4j.password),
                ),
                None,
            )

        mirror_uri = config.validator.mirror_uri
        if not mirror_uri:
            raise ValueError(
                "config.mirror is set but validator.mirror_uri is empty; "
                "set validator.mirror_uri to the bolt URI of the mirror Neo4j."
            )
        mirror_driver = GraphDatabase.driver(
            mirror_uri,
            auth=(config.neo4j.user, config.neo4j.password),
        )
        prod_driver: Driver | None = None
        if config.schema_.source == "introspect":
            prod_driver = GraphDatabase.driver(
                config.neo4j.uri,
                auth=(config.neo4j.user, config.neo4j.password),
            )
        return (prod_driver, mirror_driver)

    @staticmethod
    def _build_cost_gate(config: GateConfig, driver: Driver | None) -> CostGate | None:
        if not config.cost.enabled:
            return None
        if driver is None:
            # _open_drivers should have opened one when cost.enabled is
            # True; defensive branch.
            raise ValueError(
                "cost.enabled=True requires a Neo4j driver; Gate.from_config "
                "opens one automatically."
            )
        return CostGate(
            driver,
            threshold_rows=config.cost.threshold_rows,
            threshold_dbhits=config.cost.threshold_dbhits,
            database=config.neo4j.database,
        )

    @staticmethod
    def _maybe_build_mirror(
        config: GateConfig,
        mirror_driver: Driver | None,
        schema: Schema,
    ) -> None:
        """Invoke :class:`MirrorGraphBuilder` against the mirror driver
        when ``mirror.auto_build`` is True. No-op otherwise.

        Idempotency is the builder's responsibility — if the mirror is
        already populated under the default prefix the build short-
        circuits and emits an idempotency warning.
        """
        if config.mirror is None or not config.mirror.auto_build:
            return
        if mirror_driver is None:
            raise ValueError(
                "mirror.auto_build=True requires a mirror driver; "
                "Gate.from_config opens one when config.mirror is set."
            )
        builder = MirrorGraphBuilder(mirror_driver, database=config.neo4j.database)
        builder.build_from_schema(schema)

    @staticmethod
    def _open_driver(config: GateConfig) -> Driver:
        """Production-side driver opener used by the lazy path.

        When ``config.mirror`` is set, production connections always go
        to ``Neo4jConfig.uri`` (the mirror driver is opened separately
        in :meth:`_open_drivers`). When ``config.mirror`` is ``None``,
        the legacy ``validator.mirror_uri`` escape hatch is honoured
        so existing single-driver wiring keeps working.
        """
        if config.mirror is not None:
            uri = config.neo4j.uri
        else:
            uri = config.validator.mirror_uri or config.neo4j.uri
        return GraphDatabase.driver(
            uri,
            auth=(config.neo4j.user, config.neo4j.password),
        )

    @staticmethod
    def _load_schema_from_config(config: GateConfig, driver: Driver | None) -> Schema:
        source = config.schema_.source
        if source == "spec_file":
            if config.schema_.spec_path is None:
                raise ValueError("schema.source='spec_file' requires schema.spec_path to be set.")
            return load_schema_spec(config.schema_.spec_path)
        if source == "spec_object":
            if config.schema_.spec_object is None:
                raise ValueError(
                    "schema.source='spec_object' requires schema.spec_object to be set."
                )
            return load_schema_spec(config.schema_.spec_object)
        if source == "introspect":
            if driver is None:
                # _make_driver_if_needed should have opened one; this
                # branch is defensive in case from_config is called
                # with a non-standard driver-construction path.
                raise ValueError(
                    "schema.source='introspect' requires a driver; "
                    "Gate.from_config opens one automatically."
                )
            return introspect_schema(driver, config.neo4j.database)
        raise ValueError(f"Unknown schema source: {source!r}")

    def _get_or_create_driver(self) -> Driver:
        """Lazy driver getter for components that decide they need one
        after Gate construction. Most callers should not need this —
        ``from_config`` opens a driver up-front when the config requests
        explain or introspect. Future slices that introduce new
        driver-needing components (e.g. on-demand mirror rebuild)
        should reuse this helper rather than opening a second driver.

        Lifecycle remains Gate's responsibility; ``close()`` shuts it
        down whether it was opened up-front or lazily.
        """
        if self._driver is None:
            self._driver = Gate._open_driver(self._config)
        return self._driver

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def config(self) -> GateConfig:
        return self._config

    @property
    def chain(self) -> ValidatorChain:
        return self._chain

    def get_schema(self) -> Schema:
        """Return the currently loaded schema."""
        return self._schema

    def close(self) -> None:
        """Release the driver(s) the Gate owns: the production driver
        (when one was opened for introspection or legacy single-driver
        routing) and the mirror driver (when ``config.mirror`` is set).
        Safe to call multiple times."""
        if self._driver is not None:
            self._driver.close()
            self._driver = None
        if self._mirror_driver is not None:
            self._mirror_driver.close()
            self._mirror_driver = None

    def __enter__(self) -> Gate:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Gate methods
    # ------------------------------------------------------------------

    def validate(
        self,
        query: str,
        *,
        backends: list[ValidatorBackend] | None = None,
        collection_mode: CollectionMode | None = None,
    ) -> StructuralValidatorResult:
        """Run the structural validator chain.

        Args:
            query: The Cypher query to validate.
            backends: Optional per-call override. When set, a fresh
                chain is built for this single call from the listed
                backends.
            collection_mode: Optional per-call override.
                ``"short_circuit"`` stops at the first failing backend;
                ``"collect_all"`` runs every backend that can run on
                the input and returns the full list of errors in
                ``StructuralValidatorResult.all_errors``. When ``None``,
                the configured ``validator.collection_mode`` applies.
        """
        if backends is None and collection_mode is None:
            return self._chain.validate(query)
        override_updates: dict[str, object] = {}
        if backends is not None:
            override_updates["backends"] = backends
        if collection_mode is not None:
            override_updates["collection_mode"] = collection_mode
        override_config = self._config.validator.model_copy(update=override_updates)
        validator_driver = self._mirror_driver if self._mirror_driver is not None else self._driver
        override_chain = build_chain(override_config, self._schema, driver=validator_driver)
        return override_chain.validate(query)

    def estimate_cost(self, query: str) -> CostGateResult:
        """Run the cost gate via EXPLAIN.

        Raises ``ValueError`` when ``config.cost.enabled is False`` — the
        cost gate was deliberately disabled by configuration and
        calling this method anyway is a contract violation. Use
        ``Gate.gate(query)`` instead, which respects the toggle and
        returns ``GateResult.cost = None``.
        """
        if self._cost_gate is None:
            raise ValueError(
                "Gate.estimate_cost is not available because "
                "config.cost.enabled=False. Re-enable the cost gate or "
                "call Gate.gate(query) which respects the toggle."
            )
        return self._cost_gate.estimate(query)

    def gate(self, query: str) -> GateResult:
        """Run structural validation then (if enabled) the cost gate.

        Returns a :class:`GateResult`:

        - Structural failure: ``cost`` is ``None`` (cost gate skipped),
          ``errors`` contains the structural error(s). In
          ``collect_all`` mode every backend's error is folded into
          ``errors`` so the corrector can consume the full list.
        - Structural pass + cost gate disabled: ``cost`` is ``None``,
          ``passed=True``, ``errors`` empty.
        - Structural pass + cost pass: ``cost`` populated, ``passed=True``,
          ``errors`` empty.
        - Structural pass + cost fail: ``cost`` populated, ``passed=False``,
          ``errors`` contains a :class:`GateError` with ``category="cost"``.
        """
        structural = self._chain.validate(query)
        errors: list[GateError] = []
        if not structural.passed and structural.error_payload is not None:
            # Wrap every structural payload (one in short_circuit mode,
            # potentially several in collect_all mode) as a GateError so
            # the corrector-facing surface is uniform regardless of
            # which chain mode produced it. Order is preserved from
            # ``structural.all_errors`` — the chain has already applied
            # the public ordering policy (parse-first, then backend
            # authority).
            for payload in structural.all_errors:
                errors.append(
                    GateError(
                        category=payload.category,
                        payload=payload,
                    )
                )
            return GateResult(
                passed=False,
                structural=structural,
                cost=None,
                errors=errors,
            )

        # Structural passed; run cost gate if enabled.
        if self._cost_gate is None:
            return GateResult(
                passed=True,
                structural=structural,
                cost=None,
                errors=[],
            )

        cost = self._cost_gate.estimate(query)
        if not cost.passed:
            from cygnet.models import CostError

            errors.append(
                GateError(
                    category="cost",
                    payload=CostError(
                        estimated_rows=cost.estimated_rows,
                        estimated_dbhits=cost.estimated_dbhits,
                        threshold_used=cost.threshold_used,
                        cost_driver=cost.cost_driver or "unknown",
                        suggested_mitigations=list(cost.suggested_mitigations),
                        estimated_cost_breakdown=list(cost.estimated_cost_breakdown),
                    ),
                )
            )
        return GateResult(
            passed=cost.passed,
            structural=structural,
            cost=cost,
            errors=errors,
        )

    def correct(
        self,
        query: str,
        error: GateError,
        *,
        all_errors: list[GateError] | None = None,
        conversation_history: list[str] | None = None,
        query_id: str | None = None,
        condition: str | None = None,
        apply_default_wrapping: bool = True,
    ) -> RefinementResult:
        """Run the configured :class:`RefinementLoop` end-to-end.

        Returns :class:`RefinementResult` (outer-loop output). The
        library owns the outer refinement loop — validation-failure
        feedback, prior-attempts assembly, and max-attempts capping
        all happen inside this call.

        With the default :class:`NullCorrector` this still aborts on
        the first attempt — wire in a real corrector via
        ``CorrectorConfig.corrector`` or ``Gate.from_config(corrector=...)``
        to get actual refinement.

        Args:
            query: The failing query to refine.
            error: The primary error to surface to the corrector. In
                ``collect_all`` mode this is the first entry of
                ``all_errors``.
            all_errors: Optional full list of errors the chain found.
                When non-empty the corrector receives the full list
                and may build a fix-all-at-once prompt.
            conversation_history: Optional prior conversational
                turns to fold into the prompt.
            query_id: Optional caller-supplied identifier propagated
                into every :class:`LLMCallRecord` emitted by telemetry.
                Used to attribute LLM calls to dataset rows.
            condition: Optional caller-supplied condition tag (e.g.
                ``raw``, ``verbal``, ``structured``) propagated into
                telemetry records.
            apply_default_wrapping: when True (default), the corrector
                configured on the gate is wrapped with the
                production retry decorators
                (:class:`ProtocolRetryingCorrector` outside,
                :class:`EmptyRetryingCorrector` inside) before the
                refinement loop runs. Set ``False`` to pass the
                configured corrector through unwrapped
                (bring-your-own-wrapping). Same kwarg name as on
                :func:`cygnet.run_correction` so users only learn it
                once.

        Returns:
            :class:`RefinementResult`. ``action == "refined"`` is
            the success condition. On ``action == "abort"`` the
            ``refined_query`` field may carry the last cypher the
            loop produced (for echoed inputs or max-attempts
            exhaustion) — do not treat it as a success signal.

        Delegates to :func:`cygnet.run_correction` underneath.
        """
        from cygnet.corrector.runner import LoopOptions, run_correction

        return run_correction(
            corrector=self._corrector,
            query=query,
            error_context=error,
            validator_chain=self._chain,
            schema=self._schema,
            loop=True,
            loop_options=LoopOptions(max_attempts=self._refinement_loop_max_attempts),
            acceptance=self._refinement_acceptance,
            telemetry=self._refinement_telemetry,
            apply_default_wrapping=apply_default_wrapping,
            query_id=query_id,
            condition=condition,
            all_errors=all_errors,
            conversation_history=conversation_history,
        )

    def refresh_schema(self) -> Schema:
        """Reload the schema from its configured source and atomically
        swap it in.

        Behaviour by source:

        - ``spec_file``: re-reads the file. Useful for picking up
          edits without restarting the agent.
        - ``introspect``: re-runs ``introspect_schema`` on the live
          Neo4j. Useful when the production schema may have evolved.
        - ``spec_object``: raises ``ValueError`` — an in-memory dict
          has no external source to refresh from. Callers should
          construct a fresh ``Gate`` instead.

        After the schema is replaced, the validator chain (and the
        cost gate if enabled) is rebuilt against the new schema. New
        ``validate`` / ``gate`` calls see the new chain immediately;
        in-flight callers using the previous chain reference are not
        interrupted but will still see the old schema-derived caches
        until they exit. Replacement is a single attribute assignment
        (atomic in CPython for object references) so no reader
        blocks.
        """
        source = self._config.schema_.source
        if source == "spec_file":
            if self._config.schema_.spec_path is None:
                raise ValueError("schema.source='spec_file' requires schema.spec_path to be set.")
            new_schema = load_schema_spec(self._config.schema_.spec_path)
        elif source == "introspect":
            new_schema = introspect_schema(
                self._get_or_create_driver(),
                self._config.neo4j.database,
            )
        elif source == "spec_object":
            raise ValueError(
                "Gate.refresh_schema is not supported for schema.source="
                "'spec_object' — the spec is in memory and has no source "
                "to reread. Construct a fresh Gate with the updated spec "
                "instead."
            )
        else:
            raise ValueError(f"Unknown schema source: {source!r}")
        self._schema = new_schema
        # Rebuild the chain against the new schema. The cost gate is
        # schema-agnostic but rebuilt for consistency (and to pick up
        # any future schema-aware mitigations).
        validator_driver = self._mirror_driver if self._mirror_driver is not None else self._driver
        self._chain = build_chain(
            self._config.validator,
            new_schema,
            driver=validator_driver,
        )
        self._cost_gate = self._build_cost_gate(self._config, validator_driver)
        return new_schema
