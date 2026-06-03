# Copyright (c) 2026 Nikodem Tomczak, Thulge Labs. All rights reserved.

"""CYGNET: Cypher Gate for Neural Execution Triage.

Pre-execution gate for LLM-generated Cypher queries against Neo4j. Wraps a
structural validator (parse, schema, property, constraint), a cost gate
(via EXPLAIN), and an optional corrector, exposed as a single `Gate`
object plus a discriminated error vocabulary for agent refinement loops.
"""

from __future__ import annotations

from cygnet.config import (
    CorrectorConfig,
    CostConfig,
    GateConfig,
    MirrorConfig,
    Neo4jConfig,
    SchemaConfig,
    TransportConfig,
    ValidatorBackend,
    ValidatorConfig,
)
from cygnet.core.gate import Gate
from cygnet.corrector import (
    Corrector,
    CorrectorContext,
    NullCorrector,
    PriorAttempt,
    run_correction,
)
from cygnet.cost import CostGate, ExplainOperator, ExplainPlan, parse_explain_plan
from cygnet.mirror import MirrorBuildReport, MirrorGraphBuilder
from cygnet.models import (
    ConstraintError,
    CorrectorResult,
    CostError,
    CostGateResult,
    EmptyResultError,
    GateError,
    GateResult,
    Index,
    NodeLabel,
    OperatorCost,
    ParseError,
    Property,
    PropertyError,
    RelationshipType,
    Schema,
    SchemaConstraint,
    SchemaError,
    StructuralValidatorResult,
)
from cygnet.schema import (
    SchemaIntrospectionError,
    SchemaSpecError,
    introspect_schema,
    load_schema_spec,
)
from cygnet.validator import (
    BuiltinValidator,
    ExplainValidator,
    MirrorExecuteValidator,
    ValidatorChain,
    build_chain,
)

__version__ = "0.0.47"

__all__ = [
    "ASTValidator",
    "AnthropicClient",
    "BuiltinValidator",
    "ConstraintError",
    "Corrector",
    "CorrectorConfig",
    "CorrectorContext",
    "CorrectorResult",
    "CostConfig",
    "CostError",
    "CostGate",
    "CostGateResult",
    "EmptyResultError",
    "ExplainOperator",
    "ExplainPlan",
    "ExplainValidator",
    "Gate",
    "GateConfig",
    "GateError",
    "GateResult",
    "GoogleGeminiClient",
    "Index",
    "LLMClient",
    "MirrorBuildReport",
    "MirrorConfig",
    "MirrorExecuteValidator",
    "MirrorGraphBuilder",
    "Neo4jConfig",
    "NodeLabel",
    "NullCorrector",
    "OllamaClient",
    "OpenAIClient",
    "OperatorCost",
    "ParseError",
    "PriorAttempt",
    "Property",
    "PropertyError",
    "RampartCorrector",
    "RelationshipType",
    "Schema",
    "SchemaConfig",
    "SchemaConstraint",
    "SchemaError",
    "SchemaIntrospectionError",
    "SchemaSpecError",
    "StructuralValidatorResult",
    "TransportConfig",
    "ValidatorBackend",
    "ValidatorChain",
    "ValidatorConfig",
    "__version__",
    "build_chain",
    "create_http_app",
    "create_mcp_server",
    "introspect_schema",
    "load_schema_spec",
    "make_llm_client",
    "parse_explain_plan",
    "run_correction",
]


def __getattr__(name: str) -> object:
    """Lazy package-root access for the ``corrector``-extra surface.

    The RAMPART-backed corrector and LLM clients require the
    ``corrector`` extra (``pip install thulge-cygnet[corrector]``).
    Importing the package without the extra would otherwise fail at
    top-level due to the ``rampart`` / ``anthropic`` / ``openai``
    imports. Delegating these names to :mod:`cygnet.corrector` keeps
    the unconditional surface intact and surfaces ``ImportError``
    only when the consumer actually references one of these names.
    """
    _corrector_lazy = {
        "AnthropicClient",
        "GoogleGeminiClient",
        "LLMClient",
        "OllamaClient",
        "OpenAIClient",
        "RampartCorrector",
        "make_llm_client",
    }
    if name in _corrector_lazy:
        from cygnet import corrector

        return getattr(corrector, name)
    _transport_lazy = {"create_mcp_server", "create_http_app"}
    if name in _transport_lazy:
        from cygnet import transports

        return getattr(transports, name)
    if name == "ASTValidator":
        from cygnet.validator import ASTValidator

        return ASTValidator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
