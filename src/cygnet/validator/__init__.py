# Copyright (c) 2026 Nikodem Tomczak, Thulge Labs. All rights reserved.

"""Structural validator: parse, schema, property, and constraint stages.

The validator chain has four tiers with distinct responsibilities:

- ``builtin`` (fast filter): pure-Python regex-based, sub-millisecond,
  no I/O. Catches obvious typos before the heavier backends pay any
  cost.
- ``ast``: ANTLR4-Cypher AST walker for full grammar checking;
  statically detects property type mismatches and existence-constraint
  violations the other backends can't see. Limitation: the bundled
  Cypher 9 grammar doesn't parse ``CALL { ... }`` subqueries.
- ``explain``: Neo4j ``EXPLAIN`` against the mirror — authoritative for
  parse (covers the AST's subquery gap) and schema reference. Cannot
  detect uniqueness violations or type mismatches because EXPLAIN
  does not execute.
- ``mirror_execute`` (v0.0.23): runs the query inside a rolled-back
  transaction against the mirror. Catches runtime-only errors EXPLAIN
  doesn't surface: ``ProcedureNotFound``, ``ParameterMissing``,
  ``TypeError`` (implicit coercion failures), and ``SemanticError``
  (post-``WITH`` scoping bugs etc.). Same driver as ``explain``.

The default chain in ``ValidatorConfig.backends`` is
``["builtin", "ast", "explain", "mirror_execute"]`` — fast filter
first, then the structural backends, then the runtime backend.
"""

from typing import TYPE_CHECKING

from cygnet.config import ValidatorConfig
from cygnet.models import Schema
from cygnet.validator.builtin import BuiltinValidator
from cygnet.validator.chain import StructuralValidator, ValidatorChain
from cygnet.validator.explain_backend import ExplainValidator
from cygnet.validator.mirror_execute import MirrorExecuteValidator

# ``ASTValidator`` lives in :mod:`cygnet.validator.ast_backend`, which
# imports ``antlr4``. ``antlr4`` is in the ``ast`` extra
# (``pip install thulge-cygnet[ast]``), so referencing the symbol from
# the package root would break ``import cygnet.validator`` on a base
# install. Resolve it lazily through ``__getattr__`` instead — see
# ``cygnet/__init__.py`` for the same pattern around the
# ``corrector`` extra.

if TYPE_CHECKING:
    from neo4j import Driver

    from cygnet.validator.ast_backend import ASTValidator as ASTValidator  # noqa: F401

__all__ = [
    "ASTValidator",
    "BuiltinValidator",
    "ExplainValidator",
    "MirrorExecuteValidator",
    "StructuralValidator",
    "ValidatorChain",
    "build_chain",
]


def __getattr__(name: str) -> object:
    """Lazy access to ``ASTValidator`` (requires the ``ast`` extra)."""
    if name == "ASTValidator":
        from cygnet.validator.ast_backend import ASTValidator

        return ASTValidator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def build_chain(
    config: ValidatorConfig,
    schema: Schema,
    *,
    driver: "Driver | None" = None,
) -> ValidatorChain:
    """Construct a ``ValidatorChain`` from configuration.

    Backends listed in ``config.backends`` are instantiated and chained
    in order. Both ``explain`` and ``mirror_execute`` require a
    ``neo4j.Driver``; pass one via the ``driver`` keyword argument or
    omit those backends from the backend list. ``Gate.from_config``
    opens an appropriate driver automatically when either is in the
    chain.

    ``config.collection_mode`` is forwarded to the chain (v0.0.25);
    defaults to ``"short_circuit"``.
    """
    validators: list[StructuralValidator] = []
    for backend_name in config.backends:
        if backend_name == "builtin":
            validators.append(
                BuiltinValidator(
                    schema=schema,
                    strict_property_existence=config.strict_property_existence,
                )
            )
        elif backend_name == "ast":
            try:
                from cygnet.validator.ast_backend import ASTValidator
            except ImportError as exc:
                raise ImportError(
                    "Validator backend 'ast' requires the 'ast' extra. "
                    "Install with `pip install thulge-cygnet[ast]`, or "
                    "remove 'ast' from validator.backends."
                ) from exc
            validators.append(
                ASTValidator(
                    schema=schema,
                    strict_property_existence=config.strict_property_existence,
                )
            )
        elif backend_name == "explain":
            if driver is None:
                raise ValueError(
                    "Validator backend 'explain' requires a Neo4j driver. "
                    "Pass driver=... to build_chain (Gate.from_config does "
                    "this automatically when explain is in the chain), or "
                    "remove 'explain' from validator.backends."
                )
            validators.append(ExplainValidator(schema=schema, driver=driver))
        elif backend_name == "mirror_execute":
            if driver is None:
                raise ValueError(
                    "Validator backend 'mirror_execute' requires a Neo4j "
                    "driver. Pass driver=... to build_chain "
                    "(Gate.from_config does this automatically when "
                    "mirror_execute is in the chain), or remove "
                    "'mirror_execute' from validator.backends."
                )
            validators.append(MirrorExecuteValidator(driver=driver, schema=schema))
    return ValidatorChain(validators, collection_mode=config.collection_mode)
