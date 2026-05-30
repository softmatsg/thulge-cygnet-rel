# Copyright (c) 2026 Nikodem Tomczak, Thulge Labs. All rights reserved.

"""FastMCP server exposing CYGNET as an MCP tool surface.

After this slice, an MCP-speaking agent host (Claude Desktop,
Cursor, Anthropic Claude, custom frameworks) can spawn this server
and the six tools below appear in the agent's toolset. Users no
longer need to write Python wrapper code.

**Tool surface** (treat names + parameter names as the immutable
agent-host integration contract; renames break user configurations):

- ``validate_cypher(query)`` -> structural validator chain output.
- ``estimate_cypher_cost(query)`` -> cost gate output.
- ``gate_cypher(query)`` -> full pipeline (structural + cost).
- ``get_schema()`` -> currently loaded :class:`Schema`.
- ``correct_cypher(query, error, attempt_number=1)`` -> corrector
  output. ``error`` is the dict shape of a ``GateError``.
- ``refresh_schema()`` -> reload from configured source. Mode-gated:
  only registered when ``mode == "read_write"``.

**Mode gating.** The ``mode`` parameter governs which tools are
registered. In ``"read_only"``, ``refresh_schema`` is omitted from
the tool list — not advertised, so the LLM cannot see or call it
(lifted from the mcp-neo4j-cypher pattern documented in
``docs/inspection_mcp_neo4j.md`` section 6). No other tool is
"write" in the Neo4j sense; the gating is for the Gate-side schema-
state mutation that ``refresh_schema`` performs, not on the
database itself.

**FastMCP 3.x note.** Upstream Neo4j's mcp-neo4j-cypher (FastMCP 2.x
era) used an ``enabled=`` kwarg on ``@mcp.tool`` to hide tools at
registration time. FastMCP 3.x dropped that arg; mode-gating here is
implemented by **conditional registration** inside the factory
closure, which produces the same observable outcome (the tool is
absent from the tool list, not just rejected at call time).

Lifted patterns from ``docs/inspection_mcp_neo4j.md`` (MIT-attributed
to Neo4j Labs in that doc):

- Factory-with-closure tool registration: ``create_mcp_server(...)``
  builds a fresh :class:`FastMCP` and binds tools via decorators
  that capture the ``Gate`` by closure.
- :class:`ToolAnnotations` on every tool with accurate
  ``readOnlyHint``/``destructiveHint``/``idempotentHint``/``openWorldHint``.
- Domain-exception-to-ToolError mapping at the tool boundary.

Things deliberately not adopted from upstream (also documented in
the inspection doc): no auth at this layer (stdio is process-
boundary authenticated; HTTP deployments add auth in front),
no tiktoken-based truncation (gate outputs are small enough for
batch), no driver lifecycle hook on the FastMCP server itself
(``Gate.close()`` is the one resource we own — wired into the
lifespan below).
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, Literal

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import ValidationError

from cygnet.models import GateError

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from cygnet import Gate

__all__ = ["TOOL_NAMES", "MCPMode", "create_mcp_server"]


MCPMode = Literal["read_only", "read_write"]
"""Server mode. ``"read_only"`` hides ``refresh_schema``."""


# Tool names. These are part of the agent-host integration contract;
# treat as immutable artifacts per the brief.
TOOL_NAMES: dict[str, str] = {
    "validate": "validate_cypher",
    "estimate": "estimate_cypher_cost",
    "gate": "gate_cypher",
    "schema": "get_schema",
    "correct": "correct_cypher",
    "refresh": "refresh_schema",
}


def create_mcp_server(
    gate: Gate,
    *,
    mode: MCPMode = "read_write",
    server_name: str = "cygnet",
    own_gate: bool = False,
) -> FastMCP:
    """Build a :class:`FastMCP` server bound to a :class:`Gate`.

    Args:
        gate: an already-constructed :class:`Gate` (typically from
            :meth:`Gate.from_config`). The server delegates every
            tool call to this instance.
        mode: ``"read_only"`` omits the ``refresh_schema`` tool from
            registration; ``"read_write"`` registers all six tools.
        server_name: ``FastMCP`` instance name. Surfaces to clients
            as the server identifier.
        own_gate: when ``True``, the server's lifespan closes the
            ``Gate`` on shutdown. Set to ``True`` from the CLI
            launcher where the server owns the Gate's lifecycle;
            leave ``False`` (default) when the caller manages the
            Gate themselves (tests, embedded use).
    """

    @asynccontextmanager
    async def _lifespan(_server: FastMCP) -> AsyncIterator[None]:
        try:
            yield
        finally:
            if own_gate:
                gate.close()

    mcp = FastMCP(
        name=server_name,
        instructions=(
            "CYGNET MCP server. Validate, cost-estimate, and refine "
            "LLM-generated Cypher queries before they execute against "
            "Neo4j. Use validate_cypher / estimate_cypher_cost / "
            "gate_cypher to check a query, get_schema to introspect "
            "the loaded schema, and correct_cypher to refine a query "
            "that failed gating."
        ),
        lifespan=_lifespan,
    )

    _register_read_only_tools(mcp, gate)
    if mode == "read_write":
        _register_read_write_tools(mcp, gate)
    return mcp


# ---------------------------------------------------------------------------
# Tool registrations
# ---------------------------------------------------------------------------


def _register_read_only_tools(mcp: FastMCP, gate: Gate) -> None:
    """Register the five always-on tools.

    All read-only in the agent-host sense: they do not mutate Neo4j.
    ``validate_cypher``/``estimate_cypher_cost``/``gate_cypher``/
    ``correct_cypher`` are marked ``openWorldHint=True`` because they
    depend on external Neo4j state (mirror or production EXPLAIN
    plans). ``get_schema`` is ``openWorldHint=False`` — it returns
    the cached schema the gate is currently using.
    """

    @mcp.tool(
        name=TOOL_NAMES["validate"],
        description=(
            "Run CYGNET's structural validator chain against a Cypher "
            "query. Returns the StructuralValidatorResult shape: "
            "passed/failed_stage/error_payload."
        ),
        annotations=ToolAnnotations(
            title="Validate Cypher",
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=True,
        ),
    )
    def validate_cypher(query: str) -> dict[str, Any]:
        try:
            result = gate.validate(query)
        except Exception as exc:  # pragma: no cover - safety net
            raise ToolError(_friendly_error("validate_cypher", exc)) from exc
        return _to_dict(result)

    @mcp.tool(
        name=TOOL_NAMES["estimate"],
        description=(
            "Run CYGNET's cost gate against a Cypher query. Returns "
            "the CostGateResult shape including estimated rows, "
            "db-hits proxy, cost driver, and the top-5 operator "
            "breakdown. Raises if the cost gate is disabled in the "
            "active GateConfig."
        ),
        annotations=ToolAnnotations(
            title="Estimate Cypher Cost",
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=True,
        ),
    )
    def estimate_cypher_cost(query: str) -> dict[str, Any]:
        try:
            result = gate.estimate_cost(query)
        except ValueError as exc:
            # Cost gate disabled -> friendly ToolError instead of
            # leaking the contract-violation ValueError.
            raise ToolError(str(exc)) from exc
        except Exception as exc:  # pragma: no cover - safety net
            raise ToolError(_friendly_error("estimate_cypher_cost", exc)) from exc
        return _to_dict(result)

    @mcp.tool(
        name=TOOL_NAMES["gate"],
        description=(
            "Run the full CYGNET pipeline (structural validation then "
            "cost gating, with structural failures short-circuiting). "
            "Returns the GateResult shape: passed/structural/cost/errors."
        ),
        annotations=ToolAnnotations(
            title="Gate Cypher",
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=True,
        ),
    )
    def gate_cypher(query: str) -> dict[str, Any]:
        try:
            result = gate.gate(query)
        except Exception as exc:  # pragma: no cover - safety net
            raise ToolError(_friendly_error("gate_cypher", exc)) from exc
        return _to_dict(result)

    @mcp.tool(
        name=TOOL_NAMES["schema"],
        description=(
            "Return the currently loaded Schema (labels, "
            "relationship types, properties, constraints, indexes). "
            "Does not hit Neo4j — returns the gate's cached schema."
        ),
        annotations=ToolAnnotations(
            title="Get Schema",
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=False,
        ),
    )
    def get_schema() -> dict[str, Any]:
        return _to_dict(gate.get_schema())

    @mcp.tool(
        name=TOOL_NAMES["correct"],
        description=(
            "Invoke the configured Corrector on a failing query and "
            "its GateError. Returns the CorrectorResult shape: "
            "action/refined_query/reasoning/attempts_used. The "
            "default corrector (NullCorrector) always aborts; "
            "RampartCorrector returns a refined query when an LLM "
            "client is wired in."
        ),
        annotations=ToolAnnotations(
            title="Correct Cypher",
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=True,
        ),
    )
    def correct_cypher(
        query: str,
        error: dict[str, Any],
        attempt_number: int = 1,
    ) -> dict[str, Any]:
        try:
            gate_error = GateError.model_validate(error)
        except ValidationError as exc:
            raise ToolError(
                f"correct_cypher: 'error' argument does not match the GateError shape: {exc}"
            ) from exc
        # v0.0.31: Gate.correct() now runs the outer refinement loop
        # internally and returns RefinementResult. The transport's
        # ``attempt_number`` kwarg is retained on the request shape
        # for backwards compatibility but no longer affects the
        # library call (the loop owns attempt accounting).
        del attempt_number
        try:
            result = gate.correct(query, gate_error)
        except Exception as exc:  # pragma: no cover - safety net
            raise ToolError(_friendly_error("correct_cypher", exc)) from exc
        return _to_dict(result)


def _register_read_write_tools(mcp: FastMCP, gate: Gate) -> None:
    """Register the read-write tools (currently just ``refresh_schema``).

    ``refresh_schema`` is the only tool that mutates the Gate's
    state (it swaps the loaded :class:`Schema` and rebuilds the
    validator chain + cost gate). It does not write to Neo4j.
    """

    @mcp.tool(
        name=TOOL_NAMES["refresh"],
        description=(
            "Reload the schema from its configured source: "
            "spec_file rereads the YAML/JSON file; introspect "
            "re-runs db.schema introspection against the production "
            "Neo4j. spec_object is not refreshable and surfaces a "
            "structured error. Returns the new Schema shape."
        ),
        annotations=ToolAnnotations(
            title="Refresh Schema",
            readOnlyHint=False,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=True,
        ),
    )
    def refresh_schema() -> dict[str, Any]:
        try:
            new_schema = gate.refresh_schema()
        except ValueError as exc:
            # spec_object source raises here; surface as a structured
            # tool error rather than letting the exception bubble.
            raise ToolError(str(exc)) from exc
        except Exception as exc:  # pragma: no cover - safety net
            raise ToolError(_friendly_error("refresh_schema", exc)) from exc
        return _to_dict(new_schema)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_dict(model: Any) -> dict[str, Any]:
    """Pydantic model -> JSON-shaped dict.

    ``mode="json"`` ensures datetime / tuple / Path / etc. fields
    serialise to JSON-native primitives so the result is safe to
    return directly as an MCP tool payload.
    """
    dumped = model.model_dump(mode="json")
    if not isinstance(dumped, dict):
        # Defensive: every model we return is a BaseModel, so
        # model_dump always produces a dict. Wrap if a future
        # caller passes through a list-shaped payload.
        return {"value": dumped}
    return dumped


def _friendly_error(tool_name: str, exc: Exception) -> str:
    """Render an exception as a one-line ToolError message."""
    return f"{tool_name}: {type(exc).__name__}: {exc}"


def _format_json(payload: dict[str, Any]) -> str:
    """Pretty-print a payload to JSON. Used by the CLI launcher's
    health/debug log lines; the tool functions themselves return
    dicts and rely on FastMCP for JSON-encoding the response."""
    return json.dumps(payload, indent=2, default=str)
