# Copyright (c) 2026 Nikodem Tomczak, Thulge Labs. All rights reserved.

"""Transport surfaces: MCP server and FastAPI/HTTP server.

The Python API (the ``Gate`` object) is the primary surface; MCP is a
thin wrapper that delegates to the same ``Gate`` instance. The MCP
transport is optional and gated behind the ``[mcp]`` extra.

Public surface (lazy-loaded when fastmcp is not installed):

- :func:`create_mcp_server` — factory returning a configured
  ``fastmcp.FastMCP`` bound to a ``Gate``.
- :data:`TOOL_NAMES` — tool-name registry shared by the server and
  its tests. Treat names as immutable per the brief.
- :data:`MCPMode` — ``Literal["read_only", "read_write"]`` controlling
  which tools are registered.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from cygnet.transports.http_server import HTTPMode, create_http_app
    from cygnet.transports.mcp_server import MCPMode, create_mcp_server

__all__ = [
    "API_PREFIX",
    "REQUEST_ID_HEADER",
    "TOOL_NAMES",
    "HTTPMode",
    "MCPMode",
    "create_http_app",
    "create_mcp_server",
]


_MCP_NAMES = {"create_mcp_server", "MCPMode", "TOOL_NAMES"}
_HTTP_NAMES = {"create_http_app", "HTTPMode", "API_PREFIX", "REQUEST_ID_HEADER"}


def __getattr__(name: str) -> object:  # pragma: no cover - thin lazy proxy
    """Lazy access to the transport modules.

    Importing :mod:`cygnet.transports` must not fail when the
    ``[mcp]`` or ``[http]`` extras are absent; the underlying
    modules' ``fastmcp`` / ``fastapi`` imports would otherwise blow
    up at top-level. ``ImportError`` surfaces only when a consumer
    references one of these names.
    """
    if name in _MCP_NAMES:
        from cygnet.transports import mcp_server

        return getattr(mcp_server, name)
    if name in _HTTP_NAMES:
        from cygnet.transports import http_server

        return getattr(http_server, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
