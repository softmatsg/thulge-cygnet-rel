# Copyright (c) 2026 Nikodem Tomczak, Thulge Labs. All rights reserved.

"""``cygnet-mcp`` CLI: launch the CYGNET MCP server.

Wired up as a script entry point in ``pyproject.toml``:

    cygnet-mcp --config path/to/gate.yaml [--mode read_only|read_write] [--transport stdio|http]

The launcher constructs a :class:`Gate` from the YAML config (via
:meth:`GateConfig.from_yaml`), builds an MCP server bound to it, and
runs the FastMCP server loop. The Gate is closed by the server's
lifespan on shutdown.

Default transport is ``stdio`` — what Claude Desktop, Cursor, and
similar hosts speak. ``http`` selects FastMCP's HTTP transport
(host/port configurable via ``--host`` and ``--port``); useful for
hosted agent frameworks.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from cygnet import Gate, GateConfig
from cygnet.transports.mcp_server import create_mcp_server

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["build_parser", "main"]


logger = logging.getLogger("cygnet.cli.serve_mcp")


def build_parser() -> argparse.ArgumentParser:
    """Build the ``cygnet-mcp`` argument parser. Public for tests."""
    parser = argparse.ArgumentParser(
        prog="cygnet-mcp",
        description=(
            "Launch the CYGNET MCP server. Configure your MCP host "
            "(Claude Desktop, Cursor, ...) to spawn this command; the "
            "validate/estimate/gate/schema/correct tools become "
            "available in the agent's toolset."
        ),
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to the YAML GateConfig file.",
    )
    parser.add_argument(
        "--mode",
        choices=["read_only", "read_write"],
        default="read_write",
        help=(
            "Server mode. 'read_only' hides the refresh_schema tool "
            "from registration (so the LLM cannot see or call it). "
            "Default: read_write."
        ),
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "http"],
        default="stdio",
        help=(
            "MCP transport. 'stdio' (default) is what Claude Desktop "
            "and Cursor speak; 'http' selects FastMCP's HTTP transport."
        ),
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="HTTP transport bind host. Default 127.0.0.1.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="HTTP transport bind port. Default 8765.",
    )
    parser.add_argument(
        "--server-name",
        default="cygnet",
        help="MCP server identifier surfaced to clients. Default 'cygnet'.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Python log level for cygnet.* loggers. Default INFO.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """``cygnet-mcp`` entry point.

    Returns a Unix-style exit code: 0 on graceful shutdown, non-zero
    on configuration or startup failure. The FastMCP server loop is
    blocking; this function returns only on shutdown (Ctrl-C, EOF on
    stdin, or HTTP server stop).
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )

    try:
        config = GateConfig.from_yaml(args.config)
    except FileNotFoundError:
        logger.error("Config file not found: %s", args.config)
        return 2
    except Exception:
        logger.exception("Failed to load config from %s", args.config)
        return 2

    try:
        gate = Gate.from_config(config)
    except Exception:
        logger.exception("Failed to construct Gate from config")
        return 3

    try:
        server = create_mcp_server(
            gate,
            mode=args.mode,
            server_name=args.server_name,
            own_gate=True,
        )
    except Exception:
        logger.exception("Failed to construct MCP server")
        gate.close()
        return 4

    logger.info(
        "Starting CYGNET MCP server (transport=%s, mode=%s, name=%s)",
        args.transport,
        args.mode,
        args.server_name,
    )

    try:
        if args.transport == "stdio":
            server.run(transport="stdio")
        else:
            server.run(transport="http", host=args.host, port=args.port)
    except KeyboardInterrupt:
        logger.info("Received Ctrl-C; shutting down.")
        return 0
    except Exception:
        logger.exception("MCP server crashed")
        return 5
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
