# Copyright (c) 2026 Nikodem Tomczak, Thulge Labs. All rights reserved.

"""``cygnet-http`` CLI: launch the CYGNET HTTP server.

Wired up as a script entry point in ``pyproject.toml``:

    cygnet-http --config path/to/gate.yaml [--mode read_only|read_write]
                [--host 127.0.0.1] [--port 8080] [--workers 1]
                [--cors-origin ORIGIN]... [--log-level INFO]

The launcher constructs a :class:`Gate` from the YAML config, builds
the FastAPI app bound to it, and hands off to ``uvicorn`` to serve.
The Gate is closed by the app's lifespan on shutdown.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from cygnet import Gate, GateConfig
from cygnet.transports.http_server import create_http_app

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["build_parser", "main"]


logger = logging.getLogger("cygnet.cli.serve_http")


def build_parser() -> argparse.ArgumentParser:
    """Build the ``cygnet-http`` argument parser. Public for tests."""
    parser = argparse.ArgumentParser(
        prog="cygnet-http",
        description=(
            "Launch the CYGNET HTTP server. Exposes the validator, "
            "cost gate, schema, and corrector as REST endpoints under "
            "/api/v1/. Auth is not built in — deploy behind a reverse "
            "proxy or API gateway for production."
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
            "Server mode. 'read_only' omits the /api/v1/refresh-schema "
            "endpoint so it returns 404. Default: read_write."
        ),
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind host. Default 127.0.0.1 (localhost only).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Bind port. Default 8080.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Number of uvicorn worker processes. Default 1. "
            "Set >1 for production; uvicorn forks; each worker holds "
            "its own Gate instance."
        ),
    )
    parser.add_argument(
        "--cors-origin",
        action="append",
        default=[],
        metavar="ORIGIN",
        help=(
            "Allowed CORS origin. Repeat the flag for multiple "
            "origins. Default: empty (CORS disabled). Example: "
            "--cors-origin https://app.example.com"
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Python log level for cygnet.* loggers. Default INFO.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """``cygnet-http`` entry point.

    Returns a Unix-style exit code: 0 on graceful shutdown, non-zero
    on config / Gate / server failure. The uvicorn server loop is
    blocking; this function returns only on shutdown.
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
        app = create_http_app(
            gate,
            mode=args.mode,
            own_gate=True,
            cors_origins=tuple(args.cors_origin),
        )
    except Exception:
        logger.exception("Failed to construct HTTP app")
        gate.close()
        return 4

    logger.info(
        "Starting CYGNET HTTP server (host=%s, port=%d, mode=%s, workers=%d)",
        args.host,
        args.port,
        args.mode,
        args.workers,
    )

    try:
        import uvicorn

        # Workers >1 requires an import string, not an app instance,
        # because uvicorn forks and each worker re-imports the app.
        # The single-worker path takes the app object directly so the
        # Gate is constructed once in the parent process.
        if args.workers > 1:
            # When workers > 1, we can't use the in-memory `app` —
            # uvicorn needs to re-create it in each worker. Surface
            # this clearly rather than silently dropping to 1.
            logger.error(
                "--workers > 1 requires a process-launcher recipe. "
                "Use a process manager (systemd, supervisord) or run "
                "multiple `cygnet-http --workers 1` instances behind "
                "a load balancer instead."
            )
            return 4
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            log_level=args.log_level.lower(),
        )
    except KeyboardInterrupt:
        logger.info("Received Ctrl-C; shutting down.")
        return 0
    except Exception:
        logger.exception("HTTP server crashed")
        return 5
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
