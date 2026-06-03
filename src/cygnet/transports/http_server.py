# Copyright (c) 2026 Nikodem Tomczak, Thulge Labs. All rights reserved.

"""FastAPI HTTP transport for CYGNET.

Same Gate-delegation pattern as :mod:`cygnet.transports.mcp_server`,
exposed over HTTP for language-agnostic consumers and deployment
shapes that need a long-running networked server.

**Endpoint surface** (treat URL paths + request-body field names as
the immutable integration contract; renames break user
configurations):

- ``POST /api/v1/validate``        -> ``StructuralValidatorResult``
- ``POST /api/v1/estimate-cost``   -> ``CostGateResult``
- ``POST /api/v1/gate``            -> ``GateResult``
- ``GET  /api/v1/schema``          -> ``Schema``
- ``POST /api/v1/correct``         -> ``CorrectorResult``
- ``POST /api/v1/refresh-schema``  -> ``Schema`` (mode-gated; only
  registered when ``mode == "read_write"``)
- ``GET  /api/v1/health``          -> liveness ping
- ``GET  /api/v1/version``         -> server build + schema info

**Mode gating.** ``create_http_app(gate, mode="read_only")`` omits
the ``refresh-schema`` endpoint from registration. The mirrored
behaviour from the MCP server: in ``read_only`` mode the LLM /
client never sees the endpoint, so a POST to ``/api/v1/refresh-
schema`` returns FastAPI's default 404 Not Found rather than a
runtime rejection.

**No authentication.** Auth lives in front of this transport (reverse
proxy, API gateway, custom middleware). The stdio MCP transport is
authenticated by the process boundary; HTTP is not, and the brief
is explicit that deployments add their own auth layer. See
``docs/http_integration.md`` for deployment recipes.

**Error responses** are uniform across the API:

.. code-block:: json

   {"error_type": "<class name>", "message": "<human-readable>"}

with status codes 400 (library-side configuration / spec errors),
422 (request validation; FastAPI default), 503 (Neo4j driver
failures), or 500 (uncaught — generic message, never a stack
trace). Every response (success or error) carries an
``X-Request-Id`` header for log correlation; the client may set
its own value or the middleware generates a UUID4.
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, Literal

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from neo4j.exceptions import Neo4jError, ServiceUnavailable
from pydantic import BaseModel, Field, ValidationError

import cygnet
from cygnet.models import GateError
from cygnet.schema import SchemaIntrospectionError, SchemaSpecError

_CYGNET_VERSION = cygnet.__version__

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from cygnet import Gate

__all__ = [
    "API_PREFIX",
    "REQUEST_ID_HEADER",
    "CorrectRequest",
    "EstimateCostRequest",
    "GateRequest",
    "HTTPMode",
    "ValidateRequest",
    "create_http_app",
]


HTTPMode = Literal["read_only", "read_write"]
"""Server mode. ``"read_only"`` omits ``refresh-schema``."""


API_PREFIX: str = "/api/v1"
"""URL prefix for every endpoint. Future breaking changes get
``/api/v2`` — this prefix is treated as part of the public contract."""

REQUEST_ID_HEADER: str = "X-Request-Id"
"""Request-id header echoed back in every response (success or
error). Clients can set their own; the middleware fills in a UUID4
when absent."""


logger = logging.getLogger("cygnet.http")


# ---------------------------------------------------------------------------
# Request models
#
# Pydantic models so FastAPI auto-generates the OpenAPI schema with
# field descriptions. Field names are part of the public contract.
# ---------------------------------------------------------------------------


class ValidateRequest(BaseModel):
    """Body for ``POST /api/v1/validate``."""

    query: str = Field(..., description="Cypher query to validate.")


class EstimateCostRequest(BaseModel):
    """Body for ``POST /api/v1/estimate-cost``."""

    query: str = Field(..., description="Cypher query to cost-estimate via EXPLAIN.")


class GateRequest(BaseModel):
    """Body for ``POST /api/v1/gate``."""

    query: str = Field(..., description="Cypher query to run the full gate pipeline against.")


class CorrectRequest(BaseModel):
    """Body for ``POST /api/v1/correct``."""

    query: str = Field(..., description="The failing query to refine.")
    error: dict[str, Any] = Field(
        ...,
        description=(
            "The GateError that the query failed with, as a dict "
            "(typically `errors[0]` from a prior /gate response)."
        ),
    )
    attempt_number: int = Field(
        default=1,
        ge=1,
        description="1-based attempt index for the refinement loop.",
    )


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------


def create_http_app(
    gate: Gate,
    *,
    mode: HTTPMode = "read_write",
    own_gate: bool = False,
    cors_origins: Sequence[str] = (),
    app_title: str = "CYGNET HTTP API",
) -> FastAPI:
    """Build a FastAPI app bound to a :class:`Gate`.

    Args:
        gate: an already-constructed :class:`Gate`. Every endpoint
            delegates to this instance.
        mode: ``"read_only"`` omits the ``refresh-schema`` route;
            ``"read_write"`` registers all eight endpoints.
        own_gate: when ``True``, the app's lifespan closes the
            ``Gate`` on shutdown. Set ``True`` from the CLI launcher
            where the server owns the Gate's lifecycle; leave
            ``False`` (default) for embedded / test use.
        cors_origins: allow-list of CORS origins. Default empty
            (no cross-origin access). Pass a non-empty list to
            opt in.
        app_title: passed to ``FastAPI(title=...)``. Surfaces in the
            auto-generated OpenAPI document.
    """

    @asynccontextmanager
    async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            if own_gate:
                gate.close()

    app = FastAPI(
        title=app_title,
        version=_CYGNET_VERSION,
        description=(
            "CYGNET HTTP API. Validate, cost-estimate, and refine "
            "LLM-generated Cypher queries before they execute against "
            "Neo4j. All endpoints under /api/v1; request and response "
            "bodies are JSON. See docs/http_integration.md for "
            "deployment recipes and known limitations."
        ),
        lifespan=_lifespan,
    )

    _install_middleware(app, cors_origins=tuple(cors_origins))
    _install_exception_handlers(app)
    _register_routes(app, gate, mode=mode)

    return app


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------


def _install_middleware(app: FastAPI, *, cors_origins: tuple[str, ...]) -> None:
    """Register request-id + logging middleware; conditional CORS.

    Middleware ordering follows FastAPI semantics: the last
    ``add_middleware`` call wraps everything else. We add CORS first
    (innermost) so it only sees the rest of the stack's output, and
    request-id / logging last (outermost) so every response — error
    or success — gets a request id and is logged.
    """

    if cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(cors_origins),
            allow_credentials=False,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["*", REQUEST_ID_HEADER],
            expose_headers=[REQUEST_ID_HEADER],
        )

    @app.middleware("http")
    async def _request_id_and_log(request: Request, call_next: Any) -> Response:
        # Generate / forward the request id BEFORE call_next so the
        # registered exception handlers (which sit inside FastAPI's
        # ExceptionMiddleware, one layer in from here) can read it
        # off ``request.state`` to populate the X-Request-Id header
        # on error responses.
        #
        # No try/except here. Starlette dispatches per-exception
        # handlers inside ExceptionMiddleware; catching here and
        # re-raising would route the exception past that layer to
        # ServerErrorMiddleware (the outermost), which produces a
        # generic 500 and bypasses our registered handlers.
        request_id = request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex
        request.state.request_id = request_id
        start = time.perf_counter()
        response: Response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "request method=%s path=%s status=%d duration_ms=%.1f request_id=%s",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
            request_id,
        )
        response.headers[REQUEST_ID_HEADER] = request_id
        return response


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------


def _install_exception_handlers(app: FastAPI) -> None:
    """Map library and driver exceptions to structured JSON responses."""

    def _err(status: int, exc: Exception, request: Request) -> JSONResponse:
        body = {
            "error_type": type(exc).__name__,
            "message": str(exc),
        }
        request_id = getattr(request.state, "request_id", None)
        headers = {REQUEST_ID_HEADER: request_id} if request_id else {}
        return JSONResponse(status_code=status, content=body, headers=headers)

    @app.exception_handler(HTTPException)
    async def _http_exception(request: Request, exc: HTTPException) -> JSONResponse:
        # FastAPI's default handler returns {"detail": ...}. Override
        # to keep the uniform {"error_type": ..., "message": ...}
        # contract across the whole API.
        body = {"error_type": "HTTPException", "message": exc.detail}
        request_id = getattr(request.state, "request_id", None)
        headers = dict(exc.headers or {})
        if request_id:
            headers[REQUEST_ID_HEADER] = request_id
        return JSONResponse(status_code=exc.status_code, content=body, headers=headers)

    @app.exception_handler(SchemaSpecError)
    async def _schema_spec_error(request: Request, exc: SchemaSpecError) -> JSONResponse:
        return _err(400, exc, request)

    @app.exception_handler(SchemaIntrospectionError)
    async def _schema_introspection_error(
        request: Request, exc: SchemaIntrospectionError
    ) -> JSONResponse:
        return _err(400, exc, request)

    @app.exception_handler(ValidationError)
    async def _validation_error(request: Request, exc: ValidationError) -> JSONResponse:
        # Pydantic validation errors on malformed request bodies.
        # FastAPI maps RequestValidationError to 422 by default;
        # this catches stray ValidationError raised by handler code
        # (e.g. constructing a GateError from a malformed payload).
        return _err(422, exc, request)

    @app.exception_handler(ServiceUnavailable)
    async def _service_unavailable(request: Request, exc: ServiceUnavailable) -> JSONResponse:
        return _err(503, exc, request)

    @app.exception_handler(Neo4jError)
    async def _neo4j_error(request: Request, exc: Neo4jError) -> JSONResponse:
        return _err(503, exc, request)

    @app.exception_handler(Exception)
    async def _generic_error(request: Request, exc: Exception) -> JSONResponse:
        # Defensive fallthrough for anything that isn't a typed
        # library / driver / HTTP exception above. Never leak stack
        # traces or internal state to the client.
        request_id = getattr(request.state, "request_id", None)
        headers = {REQUEST_ID_HEADER: request_id} if request_id else {}
        logger.exception(
            "uncaught_exception path=%s request_id=%s",
            request.url.path,
            request_id,
        )
        return JSONResponse(
            status_code=500,
            content={
                "error_type": "InternalServerError",
                "message": "An unexpected error occurred. See server logs.",
            },
            headers=headers,
        )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


def _register_routes(app: FastAPI, gate: Gate, *, mode: HTTPMode) -> None:
    """Register every endpoint. Mode-gated routes go last so the
    handler functions don't leak into the read-only scope."""

    @app.get(f"{API_PREFIX}/health", tags=["meta"])
    async def health() -> dict[str, str]:
        """Liveness ping. Always available regardless of mode."""
        return {"status": "ok", "version": _CYGNET_VERSION}

    @app.get(f"{API_PREFIX}/version", tags=["meta"])
    async def version() -> dict[str, Any]:
        """Server build + loaded-schema metadata. Does not expose
        Neo4j credentials or schema contents."""
        return {
            "cygnet_version": _CYGNET_VERSION,
            "schema_loaded": gate.get_schema() is not None,
            "validator_backends": list(gate.config.validator.backends),
        }

    @app.post(f"{API_PREFIX}/validate", tags=["gate"])
    async def validate(body: ValidateRequest) -> dict[str, Any]:
        result = gate.validate(body.query)
        return _to_dict(result)

    @app.post(f"{API_PREFIX}/estimate-cost", tags=["gate"])
    async def estimate_cost(body: EstimateCostRequest) -> dict[str, Any]:
        try:
            result = gate.estimate_cost(body.query)
        except ValueError as exc:
            # Cost gate disabled -> 400 with a clean message.
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return _to_dict(result)

    @app.post(f"{API_PREFIX}/gate", tags=["gate"])
    async def gate_endpoint(body: GateRequest) -> dict[str, Any]:
        result = gate.gate(body.query)
        return _to_dict(result)

    @app.get(f"{API_PREFIX}/schema", tags=["schema"])
    async def get_schema() -> dict[str, Any]:
        return _to_dict(gate.get_schema())

    @app.post(f"{API_PREFIX}/correct", tags=["corrector"])
    async def correct(body: CorrectRequest) -> dict[str, Any]:
        try:
            gate_error = GateError.model_validate(body.error)
        except ValidationError as exc:
            raise HTTPException(
                status_code=422,
                detail=f"'error' does not match the GateError shape: {exc}",
            ) from exc
        # Gate.correct() owns the outer refinement loop.
        # ``attempt_number`` is kept on the request body for
        # compatibility but no longer threaded into the library call.
        _ = body.attempt_number
        result = gate.correct(body.query, gate_error)
        return _to_dict(result)

    if mode == "read_write":

        @app.post(f"{API_PREFIX}/refresh-schema", tags=["schema"])
        async def refresh_schema() -> dict[str, Any]:
            try:
                new_schema = gate.refresh_schema()
            except ValueError as exc:
                # spec_object source raises here.
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            return _to_dict(new_schema)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_dict(model: Any) -> dict[str, Any]:
    """Pydantic model -> JSON-shaped dict.

    ``mode="json"`` ensures datetime / tuple / Path / etc. fields
    serialise to JSON-native primitives.
    """
    dumped = model.model_dump(mode="json")
    if not isinstance(dumped, dict):
        return {"value": dumped}
    return dumped
