"""FastAPI application and safe HTTP process boundary."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import RequestResponseEndpoint

from app.api.routes.health import router as health_router
from app.api.routes.reviews import router as reviews_router
from app.api.schemas import ErrorBody, ErrorResponse
from app.bootstrap import application_container
from app.config import Settings
from app.errors import EvidenceFlowError

logger = logging.getLogger(__name__)
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


def _request_id(request: Request) -> str:
    return str(getattr(request.state, "request_id", f"request_{uuid4().hex}"))


def _error_response(
    *,
    status_code: int,
    code: str,
    message: str,
    request_id: str,
    details: object = None,
) -> JSONResponse:
    payload = ErrorResponse(
        error=ErrorBody(
            code=code,
            message=message,
            details=details,
            request_id=request_id,
        )
    )
    return JSONResponse(
        status_code=status_code,
        content=payload.model_dump(mode="json"),
        headers={
            "X-Request-ID": request_id,
            "X-Content-Type-Options": "nosniff",
        },
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        async with application_container(settings) as container:
            app.state.container = container
            yield

    app = FastAPI(
        title="EvidenceFlow",
        version="1.0.0",
        description="Local, synthetic business-document review workflow.",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def request_context(
        request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        request.state.request_id = f"request_{uuid4().hex}"
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @app.exception_handler(EvidenceFlowError)
    async def evidenceflow_error(
        request: Request, error: EvidenceFlowError
    ) -> JSONResponse:
        return _error_response(
            status_code=error.status_code,
            code=error.code,
            message=error.message,
            details=error.details,
            request_id=_request_id(request),
        )

    @app.exception_handler(RequestValidationError)
    async def request_validation_error(
        request: Request, error: RequestValidationError
    ) -> JSONResponse:
        details = [
            {
                "location": [str(item) for item in issue.get("loc", ())],
                "message": issue.get("msg", "Invalid value"),
                "type": issue.get("type", "validation_error"),
            }
            for issue in error.errors()
        ]
        return _error_response(
            status_code=422,
            code="request_validation_error",
            message="The request did not match the expected schema.",
            details=details,
            request_id=_request_id(request),
        )

    @app.exception_handler(Exception)
    async def unexpected_error(request: Request, error: Exception) -> JSONResponse:
        container = getattr(request.app.state, "container", None)
        log_sensitive = bool(
            getattr(getattr(container, "settings", None), "log_sensitive_content", False)
        )
        if log_sensitive:
            logger.exception("Unhandled API error", exc_info=error)
        else:
            logger.error("Unhandled API error (error_type=%s)", type(error).__name__)
        return _error_response(
            status_code=500,
            code="internal_error",
            message="The request could not be completed.",
            request_id=_request_id(request),
        )

    app.include_router(health_router)
    app.include_router(reviews_router)

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(FRONTEND_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
    return app


app = create_app()
