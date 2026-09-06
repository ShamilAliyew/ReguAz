"""
app/main.py

Main entry point for the ReguAZ FastAPI application.
Handles application startup configuration, router registration, CORS setup, middleware,
and centralized exception handling for production-ready frontend integration.
"""

from __future__ import annotations

import sys
from pathlib import Path

# pyrefly: ignore [missing-import]
from fastapi import FastAPI, Request, status

# pyrefly: ignore [missing-import]
from fastapi.exceptions import RequestValidationError

# pyrefly: ignore [missing-import]
from fastapi.middleware.cors import CORSMiddleware

# pyrefly: ignore [missing-import]
from fastapi.responses import JSONResponse

# pyrefly: ignore [missing-import]
from starlette.exceptions import HTTPException as StarletteHTTPException

# Add backend's parent directory to sys.path to allow backend package imports
_backend_parent = str(Path(__file__).resolve().parent.parent.parent)
if _backend_parent not in sys.path:
    sys.path.insert(0, _backend_parent)

from backend.app.api.chat import router as chat_router  # noqa: E402
from backend.app.api.auth import router as auth_router  # noqa: E402
from backend.app.api.documents import router as documents_router  # noqa: E402
from backend.app.api.health import router as health_router  # noqa: E402
from backend.app.api.speech import router as speech_router  # noqa: E402
from backend.app.core.config import get_settings  # noqa: E402
from backend.app.core.lifespan import lifespan  # noqa: E402
from backend.reguaz.utils.logger import get_logger  # noqa: E402

# Initialize project-wide logger for API errors
logger = get_logger("app.api.errors", "api_errors.log")


def create_app() -> FastAPI:
    """
    Application factory to create and configure the FastAPI application.
    """
    settings = get_settings()

    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description=settings.APP_DESCRIPTION,
        debug=settings.DEBUG,
        lifespan=lifespan,
    )

    # ── CORS Middleware ────────────────────────────────────────────────────────
    # Setup CORS middleware to permit browser access from configured origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
        allow_methods=settings.CORS_ALLOW_METHODS,
        allow_headers=settings.CORS_ALLOW_HEADERS,
        expose_headers=settings.CORS_EXPOSE_HEADERS,
    )

    # ── Routers ───────────────────────────────────────────────────────────────
    # Mount core API endpoints
    app.include_router(auth_router)
    app.include_router(chat_router)
    app.include_router(documents_router)
    app.include_router(health_router)
    app.include_router(speech_router)

    # A simple root status ping
    @app.get("/", tags=["status"], summary="Root status check")
    def read_root() -> dict[str, str]:
        """
        Fast ping check representing app package online status.
        """
        return {
            "name": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "status": "online",
        }

    # ── Global Error Handling ──────────────────────────────────────────────────

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        """
        Catches Starlette/FastAPI HTTPExceptions (e.g. 404, 400, 503).
        Returns error payloads conforming to the consistent schema.
        """
        code = "http_error"
        message = str(exc.detail)

        # Standardize specific codes if detail is structured
        if isinstance(exc.detail, dict):
            code = exc.detail.get("error", code)
            message = exc.detail.get("message", message)
        elif exc.status_code == status.HTTP_404_NOT_FOUND:
            code = "not_found"
        elif exc.status_code == status.HTTP_503_SERVICE_UNAVAILABLE:
            code = "service_unavailable"
        elif exc.status_code == status.HTTP_400_BAD_REQUEST:
            code = "bad_request"

        return JSONResponse(
            status_code=exc.status_code,
            headers=exc.headers,
            content={
                "success": False,
                "error": {
                    "code": code,
                    "message": message,
                },
            },
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """
        Handles Pydantic input validation failures (422 Unprocessable Entity).
        Formats field errors to prevent exposing internal stack traces.
        """
        errors = exc.errors()
        messages = []
        for error in errors:
            loc = ".".join(str(p) for p in error.get("loc", []))
            msg = error.get("msg", "Unknown validation error")
            messages.append(f"{loc}: {msg}")

        friendly_msg = "Validation failed: " + "; ".join(messages)

        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "success": False,
                "error": {
                    "code": "validation_error",
                    "message": friendly_msg,
                },
            },
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        """
        Catch-all handler for unhandled 500 errors.
        Prevents raw python exceptions and trace paths leaking to the frontend.
        """
        logger.exception(
            "An unhandled exception occurred during request execution: %s", exc
        )

        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "error": {
                    "code": "internal_server_error",
                    "message": "An unexpected server error occurred. Please try again later.",
                },
            },
        )

    return app


app = create_app()
