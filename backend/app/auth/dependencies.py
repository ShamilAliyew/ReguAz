"""FastAPI dependencies for auth database sessions and current users."""

from __future__ import annotations

from collections.abc import Generator

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from backend.app.auth.database import AuthDatabase, UserRecord
from backend.app.auth.service import AuthService
from backend.app.core.config import Settings, get_settings


def get_auth_database(request: Request) -> AuthDatabase:
    database = getattr(request.app.state.reguaz, "auth_database", None)
    if database is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "auth_unavailable",
                "message": "Authentication database is not configured.",
            },
        )
    return database


def get_auth_session(
    database: AuthDatabase = Depends(get_auth_database),
) -> Generator[Session, None, None]:
    with database.session() as session:
        yield session


def require_authenticated_user(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> UserRecord | None:
    """Require a valid session when auth is enabled; preserve local test mode."""

    if not settings.AUTH_ENABLED:
        return None
    database = get_auth_database(request)
    token = request.cookies.get(settings.AUTH_COOKIE_NAME, "")
    if not token:
        raise _unauthorized()
    with database.session() as session:
        user = AuthService(session, settings.AUTH_SESSION_TTL_DAYS).authenticate(token)
    if user is None:
        raise _unauthorized()
    return user


def _unauthorized() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail={
            "error": "authentication_required",
            "message": "Davam etmək üçün hesabınıza daxil olun.",
        },
    )
