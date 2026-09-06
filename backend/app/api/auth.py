"""Registration, login, current-user and logout endpoints."""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from sqlalchemy.orm import Session

from backend.app.auth.database import UserRecord
from backend.app.auth.dependencies import get_auth_session, require_authenticated_user
from backend.app.auth.service import (
    AuthService,
    EmailAlreadyRegisteredError,
    InvalidCredentialsError,
)
from backend.app.core.config import Settings, get_settings
from backend.app.schemas.auth import (
    AuthResponse,
    LoginRequest,
    RegisterRequest,
    UserResponse,
)


router = APIRouter(prefix="/auth", tags=["authentication"])


def _client_ip(request: Request) -> str | None:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",", 1)[0].strip()
    return request.client.host if request.client else None


def _set_session_cookie(
    response: Response,
    token: str,
    expires_at: datetime,
    settings: Settings,
) -> None:
    max_age = max(0, int((expires_at - datetime.now(UTC)).total_seconds()))
    response.set_cookie(
        key=settings.AUTH_COOKIE_NAME,
        value=token,
        max_age=max_age,
        expires=expires_at,
        path="/",
        secure=settings.AUTH_COOKIE_SECURE,
        httponly=True,
        samesite="lax",
    )


def _ensure_enabled(settings: Settings) -> None:
    if not settings.AUTH_ENABLED:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "auth_disabled",
                "message": "Authentication is disabled in this environment.",
            },
        )


@router.post("/register", response_model=AuthResponse, status_code=status.HTTP_201_CREATED)
def register(
    payload: RegisterRequest,
    request: Request,
    response: Response,
    session: Session = Depends(get_auth_session),
    settings: Settings = Depends(get_settings),
) -> AuthResponse:
    _ensure_enabled(settings)
    try:
        authenticated = AuthService(session, settings.AUTH_SESSION_TTL_DAYS).register(
            name=payload.name,
            email=payload.email,
            password=payload.password,
            user_agent=request.headers.get("user-agent"),
            ip_address=_client_ip(request),
        )
    except EmailAlreadyRegisteredError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "error": "email_already_registered",
                "message": "Bu e-poçt ünvanı ilə artıq hesab mövcuddur.",
            },
        ) from exc
    _set_session_cookie(response, authenticated.token, authenticated.expires_at, settings)
    return AuthResponse(user=UserResponse.model_validate(authenticated.user))


@router.post("/login", response_model=AuthResponse)
def login(
    payload: LoginRequest,
    request: Request,
    response: Response,
    session: Session = Depends(get_auth_session),
    settings: Settings = Depends(get_settings),
) -> AuthResponse:
    _ensure_enabled(settings)
    try:
        authenticated = AuthService(session, settings.AUTH_SESSION_TTL_DAYS).login(
            email=payload.email,
            password=payload.password,
            user_agent=request.headers.get("user-agent"),
            ip_address=_client_ip(request),
        )
    except InvalidCredentialsError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": "invalid_credentials",
                "message": "E-poçt və ya şifrə yanlışdır.",
            },
        ) from exc
    _set_session_cookie(response, authenticated.token, authenticated.expires_at, settings)
    return AuthResponse(user=UserResponse.model_validate(authenticated.user))


@router.get("/me", response_model=AuthResponse)
def current_user(
    user: UserRecord | None = Depends(require_authenticated_user),
    settings: Settings = Depends(get_settings),
) -> AuthResponse:
    _ensure_enabled(settings)
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
    return AuthResponse(user=UserResponse.model_validate(user))


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(
    request: Request,
    response: Response,
    session: Session = Depends(get_auth_session),
    settings: Settings = Depends(get_settings),
) -> Response:
    _ensure_enabled(settings)
    token = request.cookies.get(settings.AUTH_COOKIE_NAME, "")
    if token:
        AuthService(session, settings.AUTH_SESSION_TTL_DAYS).logout(token)
    response.delete_cookie(
        settings.AUTH_COOKIE_NAME,
        path="/",
        secure=settings.AUTH_COOKIE_SECURE,
        httponly=True,
        samesite="lax",
    )
    response.status_code = status.HTTP_204_NO_CONTENT
    return response
