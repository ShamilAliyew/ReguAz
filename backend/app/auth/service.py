"""Application service for registration, login and revocable sessions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.app.auth.database import AuthSessionRecord, UserRecord
from backend.app.auth.security import (
    create_session_token,
    hash_password,
    hash_session_token,
    verify_password,
)


class AuthError(Exception):
    pass


class EmailAlreadyRegisteredError(AuthError):
    pass


class InvalidCredentialsError(AuthError):
    pass


@dataclass(frozen=True)
class AuthenticatedSession:
    user: UserRecord
    token: str
    expires_at: datetime


class AuthService:
    def __init__(self, session: Session, session_ttl_days: int) -> None:
        self._session = session
        self._session_ttl = timedelta(days=session_ttl_days)

    def register(
        self,
        *,
        name: str,
        email: str,
        password: str,
        user_agent: str | None,
        ip_address: str | None,
    ) -> AuthenticatedSession:
        user = UserRecord(
            id=str(uuid4()),
            name=name.strip(),
            email=email,
            password_hash=hash_password(password),
        )
        self._session.add(user)
        try:
            self._session.flush()
        except IntegrityError as exc:
            self._session.rollback()
            raise EmailAlreadyRegisteredError from exc
        result = self._issue_session(user, user_agent, ip_address)
        self._session.commit()
        return result

    def login(
        self,
        *,
        email: str,
        password: str,
        user_agent: str | None,
        ip_address: str | None,
    ) -> AuthenticatedSession:
        user = self._session.scalar(select(UserRecord).where(UserRecord.email == email))
        if not verify_password(password, user.password_hash if user else None):
            raise InvalidCredentialsError
        if user is None or not user.is_active:
            raise InvalidCredentialsError
        user.last_login_at = datetime.now(UTC)
        result = self._issue_session(user, user_agent, ip_address)
        self._delete_expired_sessions()
        self._session.commit()
        return result

    def authenticate(self, token: str) -> UserRecord | None:
        now = datetime.now(UTC)
        auth_session = self._session.scalar(
            select(AuthSessionRecord).where(
                AuthSessionRecord.token_hash == hash_session_token(token),
                AuthSessionRecord.expires_at > now,
            )
        )
        if auth_session is None:
            return None
        user = self._session.get(UserRecord, auth_session.user_id)
        if user is None or not user.is_active:
            return None
        auth_session.last_seen_at = now
        self._session.commit()
        return user

    def logout(self, token: str) -> None:
        self._session.execute(
            delete(AuthSessionRecord).where(
                AuthSessionRecord.token_hash == hash_session_token(token)
            )
        )
        self._session.commit()

    def _issue_session(
        self,
        user: UserRecord,
        user_agent: str | None,
        ip_address: str | None,
    ) -> AuthenticatedSession:
        token = create_session_token()
        expires_at = datetime.now(UTC) + self._session_ttl
        self._session.add(
            AuthSessionRecord(
                id=str(uuid4()),
                user_id=user.id,
                token_hash=hash_session_token(token),
                expires_at=expires_at,
                user_agent=(user_agent or "")[:500] or None,
                ip_address=(ip_address or "")[:64] or None,
            )
        )
        return AuthenticatedSession(user=user, token=token, expires_at=expires_at)

    def _delete_expired_sessions(self) -> None:
        self._session.execute(
            delete(AuthSessionRecord).where(
                AuthSessionRecord.expires_at <= datetime.now(UTC)
            )
        )
