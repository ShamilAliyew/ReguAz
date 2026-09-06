"""Password hashing and opaque session-token primitives."""

from __future__ import annotations

import hashlib
import secrets

from pwdlib import PasswordHash


_PASSWORD_HASH = PasswordHash.recommended()
_DUMMY_HASH = _PASSWORD_HASH.hash("not-a-real-reguaz-password")


def hash_password(password: str) -> str:
    return _PASSWORD_HASH.hash(password)


def verify_password(password: str, password_hash: str | None) -> bool:
    """Use a dummy hash when the account is absent to reduce timing leakage."""

    return _PASSWORD_HASH.verify(password, password_hash or _DUMMY_HASH)


def create_session_token() -> str:
    return secrets.token_urlsafe(48)


def hash_session_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()
