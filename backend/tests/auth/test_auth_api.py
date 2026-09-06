from __future__ import annotations

from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import select

from backend.app.api.auth import router as auth_router
from backend.app.auth.database import AuthDatabase, AuthSessionRecord, UserRecord
from backend.app.core.config import Settings, get_settings


def _test_app() -> tuple[FastAPI, AuthDatabase]:
    database = AuthDatabase("sqlite+pysqlite:///:memory:")
    database.initialize()
    settings = Settings(
        AUTH_ENABLED=True,
        DATABASE_URL="sqlite+pysqlite:///:memory:",
        AUTH_COOKIE_SECURE=False,
        AUTH_SESSION_TTL_DAYS=7,
    )
    app = FastAPI()
    app.state.reguaz = SimpleNamespace(auth_database=database)
    app.dependency_overrides[get_settings] = lambda: settings
    app.include_router(auth_router)
    return app, database


def test_register_me_login_and_logout_use_revocable_cookie() -> None:
    app, database = _test_app()
    with TestClient(app) as client:
        registered = client.post(
            "/auth/register",
            json={
                "name": "Samil Əliyev",
                "email": "SAMIL@example.com",
                "password": "etibarli-sifre-123",
            },
        )
        assert registered.status_code == 201
        assert registered.json()["user"]["email"] == "samil@example.com"
        assert "httponly" in registered.headers["set-cookie"].lower()

        current = client.get("/auth/me")
        assert current.status_code == 200
        assert current.json()["user"]["name"] == "Samil Əliyev"

        with database.session() as session:
            user = session.scalar(select(UserRecord))
            auth_session = session.scalar(select(AuthSessionRecord))
            assert user is not None
            assert user.password_hash != "etibarli-sifre-123"
            assert user.password_hash.startswith("$argon2")
            assert auth_session is not None
            assert len(auth_session.token_hash) == 64

        logged_out = client.post("/auth/logout")
        assert logged_out.status_code == 204
        assert client.get("/auth/me").status_code == 401

        invalid = client.post(
            "/auth/login",
            json={"email": "samil@example.com", "password": "wrong"},
        )
        assert invalid.status_code == 401

        logged_in = client.post(
            "/auth/login",
            json={
                "email": "samil@example.com",
                "password": "etibarli-sifre-123",
            },
        )
        assert logged_in.status_code == 200

    database.close()


def test_duplicate_email_is_rejected_without_storing_plaintext_password() -> None:
    app, database = _test_app()
    payload = {
        "name": "Test User",
        "email": "test@example.com",
        "password": "another-safe-password",
    }
    with TestClient(app) as client:
        assert client.post("/auth/register", json=payload).status_code == 201
        duplicate = client.post("/auth/register", json=payload)
        assert duplicate.status_code == 409
    database.close()
