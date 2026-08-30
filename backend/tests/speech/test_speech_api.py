from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.app.api.speech import router
from backend.app.core.config import Settings
from backend.app.core.dependencies import get_app_state
from backend.app.core.lifespan import AppState, _load_speech
from backend.reguaz.services.speech.openrouter_tts import SpeechAudio


class FakeSpeechService:
    async def synthesize(self, text: str) -> SpeechAudio:
        assert text == "Cavab [1]"
        return SpeechAudio(
            content=b"ID3" + b"\x00" * 8,
            content_type="audio/mpeg",
            generation_id="gen-api",
            model_id="fish-audio/s2.1-pro-free:free",
            elapsed_ms=42.5,
            normalized_characters=13,
        )


def build_client(state: AppState) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_app_state] = lambda: state
    return TestClient(app)


def test_speech_status_is_nonfatal_without_key() -> None:
    response = build_client(AppState()).get("/speech/status")

    assert response.status_code == 200
    assert response.json()["available"] is False
    assert response.json()["reason"] == "api_key_not_configured"


def test_blank_environment_key_is_nonfatal() -> None:
    state = AppState()

    _load_speech(
        state,
        Settings(OPENROUTER_TTS_ENABLED=True, OPENROUTER_API_KEY="   "),
    )

    assert state.speech_service is None


def test_speech_endpoint_returns_inline_mp3_and_observability_headers() -> None:
    response = build_client(AppState(speech_service=FakeSpeechService())).post(
        "/speech", json={"text": "Cavab [1]"}
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/mpeg"
    assert response.headers["x-generation-id"] == "gen-api"
    assert response.headers["x-tts-model"] == "fish-audio/s2.1-pro-free:free"
    assert response.headers["cache-control"] == "private, no-store"
    assert response.content.startswith(b"ID3")


def test_speech_endpoint_fails_closed_when_unconfigured() -> None:
    response = build_client(AppState()).post("/speech", json={"text": "Cavab"})

    assert response.status_code == 503
    assert response.json()["detail"]["error"] == "speech_unavailable"
