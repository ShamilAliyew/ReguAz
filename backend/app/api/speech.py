"""Read-only TTS endpoints for finalized assistant answers."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Response, status

from backend.app.auth.dependencies import require_authenticated_user
from backend.app.core.config import get_settings
from backend.app.core.dependencies import get_app_state
from backend.app.core.lifespan import AppState
from backend.app.schemas.speech import SpeechRequest, SpeechStatusResponse
from backend.reguaz.services.speech.openrouter_tts import (
    SpeechInputError,
    SpeechInputTooLongError,
    SpeechProviderError,
    SpeechRateLimitError,
)


router = APIRouter(
    prefix="/speech",
    tags=["speech"],
    dependencies=[Depends(require_authenticated_user)],
)


@router.get(
    "/status",
    response_model=SpeechStatusResponse,
    summary="Report whether answer speech is configured",
)
def speech_status(
    state: AppState = Depends(get_app_state),
) -> SpeechStatusResponse:
    settings = get_settings()
    available = state.speech_service is not None
    reason = None
    if not settings.OPENROUTER_TTS_ENABLED:
        reason = "speech_disabled"
    elif not available:
        reason = "api_key_not_configured"
    return SpeechStatusResponse(
        available=available,
        model_id=settings.OPENROUTER_TTS_MODEL,
        reason=reason,
    )


@router.post(
    "",
    responses={
        200: {"content": {"audio/mpeg": {}}},
        413: {"description": "Answer is too long for speech synthesis"},
        429: {"description": "Provider rate limit"},
        502: {"description": "Invalid provider response"},
        503: {"description": "Speech is not configured"},
    },
    summary="Synthesize an assistant answer as MP3",
)
async def create_speech(
    request: SpeechRequest,
    state: AppState = Depends(get_app_state),
) -> Response:
    service = state.speech_service
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "speech_unavailable",
                "message": "Speech synthesis is not configured.",
            },
        )
    try:
        audio = await service.synthesize(request.text)
    except SpeechRateLimitError as exc:
        headers = {}
        if exc.retry_after_seconds is not None:
            headers["Retry-After"] = str(exc.retry_after_seconds)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "error": "speech_rate_limited",
                "message": "Speech service is busy. Please try again shortly.",
            },
            headers=headers,
        ) from exc
    except SpeechInputTooLongError as exc:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail={"error": "speech_input_invalid", "message": str(exc)},
        ) from exc
    except SpeechInputError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"error": "speech_input_invalid", "message": str(exc)},
        ) from exc
    except SpeechProviderError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={
                "error": "speech_provider_error",
                "message": "Speech could not be generated. Please try again.",
            },
        ) from exc

    headers = {
        "Cache-Control": "private, no-store",
        "Content-Disposition": 'inline; filename="reguaz-answer.mp3"',
        "X-TTS-Model": audio.model_id,
        "X-TTS-Latency-Ms": f"{audio.elapsed_ms:.1f}",
        "X-TTS-Characters": str(audio.normalized_characters),
        "Server-Timing": f"tts;dur={audio.elapsed_ms:.1f}",
    }
    if audio.generation_id:
        headers["X-Generation-Id"] = audio.generation_id
    return Response(
        content=audio.content, media_type=audio.content_type, headers=headers
    )
