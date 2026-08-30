"""Optional text-to-speech services for ReguAZ answers."""

from backend.reguaz.services.speech.openrouter_tts import (
    OpenRouterTTSService,
    SpeechAudio,
    SpeechInputError,
    SpeechInputTooLongError,
    SpeechProviderError,
    SpeechRateLimitError,
)

__all__ = [
    "OpenRouterTTSService",
    "SpeechAudio",
    "SpeechInputError",
    "SpeechInputTooLongError",
    "SpeechProviderError",
    "SpeechRateLimitError",
]
