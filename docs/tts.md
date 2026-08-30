# ReguAZ answer text-to-speech

ReguAZ exposes optional, on-demand speech for a finalized assistant answer. TTS is deliberately outside retrieval and generation: the text answer and citation metadata remain the authoritative result and are returned without waiting for audio. The frontend calls TTS only when the user presses the small speaker icon below an assistant message.

## Runtime flow

1. Existing V2 retrieval, reranking, generation and citation resolution return the text answer.
2. The UI renders that answer immediately.
3. On `Cavabı dinlə`, the UI sends only the finalized answer to `POST /speech`; it never sends the question, retrieval evidence, prompts or API credentials.
4. The backend removes visual numeric citation markers from the speech-only copy, verbalizes numeric expressions in Azerbaijani, normalizes Markdown and legal notation, then calls OpenRouter `POST /api/v1/audio/speech` with `fish-audio/s2.1-pro-free:free` and explicit `response_format=mp3`.
5. The backend validates status, MIME type, MP3 signature and size before returning `audio/mpeg`.
6. The browser keeps one in-memory Blob per rendered message for play, pause and replay. It revokes the Blob URL when the message component is removed. No audio is persisted by ReguAZ.

TTS failure never changes or removes the generated answer or citations.

## Configuration

Keep the OpenRouter key only in the repository-root ignored `.env` file:

```bash
OPENROUTER_TTS_ENABLED=true
OPENROUTER_API_KEY=replace_with_your_key
OPENROUTER_API_BASE=https://openrouter.ai/api/v1
OPENROUTER_TTS_MODEL=fish-audio/s2.1-pro-free:free
OPENROUTER_TTS_VOICE=1db26754d7d84db39b8463c322c8d162
OPENROUTER_SITE_URL=http://localhost:3000
OPENROUTER_APP_TITLE=ReguAZ
```

The current OpenRouter model metadata publishes no built-in voice list for Fish Audio S2.1. Without a reference voice, Fish Audio may select a different default speaker between generations. ReguAZ pins the public Fish Audio `Məlahətli Azərbaycan Səsi` narrator reference ID shown above so speaker identity remains stable. Set the value to an empty string only when intentionally accepting provider-selected voices.

Missing configuration is non-fatal. `GET /speech/status` returns `available=false`, the backend remains healthy, and the UI shows a disabled speaker icon.

## Safety and operations

- Provider URL and model are server-whitelisted and HTTPS-only.
- The key is never returned to or used by the frontend.
- Provider routing requests zero-data-retention and denies data collection.
- Input, output bytes, concurrency, time and retry counts are bounded.
- A `429` is returned to the UI with `Retry-After`; rate limits are not retried automatically.
- Only a transient connection failure or `502/503/504` before accepted audio may be retried once.
- Logs contain model, character count, byte count, latency and generation ID, but not answer text or credentials.
- Numeric citation markers are removed only from the TTS copy. The rendered answer, citation metadata, source cards and document highlighting are unchanged.
- Numbers are verbalized before provider submission: grouped and compact integers, `min/milyon/milyard` forms, AZN amounts, percentages, ratios, decimals, numeric ordinals, dates and dotted legal locators each have deterministic Azerbaijani rules. Alphanumeric identifiers such as `ABC123` are preserved.
- The free model is suitable for development and low-volume tests; OpenRouter does not promise production latency or availability for it.

## Offline verification

Unit and API tests use fake clients and never contact OpenRouter:

```bash
poetry run pytest -q backend/tests/speech
cd frontend && npm test -- --run
```

Inspect the exact ten synthetic Azerbaijani legal inputs without a key:

```bash
poetry run python scripts/run_tts_smoke.py --dry-run
```

After configuring the key, generate the MP3 suite and `results/tts/report.json`:

```bash
poetry run python scripts/run_tts_smoke.py
```

The report records success, latency, byte size, provider generation ID and audio SHA-256. Naturalness and pronunciation scores remain explicitly `null` until a person listens to each file. Review Azerbaijani letters, locators, dates, amounts, percentages, abbreviations, citations and long legal sentences before enabling the model for users.

## Start the application

After `poetry install`, `cd frontend && npm ci`, and root `.env` configuration,
start both development services with:

```bash
./scripts/start_dev.sh
```

Use `./scripts/start_backend.sh` and `./scripts/start_frontend.sh` when separate
terminal logs are preferable. These scripts derive paths from their own
location and contain no developer-machine paths.

## Known limitation

OpenRouter lists Fish Audio S2.1 Pro Free as multilingual and intended for prototyping, but does not publish an Azerbaijani quality guarantee for this free endpoint. ReguAZ performs conservative pronunciation normalization and avoids unsafe blanket number-to-words conversion. A paid/provider fallback should be selected only after comparing the fixed Azerbaijani suite.
