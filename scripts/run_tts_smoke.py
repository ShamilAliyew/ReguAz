#!/usr/bin/env python3
"""Run the fixed Azerbaijani TTS smoke suite after configuring OpenRouter."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from backend.app.core.config import Settings  # noqa: E402
from backend.reguaz.services.speech.openrouter_tts import (  # noqa: E402
    OpenRouterTTSService,
)


TEST_CASES: tuple[dict[str, str], ...] = (
    {
        "id": "azerbaijani_letters",
        "focus": "ə, ğ, ı, ö, ü, ç, ş",
        "text": "Əmanətçi ödənişlərin düzgün və şəffaf şəkildə həyata keçirilməsini gözləyir.",
    },
    {
        "id": "legal_locator",
        "focus": "article and clause locator",
        "text": "Qaydanın 12.3.1-ci bəndində göstərilən tələb tətbiq olunur.",
    },
    {
        "id": "currency",
        "focus": "AZN, narrow space scale and compact large integer",
        "text": (
            "Limit 500\u202fmilyon AZN, ümumi öhdəlik isə "
            "500000000000 AZN müəyyən edilmişdir."
        ),
    },
    {
        "id": "percentage_ratio",
        "focus": "percentage and ratio",
        "text": "Likvidlik göstəricisi 25% və risk nisbəti 1/5 səviyyəsindədir.",
    },
    {
        "id": "date",
        "focus": "legal date",
        "text": "Qərar 15.08.2026-cı il tarixindən qüvvəyə minir.",
    },
    {
        "id": "abbreviation",
        "focus": "AMB abbreviation",
        "text": "AMB banklardan prudensial hesabatların vaxtında təqdim edilməsini tələb edir.",
    },
    {
        "id": "citation",
        "focus": "citation marker",
        "text": "Bank risklərin idarə edilməsi siyasətini təsdiq etməlidir. [1]",
    },
    {
        "id": "decimal",
        "focus": "decimal number",
        "text": "Əmsalın minimum həddi 2.5, maksimum həddi isə 4.0 kimi müəyyən edilir.",
    },
    {
        "id": "formal_terms",
        "focus": "Azerbaijani legal terminology",
        "text": "Benefisiar mülkiyyətçi və müştərinin eyniləşdirilməsi üzrə məlumatlar yoxlanılmalıdır.",
    },
    {
        "id": "long_sentence",
        "focus": "long formal sentence and prosody",
        "text": (
            "Bank müşahidə şurasının təsdiq etdiyi siyasətə uyğun olaraq riskləri müəyyən etməli, "
            "qiymətləndirməli, monitorinq aparmalı və nəticələri mütəmadi şəkildə sənədləşdirməlidir."
        ),
    },
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a fixed Azerbaijani legal TTS smoke suite"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/tts"))
    parser.add_argument("--limit", type=int, default=len(TEST_CASES))
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the exact non-sensitive test inputs without calling OpenRouter",
    )
    return parser


async def run(args: argparse.Namespace) -> int:
    selected = TEST_CASES[: max(0, min(args.limit, len(TEST_CASES)))]
    if not selected:
        print("error: --limit must select at least one case", file=sys.stderr)
        return 2
    if args.dry_run:
        print(json.dumps(selected, ensure_ascii=False, indent=2))
        return 0

    settings = Settings()
    if settings.OPENROUTER_API_KEY is None:
        print("error: OPENROUTER_API_KEY is not configured", file=sys.stderr)
        return 2

    service = OpenRouterTTSService(
        api_key=settings.OPENROUTER_API_KEY.get_secret_value(),
        base_url=settings.OPENROUTER_API_BASE,
        model_id=settings.OPENROUTER_TTS_MODEL,
        voice=settings.OPENROUTER_TTS_VOICE,
        site_url=settings.OPENROUTER_SITE_URL,
        app_title=settings.OPENROUTER_APP_TITLE,
        timeout_seconds=settings.OPENROUTER_TTS_TIMEOUT_SECONDS,
        max_input_characters=settings.OPENROUTER_TTS_MAX_INPUT_CHARACTERS,
        max_audio_bytes=settings.OPENROUTER_TTS_MAX_AUDIO_BYTES,
        max_concurrency=1,
        max_retries=settings.OPENROUTER_TTS_MAX_RETRIES,
    )
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    try:
        for case in selected:
            record: dict[str, Any] = {
                **case,
                "success": False,
                "latency_ms": None,
                "audio_bytes": None,
                "generation_id": None,
                "sha256": None,
                "manual_naturalness_1_to_5": None,
                "manual_pronunciation_1_to_5": None,
                "manual_notes": None,
            }
            try:
                audio = await service.synthesize(case["text"])
                audio_path = output_dir / f"{case['id']}.mp3"
                audio_path.write_bytes(audio.content)
                record.update(
                    {
                        "success": True,
                        "latency_ms": round(audio.elapsed_ms, 3),
                        "audio_bytes": len(audio.content),
                        "generation_id": audio.generation_id,
                        "sha256": hashlib.sha256(audio.content).hexdigest(),
                        "audio_path": str(audio_path),
                    }
                )
            except Exception as exc:  # report every case; never fabricate success
                record["error_category"] = type(exc).__name__
            results.append(record)
    finally:
        await service.aclose()

    successful = [item for item in results if item["success"]]
    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "model_id": settings.OPENROUTER_TTS_MODEL,
        "case_count": len(results),
        "success_count": len(successful),
        "success_rate": len(successful) / len(results),
        "mean_latency_ms": (
            sum(float(item["latency_ms"]) for item in successful) / len(successful)
            if successful
            else None
        ),
        "quality_review_status": "pending_manual_review",
        "results": results,
    }
    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if len(successful) == len(results) else 1


def main() -> int:
    return asyncio.run(run(build_parser().parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
