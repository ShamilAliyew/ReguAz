#!/usr/bin/env python3
"""Thin CLI for the reusable ReguAz V2 chunking service."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from backend.reguaz.services.chunking import (  # noqa: E402
    ApproximateTokenizer,
    BgeM3Tokenizer,
    ChunkingConfig,
    ChunkingPipeline,
)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Build isolated ReguAz V2 chunks and metadata")
    value.add_argument("--input", type=Path, default=Path("data/processed/cleaned_documents"))
    value.add_argument("--output", type=Path, default=Path("data/processed/v2"))
    value.add_argument("--document", help="Exact NFC-equivalent filename, stem, or relative path")
    value.add_argument("--limit", type=int)
    value.add_argument("--dry-run", action="store_true")
    value.add_argument("--force", action="store_true")
    value.add_argument("--fail-on-quality-errors", action="store_true")
    value.add_argument("--tokenizer", default="BAAI/bge-m3", choices=["BAAI/bge-m3", "approximate"])
    value.add_argument(
        "--allow-approx-tokenizer",
        action="store_true",
        help="Explicitly permit deterministic approximate counts (recorded in manifest)",
    )
    value.add_argument("--local-files-only", action="store_true", help="Do not download BGE tokenizer files")
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser().error("--limit must be at least 1")
    if args.tokenizer == "approximate":
        if not args.allow_approx_tokenizer:
            parser().error("--tokenizer approximate requires --allow-approx-tokenizer")
        tokenizer = ApproximateTokenizer()
    else:
        try:
            tokenizer = BgeM3Tokenizer(local_files_only=args.local_files_only)
        except RuntimeError as exc:
            if not args.allow_approx_tokenizer:
                print(f"error: {exc}", file=sys.stderr)
                return 2
            tokenizer = ApproximateTokenizer()
            print(f"warning: {exc}\nUsing explicitly permitted approximate tokenizer.", file=sys.stderr)
    config = ChunkingConfig(input_root=args.input, output_root=args.output)
    pipeline = ChunkingPipeline(config, tokenizer)
    try:
        if args.dry_run:
            print(json.dumps(pipeline.dry_run(document=args.document, limit=args.limit), ensure_ascii=False, indent=2))
            return 0
        result = pipeline.run(
            document=args.document,
            limit=args.limit,
            force=args.force,
            fail_on_quality_errors=args.fail_on_quality_errors,
        )
        print(json.dumps(result.manifest.model_dump(mode="json"), ensure_ascii=False, indent=2))
        return 0 if result.succeeded else 1
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
