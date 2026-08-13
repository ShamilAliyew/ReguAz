#!/usr/bin/env python3
"""Generate validated ReguAz V2 BGE-M3 dense+sparse embedding artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from backend.reguaz.services.embeddings.bge_m3_v2 import BgeM3DenseSparseEncoder  # noqa: E402
from backend.reguaz.services.embeddings.v2_pipeline import (  # noqa: E402
    EmbeddingSettings,
    V2EmbeddingPipeline,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate isolated V2 BGE-M3 dense+sparse embedding artifacts"
    )
    parser.add_argument("--input", type=Path, default=Path("data/processed/v2"))
    parser.add_argument(
        "--output", type=Path, default=Path("data/processed/v2/embeddings")
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--shard-size", type=int, default=256)
    parser.add_argument("--device", help="FlagEmbedding device, e.g. cpu, cuda:0")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--model-revision", help="Immutable Hugging Face revision")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        encoder = BgeM3DenseSparseEncoder(
            device=args.device,
            use_fp16=args.fp16,
            local_files_only=args.local_files_only,
            model_revision=args.model_revision,
        )
        settings = EmbeddingSettings(
            input_root=args.input,
            output_root=args.output,
            batch_size=args.batch_size,
            max_length=args.max_length,
            shard_size=args.shard_size,
            force=args.force,
            resume=not args.no_resume,
        )
        pipeline = V2EmbeddingPipeline(settings, encoder)
        if args.dry_run:
            print(json.dumps(pipeline.dry_run(), ensure_ascii=False, indent=2))
        else:
            manifest = pipeline.run()
            print(
                json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False, indent=2)
            )
        return 0
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
