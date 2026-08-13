#!/usr/bin/env python3
"""Ingest ReguAz V2 dense+sparse artifacts into a versioned Qdrant collection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from backend.reguaz.services.ingestion.qdrant_v2 import (  # noqa: E402
    QdrantV2IngestionPipeline,
    QdrantV2Settings,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ingest V2 BGE-M3 dense+sparse points into isolated Qdrant storage"
    )
    parser.add_argument("--v2-root", type=Path, default=Path("data/processed/v2"))
    parser.add_argument(
        "--embeddings-root",
        type=Path,
        default=Path("data/processed/v2/embeddings"),
    )
    parser.add_argument(
        "--qdrant-path", type=Path, default=Path("data/processed/v2/qdrant")
    )
    parser.add_argument("--collection-prefix", default="reguaz_v2_bge_m3")
    parser.add_argument("--alias", default="reguaz_v2_current")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--force-collection", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        settings = QdrantV2Settings(
            v2_root=args.v2_root,
            embeddings_root=args.embeddings_root,
            qdrant_path=args.qdrant_path,
            collection_prefix=args.collection_prefix,
            alias=args.alias,
            upload_batch_size=args.batch_size,
            parallel=args.parallel,
            force_collection=args.force_collection,
        )
        pipeline = QdrantV2IngestionPipeline(settings)
        result = pipeline.dry_run() if args.dry_run else pipeline.run().model_dump(mode="json")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
