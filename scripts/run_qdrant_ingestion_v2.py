#!/usr/bin/env python3
"""Ingest ReguAz V2 dense+sparse artifacts into a versioned Qdrant collection."""

from __future__ import annotations

import argparse
import json
import os
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
    parser.add_argument(
        "--qdrant-url",
        default=os.getenv("QDRANT_URL"),
        help="Remote Qdrant endpoint; defaults to QDRANT_URL",
    )
    parser.add_argument(
        "--qdrant-api-key-env",
        default="QDRANT_API_KEY",
        help="Environment variable containing the remote API key",
    )
    parser.add_argument("--qdrant-timeout-seconds", type=float, default=60.0)
    parser.add_argument(
        "--report-path",
        type=Path,
        help="Optional report path; remote default is logs/qdrant_ingestion_v2_cloud.json",
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
        qdrant_api_key = os.getenv(args.qdrant_api_key_env)
        if args.qdrant_url and not qdrant_api_key:
            raise ValueError(
                f"{args.qdrant_api_key_env} is required for remote Qdrant ingestion"
            )
        report_path = args.report_path
        if args.qdrant_url and report_path is None:
            report_path = Path("logs/qdrant_ingestion_v2_cloud.json")
        settings = QdrantV2Settings(
            v2_root=args.v2_root,
            embeddings_root=args.embeddings_root,
            qdrant_path=args.qdrant_path,
            qdrant_url=args.qdrant_url,
            qdrant_api_key=qdrant_api_key,
            qdrant_timeout_seconds=args.qdrant_timeout_seconds,
            report_path=report_path,
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
