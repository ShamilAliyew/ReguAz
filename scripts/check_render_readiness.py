#!/usr/bin/env python3
"""Validate immutable V2 artifacts and optional Qdrant Cloud state for Render."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from backend.reguaz.retrieval.qdrant_v2_retriever import QdrantV2Retriever  # noqa: E402
from backend.reguaz.retrieval.v2_artifacts import V2ArtifactStore  # noqa: E402
from backend.reguaz.retrieval.v2_contract import V2RetrievalContract  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate Render-packaged V2 artifacts and Qdrant Cloud"
    )
    parser.add_argument(
        "--v2-root",
        type=Path,
        default=REPOSITORY_ROOT / "data/processed/v2",
    )
    parser.add_argument("--check-qdrant-cloud", action="store_true")
    parser.add_argument("--alias", default="reguaz_v2_current")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        contract = V2RetrievalContract.load(args.v2_root)
        artifacts = V2ArtifactStore(v2_root=args.v2_root, contract=contract)
        source_count = 0
        for document in artifacts.iter_document_metadata():
            document_id = str(document["document_id"])
            if artifacts.get_source_representation(document_id) is None:
                raise ValueError(
                    f"authoritative cleaned source is missing: {document_id}"
                )
            source_count += 1

        qdrant_status: dict[str, object] = {"checked": False}
        if args.check_qdrant_cloud:
            qdrant_url = os.getenv("QDRANT_URL", "").strip()
            qdrant_api_key = os.getenv("QDRANT_API_KEY", "").strip()
            if not qdrant_url or not qdrant_api_key:
                raise ValueError(
                    "QDRANT_URL and QDRANT_API_KEY are required for cloud validation"
                )
            with QdrantV2Retriever(
                v2_root=args.v2_root,
                qdrant_url=qdrant_url,
                qdrant_api_key=qdrant_api_key,
                alias=args.alias,
                contract=contract,
            ) as retriever:
                qdrant_status = {
                    "checked": True,
                    "mode": retriever.mode,
                    "alias": retriever.alias,
                    "physical_collection": retriever.physical_collection,
                }

        output = {
            "status": "ready",
            "corpus_version": contract.chunk_manifest.corpus_version,
            "child_count": artifacts.corpus_size,
            "document_count": source_count,
            "embedding_model_id": contract.embedding_manifest.model_id,
            "embedding_model_revision": contract.embedding_manifest.model_revision,
            "dense_dimension": contract.embedding_manifest.dense_dimension,
            "qdrant": qdrant_status,
        }
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return 0
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
