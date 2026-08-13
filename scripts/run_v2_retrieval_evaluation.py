#!/usr/bin/env python3
"""Evaluate the frozen ReguAZ V2 dataset and emit a reproducible JSON report."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from backend.reguaz.evaluation.retrieval_v2 import (  # noqa: E402
    aggregate_quality,
    evaluate_ranking,
    latency_summary,
)
from backend.reguaz.evaluation.v2_dataset import (  # noqa: E402
    EvaluationRecord,
    sha256_file,
    validate_dataset,
)
from backend.reguaz.retrieval.hybrid_v2 import (  # noqa: E402
    DEFAULT_V2_RERANKER_MODEL,
    DEFAULT_V2_RERANKER_REVISION,
    HybridV2Retriever,
    HybridV2Settings,
)


STAGES = ("dense", "sparse", "bm25", "rrf", "reranked")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/evaluation/reguaz_v2_evaluation_50.jsonl"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/evaluation/reguaz_v2_evaluation_50_manifest.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--strategy",
        choices=("sequential", "qdrant_batch_bm25_overlap", "concurrent_all"),
        required=True,
    )
    parser.add_argument("--warmup-queries", type=int, default=3)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--device")
    parser.add_argument("--reranker-model", default=DEFAULT_V2_RERANKER_MODEL)
    parser.add_argument("--reranker-revision", default=DEFAULT_V2_RERANKER_REVISION)
    parser.add_argument("--reranker-trust-remote-code", action="store_true")
    parser.add_argument("--reranker-code-revision")
    args = parser.parse_args()

    validation = validate_dataset(args.dataset, args.manifest)
    records = [
        EvaluationRecord.model_validate_json(line)
        for line in args.dataset.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.limit is not None:
        records = records[: args.limit]
    settings = HybridV2Settings(
        retrieval_strategy=args.strategy,
        rerank_top_k=15,
    )
    per_stage: dict[str, list[dict[str, Any]]] = defaultdict(list)
    timings: dict[str, list[float]] = defaultdict(list)
    failures = []
    traces = []
    with HybridV2Retriever(
        settings=settings,
        local_files_only=args.local_files_only,
        device=args.device,
        reranker_model=args.reranker_model,
        reranker_revision=args.reranker_revision,
        reranker_trust_remote_code=args.reranker_trust_remote_code,
        reranker_code_revision=args.reranker_code_revision,
    ) as retriever:
        for index in range(args.warmup_queries):
            retriever.retrieve(records[index % len(records)].question)
        runtime = None
        for record in records:
            try:
                run = retriever.retrieve_with_trace(record.question)
            except Exception as exc:
                failures.append(
                    {"record_id": record.id, "error": f"{type(exc).__name__}: {exc}"}
                )
                continue
            runtime = run["runtime"]
            stage_ids = {
                "dense": [item["chunk_id"] for item in run["intermediate"]["dense"]],
                "sparse": [item["chunk_id"] for item in run["intermediate"]["sparse"]],
                "bm25": [item["chunk_id"] for item in run["intermediate"]["bm25"]],
                "rrf": [
                    item["chunk_id"] for item in run["intermediate"]["fused_all"][:30]
                ],
                "reranked": [
                    item["chunk_id"]
                    for item in run["intermediate"]["reranked_candidates"]
                ],
            }
            for stage, ids in stage_ids.items():
                per_stage[stage].append(
                    {
                        "record_id": record.id,
                        "split": record.split,
                        "difficulty": record.difficulty,
                        "category": record.category,
                        "question_type": record.question_type,
                        "requires_multiple_chunks": record.metadata.requires_multiple_chunks,
                        "metrics": evaluate_ranking(ids, record, retriever._artifacts),
                    }
                )
            for key, value in run["timings"].items():
                timings[key].append(float(value))
            traces.append(
                {
                    "record_id": record.id,
                    "stage_ids": stage_ids,
                    "timings": run["timings"],
                    "counts": run["counts"],
                    "conditions": run["conditions"],
                }
            )
        report = {
            "schema_version": "2.0",
            "dataset_sha256": sha256_file(args.dataset),
            "dataset_validation": validation,
            "strategy": args.strategy,
            "reranker_model": args.reranker_model,
            "reranker_revision": args.reranker_revision,
            "reranker_code_revision": args.reranker_code_revision,
            "rerank_top_k": settings.rerank_top_k,
            "warmup_query_count": args.warmup_queries,
            "query_count": len(records),
            "successful_query_count": len(traces),
            "failed_query_count": len(failures),
            "failures": failures,
            "runtime": runtime,
            "quality": {stage: aggregate_quality(per_stage[stage]) for stage in STAGES},
            "latency_ms": {
                key: latency_summary(values) for key, values in sorted(timings.items())
            },
            "traces": traces,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "strategy",
                    "query_count",
                    "successful_query_count",
                    "failed_query_count",
                    "runtime",
                    "latency_ms",
                )
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
