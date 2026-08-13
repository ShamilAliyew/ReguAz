#!/usr/bin/env python3
"""Read-only ReguAZ V2 dense+sparse+BM25 retrieval and reranking CLI."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from statistics import fmean
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from backend.reguaz.retrieval.hybrid_v2 import (  # noqa: E402
    DEFAULT_V2_RERANKER_MODEL,
    DEFAULT_V2_RERANKER_REVISION,
    HybridV2Retriever,
    HybridV2Settings,
)
from backend.reguaz.retrieval.v2_contract import RetrievalFilters  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--query")
    source.add_argument("--query-file", type=Path)
    parser.add_argument(
        "--query-split",
        choices=("all", "dev", "test"),
        default="all",
        help="Filter JSON/JSONL object records by their split field.",
    )
    parser.add_argument("--v2-root", type=Path, default=Path("data/processed/v2"))
    parser.add_argument(
        "--qdrant-path", type=Path, default=Path("data/processed/v2/qdrant")
    )
    parser.add_argument("--alias", default="reguaz_v2_current")
    parser.add_argument("--category")
    parser.add_argument("--document-id")
    parser.add_argument(
        "--is-current",
        choices=("true", "false", "any"),
        default="true",
    )
    parser.add_argument("--dense-top-k", type=int, default=30)
    parser.add_argument("--sparse-top-k", type=int, default=30)
    parser.add_argument("--bm25-top-k", type=int, default=30)
    parser.add_argument("--dense-weight", type=float, default=1.0)
    parser.add_argument("--sparse-weight", type=float, default=1.0)
    parser.add_argument("--bm25-weight", type=float, default=1.0)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--rerank-top-k", type=int, default=15)
    parser.add_argument("--final-top-k", type=int, default=5)
    parser.add_argument("--reranker-batch-size", type=int, default=8)
    parser.add_argument(
        "--retrieval-strategy",
        choices=("sequential", "qdrant_batch_bm25_overlap", "concurrent_all"),
        default="qdrant_batch_bm25_overlap",
    )
    parser.add_argument("--device")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--reranker-revision", default=DEFAULT_V2_RERANKER_REVISION)
    parser.add_argument("--reranker-model", default=DEFAULT_V2_RERANKER_MODEL)
    parser.add_argument("--reranker-trust-remote-code", action="store_true")
    parser.add_argument("--reranker-code-revision")
    parser.add_argument("--warmup-queries", type=int, default=1)
    parser.add_argument("--benchmark-runs", type=int, default=1)
    parser.add_argument(
        "--compare-rerank-depths",
        type=int,
        nargs="+",
        help="Compare depths such as 15 20 using one shared retrieval pass.",
    )
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.warmup_queries < 0 or args.benchmark_runs <= 0:
            raise ValueError("warmup-queries must be >= 0 and benchmark-runs > 0")
        queries = (
            [args.query]
            if args.query is not None
            else _read_queries(args.query_file, split=args.query_split)
        )
        queries = [
            query for query in queries if isinstance(query, str) and query.strip()
        ]
        if not queries:
            raise ValueError("no non-empty queries were supplied")
        filters = RetrievalFilters(
            category=args.category,
            document_id=args.document_id,
            is_current={"true": True, "false": False, "any": None}[args.is_current],
        )
        settings = HybridV2Settings(
            dense_top_k=args.dense_top_k,
            sparse_top_k=args.sparse_top_k,
            bm25_top_k=args.bm25_top_k,
            dense_weight=args.dense_weight,
            sparse_weight=args.sparse_weight,
            bm25_weight=args.bm25_weight,
            rrf_k=args.rrf_k,
            rerank_top_k=args.rerank_top_k,
            final_top_k=args.final_top_k,
            reranker_batch_size=args.reranker_batch_size,
            retrieval_strategy=args.retrieval_strategy,
        )
        with HybridV2Retriever(
            v2_root=args.v2_root,
            qdrant_path=args.qdrant_path,
            alias=args.alias,
            settings=settings,
            device=args.device,
            local_files_only=args.local_files_only,
            reranker_revision=args.reranker_revision,
            reranker_model=args.reranker_model,
            reranker_trust_remote_code=args.reranker_trust_remote_code,
            reranker_code_revision=args.reranker_code_revision,
        ) as retriever:
            for index in range(args.warmup_queries):
                retriever.retrieve(queries[index % len(queries)], filters=filters)

            runs: list[dict[str, Any]] = []
            for query in queries:
                for _ in range(args.benchmark_runs):
                    if args.compare_rerank_depths:
                        run = retriever.compare_rerank_depths(
                            query,
                            depths=args.compare_rerank_depths,
                            final_top_k=args.final_top_k,
                            filters=filters,
                        )
                    else:
                        run = retriever.retrieve_with_trace(query, filters=filters)
                    runs.append(run)
            output = {
                "schema_version": "2.0",
                "pipeline": "reguaz_v2_hybrid_retrieval",
                "cold_start_ms": retriever.startup_ms,
                "warmup_query_count": args.warmup_queries,
                "benchmark_run_count": len(runs),
                "comparison_mode": bool(args.compare_rerank_depths),
                "summary": _latency_summary(runs),
                "runs": runs,
            }
        _emit(output, args.output)
        return 0
    except (FileNotFoundError, json.JSONDecodeError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def _read_queries(path: Path, *, split: str = "all") -> list[str]:
    if path.suffix.lower() == ".jsonl":
        values = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "queries" in payload:
            values = payload["queries"]
        elif isinstance(payload, list):
            values = payload
        else:
            values = [payload]
    queries = []
    for value in values:
        if split != "all" and isinstance(value, dict) and value.get("split") != split:
            continue
        if isinstance(value, str):
            queries.append(value)
        elif isinstance(value, dict) and any(
            isinstance(value.get(key), str)
            for key in ("query", "question", "user_input")
        ):
            queries.append(
                next(
                    value[key]
                    for key in ("query", "question", "user_input")
                    if isinstance(value.get(key), str)
                )
            )
        else:
            raise ValueError("query file entries must be strings or objects with query")
    return queries


def _latency_summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    if not runs:
        return {}
    if "comparisons" in runs[0]:
        depths = sorted(
            {int(item["rerank_top_k"]) for run in runs for item in run["comparisons"]}
        )
        return {
            "by_rerank_top_k": {
                str(depth): _summarize_samples(
                    [
                        item
                        for run in runs
                        for item in run["comparisons"]
                        if item["rerank_top_k"] == depth
                    ]
                )
                for depth in depths
            }
        }
    samples = [
        {
            "total_retrieval_ms": run["timings"]["total_retrieval_ms"],
            "reranker_ms": run["timings"]["reranker_ms"],
        }
        for run in runs
    ]
    return _summarize_samples(samples)


def _summarize_samples(samples: list[dict[str, Any]]) -> dict[str, float]:
    totals = [float(sample["total_retrieval_ms"]) for sample in samples]
    rerank = [float(sample["reranker_ms"]) for sample in samples]
    return {
        "average_latency_ms": fmean(totals),
        "p50_latency_ms": _percentile(totals, 0.50),
        "p95_latency_ms": _percentile(totals, 0.95),
        "max_latency_ms": max(totals),
        "average_reranker_ms": fmean(rerank),
        "reranker_percentage_of_total": (
            100.0 * sum(rerank) / sum(totals) if sum(totals) else 0.0
        ),
    }


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return ordered[index]


def _emit(output: dict[str, Any], path: Path | None) -> None:
    if path is None:
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if path.suffix.lower() == ".jsonl":
        lines = [
            json.dumps({"type": "run", **run}, ensure_ascii=False)
            for run in output["runs"]
        ]
        lines.append(
            json.dumps(
                {
                    "type": "summary",
                    **{key: value for key, value in output.items() if key != "runs"},
                },
                ensure_ascii=False,
            )
        )
        temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    else:
        temporary.write_text(
            json.dumps(output, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    temporary.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
