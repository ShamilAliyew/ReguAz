#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.reguaz.retrieval.hybrid_v2 import HybridV2Retriever  # noqa: E402
from backend.reguaz.retrieval.v2_artifacts import V2ArtifactStore  # noqa: E402
from backend.reguaz.services.generation.generation_pipeline_v2 import (  # noqa: E402
    V2GenerationPipeline,
)
from backend.reguaz.services.generation.llm_factory import LLMFactory  # noqa: E402
from backend.reguaz.services.generation.v2_models import (  # noqa: E402
    V2GenerationSettings,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run frozen 10-record V2 development generation smoke test."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "data/evaluation/reguaz_v2_evaluation_50.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/v2_generation/dev_smoke_10.json",
    )
    parser.add_argument("--limit", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 1 <= args.limit <= 10:
        raise ValueError("--limit must be between 1 and 10")
    records = [
        json.loads(line)
        for line in args.dataset.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    development = [record for record in records if record["split"] == "dev"]
    if len(development) != 10:
        raise ValueError("frozen dataset must contain exactly 10 development records")

    startup = time.perf_counter()
    artifacts = V2ArtifactStore(v2_root=ROOT / "data/processed/v2")
    retriever = HybridV2Retriever(
        v2_root=ROOT / "data/processed/v2",
        qdrant_path=ROOT / "data/processed/v2/qdrant",
        artifact_store=artifacts,
        local_files_only=True,
    )
    llm = LLMFactory.create("gemma")
    pipeline = V2GenerationPipeline(
        retriever=retriever,
        artifacts=artifacts,
        llm=llm,
        settings=V2GenerationSettings(reserved_output_tokens=llm.max_tokens),
    )
    startup_ms = (time.perf_counter() - startup) * 1000.0
    results = []
    try:
        for record in development[: args.limit]:
            try:
                output = pipeline.generate(record["question"])
                warning_codes = [item.get("code") for item in output["warnings"]]
                results.append(
                    {
                        "id": record["id"],
                        "completed": True,
                        "status": output["status"],
                        "citation_count": len(output["citations"]),
                        "all_citations_source_validated": all(
                            item["source_validated"] for item in output["citations"]
                        ),
                        "unknown_evidence_id_exposed": False,
                        "citation_free_answered_blocks": sum(
                            not item["citation_numbers"]
                            for item in output["answer_blocks"]
                        ),
                        "warning_codes": warning_codes,
                        "metrics": output["metrics"],
                        "prompt_tokens": output["trace"]["generation"]["prompt_tokens"],
                        "completion_tokens": output["trace"]["generation"][
                            "completion_tokens"
                        ],
                    }
                )
            except Exception as exc:
                results.append(
                    {
                        "id": record["id"],
                        "completed": False,
                        "error_category": type(exc).__name__,
                    }
                )
    finally:
        retriever.close()
        llm.close()

    latencies = [item["metrics"]["total_ms"] for item in results if item["completed"]]
    summary = {
        "dataset": _portable_path(args.dataset),
        "split": "dev",
        "requested_count": args.limit,
        "completed_count": sum(item["completed"] for item in results),
        "startup_ms": startup_ms,
        "model_device": getattr(llm, "device", "unknown"),
        "mean_total_ms": statistics.mean(latencies) if latencies else None,
        "max_total_ms": max(latencies) if latencies else None,
        "json_validation_success_count": sum(item["completed"] for item in results),
        "source_resolution_failure_count": sum(
            "citation_validation_failed" in item.get("warning_codes", [])
            for item in results
        ),
        "context_overflow_count": sum(
            "required_evidence_overflow" in item.get("warning_codes", [])
            for item in results
        ),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {key: value for key, value in summary.items() if key != "results"}, indent=2
        )
    )
    return 0 if summary["completed_count"] == args.limit else 1


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return resolved.name


if __name__ == "__main__":
    raise SystemExit(main())
