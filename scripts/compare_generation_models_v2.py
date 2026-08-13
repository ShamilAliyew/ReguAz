#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.reguaz.evaluation.generation_v2_comparison import (  # noqa: E402
    score_record,
    summarize,
)
from backend.reguaz.retrieval.hybrid_v2 import HybridV2Retriever  # noqa: E402
from backend.reguaz.retrieval.v2_artifacts import V2ArtifactStore  # noqa: E402
from backend.reguaz.services.generation.generation_pipeline_v2 import (  # noqa: E402
    V2GenerationPipeline,
)
from backend.reguaz.services.generation.llm_factory import LLMFactory  # noqa: E402
from backend.reguaz.services.generation.v2_models import (  # noqa: E402
    V2GenerationSettings,
)


MODELS = ("gemma", "groq_gpt_oss_20b", "groq_gpt_oss_120b")
REMOTE_MODELS = frozenset({"nvidia_gpt_oss", "groq_gpt_oss_20b", "groq_gpt_oss_120b"})


class CachedRetriever:
    def __init__(self, traces: dict[str, dict[str, Any]]) -> None:
        self.traces = traces

    def retrieve_with_trace(self, question: str) -> dict[str, Any]:
        return self.traces[question]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare selectable generation models on frozen V2 dev data."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "data/evaluation/reguaz_v2_evaluation_50.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/v2_generation/model_comparison_dev10.json",
    )
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=(
            "gemma",
            "nvidia_gpt_oss",
            "groq_gpt_oss_20b",
            "groq_gpt_oss_120b",
        ),
        default=list(MODELS),
    )
    parser.add_argument(
        "--allow-remote-evidence",
        action="store_true",
        help="Explicitly allow dev questions and selected V2 evidence to leave host.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 1 <= args.limit <= 10:
        raise ValueError("--limit must be between 1 and 10")
    if REMOTE_MODELS.intersection(args.models) and not args.allow_remote_evidence:
        raise PermissionError(
            "Remote comparison requires explicit --allow-remote-evidence consent"
        )
    records = [
        json.loads(line)
        for line in args.dataset.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    development = [item for item in records if item["split"] == "dev"]
    if len(development) != 10:
        raise ValueError("frozen dataset must contain exactly 10 development records")
    selected = development[: args.limit]

    artifacts = V2ArtifactStore(v2_root=ROOT / "data/processed/v2")
    retriever = HybridV2Retriever(
        v2_root=ROOT / "data/processed/v2",
        qdrant_path=ROOT / "data/processed/v2/qdrant",
        artifact_store=artifacts,
        local_files_only=True,
    )
    traces: dict[str, dict[str, Any]] = {}
    try:
        for record in selected:
            traces[record["question"]] = retriever.retrieve_with_trace(
                record["question"]
            )
    finally:
        retriever.close()
    del retriever
    gc.collect()
    _release_accelerator_cache()

    model_reports: dict[str, Any] = {}
    for model_name in args.models:
        startup = time.perf_counter()
        try:
            llm = LLMFactory.create(model_name)
        except Exception as exc:
            startup_ms = (time.perf_counter() - startup) * 1000.0
            failed = [
                {
                    "id": record["id"],
                    "completed": False,
                    "error_category": type(exc).__name__,
                    "attempt_ms": 0.0,
                }
                for record in selected
            ]
            model_reports[model_name] = {
                "startup_ms": startup_ms,
                "provider": model_name,
                "model": None,
                "device": "unavailable",
                "summary": summarize(failed),
                "results": failed,
            }
            continue
        startup_ms = (time.perf_counter() - startup) * 1000.0
        pipeline = V2GenerationPipeline(
            retriever=CachedRetriever(traces),
            artifacts=artifacts,
            llm=llm,
            settings=V2GenerationSettings(reserved_output_tokens=llm.max_tokens),
        )
        results: list[dict[str, Any]] = []
        try:
            for record in selected:
                common_retrieval_ms = float(
                    traces[record["question"]]["timings"]["total_retrieval_ms"]
                )
                attempt_started = time.perf_counter()
                try:
                    output = pipeline.generate(record["question"])
                    attempt_ms = (time.perf_counter() - attempt_started) * 1000.0
                    quality = score_record(record, output)
                    cached_total_ms = float(output["metrics"]["total_ms"])
                    results.append(
                        {
                            "id": record["id"],
                            "completed": True,
                            "status": output["status"],
                            "answer": output["answer"],
                            "answer_blocks": output["answer_blocks"],
                            "citations": [
                                {
                                    "citation": item["citation"],
                                    "chunk_id": item.get("chunk_id"),
                                    "parent_chunk_id": item.get("parent_chunk_id"),
                                    "document_version_id": item["document_version_id"],
                                    "canonical_locator": item["canonical_locator"],
                                    "source_validated": item["source_validated"],
                                }
                                for item in output["citations"]
                            ],
                            "warnings": output["warnings"],
                            "generation_ms": output["metrics"]["generation_ms"],
                            "attempt_ms": attempt_ms,
                            "cached_pipeline_ms": cached_total_ms,
                            "common_retrieval_ms": common_retrieval_ms,
                            "projected_e2e_ms": cached_total_ms + common_retrieval_ms,
                            "prompt_tokens": output["trace"]["generation"][
                                "prompt_tokens"
                            ],
                            "completion_tokens": output["trace"]["generation"][
                                "completion_tokens"
                            ],
                            "cached_prompt_tokens": output["trace"]["generation"].get(
                                "cached_prompt_tokens", 0
                            ),
                            "provider_total_ms": output["trace"]["generation"].get(
                                "provider_total_ms"
                            ),
                            "selected_evidence_count": output["trace"]["evidence"][
                                "selected_count"
                            ],
                            "quality": quality,
                        }
                    )
                except Exception as exc:
                    attempt_ms = (time.perf_counter() - attempt_started) * 1000.0
                    results.append(
                        {
                            "id": record["id"],
                            "completed": False,
                            "error_category": type(exc).__name__,
                            "attempt_ms": attempt_ms,
                        }
                    )
        finally:
            llm.close()
        model_reports[model_name] = {
            "startup_ms": startup_ms,
            "provider": getattr(llm, "provider", model_name),
            "model": _model_descriptor(llm),
            "device": getattr(llm, "device", "unknown"),
            "summary": summarize(results),
            "results": results,
        }

    eligible = [
        name
        for name, report in model_reports.items()
        if report["summary"]["success_rate"] == 1.0
        and report["summary"]["json_and_evidence_validation_rate"] == 1.0
        and report["summary"]["all_sources_validated_rate"] == 1.0
    ]
    latency_winner = (
        min(
            eligible,
            key=lambda name: model_reports[name]["summary"]["projected_e2e_latency_ms"][
                "mean"
            ],
        )
        if eligible
        else None
    )
    quality_leader = (
        max(
            eligible,
            key=lambda name: (
                model_reports[name]["summary"]["mean_gold_child_citation_recall"],
                model_reports[name]["summary"]["gold_document_citation_hit_rate"],
                model_reports[name]["summary"]["mean_reference_token_f1"],
                -model_reports[name]["summary"]["projected_e2e_latency_ms"]["mean"],
            ),
        )
        if eligible
        else None
    )
    report = {
        "dataset": _portable_path(args.dataset),
        "split": "dev",
        "record_count": len(selected),
        "retrieval_reused_across_models": True,
        "selection_policy": (
            "100% completion + structured/evidence validation + source validation; "
            "then lowest mean projected end-to-end latency"
        ),
        "winner": latency_winner,
        "latency_winner": latency_winner,
        "quality_proxy_leader": quality_leader,
        "models": model_reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "dataset": report["dataset"],
                "record_count": report["record_count"],
                "winner": latency_winner,
                "quality_proxy_leader": quality_leader,
                "models": {
                    name: value["summary"] for name, value in model_reports.items()
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if latency_winner else 1


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return resolved.name


def _model_descriptor(llm: Any) -> str:
    model_id = getattr(llm, "model_id", None)
    if model_id:
        return str(model_id)
    model_path = Path(llm.model_path).resolve()
    try:
        return model_path.relative_to(ROOT).as_posix()
    except ValueError:
        return model_path.name


def _release_accelerator_cache() -> None:
    """Release benchmark-only retrieval caches before loading generation model."""
    try:
        import torch

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except (ImportError, RuntimeError):
        return


if __name__ == "__main__":
    raise SystemExit(main())
