from __future__ import annotations

import math
import re
import statistics
import unicodedata
from collections import Counter
from typing import Any


TOKEN = re.compile(r"\d+(?:[.,]\d+)*|[^\W_]+", re.UNICODE)


def normalized_tokens(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFC", text).casefold()
    return TOKEN.findall(normalized)


def token_f1(prediction: str, reference: str) -> float:
    predicted = Counter(normalized_tokens(prediction))
    expected = Counter(normalized_tokens(reference))
    if not predicted or not expected:
        return 0.0
    overlap = sum((predicted & expected).values())
    precision = overlap / sum(predicted.values())
    recall = overlap / sum(expected.values())
    return 2 * precision * recall / (precision + recall) if overlap else 0.0


def score_record(record: dict[str, Any], output: dict[str, Any]) -> dict[str, Any]:
    citations = output.get("citations") or []
    cited_chunks = {str(item["chunk_id"]) for item in citations if item.get("chunk_id")}
    cited_versions = {
        str(item["document_version_id"])
        for item in citations
        if item.get("document_version_id")
    }
    gold_chunks = set(record["relevant_chunk_ids"])
    gold_versions = set(record["relevant_document_version_ids"])
    gold_chunk_recall = len(cited_chunks & gold_chunks) / len(gold_chunks)
    gold_document_hit = bool(cited_versions & gold_versions)
    answer_blocks = output.get("answer_blocks") or []
    warnings = output.get("warnings") or []
    invalid_codes = {
        "structured_output_invalid",
        "invalid_model_evidence_id",
        "citation_validation_failed",
    }
    return {
        "reference_token_f1": token_f1(
            str(output.get("answer") or ""), str(record["reference_answer"])
        ),
        "gold_child_citation_recall": gold_chunk_recall,
        "gold_document_citation_hit": gold_document_hit,
        "citation_count": len(citations),
        "all_sources_validated": (
            not citations
            if output.get("status") != "answered"
            else bool(citations)
            and all(bool(item.get("source_validated")) for item in citations)
        ),
        "citation_free_answered_blocks": sum(
            not item.get("citation_numbers") for item in answer_blocks
        ),
        "invalid_output_condition": any(
            item.get("code") in invalid_codes for item in warnings
        ),
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [item for item in results if item.get("completed")]
    generation = [float(item["generation_ms"]) for item in completed]
    projected = [float(item["projected_e2e_ms"]) for item in completed]
    attempts = [float(item["attempt_ms"]) for item in results]
    provider = [
        float(item["provider_total_ms"])
        for item in completed
        if item.get("provider_total_ms") is not None
    ]
    client_overhead = [
        float(item["generation_ms"]) - float(item["provider_total_ms"])
        for item in completed
        if item.get("provider_total_ms") is not None
    ]
    cached_prompt_tokens = [
        float(item.get("cached_prompt_tokens") or 0) for item in completed
    ]
    failures = Counter(
        str(item["error_category"]) for item in results if not item.get("completed")
    )
    return {
        "requested_count": len(results),
        "completed_count": len(completed),
        "success_rate": len(completed) / len(results) if results else 0.0,
        "failure_category_distribution": dict(sorted(failures.items())),
        "answered_rate": _mean(
            [item.get("status") == "answered" for item in completed]
        ),
        "json_and_evidence_validation_rate": _mean(
            [not item["quality"]["invalid_output_condition"] for item in completed]
        ),
        "all_sources_validated_rate": _mean(
            [item["quality"]["all_sources_validated"] for item in completed]
        ),
        "citation_free_answered_block_count": sum(
            item["quality"]["citation_free_answered_blocks"] for item in completed
        ),
        "mean_reference_token_f1": _mean(
            [item["quality"]["reference_token_f1"] for item in completed]
        ),
        "mean_gold_child_citation_recall": _mean(
            [item["quality"]["gold_child_citation_recall"] for item in completed]
        ),
        "gold_document_citation_hit_rate": _mean(
            [item["quality"]["gold_document_citation_hit"] for item in completed]
        ),
        "generation_latency_ms": _latency(generation),
        "request_attempt_latency_ms": _latency(attempts),
        "projected_e2e_latency_ms": _latency(projected),
        "provider_latency_ms": _latency(provider),
        "client_network_retry_overhead_ms": _latency(client_overhead),
        "mean_cached_prompt_tokens": _mean(cached_prompt_tokens),
    }


def _mean(values: list[Any]) -> float:
    return statistics.mean(float(value) for value in values) if values else 0.0


def _latency(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "p95": None, "min": None, "max": None}
    ordered = sorted(values)
    return {
        "mean": statistics.mean(ordered),
        "median": statistics.median(ordered),
        "p95": ordered[math.ceil(0.95 * len(ordered)) - 1],
        "min": ordered[0],
        "max": ordered[-1],
    }
