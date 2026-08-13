from __future__ import annotations

import json
import time
import uuid
from typing import Any, Protocol

from pydantic import ValidationError

from backend.reguaz.retrieval.v2_artifacts import V2ArtifactStore
from backend.reguaz.services.generation.citation_v2 import (
    BackendCitationResolver,
    CitationResolutionError,
)
from backend.reguaz.services.generation.context_budget_v2 import (
    V2ContextBudgetManager,
)
from backend.reguaz.services.generation.evidence_v2 import EvidenceSelector
from backend.reguaz.services.generation.prompt_v2 import V2PromptBuilder
from backend.reguaz.services.generation.relation_expansion_v2 import (
    RelationAwareExpander,
)
from backend.reguaz.services.generation.v2_models import (
    STRUCTURED_GENERATION_SCHEMA,
    StructuredGeneration,
    V2GenerationSettings,
)
from backend.reguaz.utils.logger import get_logger


logger = get_logger(__name__, "generation_v2.log")


class StructuredLLM(Protocol):
    context_window: int
    max_tokens: int
    model_path: Any
    device: str

    def count_chat_tokens(self, user_content: str) -> int: ...

    def generate_structured(
        self,
        user_content: str,
        schema: dict[str, Any],
        *,
        max_tokens: int | None = None,
    ) -> dict[str, Any]: ...


class V2GenerationPipeline:
    """Existing V2 retrieval extended through validated local generation."""

    def __init__(
        self,
        *,
        retriever: Any,
        artifacts: V2ArtifactStore,
        llm: StructuredLLM,
        settings: V2GenerationSettings | None = None,
    ) -> None:
        self.retriever = retriever
        self.artifacts = artifacts
        self.llm = llm
        self.settings = settings or V2GenerationSettings(
            reserved_output_tokens=llm.max_tokens
        )
        self.expander = RelationAwareExpander(artifacts, self.settings)
        self.evidence_selector = EvidenceSelector(artifacts, self.settings)
        self.budget = V2ContextBudgetManager(llm, self.settings)
        self.citations = BackendCitationResolver(artifacts)

    def generate(self, question: str) -> dict[str, Any]:
        if not question.strip():
            raise ValueError("question must not be empty")
        request_id = str(uuid.uuid4())
        total_started = time.perf_counter()
        warnings: list[dict[str, Any]] = []

        stage = time.perf_counter()
        retrieval_trace = self.retriever.retrieve_with_trace(question)
        retrieval_ms = _elapsed_ms(stage)

        stage = time.perf_counter()
        expanded_candidates, expansion_diagnostics = self.expander.expand(
            retrieval_trace["results"]
        )
        relation_expansion_ms = _elapsed_ms(stage)

        stage = time.perf_counter()
        evidence, evidence_diagnostics = self.evidence_selector.select(
            expanded_candidates
        )
        evidence_selection_ms = _elapsed_ms(stage)
        warnings.extend(
            {"code": item["code"], "details": item}
            for item in evidence_diagnostics
            if item["code"] == "invalid_evidence_candidate"
        )

        stage = time.perf_counter()
        selected, user_content, budget_diagnostics = self.budget.fit(question, evidence)
        context_budget_ms = _elapsed_ms(stage)

        stage = time.perf_counter()
        user_content = V2PromptBuilder.build_user_content(question, selected)
        prompt_tokens = self.llm.count_chat_tokens(user_content)
        prompt_build_ms = _elapsed_ms(stage)

        generated: StructuredGeneration
        generation_ms = 0.0
        parsing_ms = 0.0
        evidence_validation_ms = 0.0
        generation_usage: dict[str, Any] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": 0,
            "cached_prompt_tokens": 0,
            "provider_queue_ms": None,
            "provider_prompt_ms": None,
            "provider_completion_ms": None,
            "provider_total_ms": None,
        }
        if budget_diagnostics["required_seed_overflow"] or not selected:
            generated = StructuredGeneration(
                status="insufficient_evidence",
                limitations=[
                    "Tələb olunan əsas sübut təhlükəsiz context büdcəsinə sığmadı."
                ],
            )
            warnings.append({"code": "required_evidence_overflow"})
        else:
            generated, generation_stats, generation_warnings = self._generate_validated(
                user_content, {item.evidence_id for item in selected}
            )
            generation_ms = generation_stats["generation_ms"]
            parsing_ms = generation_stats["structured_output_parsing_ms"]
            evidence_validation_ms = generation_stats["evidence_id_validation_ms"]
            generation_usage = {
                "prompt_tokens": generation_stats["prompt_tokens"],
                "completion_tokens": generation_stats["completion_tokens"],
                "cached_prompt_tokens": generation_stats["cached_prompt_tokens"],
                "provider_queue_ms": generation_stats["provider_queue_ms"],
                "provider_prompt_ms": generation_stats["provider_prompt_ms"],
                "provider_completion_ms": generation_stats["provider_completion_ms"],
                "provider_total_ms": generation_stats["provider_total_ms"],
            }
            warnings.extend(generation_warnings)

        stage = time.perf_counter()
        try:
            citation_output = self.citations.resolve(generated, selected)
        except CitationResolutionError as exc:
            generated = StructuredGeneration(
                status="insufficient_evidence",
                limitations=["Mənbə doğrulaması tamamlanmadı."],
            )
            citation_output = self.citations.resolve(generated, selected)
            warnings.append(
                {"code": "citation_validation_failed", "error_category": str(exc)}
            )
        citation_resolution_ms = _elapsed_ms(stage)

        if (
            generated.status == "insufficient_evidence"
            and not citation_output["answer"]
        ):
            citation_output["answer"] = (
                "Mövcud sübutlar əsasında etibarlı cavab vermək mümkün deyil."
            )
        elif (
            generated.status == "conflicting_evidence" and not citation_output["answer"]
        ):
            citation_output["answer"] = "Mənbələr arasında ziddiyyət aşkarlandı."

        total_ms = _elapsed_ms(total_started)
        timings = {
            "retrieval_ms": retrieval_ms,
            "relation_expansion_ms": relation_expansion_ms,
            "evidence_selection_ms": evidence_selection_ms,
            "context_budget_ms": context_budget_ms,
            "prompt_build_ms": prompt_build_ms,
            "generation_ms": generation_ms,
            "provider_queue_ms": generation_usage["provider_queue_ms"],
            "provider_prompt_ms": generation_usage["provider_prompt_ms"],
            "provider_completion_ms": generation_usage["provider_completion_ms"],
            "provider_total_ms": generation_usage["provider_total_ms"],
            "structured_output_parsing_ms": parsing_ms,
            "evidence_id_validation_ms": evidence_validation_ms,
            "citation_resolution_ms": citation_resolution_ms,
            "api_mapping_ms": 0.0,
            "total_ms": total_ms,
            "retrieval_time": retrieval_ms / 1000.0,
            "generation_time": generation_ms / 1000.0,
            "total_time": total_ms / 1000.0,
        }
        logger.info(
            "V2_GENERATION request_id=%s status=%s total_ms=%.2f prompt_tokens=%d evidence=%d citations=%d warnings=%d device=%s",
            request_id,
            generated.status,
            total_ms,
            generation_usage["prompt_tokens"],
            len(selected),
            len(citation_output["citations"]),
            len(warnings),
            getattr(self.llm, "device", "unknown"),
        )
        return {
            "request_id": request_id,
            "question": question,
            "status": generated.status,
            **citation_output,
            "limitations": generated.limitations,
            "metrics": timings,
            "warnings": warnings,
            "pipeline_version": "v2",
            "model": {
                "provider": getattr(self.llm, "provider", "unknown"),
                "model_file": getattr(
                    self.llm.model_path, "name", str(self.llm.model_path)
                ),
                "model_id": getattr(self.llm, "model_id", None),
                "runtime": getattr(self.llm, "runtime", "unknown"),
                "device": getattr(self.llm, "device", "unknown"),
            },
            "trace": {
                "retrieval": {
                    "timings": retrieval_trace.get("timings", {}),
                    "counts": retrieval_trace.get("counts", {}),
                },
                "relation_expansion": {
                    "candidate_count": len(expanded_candidates),
                    "diagnostics": [
                        item.model_dump(mode="json") for item in expansion_diagnostics
                    ],
                },
                "evidence": {
                    "selected_count": len(selected),
                    "diagnostics": evidence_diagnostics,
                    "budget": budget_diagnostics,
                },
                "generation": generation_usage,
            },
        }

    def _generate_validated(
        self,
        user_content: str,
        allowed_evidence_ids: set[str],
    ) -> tuple[StructuredGeneration, dict[str, Any], list[dict[str, Any]]]:
        warnings: list[dict[str, Any]] = []
        total_generation_ms = 0.0
        total_parsing_ms = 0.0
        total_validation_ms = 0.0
        prompt_tokens = self.llm.count_chat_tokens(user_content)
        completion_tokens = 0
        provider_stats: dict[str, float | int | None] = {
            "cached_prompt_tokens": 0,
            "provider_queue_ms": None,
            "provider_prompt_ms": None,
            "provider_completion_ms": None,
            "provider_total_ms": None,
        }
        attempts = self.settings.max_generation_retries + 1
        for attempt in range(attempts):
            generated = self.llm.generate_structured(
                user_content,
                STRUCTURED_GENERATION_SCHEMA,
                max_tokens=self.settings.reserved_output_tokens,
            )
            total_generation_ms += float(generated["generation_ms"])
            prompt_tokens = int(generated["prompt_tokens"])
            completion_tokens += int(generated["completion_tokens"])
            provider_stats["cached_prompt_tokens"] = int(
                provider_stats["cached_prompt_tokens"] or 0
            ) + int(generated.get("cached_prompt_tokens") or 0)
            for key in (
                "provider_queue_ms",
                "provider_prompt_ms",
                "provider_completion_ms",
                "provider_total_ms",
            ):
                value = generated.get(key)
                if value is not None:
                    provider_stats[key] = float(provider_stats[key] or 0.0) + float(
                        value
                    )
            stage = time.perf_counter()
            try:
                parsed = json.loads(str(generated["text"]))
                result = StructuredGeneration.model_validate(parsed)
            except (json.JSONDecodeError, ValidationError, ValueError):
                total_parsing_ms += _elapsed_ms(stage)
                if attempt + 1 < attempts:
                    warnings.append({"code": "structured_output_retry"})
                    continue
                warnings.append({"code": "structured_output_invalid"})
                return (
                    self._safe_failure(),
                    {
                        "generation_ms": total_generation_ms,
                        "structured_output_parsing_ms": total_parsing_ms,
                        "evidence_id_validation_ms": total_validation_ms,
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        **provider_stats,
                    },
                    warnings,
                )
            total_parsing_ms += _elapsed_ms(stage)
            stage = time.perf_counter()
            try:
                self._validate_evidence_ids(result, allowed_evidence_ids)
            except ValueError:
                total_validation_ms += _elapsed_ms(stage)
                if attempt + 1 < attempts:
                    warnings.append({"code": "evidence_id_retry"})
                    continue
                warnings.append({"code": "invalid_model_evidence_id"})
                return (
                    self._safe_failure(),
                    {
                        "generation_ms": total_generation_ms,
                        "structured_output_parsing_ms": total_parsing_ms,
                        "evidence_id_validation_ms": total_validation_ms,
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        **provider_stats,
                    },
                    warnings,
                )
            total_validation_ms += _elapsed_ms(stage)
            return (
                result,
                {
                    "generation_ms": total_generation_ms,
                    "structured_output_parsing_ms": total_parsing_ms,
                    "evidence_id_validation_ms": total_validation_ms,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    **provider_stats,
                },
                warnings,
            )
        raise AssertionError("unreachable generation validation state")

    @staticmethod
    def _validate_evidence_ids(
        generated: StructuredGeneration, allowed: set[str]
    ) -> None:
        for block in generated.answer_blocks:
            if len(block.evidence_ids) != len(set(block.evidence_ids)):
                raise ValueError("duplicate evidence ID in answer block")
            unknown = set(block.evidence_ids) - allowed
            if unknown:
                raise ValueError("model returned unknown evidence ID")

    @staticmethod
    def _safe_failure() -> StructuredGeneration:
        return StructuredGeneration(
            status="insufficient_evidence",
            limitations=["Modelin strukturlaşdırılmış cavabı doğrulanmadı."],
        )


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000.0
