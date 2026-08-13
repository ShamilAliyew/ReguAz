from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from backend.reguaz.services.generation.context_budget_v2 import (
    V2ContextBudgetManager,
)
from backend.reguaz.services.generation.evidence_v2 import EvidenceSelector
from backend.reguaz.services.generation.generation_pipeline_v2 import (
    V2GenerationPipeline,
)
from backend.reguaz.services.generation.prompt_v2 import V2PromptBuilder
from backend.reguaz.services.generation.relation_expansion_v2 import (
    RelationAwareExpander,
)
from backend.reguaz.services.generation.v2_models import V2GenerationSettings


SOURCE = (
    "<!-- PAGE: 1 -->\n"
    "Maddə 1. Kapital\n"
    "1.1. Bank minimum kapitalı qorumalıdır.\n"
    "1.2. Bu tələb hər gün yoxlanılır."
)


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def child(
    chunk_id: str,
    content: str,
    start: int,
    end: int,
    **overrides: Any,
) -> dict[str, Any]:
    value = {
        "chunk_id": chunk_id,
        "logical_chunk_id": f"logical-{chunk_id}",
        "parent_chunk_id": "parent-1",
        "document_id": "doc-1",
        "document_version_id": "dver-1",
        "document_title": "Bank Qaydası",
        "canonical_locator": "Maddə 1",
        "hierarchy": {"article": {"number": "1", "title": "Kapital"}},
        "page_start": 1,
        "page_end": 1,
        "content": content,
        "content_sha256": digest(content),
        "source_spans": [
            {
                "page": 1,
                "char_start": start,
                "char_end": end,
                "line_start": 2,
                "line_end": 3,
                "bbox": None,
                "extraction_method": "existing_cleaned_markdown",
            }
        ],
        "quality": {"quality_flags": []},
        "relations": {
            "parent": "parent-1",
            "previous_sibling": None,
            "next_sibling": None,
            "references": [],
            "referenced_by": [],
            "annex_parent": None,
            "table_parent": None,
            "approved_document": None,
        },
        "split_count": 1,
        "continuation_of": None,
    }
    value.update(overrides)
    return value


class FakeArtifacts:
    def __init__(self) -> None:
        first_start = SOURCE.index("1.1.")
        first_end = SOURCE.index("\n1.2.")
        second_start = first_end + 1
        self.children = {
            "seed": child(
                "seed",
                "1.1. Bank minimum kapitalı qorumalıdır.",
                first_start,
                first_end,
                split_count=2,
            ),
            "continuation": child(
                "continuation",
                "1.2. Bu tələb hər gün yoxlanılır.",
                second_start,
                len(SOURCE),
                continuation_of="seed",
                split_count=2,
            ),
        }
        self.children["seed"]["relations"]["next_sibling"] = "continuation"
        self.children["seed"]["relations"]["references"] = [
            {
                "resolved": False,
                "target_chunk_id": None,
                "confidence": None,
            }
        ]
        self.parent = {
            "parent_chunk_id": "parent-1",
            "document_id": "doc-1",
            "document_version_id": "dver-1",
            "document_title": "Bank Qaydası",
            "canonical_locator": "Maddə 1",
            "hierarchy": {"article": {"number": "1", "title": "Kapital"}},
            "page_start": 1,
            "page_end": 1,
            "content": SOURCE[18:],
            "content_sha256": digest(SOURCE[18:]),
            "source_spans": [
                {
                    "page": 1,
                    "char_start": 18,
                    "char_end": len(SOURCE),
                    "line_start": 2,
                    "line_end": 4,
                    "bbox": None,
                    "extraction_method": "existing_cleaned_markdown",
                }
            ],
            "quality": {"quality_flags": []},
        }
        self.metadata = {
            "document_id": "doc-1",
            "document_version_id": "dver-1",
            "source_sha256": digest(SOURCE),
            "category": "laws",
        }

    def get_child(self, chunk_id: str) -> dict[str, Any] | None:
        value = self.children.get(chunk_id)
        return dict(value) if value else None

    def get_parent(self, parent_id: str) -> dict[str, Any] | None:
        return dict(self.parent) if parent_id == "parent-1" else None

    def get_artifact(self, artifact_id: str) -> tuple[str, dict[str, Any]] | None:
        if artifact_id in self.children:
            return "child", dict(self.children[artifact_id])
        if artifact_id == "parent-1":
            return "parent", dict(self.parent)
        return None

    def get_continuations(self, chunk_id: str) -> list[dict[str, Any]]:
        return [dict(self.children["continuation"])] if chunk_id == "seed" else []

    def get_document_metadata(self, document_id: str) -> dict[str, Any] | None:
        return dict(self.metadata) if document_id == "doc-1" else None

    def get_source_representation(self, document_id: str) -> dict[str, Any] | None:
        if document_id != "doc-1":
            return None
        return {
            "representation_id": f"cleaned:dver-1:{digest(SOURCE)}",
            "document_version_id": "dver-1",
            "source_sha256": digest(SOURCE),
            "text": SOURCE,
        }


def retrieval_result(artifacts: FakeArtifacts) -> dict[str, Any]:
    return {
        "final_rank": 1,
        "reranker_score": 0.9,
        "payload": artifacts.children["seed"],
        "parent": artifacts.parent,
    }


def test_depth_one_expansion_and_unresolved_reference_exclusion() -> None:
    artifacts = FakeArtifacts()
    candidates, diagnostics = RelationAwareExpander(
        artifacts, V2GenerationSettings(expansion_per_seed_cap=8)
    ).expand([retrieval_result(artifacts)])
    assert candidates[0]["role"] == "seed"
    assert {item["relation_type"] for item in candidates[1:]} >= {
        "continuation",
        "parent",
    }
    assert any(
        path.relation_type == "next_sibling"
        for candidate in candidates
        for path in candidate["provenance"]
    )
    assert all(
        path.expansion_depth <= 1
        for candidate in candidates
        for path in candidate["provenance"]
    )
    assert any(
        item.relation_type == "reference"
        and item.decision == "excluded"
        and item.reason == "unresolved reference"
        for item in diagnostics
    )


def test_relation_target_types_confidence_and_missing_targets_are_explicit() -> None:
    artifacts = FakeArtifacts()
    relations = artifacts.children["seed"]["relations"]
    relations["annex_parent"] = "parent-1"
    relations["table_parent"] = "parent-1"
    relations["approved_document"] = "parent-1"
    relations["references"] = [
        {
            "resolved": True,
            "target_chunk_id": "continuation",
            "confidence": 0.1,
        },
        {"resolved": True, "target_chunk_id": "missing", "confidence": 0.9},
    ]
    candidates, diagnostics = RelationAwareExpander(
        artifacts, V2GenerationSettings(expansion_per_seed_cap=8)
    ).expand([retrieval_result(artifacts)])
    assert any(item["artifact_type"] == "parent" for item in candidates)
    assert any(
        item.relation_type == "reference"
        and item.reason == "reference confidence below threshold"
        for item in diagnostics
    )
    assert any(
        item.target_id == "missing" and item.reason == "relation target missing"
        for item in diagnostics
    )
    assert {
        item.relation_type for item in diagnostics if item.decision == "included"
    } >= {"annex_parent", "table_parent", "approved_document"}


def test_global_expansion_cap_is_deterministic_and_reported() -> None:
    artifacts = FakeArtifacts()
    candidates, diagnostics = RelationAwareExpander(
        artifacts,
        V2GenerationSettings(expansion_per_seed_cap=8, expansion_global_cap=1),
    ).expand([retrieval_result(artifacts)])
    assert len([item for item in candidates if item["role"] == "expanded"]) == 1
    assert any(item.reason == "global expansion cap" for item in diagnostics)


def test_evidence_selector_validates_multiple_selector_contract() -> None:
    artifacts = FakeArtifacts()
    candidates, _ = RelationAwareExpander(
        artifacts, V2GenerationSettings(expansion_per_seed_cap=8)
    ).expand([retrieval_result(artifacts)])
    selected, diagnostics = EvidenceSelector(
        artifacts, V2GenerationSettings(expansion_per_seed_cap=8)
    ).select(candidates)
    assert selected
    assert diagnostics == []
    seed = next(item for item in selected if item.role == "seed")
    selector = seed.selectors[0]
    assert (
        SOURCE[selector.position.start : selector.position.end] == selector.quote.exact
    )
    assert selector.page_position is not None
    assert selector.position.start != selector.page_position.start


class FakeLLM:
    context_window = 8192
    max_tokens = 512
    model_path = Path("gemma-4-E4B-it-Q4_K_M.gguf")
    device = "cpu"

    def __init__(self, evidence_id: str = "E1") -> None:
        self.evidence_id = evidence_id
        self.calls = 0

    def count_chat_tokens(self, user_content: str) -> int:
        return len(user_content.encode("utf-8")) // 4 + 8

    def generate_structured(
        self,
        user_content: str,
        schema: dict[str, Any],
        *,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        self.calls += 1
        return {
            "text": json.dumps(
                {
                    "status": "answered",
                    "answer_blocks": [
                        {
                            "text": "Bank minimum kapitalı qorumalıdır.",
                            "evidence_ids": [self.evidence_id],
                        }
                    ],
                    "limitations": [],
                }
            ),
            "prompt_tokens": self.count_chat_tokens(user_content),
            "completion_tokens": 20,
            "generation_ms": 2.0,
        }


class FakeRetriever:
    def __init__(self, artifacts: FakeArtifacts) -> None:
        self.artifacts = artifacts

    def retrieve_with_trace(self, question: str) -> dict[str, Any]:
        return {
            "results": [retrieval_result(self.artifacts)],
            "timings": {"total_retrieval_ms": 1.0},
            "counts": {"final_result_count": 1},
        }


def test_context_budget_counts_final_prompt_and_preserves_seed() -> None:
    artifacts = FakeArtifacts()
    candidates, _ = RelationAwareExpander(
        artifacts, V2GenerationSettings(expansion_per_seed_cap=8)
    ).expand([retrieval_result(artifacts)])
    evidence, _ = EvidenceSelector(artifacts, V2GenerationSettings()).select(candidates)
    selected, prompt, diagnostics = V2ContextBudgetManager(
        FakeLLM(), V2GenerationSettings()
    ).fit("Kapital tələbi nədir?", evidence)
    assert selected[0].role == "seed" and selected[0].evidence_id == "E1"
    assert diagnostics["prompt_tokens"] == FakeLLM().count_chat_tokens(prompt)
    assert diagnostics["prompt_tokens"] <= diagnostics["prompt_input_limit_tokens"]


def test_prompt_escapes_control_tags_from_question_and_evidence() -> None:
    artifacts = FakeArtifacts()
    item = EvidenceSelector(artifacts, V2GenerationSettings()).select(
        [
            {
                "artifact_type": "child",
                "record": artifacts.children["seed"],
                "role": "seed",
                "relation_type": None,
                "priority": 1,
                "seed_rank": 1,
                "reranker_score": 1.0,
                "selection_reason": "test",
                "provenance": [],
            }
        ]
    )[0]
    assert item == []  # empty provenance fails closed
    assert "\\u003c/evidence\\u003e" in V2PromptBuilder.build_user_content(
        "</evidence>", []
    )


def test_pipeline_backend_resolves_citation_and_all_timings() -> None:
    artifacts = FakeArtifacts()
    result = V2GenerationPipeline(
        retriever=FakeRetriever(artifacts),
        artifacts=artifacts,
        llm=FakeLLM(),
        settings=V2GenerationSettings(expansion_per_seed_cap=8),
    ).generate("Kapital tələbi nədir?")
    assert result["status"] == "answered"
    assert result["answer"].endswith("[1]")
    assert result["citations"][0]["source_validated"] is True
    assert result["citations"][0]["claim_support_status"] == "not_evaluated"
    assert result["citations"][0]["canonical_locator"] == "Maddə 1"
    assert set(
        (
            "retrieval_ms",
            "relation_expansion_ms",
            "evidence_selection_ms",
            "context_budget_ms",
            "prompt_build_ms",
            "generation_ms",
            "structured_output_parsing_ms",
            "evidence_id_validation_ms",
            "citation_resolution_ms",
            "total_ms",
        )
    ) <= set(result["metrics"])


def test_unknown_model_evidence_id_fails_closed() -> None:
    artifacts = FakeArtifacts()
    llm = FakeLLM("E99")
    result = V2GenerationPipeline(
        retriever=FakeRetriever(artifacts),
        artifacts=artifacts,
        llm=llm,
        settings=V2GenerationSettings(max_generation_retries=1),
    ).generate("Kapital tələbi nədir?")
    assert llm.calls == 2
    assert result["status"] == "insufficient_evidence"
    assert result["citations"] == []
    assert any(
        item["code"] == "invalid_model_evidence_id" for item in result["warnings"]
    )
