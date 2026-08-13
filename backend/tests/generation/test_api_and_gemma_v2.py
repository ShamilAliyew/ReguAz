from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from backend.app.core.config import Settings
from backend.app.schemas.chat import ChatRequest
from backend.app.services.chat_service import ChatService
from backend.reguaz.services.generation.gemma_service import GemmaService
from backend.reguaz.services.generation import gemma_service as gemma_module


class FakePipeline:
    def generate(self, question: str) -> dict[str, Any]:
        return {
            "request_id": "request-1",
            "question": question,
            "status": "answered",
            "answer": "Tələb mövcuddur. [1]",
            "answer_blocks": [
                {
                    "text": "Tələb mövcuddur.",
                    "evidence_ids": ["E1"],
                    "citation_numbers": [1],
                }
            ],
            "sources": [
                {
                    "citation": 1,
                    "chunk_id": "chunk-1",
                    "document_id": "doc-1",
                    "document_name": "Bank Qaydası",
                    "category": "laws",
                    "page": 1,
                    "chunk_preview": "Exact evidence",
                    "canonical_locator": "Maddə 1",
                    "role": "seed",
                    "selectors": [],
                }
            ],
            "citations": [
                {
                    "citation": 1,
                    "evidence_id": "E1",
                    "role": "seed",
                    "relation_type": None,
                    "chunk_id": "chunk-1",
                    "logical_chunk_id": "logical-1",
                    "parent_chunk_id": "parent-1",
                    "document_id": "doc-1",
                    "document_version_id": "dver-1",
                    "document_title": "Bank Qaydası",
                    "category": "laws",
                    "canonical_locator": "Maddə 1",
                    "hierarchy": {},
                    "page_start": 1,
                    "page_end": 1,
                    "exact_quotes": ["Exact evidence"],
                    "selectors": [],
                    "content_sha256": "a" * 64,
                    "source_sha256": "b" * 64,
                    "source_validated": True,
                    "claim_support_status": "not_evaluated",
                    "provenance": [],
                }
            ],
            "limitations": [],
            "warnings": [],
            "pipeline_version": "v2",
            "model": {"provider": "gemma", "device": "cpu"},
            "metrics": {
                "retrieval_time": 0.001,
                "generation_time": 0.002,
                "total_time": 0.004,
                "retrieval_ms": 1.0,
                "generation_ms": 2.0,
                "api_mapping_ms": 0.0,
                "total_ms": 4.0,
            },
        }


def test_chat_service_preserves_v1_fields_and_adds_v2_contract() -> None:
    response = ChatService(FakePipeline()).process_query(
        ChatRequest(question="Məxfi istifadəçi sualı")
    )
    assert response.answer == "Tələb mövcuddur. [1]"
    assert response.sources[0].document_name == "Bank Qaydası"
    assert response.pipeline_version == "v2"
    assert response.citations[0].source_validated is True
    assert response.citations[0].claim_support_status == "not_evaluated"
    assert response.metrics.api_mapping_ms is not None
    assert response.metrics.total_ms is not None
    assert response.metrics.total_ms > 4.0


def test_pipeline_version_defaults_to_v2_and_invalid_value_fails() -> None:
    assert Settings(_env_file=None).REGUAZ_PIPELINE_VERSION == "v2"
    assert Settings(_env_file=None, LLM_TYPE="nvidia_gpt_oss").LLM_TYPE == (
        "nvidia_gpt_oss"
    )
    assert Settings(_env_file=None, LLM_TYPE="groq_gpt_oss_20b").LLM_TYPE == (
        "groq_gpt_oss_20b"
    )
    with pytest.raises(ValueError):
        Settings(_env_file=None, REGUAZ_PIPELINE_VERSION="v3")


def test_chat_service_passes_selected_v2_model() -> None:
    pipeline = FakePipeline()
    calls: list[tuple[str, str]] = []

    def generate_for_model(question: str, model: str) -> dict[str, Any]:
        calls.append((question, model))
        return pipeline.generate(question)

    pipeline.generate_for_model = generate_for_model  # type: ignore[attr-defined]
    ChatService(pipeline).process_query(
        ChatRequest(question="Tələb nədir?", llm_model="groq_gpt_oss_20b")
    )
    assert calls == [("Tələb nədir?", "groq_gpt_oss_20b")]


class FakeLlama:
    def __init__(self, template: str | None) -> None:
        self.metadata = {
            "tokenizer.ggml.bos_token_id": "2",
            "tokenizer.ggml.eos_token_id": "1",
        }
        if template is not None:
            self.metadata["tokenizer.chat_template"] = template
        self.chat_handler = None

    def detokenize(self, tokens: list[int], special: bool = False) -> bytes:
        return b"<bos>" if tokens == [2] else b"<eos>"


GEMMA_TEMPLATE = (
    "{{ bos_token }}{% for message in messages %}"
    "<start_of_turn>{{ message['role'] }}\n{{ message['content'] }}<end_of_turn>\n"
    "{% endfor %}{% if add_generation_prompt %}<start_of_turn>model\n{% endif %}"
)


def service_with_fake_llama(template: str | None) -> GemmaService:
    service = GemmaService.__new__(GemmaService)
    service._llm = FakeLlama(template)
    service._chat_formatter = None
    return service


def test_gemma_uses_verified_gguf_template_with_only_user_model_roles() -> None:
    service = service_with_fake_llama(GEMMA_TEMPLATE)
    service._configure_verified_chat_template()
    prompt = service.render_chat_prompt("Qayda")
    assert "<start_of_turn>user\nQayda" in prompt
    assert prompt.endswith("<start_of_turn>model\n")
    assert "<start_of_turn>system" not in prompt


def test_gemma_rejects_missing_gguf_chat_template() -> None:
    service = service_with_fake_llama(None)
    with pytest.raises(RuntimeError, match="tokenizer.chat_template"):
        service._configure_verified_chat_template()


def test_gemma_accelerator_failure_retries_with_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = GemmaService.__new__(GemmaService)
    service.model_path = Path("model.gguf")
    service.context_window = 2048
    service.temperature = 0.1
    service.max_tokens = 256
    service.top_p = 0.9
    service.top_k = 40
    service.repeat_penalty = 1.1
    service.seed = 42
    service.gpu_layers = -1
    service._llm = None
    calls: list[int] = []

    def create(gpu_layers: int) -> SimpleNamespace:
        calls.append(gpu_layers)
        if gpu_layers == -1:
            raise ValueError("accelerator unavailable")
        return SimpleNamespace(chat_format="fake")

    monkeypatch.setattr(service, "_create_llama", create)
    monkeypatch.setattr(service, "_configure_verified_chat_template", lambda: None)
    monkeypatch.setattr(
        gemma_module.llama_cpp,
        "llama_print_system_info",
        lambda: b"MTL available",
    )
    service._load_model()
    assert calls == [-1, 0]
    assert service.device == "cpu"
