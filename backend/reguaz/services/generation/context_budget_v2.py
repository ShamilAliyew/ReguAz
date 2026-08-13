from __future__ import annotations

from typing import Any, Protocol

from backend.reguaz.services.generation.prompt_v2 import V2PromptBuilder
from backend.reguaz.services.generation.v2_models import (
    EvidenceItem,
    V2GenerationSettings,
)


class ChatTokenizer(Protocol):
    context_window: int

    def count_chat_tokens(self, user_content: str) -> int: ...


class V2ContextBudgetManager:
    def __init__(
        self,
        llm: ChatTokenizer,
        settings: V2GenerationSettings,
    ) -> None:
        self.llm = llm
        self.settings = settings

    def fit(
        self,
        question: str,
        evidence: list[EvidenceItem],
    ) -> tuple[list[EvidenceItem], str, dict[str, Any]]:
        hard_input_limit = (
            self.llm.context_window
            - self.settings.reserved_output_tokens
            - self.settings.context_safety_margin_tokens
        )
        if hard_input_limit <= 0:
            raise ValueError("generation token reservation leaves no prompt budget")

        included: list[EvidenceItem] = []
        excluded: list[dict[str, str]] = []
        for item in evidence:
            candidate = item.model_copy(deep=True)
            candidate.evidence_id = f"E{len(included) + 1}"
            trial = [*included, candidate]
            user_content = V2PromptBuilder.build_user_content(question, trial)
            token_count = self.llm.count_chat_tokens(user_content)
            if token_count <= hard_input_limit:
                included.append(candidate)
            else:
                excluded.append(
                    {
                        "artifact_id": str(item.chunk_id or item.parent_chunk_id),
                        "reason": "final rendered chat prompt would exceed budget",
                    }
                )

        missing_seeds = [
            item
            for item in evidence
            if item.role == "seed" and not _contains(included, item)
        ]
        user_content = V2PromptBuilder.build_user_content(question, included)
        final_tokens = self.llm.count_chat_tokens(user_content)
        if final_tokens > hard_input_limit:
            raise ValueError("final rendered prompt exceeds context budget")
        diagnostics = {
            "context_window_tokens": self.llm.context_window,
            "prompt_input_limit_tokens": hard_input_limit,
            "prompt_tokens": final_tokens,
            "reserved_output_tokens": self.settings.reserved_output_tokens,
            "safety_margin_tokens": self.settings.context_safety_margin_tokens,
            "selected_evidence_count": len(included),
            "seed_count": sum(item.role == "seed" for item in included),
            "expanded_evidence_count": sum(
                item.role == "expanded" for item in included
            ),
            "excluded_evidence_count": len(excluded),
            "excluded": excluded,
            "required_seed_overflow": bool(missing_seeds),
        }
        return included, user_content, diagnostics


def _contains(selected: list[EvidenceItem], target: EvidenceItem) -> bool:
    target_id = target.chunk_id or target.parent_chunk_id
    return any(
        (item.chunk_id or item.parent_chunk_id) == target_id for item in selected
    )
