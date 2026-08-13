"""
backend/reguaz/services/generation/context_budget_manager.py
"""

from typing import Any

from backend.reguaz.services.generation.base_llm import BaseLLM
from backend.reguaz.utils.logger import get_logger

logger = get_logger(__name__, "generation_pipeline.log")


class ContextBudgetManager:
    """
    Manages the prompt token budget for the context chunks in the Generation Pipeline.
    
    Acts as a filter between the Retriever and PromptBuilder. It drops oversized chunks
    if they exceed the remaining budget, ensuring the final assembled prompt does not 
    exceed the LLM's context window.
    """

    def __init__(self, llm: BaseLLM) -> None:
        """
        Parameters
        ----------
        llm : BaseLLM
            The language model service, used to count exact tokens and query the budget.
        """
        self.llm = llm

    def filter_chunks(self, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Filter the chunks based on the LLM's context budget.
        
        Iterates over the chunks in their retrieved order. If a chunk fits within the
        remaining budget, it is kept. Otherwise, it is skipped. The chunk ordering and
        metadata remain completely unmodified.

        Parameters
        ----------
        chunks : list[dict[str, Any]]
            The list of hydrated chunk metadata dictionaries.

        Returns
        -------
        list[dict[str, Any]]
            The filtered list of chunks that safely fit into the prompt budget.
        """
        prompt_budget = self.llm.get_prompt_budget()
        logger.info("ContextBudgetManager initialized filter with budget: %d tokens.", prompt_budget)
        
        filtered_chunks = []
        current_tokens = 0
        
        initial_chunks_count = len(chunks)
        kept_chunks_count = 0
        skipped_chunks_count = 0
        
        kept_chunk_ids = []
        skipped_chunk_ids = []
        
        initial_context_tokens = 0
        
        for chunk in chunks:
            chunk_id = chunk.get("chunk_id") or chunk.get("id", "UNKNOWN")
            
            # The exact content being injected into the prompt.
            # We estimate the token count precisely based on the raw text content.
            text_content = chunk.get("text") or chunk.get("content") or ""
            
            chunk_tokens = self.llm.count_tokens(text_content)
            initial_context_tokens += chunk_tokens
            
            if current_tokens + chunk_tokens <= prompt_budget:
                filtered_chunks.append(chunk)
                current_tokens += chunk_tokens
                kept_chunks_count += 1
                kept_chunk_ids.append(chunk_id)
                decision = "KEPT"
            else:
                skipped_chunks_count += 1
                skipped_chunk_ids.append(chunk_id)
                decision = "SKIPPED"
                
            logger.info(
                "Chunk %s\n"
                "Tokens: %d\n"
                "Current Budget: %d / %d\n"
                "Decision: %s",
                chunk_id, chunk_tokens, current_tokens, prompt_budget, decision
            )
            
        remaining_budget = prompt_budget - current_tokens
        reduction_pct = 0.0
        if initial_context_tokens > 0:
            reduction_pct = ((initial_context_tokens - current_tokens) / initial_context_tokens) * 100
            
        logger.info(
            "ContextBudgetManager Final Summary\n"
            "----------------------------------\n"
            "Initial Chunks: %d\n"
            "Kept Chunks: %d\n"
            "Skipped Chunks: %d\n"
            "Initial Context Tokens: %d\n"
            "Final Context Tokens: %d\n"
            "Remaining Budget: %d\n"
            "Reduction: %.1f%%\n\n"
            "Kept Chunk IDs: %s\n"
            "Skipped Chunk IDs: %s",
            initial_chunks_count, kept_chunks_count, skipped_chunks_count,
            initial_context_tokens, current_tokens, remaining_budget, reduction_pct,
            kept_chunk_ids, skipped_chunk_ids
        )
        
        return filtered_chunks
