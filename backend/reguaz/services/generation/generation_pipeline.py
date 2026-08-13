"""
backend/reguaz/services/generation/generation_pipeline.py
"""

import time
from typing import Any

from backend.reguaz.services.generation.base_llm import BaseLLM
from backend.reguaz.services.generation.context_budget_manager import ContextBudgetManager
from backend.reguaz.services.generation.prompt_builder import PromptBuilder
from backend.reguaz.utils.logger import get_logger

logger = get_logger(__name__, "generation_pipeline.log")


class GenerationPipeline:
    """
    Orchestrates the complete RAG generation process.

    Coordinates between the retriever, metadata lookup, ContextBudgetManager, PromptBuilder, and the LLM.
    """

    def __init__(
        self,
        retriever: Any,
        chunk_lookup: dict[str, dict[str, Any]],
        llm: BaseLLM,
        top_k: int = 5,
    ) -> None:
        """
        Parameters
        ----------
        retriever : Any
            An instance of the retriever (e.g., HybridQdrantRetriever) that provides a `retrieve(query, top_k)` method.
        chunk_lookup : dict[str, dict[str, Any]]
            A lookup table mapping chunk_id to its full metadata dict.
        llm : BaseLLM
            The language model service.
        top_k : int
            The number of chunks to retrieve and include in the context.
        """
        self.retriever = retriever
        self.chunk_lookup = chunk_lookup
        self.llm = llm
        self.top_k = top_k
        self.budget_manager = ContextBudgetManager(llm=self.llm)
        logger.info("GenerationPipeline initialized with top_k=%d.", self.top_k)

    def generate(self, question: str) -> dict[str, Any]:
        """
        Run the full generation pipeline for a given question.

        Parameters
        ----------
        question : str
            The user's query.

        Returns
        -------
        dict[str, Any]
            The generation result containing the question, answer, hydrated sources, and timing metrics.
        """
        logger.info("Processing question: '%s...'", question[:50])
        total_t0 = time.perf_counter()
        
        # 1. Retrieval
        retrieval_t0 = time.perf_counter()
        retrieval_results = self.retriever.retrieve(question, top_k=self.top_k)
        retrieval_time = time.perf_counter() - retrieval_t0
        logger.info("Retrieved %d chunks in %.3fs.", len(retrieval_results), retrieval_time)
        
        # 2. Metadata hydration
        hydrated_sources = []
        logger.info("Retrieved %d chunks. Chunk IDs: %s", len(retrieval_results), [res["id"] for res in retrieval_results])
        
        for idx, res in enumerate(retrieval_results, start=1):
            chunk_id = res["id"]
            if chunk_id in self.chunk_lookup:
                metadata = self.chunk_lookup[chunk_id].copy()
                # Merge retrieval metrics and scores into source metadata
                metadata["rerank_score"] = res.get("rerank_score")
                metadata["rrf_score"] = res.get("rrf_score")
                metadata["semantic_rank"] = res.get("semantic_rank")
                metadata["bm25_rank"] = res.get("bm25_rank")
                hydrated_sources.append(metadata)
            else:
                logger.warning("Chunk ID '%s' not found in chunk lookup. Skipping.", chunk_id)
                
        # 3. Context Budget Management
        filtered_sources = self.budget_manager.filter_chunks(hydrated_sources)
                
        # 4. Prompt construction
        prompt_t0 = time.perf_counter()
        prompt = PromptBuilder.build_prompt(question, filtered_sources, token_counter=self.llm.count_tokens)
        prompt_time = time.perf_counter() - prompt_t0
        logger.debug("Prompt created in %.3fs.", prompt_time)
        
        # 4. LLM generation
        gen_t0 = time.perf_counter()
        try:
            answer = self.llm.generate(prompt)
        except Exception as e:
            logger.error("LLM generation failed: %s", e)
            raise
        gen_time = time.perf_counter() - gen_t0
        logger.info("Generation completed in %.3fs.", gen_time)
        
        total_time = time.perf_counter() - total_t0
        logger.info("Total pipeline time: %.3fs.", total_time)
        
        return {
            "question": question,
            "answer": answer.strip(),
            "sources": filtered_sources,
            "metrics": {
                "retrieval_time": retrieval_time,
                "prompt_build_time": prompt_time,
                "generation_time": gen_time,
                "total_time": total_time,
            }
        }
