"""
app/services/chat_service.py

Service layer for Chat functionality.
Acts as a buffer between the API controller/router and the core AI RAG pipeline.
"""

from __future__ import annotations

import uuid
import time
from typing import Any

from backend.app.schemas.chat import (
    AnswerBlockResponse,
    ChatRequest,
    ChatResponse,
    CitationResponse,
    SourceDocument,
    MetricsResponse,
)
from backend.reguaz.utils.logger import get_logger

# Initialize project-wide logger for API requests
logger = get_logger("app.api.requests", "api_requests.log")


class ChatService:
    """
    Service responsible for orchestrating chat queries and mapping
    the output of the GenerationPipeline to clean API schemas.
    """

    def __init__(self, generation_pipeline: Any) -> None:
        """
        Initialize the ChatService with an instance of the GenerationPipeline.

        Args:
            generation_pipeline: The backend RAG generation pipeline instance.
        """
        self.generation_pipeline = generation_pipeline

    def process_query(self, request: ChatRequest) -> ChatResponse:
        """
        Executes the query through the RAG generation pipeline and formats the response.

        Args:
            request: The validated ChatRequest containing the user's question.

        Returns:
            A formatted ChatResponse.
        """
        try:
            # Call selected GenerationPipeline singleton.
            # The pipeline returns:
            # {
            #     "question": str,
            #     "answer": str,
            #     "sources": list[dict],
            #     "metrics": {
            #         "retrieval_time": float,
            #         "prompt_build_time": float,
            #         "generation_time": float,
            #         "total_time": float
            #     }
            # }
            generate_for_model = getattr(
                self.generation_pipeline, "generate_for_model", None
            )
            if request.llm_model is not None and callable(generate_for_model):
                pipeline_output = generate_for_model(
                    request.question, request.llm_model
                )
            else:
                pipeline_output = self.generation_pipeline.generate(request.question)
            mapping_started = time.perf_counter()

            # Map metrics
            metrics_data = pipeline_output.get("metrics", {})
            retrieval_time = metrics_data.get("retrieval_time", 0.0)
            generation_time = metrics_data.get("generation_time", 0.0)
            total_time = metrics_data.get("total_time", 0.0)

            metrics = MetricsResponse.model_validate(
                {
                    **metrics_data,
                    "retrieval_time": retrieval_time,
                    "generation_time": generation_time,
                    "total_time": total_time,
                }
            )

        except Exception as exc:
            # Log request failure details
            logger.error(
                "CHAT_REQUEST FAILURE | error_category='%s'",
                type(exc).__name__,
                exc_info=True,
            )
            raise

        # Map sources and construct citation indices
        sources = []
        for idx, source_dict in enumerate(pipeline_output.get("sources", []), start=1):
            text_content = (
                source_dict.get("text")
                or source_dict.get("content")
                or source_dict.get("chunk_preview")
                or ""
            )
            preview = text_content[:300]

            source_doc = SourceDocument(
                citation=source_dict.get("citation", idx),
                chunk_id=source_dict.get("chunk_id") or source_dict.get("id") or "",
                document_id=source_dict.get("document_id"),
                document_name=source_dict.get("title")
                or source_dict.get("document_name")
                or "Unknown Document",
                category=source_dict.get("category", "unknown"),
                chapter=source_dict.get("chapter"),
                article=source_dict.get("article"),
                page=source_dict.get("page_start") or source_dict.get("page"),
                chunk_preview=preview,
                # Optional scoring metadata if present in pipeline output
                rerank_score=source_dict.get("rerank_score"),
                rrf_score=source_dict.get("rrf_score"),
                semantic_rank=source_dict.get("semantic_rank"),
                bm25_rank=source_dict.get("bm25_rank"),
                canonical_locator=source_dict.get("canonical_locator"),
                role=source_dict.get("role"),
                relation_type=source_dict.get("relation_type"),
                selectors=source_dict.get("selectors") or [],
            )
            sources.append(source_doc)

        # Generate or reuse session id
        resp_session_id = request.session_id or str(uuid.uuid4())

        citations = [
            CitationResponse.model_validate(item)
            for item in pipeline_output.get("citations", [])
        ]
        answer_blocks = [
            AnswerBlockResponse.model_validate(item)
            for item in pipeline_output.get("answer_blocks", [])
        ]
        mapping_ms = (time.perf_counter() - mapping_started) * 1000.0
        if metrics.api_mapping_ms is not None:
            metrics.api_mapping_ms = mapping_ms
            if metrics.total_ms is not None:
                metrics.total_ms += mapping_ms
                metrics.total_time = metrics.total_ms / 1000.0
        logger.info(
            "CHAT_REQUEST SUCCESS | request_id=%s | pipeline=%s | status=%s | retrieval_time=%.3fs | generation_time=%.3fs | total_time=%.3fs",
            pipeline_output.get("request_id", "unknown"),
            pipeline_output.get("pipeline_version", "v1"),
            pipeline_output.get("status", "answered"),
            metrics.retrieval_time,
            metrics.generation_time,
            metrics.total_time,
        )
        return ChatResponse(
            session_id=resp_session_id,
            question=request.question,
            answer=pipeline_output.get("answer", ""),
            sources=sources,
            metrics=metrics,
            status=pipeline_output.get("status", "answered"),
            answer_blocks=answer_blocks,
            citations=citations,
            limitations=pipeline_output.get("limitations", []),
            warnings=pipeline_output.get("warnings", []),
            pipeline_version=pipeline_output.get("pipeline_version", "v1"),
            model=pipeline_output.get("model"),
        )
