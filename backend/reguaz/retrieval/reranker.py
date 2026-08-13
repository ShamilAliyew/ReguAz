"""
backend/reguaz/retrieval/reranker.py

Cross-Encoder reranker wrapper.
Performs semantic reranking of candidate documents using a Cross-Encoder model.
"""

from __future__ import annotations

import logging
import time

# pyrefly: ignore [missing-import]
import torch

# pyrefly: ignore [missing-import]
from sentence_transformers import CrossEncoder
from backend.reguaz.utils.devices import select_inference_device

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """
    Reranks a list of candidate documents relative to a query using a Cross-Encoder.

    This class loads a sentence-transformers Cross-Encoder model (defaulting to
    ``"BAAI/bge-reranker-v2-m3"``) and computes scores for (query, document) pairs.

    Parameters
    ----------
    model_name : str
        Name/path of the Cross-Encoder model to load (e.g. ``"BAAI/bge-reranker-v2-m3"``).
    device : str | None
        Target device for inference (e.g. ``"cuda"``, ``"cpu"``). If None, defaults
        to auto-detection.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: str | None = None,
        *,
        batch_size: int = 32,
        max_length: int = 1024,
        revision: str | None = None,
        local_files_only: bool = False,
        trust_remote_code: bool = False,
        code_revision: str | None = None,
        show_progress_bar: bool = True,
    ) -> None:
        if batch_size <= 0 or max_length <= 0:
            raise ValueError("reranker batch_size and max_length must be positive")
        self._batch_size = batch_size
        self._show_progress_bar = show_progress_bar
        logger.info(
            "CrossEncoderReranker: loading Cross-Encoder model '%s' on device %s ...",
            model_name,
            device or "auto",
        )

        try:
            device = select_inference_device(device, torch_module=torch)

            logger.info(
                "CrossEncoderReranker: loading Cross-Encoder model '%s' on device '%s' ...",
                model_name,
                device,
            )

            self._model = CrossEncoder(
                model_name,
                device=device,
                max_length=max_length,
                revision=revision,
                local_files_only=local_files_only,
                trust_remote_code=trust_remote_code,
                model_kwargs=(
                    {"code_revision": code_revision} if code_revision else None
                ),
                processor_kwargs=(
                    {"code_revision": code_revision} if code_revision else None
                ),
                config_kwargs=(
                    {"code_revision": code_revision} if code_revision else None
                ),
            )
            self.model_name = model_name
            self.model_revision = revision
            self.code_revision = code_revision
            self._repair_alibaba_position_ids(model_name)
            self.device = str(self._model.model.device)

            logger.info(
                "CrossEncoderReranker: actual model device = %s",
                self._model.model.device,
            )

            logger.info(
                "CrossEncoderReranker: torch mps available = %s | cuda available = %s",
                torch.backends.mps.is_available(),
                torch.cuda.is_available(),
            )

            logger.info(
                "CrossEncoderReranker: max_length = %d",
                max_length,
            )

            logger.info(
                "CrossEncoderReranker: model '%s' loaded successfully.",
                model_name,
            )

        except Exception as exc:
            logger.error(
                "CrossEncoderReranker: failed to load model '%s': %s",
                model_name,
                exc,
            )
            raise

    def _repair_alibaba_position_ids(self, model_name: str) -> None:
        """Repair a Transformers 5 buffer-loading incompatibility in pinned mGTE code."""
        if model_name != "Alibaba-NLP/gte-multilingual-reranker-base":
            return
        model = self._model.model
        embeddings = getattr(getattr(model, "new", None), "embeddings", None)
        if embeddings is None:
            raise RuntimeError("Alibaba reranker embeddings module is missing")
        expected_size = int(model.config.max_position_embeddings)
        expected = torch.arange(
            expected_size,
            device=embeddings.word_embeddings.weight.device,
        )
        position_ids = getattr(embeddings, "position_ids", None)
        if (
            not isinstance(position_ids, torch.Tensor)
            or position_ids.shape != expected.shape
            or not torch.equal(position_ids, expected)
        ):
            embeddings.register_buffer(
                "position_ids",
                expected,
                persistent=False,
            )

    def rerank(
        self,
        query: str,
        documents: list[str],
        *,
        batch_size: int | None = None,
    ) -> list[float]:
        """
        Compute similarity scores for a query and a list of candidate documents.

        Parameters
        ----------
        query : str
            The query text.
        documents : list[str]
            A list of document texts to rerank.

        Returns
        -------
        list[float]
            List of similarity scores (higher = more relevant) corresponding to
            the input documents in the same order.
        """
        if not documents:
            return []
        if not query.strip():
            raise ValueError("reranker query must not be empty")
        effective_batch_size = batch_size or self._batch_size
        if effective_batch_size <= 0:
            raise ValueError("reranker batch_size must be positive")

        logger.info(
            "CrossEncoderReranker: reranking %d documents.",
            len(documents),
        )

        # Prepare (query, document) pairs for the cross-encoder.
        pairs = [[query, doc] for doc in documents]

        logger.info(
            "CrossEncoderReranker: %d query-document pairs prepared.",
            len(pairs),
        )

        start = time.perf_counter()

        lengths = [len(doc.split()) for doc in documents]

        if lengths:
            logger.info(
                "CrossEncoderReranker: document lengths (words) | "
                "min=%d max=%d avg=%.1f",
                min(lengths),
                max(lengths),
                sum(lengths) / len(lengths),
            )

        # Compute raw scores.
        raw_scores = self._model.predict(
            pairs,
            batch_size=effective_batch_size,
            show_progress_bar=self._show_progress_bar,
        )

        elapsed = time.perf_counter() - start

        logger.info(
            "CrossEncoderReranker: inference finished in %.3f s "
            "(%.3f s / document).",
            elapsed,
            elapsed / len(documents),
        )

        # Ensure we return Python float type.
        if isinstance(raw_scores, list):
            return [float(score) for score in raw_scores]

        return [float(score) for score in raw_scores.tolist()]
