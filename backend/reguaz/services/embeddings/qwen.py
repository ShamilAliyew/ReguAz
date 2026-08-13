import logging
from typing import List

from sentence_transformers import SentenceTransformer

from backend.reguaz.services.embeddings.base import BaseEmbeddingService

logger = logging.getLogger(__name__)


class QwenEmbeddingService(BaseEmbeddingService):
    """
    Embedding service using Qwen/Qwen3-Embedding-0.6B.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-0.6B"
    ):
        logger.info(f"Loading Qwen model: {model_name}")
        self.model = SentenceTransformer(model_name)

    def embed_text(
        self,
        text: str
    ) -> List[float]:

        embedding = self.model.encode(
            text,
            normalize_embeddings=True,
        )

        return embedding.tolist()

    def embed_batch(
        self,
        texts: List[str]
    ) -> List[List[float]]:

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
        )

        return embeddings.tolist()