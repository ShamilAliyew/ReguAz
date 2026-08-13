import logging
from typing import List

from sentence_transformers import SentenceTransformer

from backend.reguaz.services.embeddings.base import BaseEmbeddingService

logger = logging.getLogger(__name__)

class JinaV3EmbeddingService(BaseEmbeddingService):
    """
    Embedding service using jinaai/jina-embeddings-v3.
    """
    
    def __init__(self, model_name: str = "jinaai/jina-embeddings-v3"):
        logger.info(f"Loading Jina v3 model: {model_name}")
        # Jina v3 requires trust_remote_code=True
        self.model = SentenceTransformer(model_name, trust_remote_code=True)
        # Jina v3 supports task specific embeddings. Defaulting to 'text-matching' or leaving it empty.
        # Often it accepts a `task` argument in encode(), but we'll use the default settings here.

    def embed_text(self, text: str) -> List[float]:
        # Using task="retrieval.passage" as we are encoding document chunks.
        # Note: If sentence-transformers version doesn't support the task argument natively yet,
        # it might just ignore it or throw an error. In recent versions with Jina v3, it is supported via kwargs.
        try:
            embedding = self.model.encode(text, task="retrieval.passage", normalize_embeddings=True)
        except TypeError:
            # Fallback if task argument is not supported
            embedding = self.model.encode(text, normalize_embeddings=True)
        return embedding.tolist()
        
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        try:
            embeddings = self.model.encode(texts, task="retrieval.passage", normalize_embeddings=True)
        except TypeError:
            embeddings = self.model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()
