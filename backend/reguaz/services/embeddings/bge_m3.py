import logging
from typing import List

from sentence_transformers import SentenceTransformer

from backend.reguaz.services.embeddings.base import BaseEmbeddingService

logger = logging.getLogger(__name__)

class BGEM3EmbeddingService(BaseEmbeddingService):
    """
    Embedding service using BAAI/bge-m3.
    """
    
    def __init__(self, model_name: str = "BAAI/bge-m3"):
        logger.info(f"Loading BGE-M3 model: {model_name}")
        # bge-m3 does not strictly require a specific query/passage prefix for general retrieval in many implementations,
        # though sometimes "query: " is used for queries. For chunks, plain text is fine.
        self.model = SentenceTransformer(model_name)

    def embed_text(self, text: str) -> List[float]:
        embedding = self.model.encode(text, normalize_embeddings=True)
        return embedding.tolist()
        
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()
