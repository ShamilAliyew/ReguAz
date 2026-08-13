import logging
from typing import List

from sentence_transformers import SentenceTransformer

from backend.reguaz.services.embeddings.base import BaseEmbeddingService

logger = logging.getLogger(__name__)

class E5EmbeddingService(BaseEmbeddingService):
    """
    Embedding service using intfloat/multilingual-e5-large.
    """
    
    def __init__(self, model_name: str = "intfloat/multilingual-e5-large"):
        logger.info(f"Loading E5 model: {model_name}")
        self.model = SentenceTransformer(model_name)
        # E5 models generally require a prefix for asymmetric tasks.
        # For indexing documents, "passage: " is typically used.
        self.prefix = "passage: "

    def embed_text(self, text: str) -> List[float]:
        prefixed_text = self.prefix + text
        embedding = self.model.encode(prefixed_text, normalize_embeddings=True)
        return embedding.tolist()
        
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        prefixed_texts = [self.prefix + text for text in texts]
        embeddings = self.model.encode(prefixed_texts, normalize_embeddings=True)
        return embeddings.tolist()
