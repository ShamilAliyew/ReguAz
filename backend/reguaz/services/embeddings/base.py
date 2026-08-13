from abc import ABC, abstractmethod
from typing import List

class BaseEmbeddingService(ABC):
    """
    Abstract base class for all embedding services.
    """
    
    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """
        Embed a single text string into a vector.
        
        Args:
            text: The text to embed.
            
        Returns:
            A list of floats representing the embedding vector.
        """
        pass
    
    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a batch of text strings.
        
        Args:
            texts: A list of text strings to embed.
            
        Returns:
            A list of embedding vectors.
        """
        pass
