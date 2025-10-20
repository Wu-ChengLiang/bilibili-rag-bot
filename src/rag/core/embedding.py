"""Abstract base class for embedding models"""

from abc import ABC, abstractmethod
from typing import List, Union


class BaseEmbedding(ABC):
    """Abstract base class for embedding models

    All embedding implementations should inherit from this class
    and implement the encode method.
    """

    @abstractmethod
    def encode(self, texts: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """Generate embeddings for input text(s)

        Args:
            texts: Single text string or list of text strings

        Returns:
            Single embedding vector or list of embedding vectors

        Examples:
            >>> embedding = model.encode("hello world")
            >>> embeddings = model.encode(["hello", "world"])
        """
        pass

    def encode_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Generate embeddings in batches

        Args:
            texts: List of text strings
            batch_size: Number of texts to process in each batch

        Returns:
            List of embedding vectors
        """
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.encode(batch)
            if isinstance(batch_embeddings[0], list):
                embeddings.extend(batch_embeddings)
            else:
                embeddings.append(batch_embeddings)
        return embeddings

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Get embedding dimension

        Returns:
            Dimension of the embedding vectors
        """
        pass

    @property
    def model_name(self) -> str:
        """Get model name

        Returns:
            Name of the embedding model
        """
        return self.__class__.__name__
