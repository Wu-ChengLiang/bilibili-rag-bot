"""Text2Vec embedding implementation"""

from typing import List, Union
from text2vec import SentenceModel
from ..core.embedding import BaseEmbedding


class Text2VecEmbedding(BaseEmbedding):
    """Text2Vec embedding model for Chinese text

    This implementation wraps the text2vec library and provides
    embeddings optimized for Chinese language processing.
    """

    def __init__(self, model_name: str = "shibing624/text2vec-base-chinese"):
        """Initialize Text2Vec embedding model

        Args:
            model_name: Name of the text2vec model to use
        """
        self._model_name = model_name
        self._model = SentenceModel(model_name)
        # Get dimension from first encoding (lazy evaluation)
        self._dimension = None

    def encode(self, texts: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """Generate embeddings for input text(s)

        Args:
            texts: Single text string or list of text strings

        Returns:
            Single embedding vector or list of embedding vectors
        """
        is_single = isinstance(texts, str)

        if is_single:
            texts = [texts]

        # Generate embeddings
        embeddings = self._model.encode(texts)

        # Convert to list format
        if hasattr(embeddings, 'tolist'):
            embeddings = embeddings.tolist()

        # Cache dimension on first use
        if self._dimension is None and len(embeddings) > 0:
            self._dimension = len(embeddings[0])

        return embeddings[0] if is_single else embeddings

    @property
    def dimension(self) -> int:
        """Get embedding dimension

        Returns:
            Dimension of the embedding vectors
        """
        if self._dimension is None:
            # Initialize dimension by encoding a dummy text
            dummy_embedding = self.encode("test")
            self._dimension = len(dummy_embedding)
        return self._dimension

    @property
    def model_name(self) -> str:
        """Get model name

        Returns:
            Name of the embedding model
        """
        return self._model_name
