"""GTE (General Text Embeddings) embedding implementation"""

from typing import List, Union
from sentence_transformers import SentenceTransformer
import numpy as np
from ..core.embedding import BaseEmbedding


class GTEEmbedding(BaseEmbedding):
    """GTE embedding model for Chinese text

    This implementation uses sentence-transformers library to provide
    efficient embeddings optimized for Chinese language processing.

    Models:
        - gte-base-zh: Fast, lightweight (0.2GB, 768 dims)
        - gte-large-zh: Better accuracy (0.67GB, 1024 dims)
    """

    def __init__(self, model_name: str = "thenlper/gte-base-zh"):
        """Initialize GTE embedding model

        Args:
            model_name: Name of the GTE model to use
                Defaults to gte-base-zh for CPU efficiency
        """
        self._model_name = model_name
        self._model = SentenceTransformer(model_name)
        self._dimension = None

        # Cache dimension info based on model
        if "base" in model_name:
            self._dimension = 768
        elif "large" in model_name:
            self._dimension = 1024
        else:
            # Fallback: compute on first use
            self._dimension = None

    def encode(self, texts: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """Generate embeddings for input text(s)

        Args:
            texts: Single text string or list of text strings

        Returns:
            Single embedding vector or list of embedding vectors

        Raises:
            ValueError: If input is empty or invalid
            TypeError: If input type is not supported
        """
        is_single = isinstance(texts, str)

        # Input validation and type conversion
        if is_single:
            if texts is None:
                raise ValueError("Text cannot be None")
            if not isinstance(texts, str):
                # Try to convert to string
                texts = str(texts)
            texts = [texts]
        else:
            if not isinstance(texts, (list, tuple)):
                raise TypeError(
                    f"Expected str or list of str, got {type(texts).__name__}"
                )
            # Convert and filter: remove None values and convert to strings
            cleaned_texts = []
            for i, text in enumerate(texts):
                if text is None:
                    continue
                if isinstance(text, bytes):
                    try:
                        text = text.decode('utf-8')
                    except UnicodeDecodeError:
                        continue
                elif not isinstance(text, str):
                    text = str(text)
                if text.strip():  # Skip empty strings
                    cleaned_texts.append(text)

            if not cleaned_texts:
                raise ValueError("No valid text to encode after filtering")
            texts = cleaned_texts

        # Generate embeddings using sentence-transformers
        try:
            embeddings = self._model.encode(texts, normalize_embeddings=True)
        except Exception as e:
            raise RuntimeError(f"Failed to encode texts: {str(e)}")

        # Convert to list if numpy array
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
