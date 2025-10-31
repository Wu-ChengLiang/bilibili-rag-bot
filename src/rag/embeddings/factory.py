"""Factory for creating embedding model instances"""

from typing import Literal
from ..core.embedding import BaseEmbedding
from .text2vec import Text2VecEmbedding
from .gte import GTEEmbedding


def create_embedding(
    provider: Literal["text2vec", "gte"],
    model_name: str
) -> BaseEmbedding:
    """Create embedding model instance from provider and model name

    Args:
        provider: Embedding provider ("text2vec" or "gte")
        model_name: Model name/identifier on HuggingFace

    Returns:
        BaseEmbedding instance

    Raises:
        ValueError: If provider is unsupported

    Examples:
        >>> embedding = create_embedding("gte", "thenlper/gte-base-zh")
        >>> embedding = create_embedding("text2vec", "shibing624/text2vec-base-chinese")
    """
    providers = {
        "text2vec": Text2VecEmbedding,
        "gte": GTEEmbedding,
    }

    if provider not in providers:
        raise ValueError(
            f"Unsupported embedding provider: {provider}. "
            f"Supported providers: {list(providers.keys())}"
        )

    return providers[provider](model_name=model_name)
