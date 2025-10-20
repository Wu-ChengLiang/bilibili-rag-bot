"""Reranker for improving retrieval accuracy"""

from typing import List, Dict, Any
from .types import SearchResult


class Reranker:
    """Simple reranker based on keyword matching and semantic similarity"""

    def __init__(self):
        """Initialize reranker"""
        pass

    def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: int = 5
    ) -> List[SearchResult]:
        """Rerank search results

        Args:
            query: Original query
            results: Initial search results
            top_k: Number of top results to return

        Returns:
            Reranked results
        """
        # Score each result
        scored_results = []
        for result in results:
            score = self._calculate_relevance_score(query, result)
            result_copy = result.copy()
            result_copy["rerank_score"] = score
            scored_results.append(result_copy)

        # Sort by rerank score
        scored_results.sort(key=lambda x: x["rerank_score"], reverse=True)

        return scored_results[:top_k]

    def _calculate_relevance_score(
        self,
        query: str,
        result: SearchResult
    ) -> float:
        """Calculate relevance score combining multiple factors

        Args:
            query: Query text
            result: Search result

        Returns:
            Relevance score
        """
        content = result["content"]
        base_score = result["score"]

        # Factor 1: Original vector similarity (0.5 weight)
        vector_score = base_score * 0.5

        # Factor 2: Keyword overlap (0.3 weight)
        keyword_score = self._keyword_overlap_score(query, content) * 0.3

        # Factor 3: Position/length bonus (0.2 weight)
        length_score = self._length_score(content) * 0.2

        total_score = vector_score + keyword_score + length_score

        return total_score

    def _keyword_overlap_score(self, query: str, content: str) -> float:
        """Calculate keyword overlap score

        Args:
            query: Query text
            content: Document content

        Returns:
            Overlap score (0-1)
        """
        # Simple character-level overlap for Chinese
        query_chars = set(query.replace(" ", ""))
        content_chars = set(content.replace(" ", ""))

        if not query_chars:
            return 0.0

        overlap = len(query_chars & content_chars)
        score = overlap / len(query_chars)

        return min(score, 1.0)

    def _length_score(self, content: str) -> float:
        """Calculate length-based score (prefer moderate length)

        Args:
            content: Document content

        Returns:
            Length score (0-1)
        """
        length = len(content)

        # Prefer moderate length (200-500 chars)
        if 200 <= length <= 500:
            return 1.0
        elif length < 200:
            return length / 200
        else:
            # Penalize very long documents
            return max(0.5, 1.0 - (length - 500) / 1000)
