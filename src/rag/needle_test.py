"""Needle In a Haystack Test Framework"""

import random
from typing import List, Dict, Any, Tuple
from .client import RAGClient


class NeedleTest:
    """Framework for testing RAG retrieval accuracy using Needle In a Haystack methodology"""

    def __init__(self, client: RAGClient):
        """Initialize needle test framework

        Args:
            client: RAG client instance
        """
        self.client = client

    def generate_haystack(
        self,
        size: int,
        template: str = "这是文档编号{}的内容。"
    ) -> List[str]:
        """Generate haystack documents

        Args:
            size: Number of documents to generate
            template: Template string for generating documents

        Returns:
            List of generated documents
        """
        return [template.format(i) for i in range(size)]

    def insert_needle(
        self,
        haystack: List[str],
        needle: str,
        position: int = None
    ) -> Tuple[List[str], int]:
        """Insert needle document into haystack

        Args:
            haystack: List of haystack documents
            needle: The needle document to insert
            position: Position to insert (random if None)

        Returns:
            Tuple of (documents with needle, needle position)
        """
        if position is None:
            position = random.randint(0, len(haystack))

        haystack_copy = haystack.copy()
        haystack_copy.insert(position, needle)

        return haystack_copy, position

    def run_test(
        self,
        needle: str,
        haystack_size: int = 100,
        needle_position: int = None,
        query: str = None,
        limit: int = 5
    ) -> Dict[str, Any]:
        """Run a needle in haystack test

        Args:
            needle: The needle document
            haystack_size: Size of the haystack
            needle_position: Position to insert needle (random if None)
            query: Query to search for (uses needle if None)
            limit: Number of results to retrieve

        Returns:
            Test results dictionary
        """
        # Generate haystack
        haystack = self.generate_haystack(haystack_size)

        # Insert needle
        documents, actual_position = self.insert_needle(
            haystack,
            needle,
            needle_position
        )

        # Add documents to RAG system
        doc_ids = self.client.add_documents(documents)

        # Determine query
        if query is None:
            query = needle

        # Search
        results = self.client.search(query, limit=limit)

        # Analyze results
        needle_found = False
        needle_rank = None

        for i, result in enumerate(results):
            if result["content"] == needle:
                needle_found = True
                needle_rank = i + 1  # Rank starts from 1
                break

        return {
            "needle": needle,
            "haystack_size": haystack_size,
            "needle_position": actual_position,
            "query": query,
            "limit": limit,
            "needle_found": needle_found,
            "needle_rank": needle_rank,
            "success": needle_found and needle_rank == 1,
            "retrieved_documents": [r["content"] for r in results]
        }

    def run_multiple_tests(
        self,
        needle: str,
        haystack_sizes: List[int] = None,
        trials_per_size: int = 5
    ) -> List[Dict[str, Any]]:
        """Run multiple tests with different configurations

        Args:
            needle: The needle document
            haystack_sizes: List of haystack sizes to test
            trials_per_size: Number of trials per haystack size

        Returns:
            List of test results
        """
        if haystack_sizes is None:
            haystack_sizes = [10, 50, 100, 500]

        results = []
        for size in haystack_sizes:
            for trial in range(trials_per_size):
                result = self.run_test(needle=needle, haystack_size=size)
                result["trial"] = trial
                results.append(result)

        return results
