"""Needle In a Haystack Test Implementation"""

import random
from typing import List, Dict, Any
from .rag_system import RAGSystem


class NeedleInHaystackTest:
    """Test framework for evaluating RAG retrieval accuracy"""

    def __init__(self, rag_system: RAGSystem):
        """Initialize the test framework

        Args:
            rag_system: An instance of RAGSystem
        """
        self.rag_system = rag_system

    def generate_haystack(self, size: int, template: str = "This is document number {}.") -> List[str]:
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
    ) -> tuple[List[str], int]:
        """Insert a needle document into the haystack

        Args:
            haystack: List of haystack documents
            needle: The needle document to insert
            position: Position to insert the needle (random if None)

        Returns:
            Tuple of (modified haystack, needle position)
        """
        if position is None:
            position = random.randint(0, len(haystack))

        haystack_with_needle = haystack.copy()
        haystack_with_needle.insert(position, needle)

        return haystack_with_needle, position

    def run_test(
        self,
        needle: str,
        haystack_size: int = 100,
        needle_position: int = None,
        query: str = None,
        n_results: int = 5
    ) -> Dict[str, Any]:
        """Run a needle in haystack test

        Args:
            needle: The needle document
            haystack_size: Size of the haystack
            needle_position: Position to insert needle (random if None)
            query: Query to search for (uses needle if None)
            n_results: Number of results to retrieve

        Returns:
            Dictionary containing test results
        """
        # Clear previous data
        self.rag_system.clear()

        # Generate haystack
        haystack = self.generate_haystack(haystack_size)

        # Insert needle
        documents, actual_position = self.insert_needle(haystack, needle, needle_position)

        # Add to RAG system
        self.rag_system.add_documents(documents)

        # Query
        if query is None:
            query = needle

        results = self.rag_system.query(query, n_results=n_results)

        # Analyze results
        retrieved_docs = results.get('documents', [[]])[0]
        needle_found = needle in retrieved_docs
        needle_rank = None

        if needle_found:
            needle_rank = retrieved_docs.index(needle) + 1

        return {
            "needle": needle,
            "haystack_size": haystack_size,
            "needle_position": actual_position,
            "query": query,
            "n_results": n_results,
            "needle_found": needle_found,
            "needle_rank": needle_rank,
            "retrieved_documents": retrieved_docs,
            "success": needle_found and needle_rank == 1
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
