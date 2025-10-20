"""Long Context Test for LLM (Arize-style Needle In Haystack)

This test evaluates an LLM's ability to retrieve specific information
from a long context window, similar to Arize's implementation.

Key differences from RAG vector search test:
- No embedding or vector search
- Direct LLM context test
- Tests long context understanding, not retrieval
"""

import random
from typing import List, Dict, Any, Optional
from .llm_client import LLMClient


class LongContextTest:
    """Test LLM's ability to find needles in long contexts

    This follows the Arize Needle-In-Haystack methodology:
    1. Build a long context document (haystack)
    2. Insert a specific fact (needle) at various depths
    3. Ask LLM to retrieve the specific fact
    4. Evaluate accuracy
    """

    def __init__(self, llm_client: LLMClient):
        """Initialize long context test

        Args:
            llm_client: LLM client for testing
        """
        self.llm_client = llm_client

    def create_long_context(
        self,
        documents: List[str],
        needle: str,
        needle_position: Optional[int] = None
    ) -> tuple[str, int]:
        """Create long context by concatenating documents and inserting needle

        Args:
            documents: List of document chunks to form haystack
            needle: The specific fact to insert
            needle_position: Position to insert (random if None)

        Returns:
            Tuple of (long_context_text, actual_needle_position)
        """
        if needle_position is None:
            needle_position = random.randint(0, len(documents))

        # Insert needle at specified position
        docs_with_needle = documents.copy()
        docs_with_needle.insert(needle_position, needle)

        # Concatenate into long context
        long_context = "\n\n".join(docs_with_needle)

        return long_context, needle_position

    def run_test(
        self,
        documents: List[str],
        needle: str,
        query: str,
        expected_answer: str,
        needle_position: Optional[int] = None,
        context_limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """Run a single long context test

        Args:
            documents: List of haystack documents
            needle: The needle sentence containing the answer
            query: Question to ask the LLM
            expected_answer: Expected answer (e.g., company name, number)
            needle_position: Position to insert needle (random if None)
            context_limit: Limit context to first N documents (None = use all)

        Returns:
            Test results dictionary
        """
        # Limit context if specified
        if context_limit is not None:
            documents = documents[:context_limit]

        # Create long context
        long_context, actual_position = self.create_long_context(
            documents, needle, needle_position
        )

        # Calculate context metrics
        context_length = len(long_context)
        context_tokens = len(long_context.split())  # Rough token estimate
        needle_depth_percent = (actual_position / len(documents)) * 100 if documents else 0

        # Build prompt for LLM
        system_prompt = """你是一个智能助手。请仔细阅读提供的文档，从中找到准确的信息来回答问题。

只回答问题，不要添加额外的解释。"""

        user_message = f"""文档内容：
{long_context}

问题：{query}"""

        # Call LLM
        try:
            response = self.llm_client.client.chat.completions.create(
                model=self.llm_client.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.1  # Low temperature for factual retrieval
            )

            llm_answer = response.choices[0].message.content.strip()

            # Check if answer is correct
            success = expected_answer.lower() in llm_answer.lower()

        except Exception as e:
            llm_answer = f"ERROR: {str(e)}"
            success = False

        return {
            "success": success,
            "expected_answer": expected_answer,
            "llm_answer": llm_answer,
            "needle": needle,
            "needle_position": actual_position,
            "needle_depth_percent": needle_depth_percent,
            "context_length_chars": context_length,
            "context_length_tokens": context_tokens,
            "haystack_size": len(documents),
            "query": query,
        }

    def run_multiple_tests(
        self,
        documents: List[str],
        needle: str,
        query: str,
        expected_answer: str,
        context_sizes: List[int] = None,
        depth_percentages: List[int] = None,
        trials_per_config: int = 3
    ) -> List[Dict[str, Any]]:
        """Run multiple tests with varying context sizes and needle depths

        This creates a test matrix similar to Arize's visualization:
        - X-axis: Context length (token/document count)
        - Y-axis: Needle depth (percentage through document)

        Args:
            documents: Full haystack documents
            needle: The needle sentence
            query: Question to ask
            expected_answer: Expected answer
            context_sizes: List of context sizes to test (in document count)
            depth_percentages: List of depth percentages to test (0-100)
            trials_per_config: Number of trials per configuration

        Returns:
            List of test results
        """
        if context_sizes is None:
            context_sizes = [10, 30, 50, 70, len(documents)]

        if depth_percentages is None:
            depth_percentages = [0, 25, 50, 75, 100]

        results = []

        for context_size in context_sizes:
            for depth_pct in depth_percentages:
                for trial in range(trials_per_config):
                    # Calculate needle position based on depth percentage
                    needle_pos = int((depth_pct / 100) * min(context_size, len(documents)))
                    needle_pos = max(0, min(needle_pos, context_size - 1))

                    # Run test
                    result = self.run_test(
                        documents=documents,
                        needle=needle,
                        query=query,
                        expected_answer=expected_answer,
                        needle_position=needle_pos,
                        context_limit=context_size
                    )

                    result["trial"] = trial
                    result["target_depth_percent"] = depth_pct
                    results.append(result)

        return results

    def generate_random_needle(self, cities: List[str] = None) -> tuple[str, str, str]:
        """Generate a random needle with unique number (Arize style)

        This prevents caching and ensures each test is independent.

        Args:
            cities: List of cities to choose from

        Returns:
            Tuple of (needle_sentence, query, expected_answer)
        """
        if cities is None:
            cities = ["杭州", "北京", "上海", "深圳", "广州", "成都"]

        random_num = random.randint(10000, 99999)
        city = random.choice(cities)

        needle = f"特殊信息：在{city}市，魔法数字是{random_num}。这是一个重要的标记。"
        query = "魔法数字是多少？"
        expected_answer = str(random_num)

        return needle, query, expected_answer

    def print_results_summary(self, results: List[Dict[str, Any]]) -> None:
        """Print a summary of test results

        Args:
            results: List of test results
        """
        total = len(results)
        successful = sum(1 for r in results if r["success"])
        success_rate = (successful / total * 100) if total > 0 else 0

        print("=" * 80)
        print("长上下文测试结果摘要")
        print("=" * 80)
        print(f"总测试数: {total}")
        print(f"成功: {successful}/{total} ({success_rate:.1f}%)")

        # Group by context size
        context_sizes = sorted(set(r["haystack_size"] for r in results))
        print("\n按上下文大小分组:")
        for size in context_sizes:
            size_results = [r for r in results if r["haystack_size"] == size]
            size_success = sum(1 for r in size_results if r["success"])
            size_total = len(size_results)
            print(f"  {size} 文档: {size_success}/{size_total} " +
                  f"({size_success/size_total*100:.1f}%)")

        # Group by depth
        if results and "target_depth_percent" in results[0]:
            depths = sorted(set(r["target_depth_percent"] for r in results))
            print("\n按 Needle 深度分组:")
            for depth in depths:
                depth_results = [r for r in results if r["target_depth_percent"] == depth]
                depth_success = sum(1 for r in depth_results if r["success"])
                depth_total = len(depth_results)
                print(f"  {depth}% 深度: {depth_success}/{depth_total} " +
                      f"({depth_success/depth_total*100:.1f}%)")

        print("=" * 80)
