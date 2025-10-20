"""Example usage of the RAG Needle In a Haystack Test"""

from src.rag_system import RAGSystem
from src.needle_test import NeedleInHaystackTest


def main():
    # Initialize RAG system
    print("Initializing RAG system...")
    rag = RAGSystem()

    # Initialize test framework
    tester = NeedleInHaystackTest(rag)

    # Define the needle
    needle = "The secret password is: BANANA_SPLIT_2024"

    # Run a single test
    print("\nRunning single test...")
    result = tester.run_test(
        needle=needle,
        haystack_size=100,
        query="What is the secret password?"
    )

    print(f"\nTest Results:")
    print(f"  Needle found: {result['needle_found']}")
    print(f"  Needle rank: {result['needle_rank']}")
    print(f"  Success: {result['success']}")
    print(f"  Haystack size: {result['haystack_size']}")
    print(f"  Needle position: {result['needle_position']}")

    # Run multiple tests
    print("\n" + "="*50)
    print("Running multiple tests with different haystack sizes...")
    results = tester.run_multiple_tests(
        needle=needle,
        haystack_sizes=[10, 50, 100],
        trials_per_size=3
    )

    # Calculate success rate
    total_tests = len(results)
    successful_tests = sum(1 for r in results if r['needle_found'])
    success_rate = (successful_tests / total_tests) * 100

    print(f"\nOverall Results:")
    print(f"  Total tests: {total_tests}")
    print(f"  Successful retrievals: {successful_tests}")
    print(f"  Success rate: {success_rate:.2f}%")

    # Breakdown by haystack size
    print("\nBreakdown by haystack size:")
    for size in [10, 50, 100]:
        size_results = [r for r in results if r['haystack_size'] == size]
        size_success = sum(1 for r in size_results if r['needle_found'])
        size_rate = (size_success / len(size_results)) * 100
        print(f"  Size {size}: {size_success}/{len(size_results)} ({size_rate:.2f}%)")


if __name__ == "__main__":
    main()
