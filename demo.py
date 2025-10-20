"""Demo: Needle In a Haystack test with life.txt"""

from src.rag.client import RAGClient
from src.rag.needle_test import NeedleTest
from src.rag.document_loader import DocumentLoader


def load_life_txt(strategy="smart", chunk_size=300):
    """Load and split life.txt into paragraphs

    Args:
        strategy: Chunking strategy ("smart", "sentences", "fixed_size")
        chunk_size: Target size for each chunk in characters

    Returns:
        List of text chunks
    """
    loader = DocumentLoader()
    paragraphs = loader.load_file(
        "data/life4.txt",  # Needle is in life4.txt
        strategy=strategy,
        chunk_size=chunk_size
    )
    return paragraphs


def main():
    print("=" * 80)
    print("RAG Needle In a Haystack Test - Demo with life.txt")
    print("=" * 80)

    # Initialize RAG client
    print("\n[1] Initializing RAG client with Text2Vec Chinese embeddings...")
    client = RAGClient(
        persist_directory="./demo_chroma_db",
        collection_name="life_documents"
    )

    # Load life.txt
    print("\n[2] Loading life.txt...")
    paragraphs = load_life_txt()
    print(f"    Loaded {len(paragraphs)} paragraphs from life.txt")

    # The needle - a specific sentence from the text
    needle = "后来，在某个城市漫步的我，进入了一个初创公司，名叫梦醒，开始了旅程"
    print(f"\n[3] Needle to find:")
    print(f"    '{needle}'")

    # Remove the needle from paragraphs to use as haystack
    haystack = [p for p in paragraphs if needle not in p]
    print(f"\n[4] Haystack size: {len(haystack)} paragraphs")

    # Create needle tester
    tester = NeedleTest(client)

    # Run test with real haystack documents
    print("\n[5] Running Needle In a Haystack test...")
    print("    Query: '我进入了什么公司'")

    # Insert needle into haystack
    import random
    position = random.randint(0, len(haystack))
    all_documents = haystack.copy()
    all_documents.insert(position, needle)

    # Add to RAG system
    client.add_documents(all_documents)

    # Search
    search_results = client.search("我进入了什么公司", limit=5)

    # Analyze results
    needle_found = False
    needle_rank = None
    for i, res in enumerate(search_results):
        if needle in res["content"]:
            needle_found = True
            needle_rank = i + 1
            break

    result = {
        "haystack_size": len(haystack),
        "needle_position": position,
        "needle_found": needle_found,
        "needle_rank": needle_rank,
        "success": needle_found and needle_rank == 1,
        "retrieved_documents": [r["content"] for r in search_results]
    }

    # Print results
    print("\n" + "=" * 80)
    print("TEST RESULTS")
    print("=" * 80)
    print(f"Haystack Size: {result['haystack_size']}")
    print(f"Needle Position: {result['needle_position']}")
    print(f"Needle Found: {result['needle_found']}")
    print(f"Needle Rank: {result['needle_rank']}")
    print(f"Success (Rank #1): {result['success']}")

    print("\n" + "-" * 80)
    print("Top 5 Retrieved Documents:")
    print("-" * 80)
    for i, doc in enumerate(result['retrieved_documents'][:5], 1):
        is_needle = "(NEEDLE)" if doc == needle else ""
        print(f"{i}. {doc[:80]}... {is_needle}")

    # Run multiple tests with different haystack sizes
    # Note: Commented out to avoid using generated haystack
    # TODO: Implement run_multiple_tests with real documents from life4.txt
    # print("\n" + "=" * 80)
    # print("RUNNING MULTIPLE TESTS")
    # print("=" * 80)
    #
    # sizes = [50, 100, 150]
    # print(f"Testing with haystack sizes: {sizes}")
    # print("Trials per size: 3\n")
    #
    # results = tester.run_multiple_tests(
    #     needle=needle,
    #     haystack_sizes=sizes,
    #     trials_per_size=3
    # )
    #
    # # Calculate statistics
    # total_tests = len(results)
    # found_count = sum(1 for r in results if r['needle_found'])
    # success_count = sum(1 for r in results if r['success'])
    #
    # print(f"Total tests: {total_tests}")
    # print(f"Needle found: {found_count}/{total_tests} ({found_count/total_tests*100:.1f}%)")
    # print(f"Ranked #1: {success_count}/{total_tests} ({success_count/total_tests*100:.1f}%)")
    #
    # # Breakdown by size
    # print("\nBreakdown by haystack size:")
    # for size in sizes:
    #     size_results = [r for r in results if r['haystack_size'] == size]
    #     size_found = sum(1 for r in size_results if r['needle_found'])
    #     size_success = sum(1 for r in size_results if r['success'])
    #     print(f"  Size {size}: Found {size_found}/3, Rank#1 {size_success}/3")

    print("\n" + "=" * 80)
    print("Demo completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
