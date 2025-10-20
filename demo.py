"""Demo: Needle In a Haystack test with life.txt"""

from src.rag.client import RAGClient
from src.rag.needle_test import NeedleTest


def load_life_txt():
    """Load and split life.txt into paragraphs"""
    with open("data/life3.txt", "r", encoding="utf-8") as f:
        content = f.read()

    # Split by line numbers (format: "1→text")
    lines = content.strip().split("\n")
    paragraphs = []

    for line in lines:
        if "→" in line:
            # Extract text after arrow
            text = line.split("→", 1)[1].strip()
            if text:  # Only add non-empty paragraphs
                paragraphs.append(text)

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
    needle = "在2025年，阿良曾遇见了一个叫雅薇的女孩，当时是在杭州钱塘江漫步的绿道上，似乎一切都有着阳光。"
    print(f"\n[3] Needle to find:")
    print(f"    '{needle}'")

    # Remove the needle from paragraphs to use as haystack
    haystack = [p for p in paragraphs if needle not in p]
    print(f"\n[4] Haystack size: {len(haystack)} paragraphs")

    # Create needle tester
    tester = NeedleTest(client)

    # Run test
    print("\n[5] Running Needle In a Haystack test...")
    print("    Query: '阿良在2025年遇见了谁'")

    result = tester.run_test(
        needle=needle,
        haystack_size=len(haystack),
        query="阿良在2025年遇见了谁"
    )

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
    print("\n" + "=" * 80)
    print("RUNNING MULTIPLE TESTS")
    print("=" * 80)

    sizes = [50, 100, 150]
    print(f"Testing with haystack sizes: {sizes}")
    print("Trials per size: 3\n")

    results = tester.run_multiple_tests(
        needle=needle,
        haystack_sizes=sizes,
        trials_per_size=3
    )

    # Calculate statistics
    total_tests = len(results)
    found_count = sum(1 for r in results if r['needle_found'])
    success_count = sum(1 for r in results if r['success'])

    print(f"Total tests: {total_tests}")
    print(f"Needle found: {found_count}/{total_tests} ({found_count/total_tests*100:.1f}%)")
    print(f"Ranked #1: {success_count}/{total_tests} ({success_count/total_tests*100:.1f}%)")

    # Breakdown by size
    print("\nBreakdown by haystack size:")
    for size in sizes:
        size_results = [r for r in results if r['haystack_size'] == size]
        size_found = sum(1 for r in size_results if r['needle_found'])
        size_success = sum(1 for r in size_results if r['success'])
        print(f"  Size {size}: Found {size_found}/3, Rank#1 {size_success}/3")

    print("\n" + "=" * 80)
    print("Demo completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
