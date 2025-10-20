"""Quick test of document loader"""

from src.rag.document_loader import DocumentLoader

# Test loading
loader = DocumentLoader()

print("Testing different chunking strategies:")
print("=" * 80)

strategies = ["smart", "sentences", "fixed_size"]

for strategy in strategies:
    chunks = loader.load_file(
        "data/life3.txt",
        strategy=strategy,
        chunk_size=300
    )

    print(f"\nStrategy: {strategy}")
    print(f"Total chunks: {len(chunks)}")
    print(f"First chunk length: {len(chunks[0])} chars")
    print(f"First chunk preview:")
    print(f"  {chunks[0][:150]}...")

    # Show size distribution
    sizes = [len(c) for c in chunks]
    avg_size = sum(sizes) / len(sizes)
    print(f"Average chunk size: {avg_size:.1f} chars")
    print(f"Min/Max: {min(sizes)}/{max(sizes)} chars")

# Check if needle exists
needle = "在2025年，阿良曾遇见了一个叫雅薇的女孩，当时是在杭州钱塘江漫步的绿道上，似乎一切都有着阳光。"
chunks = loader.load_file("data/life4.txt", strategy="smart", chunk_size=300)

found = False
for i, chunk in enumerate(chunks):
    if needle in chunk:
        print(f"\n{'='*80}")
        print(f"Found needle in chunk #{i}:")
        print(f"Chunk size: {len(chunk)} chars")
        print(f"Chunk content:")
        print(chunk)
        found = True
        break

if not found:
    print(f"\n{'='*80}")
    print("WARNING: Needle not found in any chunk!")
