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
needle = "后来，在某个城市漫步的我，进入了一个初创公司，名叫梦醒，开始了旅程"
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
