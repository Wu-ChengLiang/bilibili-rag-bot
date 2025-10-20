"""Debug demo to check what's happening"""

from src.rag.document_loader import DocumentLoader

loader = DocumentLoader()
chunks = loader.load_file("data/life4.txt", strategy="smart", chunk_size=300)

needle = "在2025年，阿良曾遇见了一个叫雅薇的女孩，当时是在杭州钱塘江漫步的绿道上，似乎一切都有着阳光。"

print(f"Total chunks: {len(chunks)}")
print(f"\nSearching for needle occurrences...")

needle_chunks = []
for i, chunk in enumerate(chunks):
    if needle in chunk:
        needle_chunks.append((i, chunk))
        print(f"\nChunk #{i} contains needle:")
        print(f"Length: {len(chunk)} chars")
        print(f"Content: {chunk[:200]}...")

print(f"\n{'='*80}")
print(f"Total chunks containing needle: {len(needle_chunks)}")

# Check haystack
haystack = [c for c in chunks if needle not in c]
print(f"Haystack (chunks without needle): {len(haystack)}")
print(f"Expected haystack size from demo: 30")
