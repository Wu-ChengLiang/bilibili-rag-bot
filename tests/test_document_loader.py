"""Tests for Document Loader"""

import pytest
import tempfile
import os
from src.rag.document_loader import DocumentLoader


@pytest.fixture
def sample_text_file():
    """Create a temporary text file"""
    content = """这是第一段文字。这是第一段的第二句话。

这是第二段文字。它包含多个句子。第三句话在这里。

这是第三段。"""

    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8') as f:
        f.write(content)
        temp_path = f.name

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.remove(temp_path)


class TestDocumentLoader:
    """Test DocumentLoader"""

    def test_loader_creation(self):
        """Test that loader can be created"""
        loader = DocumentLoader()
        assert loader is not None

    def test_load_file_exists(self, sample_text_file):
        """Test loading existing file"""
        loader = DocumentLoader()
        chunks = loader.load_file(sample_text_file)
        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)

    def test_load_file_not_found(self):
        """Test loading non-existent file raises error"""
        loader = DocumentLoader()
        with pytest.raises(FileNotFoundError):
            loader.load_file("nonexistent.txt")


class TestChunkingStrategies:
    """Test different chunking strategies"""

    def test_smart_chunking(self, sample_text_file):
        """Test smart chunking by paragraphs"""
        loader = DocumentLoader()
        chunks = loader.load_file(sample_text_file, strategy="smart")

        assert len(chunks) >= 3  # At least 3 paragraphs
        assert all(len(chunk.strip()) > 0 for chunk in chunks)

    def test_sentence_chunking(self, sample_text_file):
        """Test sentence-based chunking"""
        loader = DocumentLoader()
        chunks = loader.load_file(sample_text_file, strategy="sentences")

        # Should have multiple sentence chunks
        assert len(chunks) > 3

    def test_fixed_size_chunking(self, sample_text_file):
        """Test fixed-size chunking"""
        loader = DocumentLoader()
        chunk_size = 50
        chunks = loader.load_file(
            sample_text_file,
            strategy="fixed_size",
            chunk_size=chunk_size
        )

        # Most chunks should be around the target size
        assert all(len(chunk) <= chunk_size * 1.5 for chunk in chunks)

    def test_chunk_overlap(self, sample_text_file):
        """Test chunking with overlap"""
        loader = DocumentLoader()
        chunks = loader.load_file(
            sample_text_file,
            strategy="fixed_size",
            chunk_size=30,
            overlap=10
        )

        # Should have overlapping content
        assert len(chunks) > 1


class TestChineseTextProcessing:
    """Test Chinese text processing"""

    def test_chinese_sentence_split(self):
        """Test splitting Chinese sentences"""
        content = "这是第一句。这是第二句！这是第三句？这是第四句"

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8') as f:
            f.write(content)
            temp_path = f.name

        try:
            loader = DocumentLoader()
            chunks = loader.load_file(temp_path, strategy="sentences")

            # Should split by Chinese punctuation
            assert len(chunks) >= 3
        finally:
            os.remove(temp_path)

    def test_preserve_chinese_characters(self, sample_text_file):
        """Test that Chinese characters are preserved"""
        loader = DocumentLoader()
        chunks = loader.load_file(sample_text_file)

        # All chunks should contain Chinese characters
        combined = "".join(chunks)
        assert any('\u4e00' <= char <= '\u9fff' for char in combined)
