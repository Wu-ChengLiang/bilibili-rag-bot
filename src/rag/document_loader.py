"""Document loader with flexible text chunking strategies"""

import re
from typing import List, Literal


class DocumentLoader:
    """Load and chunk text documents for RAG systems"""

    @staticmethod
    def load_file(
        file_path: str,
        strategy: Literal["sentences", "fixed_size", "smart"] = "smart",
        chunk_size: int = 300,
        overlap: int = 50
    ) -> List[str]:
        """Load and chunk a text file

        Args:
            file_path: Path to the text file
            strategy: Chunking strategy to use
                - "sentences": Group by sentence boundaries
                - "fixed_size": Fixed character size chunks with overlap
                - "smart": Smart chunking based on content structure
            chunk_size: Target size for chunks (in characters)
            overlap: Character overlap between chunks (for fixed_size)

        Returns:
            List of text chunks
        """
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Normalize line endings (CRLF -> LF)
        content = content.replace("\r\n", "\n").replace("\r", "\n")

        if strategy == "sentences":
            return DocumentLoader._chunk_by_sentences(content, chunk_size)
        elif strategy == "fixed_size":
            return DocumentLoader._chunk_by_fixed_size(content, chunk_size, overlap)
        elif strategy == "smart":
            return DocumentLoader._chunk_smart(content, chunk_size)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    @staticmethod
    def _chunk_by_sentences(content: str, target_size: int) -> List[str]:
        """Chunk text by grouping sentences to target size

        Args:
            content: Full text content
            target_size: Target chunk size in characters

        Returns:
            List of chunks
        """
        # Split by Chinese sentence endings
        sentence_pattern = r'[。！？；\n]+'
        sentences = re.split(sentence_pattern, content)
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks = []
        current_chunk = []
        current_size = 0

        for sentence in sentences:
            sentence_len = len(sentence)

            # If single sentence exceeds target, add it as its own chunk
            if sentence_len > target_size:
                if current_chunk:
                    chunks.append("".join(current_chunk))
                    current_chunk = []
                    current_size = 0
                chunks.append(sentence)
                continue

            # If adding this sentence exceeds target, start new chunk
            if current_size + sentence_len > target_size and current_chunk:
                chunks.append("".join(current_chunk))
                current_chunk = [sentence]
                current_size = sentence_len
            else:
                current_chunk.append(sentence)
                current_size += sentence_len

        # Add remaining chunk
        if current_chunk:
            chunks.append("".join(current_chunk))

        return chunks

    @staticmethod
    def _chunk_by_fixed_size(content: str, chunk_size: int, overlap: int) -> List[str]:
        """Chunk text by fixed character size with overlap

        Args:
            content: Full text content
            chunk_size: Size of each chunk
            overlap: Character overlap between chunks

        Returns:
            List of chunks
        """
        # Remove excessive whitespace
        content = re.sub(r'\s+', ' ', content.strip())

        chunks = []
        start = 0
        content_len = len(content)

        while start < content_len:
            end = start + chunk_size
            chunk = content[start:end].strip()

            if chunk:
                chunks.append(chunk)

            # Move start position with overlap
            start = end - overlap if overlap > 0 else end

        return chunks

    @staticmethod
    def _chunk_smart(content: str, target_size: int) -> List[str]:
        """Smart chunking that preserves dialogue and narrative structure

        This strategy:
        1. Identifies short lines (likely dialogue or single sentences)
        2. Identifies long lines (narrative paragraphs)
        3. Groups them intelligently to maintain context

        Args:
            content: Full text content
            target_size: Target chunk size in characters

        Returns:
            List of chunks
        """
        lines = content.split('\n')
        lines = [line.strip() for line in lines if line.strip()]

        chunks = []
        current_chunk = []
        current_size = 0

        for line in lines:
            line_len = len(line)

            # Very long line - split it if needed
            if line_len > target_size * 1.5:
                # Save current chunk
                if current_chunk:
                    chunks.append("\n".join(current_chunk))
                    current_chunk = []
                    current_size = 0

                # Split long line by sentences
                sub_chunks = DocumentLoader._chunk_by_sentences(line, target_size)
                chunks.extend(sub_chunks)
                continue

            # Check if adding this line exceeds target
            if current_size + line_len > target_size and current_chunk:
                # Special case: if current line is short (dialogue),
                # and we have content, try to include it
                is_short_line = line_len < 100
                is_dialogue = '"' in line or '"' in line or '「' in line

                if is_short_line and is_dialogue and current_size > target_size * 0.5:
                    # Start new chunk with this dialogue
                    chunks.append("\n".join(current_chunk))
                    current_chunk = [line]
                    current_size = line_len
                elif current_size > target_size * 0.3:
                    # We have enough content, start new chunk
                    chunks.append("\n".join(current_chunk))
                    current_chunk = [line]
                    current_size = line_len
                else:
                    # Add to current chunk even if it exceeds slightly
                    current_chunk.append(line)
                    current_size += line_len
            else:
                # Add to current chunk
                current_chunk.append(line)
                current_size += line_len

        # Add remaining chunk
        if current_chunk:
            chunks.append("\n".join(current_chunk))

        return chunks

    @staticmethod
    def load_multiple_files(
        file_paths: List[str],
        strategy: Literal["sentences", "fixed_size", "smart"] = "smart",
        chunk_size: int = 300
    ) -> List[str]:
        """Load and chunk multiple text files

        Args:
            file_paths: List of file paths
            strategy: Chunking strategy
            chunk_size: Target chunk size

        Returns:
            Combined list of chunks from all files
        """
        all_chunks = []
        for file_path in file_paths:
            chunks = DocumentLoader.load_file(file_path, strategy, chunk_size)
            all_chunks.extend(chunks)
        return all_chunks
