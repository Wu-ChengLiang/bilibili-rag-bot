"""Local file data loader for txt and markdown files"""

import logging
from pathlib import Path
from typing import List, Optional
from ..document import Document
from .base import BaseDataLoader

logger = logging.getLogger(__name__)


class LocalFileLoader(BaseDataLoader):
    """Load documents from local txt and markdown files"""

    def __init__(
        self,
        file_paths: Optional[List[str]] = None,
        directory: Optional[str] = None,
        file_pattern: str = "*.txt",
    ):
        """
        Initialize LocalFileLoader

        Args:
            file_paths: List of specific file paths to load
            directory: Directory to search for files
            file_pattern: Pattern to match files (default: *.txt)
                         Can be "*.txt", "*.md", "*.{txt,md}", etc.
        """
        self.file_paths = file_paths or []
        self.directory = Path(directory) if directory else None
        self.file_pattern = file_pattern

    def load(self) -> List[Document]:
        """Load documents from local files"""
        documents = []

        # Collect all files to load
        files_to_load = []

        # Add explicitly specified files
        for file_path in self.file_paths:
            path = Path(file_path)
            if path.exists() and path.is_file():
                files_to_load.append(path)
            else:
                logger.warning(f"File not found: {file_path}")

        # Add files from directory if specified
        if self.directory and self.directory.exists():
            logger.info(f"Searching for files matching '{self.file_pattern}' in {self.directory}")
            for file_path in self.directory.glob(self.file_pattern):
                if file_path.is_file():
                    files_to_load.append(file_path)

        if not files_to_load:
            logger.warning("No files found to load")
            return []

        logger.info(f"Found {len(files_to_load)} file(s) to load")

        # Load content from each file
        for file_path in files_to_load:
            try:
                logger.info(f"Loading file: {file_path}")
                content = file_path.read_text(encoding="utf-8")

                if not content.strip():
                    logger.warning(f"File is empty: {file_path}")
                    continue

                document = Document(
                    content=content,
                    doc_id=file_path.stem,  # Use filename as ID
                    source="local_file",
                    title=file_path.name,
                    url=str(file_path.absolute()),
                    metadata={
                        "file_path": str(file_path.absolute()),
                        "file_size": len(content),
                    },
                )

                documents.append(document)
                logger.info(
                    f"✅ Loaded document: {file_path.name} ({len(content)} bytes)"
                )

            except Exception as e:
                logger.error(f"❌ Failed to load file {file_path}: {e}", exc_info=True)

        # Validate documents
        valid_documents = self._validate_documents(documents)
        logger.info(f"✅ Loaded {len(valid_documents)} valid document(s)")
        return valid_documents
