"""Configuration management for data loaders"""

import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass
class FeishuConfig:
    """Feishu configuration"""

    app_id: str
    app_secret: str
    wiki_space_id: Optional[str] = None

    @classmethod
    def from_env(cls) -> "FeishuConfig":
        """Load Feishu config from environment variables"""
        app_id = os.getenv("FEISHU_APP_ID")
        app_secret = os.getenv("FEISHU_APP_SECRET")
        wiki_space_id = os.getenv("FEISHU_WIKI_SPACE_ID")

        if not app_id or not app_secret:
            raise ValueError(
                "FEISHU_APP_ID and FEISHU_APP_SECRET must be set in .env file"
            )

        return cls(
            app_id=app_id,
            app_secret=app_secret,
            wiki_space_id=wiki_space_id,
        )


@dataclass
class RAGConfig:
    """RAG system configuration"""

    embedding_model: str = "shibing624/text2vec-base-chinese"
    vector_store: str = "chromadb"
    persist_directory: str = "./chroma_db"

    @classmethod
    def from_env(cls) -> "RAGConfig":
        """Load RAG config from environment variables"""
        return cls(
            embedding_model=os.getenv(
                "RAG_EMBEDDING_MODEL", "shibing624/text2vec-base-chinese"
            ),
            vector_store=os.getenv("RAG_VECTOR_STORE", "chromadb"),
            persist_directory=os.getenv("RAG_PERSIST_DIRECTORY", "./chroma_db"),
        )
