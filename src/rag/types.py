"""Type definitions for RAG system"""

from typing import TypedDict, List, Dict, Any, Optional


class Document(TypedDict):
    """Document record"""
    doc_id: str
    content: str
    metadata: Optional[Dict[str, Any]]


class SearchResult(TypedDict):
    """Search result"""
    doc_id: str
    content: str
    metadata: Optional[Dict[str, Any]]
    score: float
