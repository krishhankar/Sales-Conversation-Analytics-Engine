"""ChromaDB vector store for semantic search across sales conversations.

Uses an embedded PersistentClient (no separate server) with HuggingFace
embeddings (all-MiniLM-L6-v2, runs locally, no API key needed).
"""

from __future__ import annotations

import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import settings
from app.dependencies import get_embedding_model

# ── Singletons (initialized once, reused across requests) ──

_embeddings: HuggingFaceEmbeddings | None = None
_chroma_client: chromadb.ClientAPI | None = None
_langchain_store: Chroma | None = None

COLLECTION_NAME = "sales_conversations"


def get_embeddings() -> HuggingFaceEmbeddings:
    """Return the shared embedding model instance."""
    global _embeddings
    if _embeddings is None:
        _embeddings = get_embedding_model()
    return _embeddings


def get_chroma_client() -> chromadb.ClientAPI:
    """Return the shared ChromaDB persistent client."""
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)
    return _chroma_client


def get_langchain_store() -> Chroma:
    """Return a LangChain Chroma wrapper for high-level operations."""
    global _langchain_store
    if _langchain_store is None:
        _langchain_store = Chroma(
            client=get_chroma_client(),
            collection_name=COLLECTION_NAME,
            embedding_function=get_embeddings(),
        )
    return _langchain_store


def add_document(call_id: str, text: str, metadata: dict | None = None) -> None:
    """Add a transcript to the vector store with associated metadata."""
    store = get_langchain_store()
    meta = metadata or {}
    meta["call_id"] = call_id
    store.add_texts(
        texts=[text],
        metadatas=[meta],
        ids=[call_id],
    )


def search(query: str, n_results: int = 10) -> list[dict]:
    """Perform semantic similarity search and return results with scores.

    Returns a list of dicts with keys: call_id, excerpt, score, metadata.
    """
    store = get_langchain_store()
    results = store.similarity_search_with_relevance_scores(
        query=query,
        k=n_results,
    )

    output = []
    for doc, score in results:
        output.append(
            {
                "call_id": doc.metadata.get("call_id", ""),
                "excerpt": doc.page_content[:500],
                "score": round(float(score), 4),
                "metadata": doc.metadata,
            }
        )
    return output
