"""Embed and store node — saves transcript to ChromaDB and PostgreSQL.

This is the final node in the pipeline. It:
1. Embeds the transcript in ChromaDB for semantic search
2. Persists structured metadata to PostgreSQL
3. Caches the result in Redis for fast retrieval
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from app import cache, vectorstore
from app.db.crud import create_call_record
from app.db.engine import get_session_factory

logger = logging.getLogger(__name__)


def embed_and_store(state: dict[str, Any]) -> dict[str, Any]:
    """Embed the transcript and persist all data to storage backends.

    Reads the full state, writes to ChromaDB, PostgreSQL, and Redis.
    """
    call_id = state["call_id"]
    transcript = state.get("raw_transcript", "")

    if not transcript:
        logger.warning("embed_and_store: no transcript to store")
        state["embedded"] = False
        return state

    # ── 1. Store in ChromaDB (vector store) ──
    try:
        metadata = {
            "language": state.get("language", "unknown"),
            "source_type": state.get("input_type", "text"),
            "predicted_outcome": (state.get("crm_record") or {}).get(
                "predicted_outcome", "unknown"
            ),
        }
        # Add rubric score fields to metadata for filtered search
        rubric = state.get("rubric_scores")
        if rubric:
            metadata["overall_score"] = rubric.get("overall_score", 0.0)

        vectorstore.add_document(call_id=call_id, text=transcript, metadata=metadata)
        logger.info("Stored call %s in ChromaDB", call_id)
    except Exception:
        logger.error("Failed to store in ChromaDB", exc_info=True)

    # ── 2. Store in PostgreSQL (structured metadata) ──
    try:
        db_data = {
            "id": call_id,
            "language": state.get("language", "unknown"),
            "language_confidence": state.get("language_confidence", 0.0),
            "transcript": transcript,
            "source_type": state.get("input_type", "text"),
            "rubric_scores": state.get("rubric_scores"),
            "crm_record": state.get("crm_record"),
            "predicted_outcome": (state.get("crm_record") or {}).get(
                "predicted_outcome"
            ),
        }

        # Run the async DB write from this sync context
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # We're inside an async context (FastAPI), schedule as a task
            asyncio.ensure_future(_store_in_db(db_data))
        else:
            loop.run_until_complete(_store_in_db(db_data))

        logger.info("Stored call %s in PostgreSQL", call_id)
    except Exception:
        logger.error("Failed to store in PostgreSQL", exc_info=True)

    # ── 3. Cache in Redis ──
    try:
        cache_data = {
            "call_id": call_id,
            "language": state.get("language", "unknown"),
            "language_confidence": state.get("language_confidence", 0.0),
            "transcript": transcript,
            "source_type": state.get("input_type", "text"),
            "rubric_scores": state.get("rubric_scores"),
            "crm_record": state.get("crm_record"),
        }

        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.ensure_future(cache.cache_call(call_id, cache_data))
        else:
            loop.run_until_complete(cache.cache_call(call_id, cache_data))
    except Exception:
        logger.warning("Failed to cache in Redis", exc_info=True)

    state["embedded"] = True
    return state


async def _store_in_db(data: dict) -> None:
    """Async helper to write a call record to PostgreSQL."""
    factory = get_session_factory()
    async with factory() as session:
        await create_call_record(session, data)
