"""FastAPI application entry point for the Sales Conversation Analytics Engine.

This service transcribes, scores, and indexes sales conversations using:
- faster-whisper for local multilingual transcription
- LangGraph for pipeline orchestration
- Groq (free tier) for LLM-based scoring and CRM extraction
- ChromaDB for semantic search
- PostgreSQL for structured metadata
- Redis for caching
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.cache import close_redis
from app.db.engine import get_engine
from app.db.models import Base
from app.routes import analytics, calls, search

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: initialize resources on startup, clean up on shutdown."""
    # ── Startup ──
    logger.info("Starting Sales Conversation Analytics Engine")

    # Create database tables (idempotent)
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables created / verified")

    yield

    # ── Shutdown ──
    logger.info("Shutting down...")
    await close_redis()
    await get_engine().dispose()


app = FastAPI(
    title="Sales Conversation Analytics Engine",
    description=(
        "Transcribes, scores, and indexes sales conversations using a LangGraph pipeline. "
        "Features multilingual transcription (faster-whisper), rubric-based scoring (Groq LLM), "
        "CRM auto-fill extraction, and semantic search (ChromaDB). "
        "All components are free-tier or locally hosted."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ── Include routers ──
app.include_router(calls.router)
app.include_router(search.router)
app.include_router(analytics.router)


@app.get("/health", tags=["system"])
async def health_check():
    """Simple health check endpoint."""
    return {"status": "healthy", "service": "sales-conversation-analytics-engine"}
