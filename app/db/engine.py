"""Async SQLAlchemy engine and session factory for PostgreSQL.

Engine and session factory are created lazily on first access to avoid
import-time side effects (important for testing with mocked databases).
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import settings

# Lazy-initialized singletons
_engine = None
_async_session = None


def get_engine():
    """Return the async engine, creating it on first call."""
    global _engine
    if _engine is None:
        _engine = create_async_engine(
            settings.DATABASE_URL,
            echo=False,
            pool_size=10,
            max_overflow=20,
        )
    return _engine


def get_session_factory():
    """Return the async session factory, creating it on first call."""
    global _async_session
    if _async_session is None:
        _async_session = async_sessionmaker(
            get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
        )
    return _async_session


async def get_session() -> AsyncSession:  # type: ignore[misc]
    """FastAPI dependency that yields a DB session and closes it after the request."""
    factory = get_session_factory()
    async with factory() as session:
        yield session
