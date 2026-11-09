"""Redis cache for call results and search queries.

Uses redis.asyncio for non-blocking operations within FastAPI.
"""

from __future__ import annotations

import json
import hashlib
import logging

import redis.asyncio as aioredis

from app.config import settings

logger = logging.getLogger(__name__)

_redis_client: aioredis.Redis | None = None


async def get_redis() -> aioredis.Redis:
    """Return the shared async Redis client, creating it on first call."""
    global _redis_client
    if _redis_client is None:
        _redis_client = aioredis.from_url(
            settings.REDIS_URL,
            decode_responses=True,
        )
    return _redis_client


async def close_redis() -> None:
    """Gracefully close the Redis connection."""
    global _redis_client
    if _redis_client is not None:
        await _redis_client.aclose()
        _redis_client = None


async def cache_call(call_id: str, data: dict, ttl: int = 3600) -> None:
    """Cache a call response in Redis with a TTL (default 1 hour)."""
    try:
        client = await get_redis()
        key = f"call:{call_id}"
        await client.setex(key, ttl, json.dumps(data))
    except Exception:
        logger.warning("Failed to cache call %s in Redis", call_id, exc_info=True)


async def get_cached_call(call_id: str) -> dict | None:
    """Retrieve a cached call response, or None if not found / expired."""
    try:
        client = await get_redis()
        key = f"call:{call_id}"
        value = await client.get(key)
        if value:
            return json.loads(value)
    except Exception:
        logger.warning("Failed to read cache for call %s", call_id, exc_info=True)
    return None


def _query_hash(query: str, n_results: int) -> str:
    """Create a deterministic cache key for a search query and result limit."""
    cache_input = f"{query.strip().casefold()}:{n_results}"
    return hashlib.sha256(cache_input.encode()).hexdigest()[:16]


async def cache_search(
    query: str, n_results: int, results: list[dict], ttl: int = 300
) -> None:
    """Cache semantic search results for a query (default 5 min TTL)."""
    try:
        client = await get_redis()
        key = f"search:{_query_hash(query, n_results)}"
        await client.setex(key, ttl, json.dumps(results))
    except Exception:
        logger.warning("Failed to cache search results", exc_info=True)


async def get_cached_search(query: str, n_results: int) -> list[dict] | None:
    """Retrieve cached search results, or None if not found / expired."""
    try:
        client = await get_redis()
        key = f"search:{_query_hash(query, n_results)}"
        value = await client.get(key)
        if value:
            return json.loads(value)
    except Exception:
        logger.warning("Failed to read cached search results", exc_info=True)
    return None
