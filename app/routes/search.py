"""Semantic search endpoint.

POST /search — search across all stored call transcripts using natural
               language queries, powered by ChromaDB + HuggingFace embeddings.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter

from app import cache, vectorstore
from app.schemas import RubricScores, SearchRequest, SearchResponse, SearchResult

logger = logging.getLogger(__name__)

router = APIRouter(tags=["search"])


@router.post("/search", response_model=SearchResponse)
async def semantic_search(request: SearchRequest):
    """Search across all processed calls using semantic similarity.

    Example queries:
    - "find calls where the customer objected to price but still converted"
    - "calls discussing integration capabilities"
    - "conversations with confirmed next steps"
    """
    # Check cache first
    cached = await cache.get_cached_search(request.query)
    if cached:
        return SearchResponse(
            query=request.query,
            results=[SearchResult(**r) for r in cached],
            total=len(cached),
        )

    # Perform semantic search via ChromaDB
    raw_results = vectorstore.search(
        query=request.query,
        n_results=request.n_results,
    )

    results = []
    for hit in raw_results:
        # Try to reconstruct rubric scores from metadata if available
        rubric = None
        meta = hit.get("metadata", {})
        if "overall_score" in meta:
            # We only stored overall_score in metadata; for full rubric, we'd need DB lookup
            rubric = None  # Partial data not worth returning

        results.append(
            SearchResult(
                call_id=hit["call_id"],
                excerpt=hit["excerpt"],
                score=hit["score"],
                rubric_scores=rubric,
            )
        )

    # Cache results
    await cache.cache_search(
        request.query,
        [r.model_dump() for r in results],
    )

    return SearchResponse(
        query=request.query,
        results=results,
        total=len(results),
    )
