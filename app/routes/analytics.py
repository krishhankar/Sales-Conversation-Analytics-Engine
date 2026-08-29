"""Analytics endpoints.

GET /analytics/summary — aggregate statistics across all processed calls.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.crud import get_analytics_summary
from app.db.engine import get_session
from app.schemas import AnalyticsSummary

router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get("/summary", response_model=AnalyticsSummary)
async def analytics_summary(session: AsyncSession = Depends(get_session)):
    """Return aggregate statistics across all processed calls.

    Includes: total calls, average rubric scores, outcome distribution,
    and language distribution.
    """
    data = await get_analytics_summary(session)

    return AnalyticsSummary(
        total_calls=data["total_calls"],
        avg_overall_score=data["avg_overall_score"],
        outcome_distribution=data["outcome_distribution"],
        avg_rubric_scores=data["avg_rubric_scores"],
        language_distribution=data["language_distribution"],
    )
