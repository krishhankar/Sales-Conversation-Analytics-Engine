"""CRUD operations for call records in PostgreSQL."""

from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import CallRecord


async def create_call_record(session: AsyncSession, data: dict) -> CallRecord:
    """Insert a new call record and return it."""
    record = CallRecord(**data)
    session.add(record)
    await session.commit()
    await session.refresh(record)
    return record


async def get_call_record(session: AsyncSession, call_id: str) -> CallRecord | None:
    """Fetch a single call record by ID."""
    result = await session.execute(select(CallRecord).where(CallRecord.id == call_id))
    return result.scalar_one_or_none()


async def get_all_calls(
    session: AsyncSession, limit: int = 100, offset: int = 0
) -> list[CallRecord]:
    """Fetch a paginated list of call records."""
    result = await session.execute(
        select(CallRecord)
        .order_by(CallRecord.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    return list(result.scalars().all())


async def get_analytics_summary(session: AsyncSession) -> dict:
    """Compute aggregate analytics across all call records.

    Returns total count, average overall score, outcome distribution,
    average rubric scores, and language distribution.
    """
    # Total count
    total_result = await session.execute(select(func.count(CallRecord.id)))
    total_calls = total_result.scalar() or 0

    if total_calls == 0:
        return {
            "total_calls": 0,
            "avg_overall_score": 0.0,
            "outcome_distribution": {},
            "avg_rubric_scores": {},
            "language_distribution": {},
        }

    # Outcome distribution
    outcome_rows = await session.execute(
        select(CallRecord.predicted_outcome, func.count(CallRecord.id)).group_by(
            CallRecord.predicted_outcome
        )
    )
    outcome_distribution = {
        row[0] or "unknown": row[1] for row in outcome_rows.all()
    }

    # Language distribution
    lang_rows = await session.execute(
        select(CallRecord.language, func.count(CallRecord.id)).group_by(
            CallRecord.language
        )
    )
    language_distribution = {row[0]: row[1] for row in lang_rows.all()}

    # Fetch all records for rubric score averaging (JSON columns need app-level aggregation)
    all_records = await session.execute(select(CallRecord.rubric_scores))
    rubric_list = [row[0] for row in all_records.all() if row[0] is not None]

    avg_rubric_scores: dict[str, float] = {}
    avg_overall_score = 0.0

    if rubric_list:
        # Aggregate rubric fields
        rubric_keys = [
            "pricing_discussed",
            "objection_handled",
            "next_steps_confirmed",
            "discovery_questions_asked",
            "product_demo_offered",
            "urgency_created",
            "competitor_mentioned",
            "overall_score",
        ]
        for key in rubric_keys:
            values = []
            for rubric in rubric_list:
                val = rubric.get(key)
                if val is not None:
                    # Booleans become 1.0/0.0 for averaging
                    values.append(float(val) if isinstance(val, (int, float)) else (1.0 if val else 0.0))
            avg_rubric_scores[key] = round(sum(values) / len(values), 3) if values else 0.0

        avg_overall_score = avg_rubric_scores.get("overall_score", 0.0)

    return {
        "total_calls": total_calls,
        "avg_overall_score": avg_overall_score,
        "outcome_distribution": outcome_distribution,
        "avg_rubric_scores": avg_rubric_scores,
        "language_distribution": language_distribution,
    }
