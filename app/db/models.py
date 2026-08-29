"""SQLAlchemy ORM models for PostgreSQL storage."""

from datetime import datetime

from sqlalchemy import DateTime, Float, String, Text, func
from sqlalchemy.dialects.postgresql import JSON
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""

    pass


class CallRecord(Base):
    """Stores structured metadata and transcript for each processed call."""

    __tablename__ = "call_records"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    language: Mapped[str] = mapped_column(String(10), nullable=False)
    language_confidence: Mapped[float] = mapped_column(Float, default=0.0)
    transcript: Mapped[str] = mapped_column(Text, nullable=False)
    source_type: Mapped[str] = mapped_column(
        String(10), nullable=False, default="text"
    )  # "audio" | "text"
    rubric_scores: Mapped[dict] = mapped_column(JSON, nullable=True)
    crm_record: Mapped[dict] = mapped_column(JSON, nullable=True)
    predicted_outcome: Mapped[str | None] = mapped_column(
        String(20), nullable=True
    )  # "conversion" | "no_conversion"
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
