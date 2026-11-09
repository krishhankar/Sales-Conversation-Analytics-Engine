"""Pydantic models for structured LLM output, API requests, and responses."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ═══════════════════════════════════════════════════════════════════════
# Structured LLM Output Models (used with ChatGroq.with_structured_output)
# ═══════════════════════════════════════════════════════════════════════


class RubricScores(BaseModel):
    """Rubric-based evaluation of a sales call transcript.

    Each boolean field maps to a specific sales best-practice criterion.
    The overall_score is a composite effectiveness rating from 0.0 to 1.0.
    """

    pricing_discussed: bool = Field(
        description="Whether the sales rep discussed pricing or cost details"
    )
    objection_handled: bool = Field(
        description="Whether the sales rep addressed customer objections"
    )
    next_steps_confirmed: bool = Field(
        description="Whether clear next steps were confirmed at the end"
    )
    discovery_questions_asked: bool = Field(
        description="Whether the sales rep asked discovery/qualifying questions"
    )
    product_demo_offered: bool = Field(
        description="Whether a product demo or trial was offered"
    )
    urgency_created: bool = Field(
        description="Whether the rep created a sense of urgency or scarcity"
    )
    competitor_mentioned: bool = Field(
        description="Whether a competitor product or brand was mentioned"
    )
    overall_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Overall sales effectiveness score from 0.0 (poor) to 1.0 (excellent)",
    )


class CRMRecord(BaseModel):
    """CRM-ready structured extraction from a sales conversation.

    Designed to auto-fill CRM fields without manual data entry.
    """

    customer_name_mentioned: str | None = Field(
        default=None,
        description="Customer or prospect name if mentioned in the conversation",
    )
    company_mentioned: str | None = Field(
        default=None,
        description="Customer's company or organization name if mentioned",
    )
    pricing_discussed: bool = Field(
        description="Whether pricing, cost, or budget was discussed"
    )
    objections_raised: list[str] = Field(
        default_factory=list,
        description="List of specific objections raised by the customer",
    )
    next_steps_confirmed: bool = Field(
        description="Whether concrete next steps were agreed upon"
    )
    follow_up_date_mentioned: str | None = Field(
        default=None,
        description="Any follow-up date or timeline mentioned (free text)",
    )
    predicted_outcome: Literal["conversion", "no_conversion"] = Field(
        description="Predicted conversion outcome: 'conversion' or 'no_conversion'"
    )
    key_topics: list[str] = Field(
        default_factory=list,
        description="Key topics or themes discussed in the conversation",
    )


# ═══════════════════════════════════════════════════════════════════════
# API Request / Response Models
# ═══════════════════════════════════════════════════════════════════════


class TextIngestRequest(BaseModel):
    """Request body for ingesting a text transcript."""

    model_config = ConfigDict(extra="forbid")

    transcript: str = Field(
        min_length=10,
        description="The full conversation transcript text to analyze",
    )

    @field_validator("transcript")
    @classmethod
    def transcript_must_contain_text(cls, value: str) -> str:
        value = value.strip()
        if len(value) < 10:
            raise ValueError("Transcript must contain at least 10 non-whitespace characters")
        return value


class CallResponse(BaseModel):
    """Response returned after processing a call via POST /calls."""

    call_id: str
    language: str
    language_confidence: float
    rubric_scores: RubricScores
    crm_record: CRMRecord
    transcript: str


class CallDetail(BaseModel):
    """Full detail response for GET /calls/{id}."""

    call_id: str
    language: str
    language_confidence: float
    transcript: str
    rubric_scores: RubricScores
    crm_record: CRMRecord
    created_at: str
    source_type: str  # "audio" | "text"


class SearchRequest(BaseModel):
    """Request body for semantic search."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        min_length=3,
        description="Natural language search query",
    )
    n_results: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Number of results to return",
    )

    @field_validator("query")
    @classmethod
    def query_must_contain_text(cls, value: str) -> str:
        value = value.strip()
        if len(value) < 3:
            raise ValueError("Query must contain at least 3 non-whitespace characters")
        return value


class SearchResult(BaseModel):
    """A single semantic search hit."""

    call_id: str
    excerpt: str
    score: float
    rubric_scores: RubricScores | None = None


class SearchResponse(BaseModel):
    """Response for POST /search."""

    query: str
    results: list[SearchResult]
    total: int


class AnalyticsSummary(BaseModel):
    """Aggregate analytics across all processed calls."""

    total_calls: int
    avg_overall_score: float
    outcome_distribution: dict[str, int]
    avg_rubric_scores: dict[str, float]
    language_distribution: dict[str, int]
