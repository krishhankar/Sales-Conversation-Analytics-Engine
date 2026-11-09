"""Tests for Pydantic schema models — validation, serialization, edge cases."""

import pytest
from pydantic import ValidationError

from app.schemas import (
    AnalyticsSummary,
    CallResponse,
    CRMRecord,
    RubricScores,
    SearchRequest,
    TextIngestRequest,
)


class TestRubricScores:
    """Tests for the RubricScores structured output model."""

    def test_valid_scores(self, sample_rubric_scores):
        """Valid scores should parse without error."""
        scores = RubricScores(**sample_rubric_scores)
        assert scores.pricing_discussed is True
        assert scores.overall_score == 0.85

    def test_overall_score_bounds(self):
        """overall_score must be between 0.0 and 1.0."""
        with pytest.raises(ValidationError):
            RubricScores(
                pricing_discussed=True,
                objection_handled=False,
                next_steps_confirmed=True,
                discovery_questions_asked=True,
                product_demo_offered=False,
                urgency_created=False,
                competitor_mentioned=False,
                overall_score=1.5,  # Out of bounds
            )

    def test_serialization_roundtrip(self, sample_rubric_scores):
        """model_dump → RubricScores should be lossless."""
        original = RubricScores(**sample_rubric_scores)
        dumped = original.model_dump()
        restored = RubricScores(**dumped)
        assert original == restored


class TestCRMRecord:
    """Tests for the CRMRecord structured output model."""

    def test_valid_record(self, sample_crm_record):
        """Valid CRM record should parse correctly."""
        record = CRMRecord(**sample_crm_record)
        assert record.customer_name_mentioned == "John"
        assert record.predicted_outcome == "conversion"
        assert len(record.objections_raised) == 1

    def test_nullable_fields(self):
        """Optional fields should default to None / empty list."""
        record = CRMRecord(
            pricing_discussed=False,
            next_steps_confirmed=False,
            predicted_outcome="no_conversion",
        )
        assert record.customer_name_mentioned is None
        assert record.company_mentioned is None
        assert record.objections_raised == []
        assert record.key_topics == []

    def test_serialization_roundtrip(self, sample_crm_record):
        """model_dump → CRMRecord should be lossless."""
        original = CRMRecord(**sample_crm_record)
        dumped = original.model_dump()
        restored = CRMRecord(**dumped)
        assert original == restored


class TestTextIngestRequest:
    """Tests for the text ingestion request model."""

    def test_valid_transcript(self):
        """Valid transcript should parse."""
        req = TextIngestRequest(transcript="This is a valid transcript for testing")
        assert len(req.transcript) > 10

    def test_too_short_transcript(self):
        """Transcripts shorter than 10 chars should be rejected."""
        with pytest.raises(ValidationError):
            TextIngestRequest(transcript="Short")

    def test_whitespace_only_content_is_rejected(self):
        with pytest.raises(ValidationError):
            TextIngestRequest(transcript="          ")

    def test_unknown_request_fields_are_rejected(self):
        with pytest.raises(ValidationError):
            TextIngestRequest(transcript="A valid enough transcript.", transcrip="typo")


class TestSearchRequest:
    """Tests for the search request model."""

    def test_valid_query(self):
        """Valid search query should parse."""
        req = SearchRequest(query="find pricing discussions")
        assert req.n_results == 10  # default

    def test_n_results_bounds(self):
        """n_results must be between 1 and 50."""
        with pytest.raises(ValidationError):
            SearchRequest(query="test query", n_results=0)

        with pytest.raises(ValidationError):
            SearchRequest(query="test query", n_results=100)

    def test_whitespace_query_is_rejected(self):
        with pytest.raises(ValidationError):
            SearchRequest(query="   ")

    def test_query_is_trimmed(self):
        assert SearchRequest(query="  pricing discussions  ").query == "pricing discussions"


class TestCRMRecordValidation:
    def test_unknown_outcome_is_rejected(self):
        with pytest.raises(ValidationError):
            CRMRecord(
                pricing_discussed=False,
                next_steps_confirmed=False,
                predicted_outcome="maybe",
            )


class TestAnalyticsSummary:
    """Tests for the analytics summary response model."""

    def test_valid_summary(self):
        """Valid analytics summary should parse."""
        summary = AnalyticsSummary(
            total_calls=100,
            avg_overall_score=0.72,
            outcome_distribution={"conversion": 60, "no_conversion": 40},
            avg_rubric_scores={"pricing_discussed": 0.8, "overall_score": 0.72},
            language_distribution={"en": 90, "hi": 10},
        )
        assert summary.total_calls == 100
