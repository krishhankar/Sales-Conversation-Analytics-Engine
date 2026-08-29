"""Tests for the scoring and CRM extraction pipeline nodes.

These tests mock the Groq API to verify structured output parsing
without requiring a real API key.
"""

import pytest
from unittest.mock import MagicMock, patch

from app.schemas import CRMRecord, RubricScores
from app.nodes.score_call import score_call
from app.nodes.extract_crm_fields import extract_crm_fields
from app.nodes.ingest import ingest
from app.nodes.detect_language import detect_language


class TestIngestNode:
    """Tests for the ingest node."""

    def test_text_input(self):
        """Text input should set input_type to 'text'."""
        state = {"raw_transcript": "Hello, this is a test transcript."}
        result = ingest(state)
        assert result["input_type"] == "text"
        assert result["call_id"]  # UUID should be generated

    def test_audio_input(self):
        """Audio path should set input_type to 'audio'."""
        state = {"audio_path": "/tmp/test.wav"}
        result = ingest(state)
        assert result["input_type"] == "audio"

    def test_no_input_raises(self):
        """Missing both inputs should raise ValueError."""
        with pytest.raises(ValueError, match="Either audio_path or raw_transcript"):
            ingest({})


class TestDetectLanguageNode:
    """Tests for the detect_language node."""

    def test_english_text(self):
        """English text should be detected as 'en'."""
        state = {
            "raw_transcript": "Hello, thank you for joining this call today.",
            "input_type": "text",
        }
        result = detect_language(state)
        assert result["language"] == "en"

    def test_audio_passthrough(self):
        """For audio inputs with pre-set language, should pass through."""
        state = {
            "language": "hi",
            "language_confidence": 0.95,
            "input_type": "audio",
        }
        result = detect_language(state)
        assert result["language"] == "hi"
        assert result["language_confidence"] == 0.95


class TestScoreCallNode:
    """Tests for the score_call node with mocked LLM."""

    @patch("app.nodes.score_call.ChatGroq")
    def test_successful_scoring(self, mock_groq_cls, sample_transcript, sample_rubric_scores):
        """Verify structured output is correctly parsed from mocked LLM response."""
        # Set up mock
        mock_llm = MagicMock()
        mock_groq_cls.return_value = mock_llm

        mock_structured = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured
        mock_structured.invoke.return_value = RubricScores(**sample_rubric_scores)

        # Run node
        state = {"raw_transcript": sample_transcript}
        result = score_call(state)

        assert result["rubric_scores"] is not None
        assert result["rubric_scores"]["pricing_discussed"] is True
        assert result["rubric_scores"]["overall_score"] == 0.85

    @patch("app.nodes.score_call.ChatGroq")
    def test_scoring_failure_fallback(self, mock_groq_cls, sample_transcript):
        """On LLM failure, should return default zero scores, not crash."""
        mock_llm = MagicMock()
        mock_groq_cls.return_value = mock_llm

        mock_structured = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured
        mock_structured.invoke.side_effect = Exception("API error")

        state = {"raw_transcript": sample_transcript}
        result = score_call(state)

        assert result["rubric_scores"] is not None
        assert result["rubric_scores"]["overall_score"] == 0.0

    def test_empty_transcript(self):
        """Empty transcript should produce None scores."""
        state = {"raw_transcript": ""}
        result = score_call(state)
        assert result["rubric_scores"] is None


class TestExtractCRMFieldsNode:
    """Tests for the extract_crm_fields node with mocked LLM."""

    @patch("app.nodes.extract_crm_fields.ChatGroq")
    def test_successful_extraction(self, mock_groq_cls, sample_transcript, sample_crm_record):
        """Verify CRM fields are correctly extracted from mocked LLM response."""
        mock_llm = MagicMock()
        mock_groq_cls.return_value = mock_llm

        mock_structured = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured
        mock_structured.invoke.return_value = CRMRecord(**sample_crm_record)

        state = {"raw_transcript": sample_transcript}
        result = extract_crm_fields(state)

        assert result["crm_record"] is not None
        assert result["crm_record"]["customer_name_mentioned"] == "John"
        assert result["crm_record"]["predicted_outcome"] == "conversion"

    @patch("app.nodes.extract_crm_fields.ChatGroq")
    def test_extraction_failure_fallback(self, mock_groq_cls, sample_transcript):
        """On failure, should return default empty CRM record."""
        mock_llm = MagicMock()
        mock_groq_cls.return_value = mock_llm

        mock_structured = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured
        mock_structured.invoke.side_effect = Exception("API error")

        state = {"raw_transcript": sample_transcript}
        result = extract_crm_fields(state)

        assert result["crm_record"] is not None
        assert result["crm_record"]["predicted_outcome"] == "no_conversion"
