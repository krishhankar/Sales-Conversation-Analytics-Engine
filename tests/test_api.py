"""Integration tests for the API endpoints.

Tests the FastAPI endpoints using httpx AsyncClient with mocked
pipeline, database, and cache dependencies.
"""

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from httpx import ASGITransport, AsyncClient


@pytest.fixture
def mock_pipeline_result(sample_rubric_scores, sample_crm_record):
    """A complete pipeline result dict."""
    return {
        "call_id": "test-uuid-1234",
        "input_type": "text",
        "raw_transcript": "Hello, this is a test conversation.",
        "language": "en",
        "language_confidence": 0.95,
        "rubric_scores": sample_rubric_scores,
        "crm_record": sample_crm_record,
        "embedded": True,
    }


@pytest.mark.asyncio
class TestHealthEndpoint:
    """Tests for the health check endpoint."""

    async def test_health_check(self, async_client):
        """GET /health should return 200 with status healthy."""
        response = await async_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    async def test_dashboard_is_served(self, async_client):
        response = await async_client.get("/")
        assert response.status_code == 200
        assert "Sales Conversation Analytics" in response.text


@pytest.mark.asyncio
class TestCallsEndpoint:
    """Tests for the /calls endpoints."""

    @patch("app.routes.calls.pipeline")
    async def test_ingest_text(self, mock_pipeline, async_client, mock_pipeline_result):
        """POST /calls with text body should return scored results."""
        mock_pipeline.invoke.return_value = mock_pipeline_result

        response = await async_client.post(
            "/calls",
            json={"transcript": "Hello, this is a test sales conversation transcript for analysis."},
        )

        assert response.status_code == 201
        data = response.json()
        assert data["call_id"] == "test-uuid-1234"
        assert data["language"] == "en"
        assert data["rubric_scores"]["pricing_discussed"] is True
        assert data["crm_record"]["predicted_outcome"] == "conversion"

    async def test_ingest_no_input(self, async_client):
        """POST /calls with empty or invalid body should return 422."""
        response = await async_client.post("/calls", json={"transcript": "short"})
        assert response.status_code == 422  # min_length=10 validation

    async def test_ingest_whitespace_transcript(self, async_client):
        response = await async_client.post("/calls", json={"transcript": "          "})
        assert response.status_code == 422

    async def test_empty_audio_upload_is_rejected(self, async_client):
        response = await async_client.post(
            "/calls/audio",
            files={"audio": ("empty.wav", b"", "audio/wav")},
        )
        assert response.status_code == 400

    @patch("app.routes.calls.MAX_AUDIO_BYTES", 4)
    async def test_oversized_audio_upload_is_rejected(self, async_client):
        response = await async_client.post(
            "/calls/audio",
            files={"audio": ("large.wav", b"12345", "audio/wav")},
        )
        assert response.status_code == 413

    @patch("app.routes.calls.cache")
    @patch("app.routes.calls.get_call_record")
    async def test_get_call_from_db(
        self, mock_get_record, mock_cache, async_client, sample_rubric_scores, sample_crm_record
    ):
        """GET /calls/{id} should return call details from DB when not cached."""
        mock_cache.get_cached_call = AsyncMock(return_value=None)

        # Create a mock DB record
        mock_record = MagicMock()
        mock_record.id = "test-uuid-1234"
        mock_record.language = "en"
        mock_record.language_confidence = 0.95
        mock_record.transcript = "Test transcript"
        mock_record.rubric_scores = sample_rubric_scores
        mock_record.crm_record = sample_crm_record
        mock_record.created_at = MagicMock()
        mock_record.created_at.isoformat.return_value = "2026-08-29T12:00:00"
        mock_record.source_type = "text"

        mock_get_record.return_value = mock_record

        response = await async_client.get("/calls/test-uuid-1234")
        assert response.status_code == 200
        data = response.json()
        assert data["call_id"] == "test-uuid-1234"

    @patch("app.routes.calls.cache")
    @patch("app.routes.calls.get_call_record")
    async def test_get_call_not_found(self, mock_get_record, mock_cache, async_client):
        """GET /calls/{id} with unknown ID should return 404."""
        mock_cache.get_cached_call = AsyncMock(return_value=None)
        mock_get_record.return_value = None

        response = await async_client.get("/calls/nonexistent-id")
        assert response.status_code == 404


@pytest.mark.asyncio
class TestSearchEndpoint:
    """Tests for the /search endpoint."""

    @patch("app.routes.search.cache")
    @patch("app.routes.search.vectorstore")
    async def test_semantic_search(self, mock_vs, mock_cache, async_client):
        """POST /search should return matching results."""
        mock_cache.get_cached_search = AsyncMock(return_value=None)
        mock_cache.cache_search = AsyncMock()

        mock_vs.search.return_value = [
            {
                "call_id": "test-uuid-1",
                "excerpt": "Customer discussed pricing concerns...",
                "score": 0.89,
                "metadata": {"language": "en"},
            }
        ]

        response = await async_client.post(
            "/search",
            json={"query": "pricing objections", "n_results": 5},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert data["results"][0]["call_id"] == "test-uuid-1"
        mock_cache.get_cached_search.assert_awaited_once_with("pricing objections", 5)
        mock_cache.cache_search.assert_awaited_once()

    @patch("app.routes.search.cache")
    async def test_cached_empty_search_is_returned(self, mock_cache, async_client):
        mock_cache.get_cached_search = AsyncMock(return_value=[])
        response = await async_client.post("/search", json={"query": "pricing concerns"})
        assert response.status_code == 200
        assert response.json()["results"] == []
        assert response.json()["total"] == 0

    async def test_whitespace_search_query_is_rejected(self, async_client):
        response = await async_client.post("/search", json={"query": "   "})
        assert response.status_code == 422


@pytest.mark.asyncio
class TestAnalyticsEndpoint:
    """Tests for the /analytics/summary endpoint."""

    @patch("app.routes.analytics.get_analytics_summary")
    async def test_analytics_summary(self, mock_summary, async_client):
        """GET /analytics/summary should return aggregate stats."""
        mock_summary.return_value = {
            "total_calls": 50,
            "avg_overall_score": 0.72,
            "outcome_distribution": {"conversion": 30, "no_conversion": 20},
            "avg_rubric_scores": {"pricing_discussed": 0.8, "overall_score": 0.72},
            "language_distribution": {"en": 45, "hi": 5},
        }

        response = await async_client.get("/analytics/summary")
        assert response.status_code == 200
        data = response.json()
        assert data["total_calls"] == 50
        assert data["avg_overall_score"] == 0.72
