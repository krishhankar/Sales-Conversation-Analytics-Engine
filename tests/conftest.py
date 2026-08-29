"""Pytest fixtures for the Sales Conversation Analytics Engine tests."""

import os
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from httpx import ASGITransport, AsyncClient


# Set test environment variables before importing app modules
os.environ.setdefault("GROQ_API_KEY", "test_key_not_real")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///test.db")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("CHROMA_PERSIST_DIR", "./test_chroma")


@pytest.fixture
def sample_transcript() -> str:
    """A realistic sample sales conversation transcript for testing."""
    return """
    Sales Rep: Hi, thank you for your time today. I'm Sarah from CloudMetrics. 
    How are you doing?
    
    Customer: Hi Sarah, I'm doing well. I'm John from TechCorp. We've been 
    looking into analytics platforms for our team.
    
    Sales Rep: That's great to hear, John. Can you tell me a bit about what 
    challenges your team is currently facing with data reporting?
    
    Customer: Sure. Our biggest pain point is that our team spends about 
    20 hours a week on manual report generation. We need something automated.
    
    Sales Rep: I completely understand. Many of our customers had the same issue 
    before switching to CloudMetrics. Our platform can reduce that manual work 
    by up to 60 percent through automated dashboards and scheduled reports.
    
    Customer: That sounds promising. But I'm a bit concerned about the pricing. 
    We're a mid-size company and our budget is tight this quarter.
    
    Sales Rep: I totally get that concern. Let me walk you through our pricing 
    tiers. Our Professional plan starts at $299 per month and includes up to 
    50 users. For your team size, that would work perfectly. We also offer a 
    14-day free trial so you can evaluate the platform risk-free.
    
    Customer: A free trial sounds good. What about integration with Salesforce? 
    We're heavily invested in that ecosystem.
    
    Sales Rep: Great question. We have a native Salesforce integration that 
    takes about 15 minutes to set up. Unlike some competitors like DataDog 
    or Tableau, our integration is bi-directional, meaning data flows both ways.
    
    Customer: That's actually one of the things that frustrated us about Tableau.
    
    Sales Rep: I hear that a lot. Would you be interested in setting up a 
    product demo for your team next Tuesday? I can show you the Salesforce 
    integration live.
    
    Customer: Yes, let's do that. Tuesday at 2 PM works for us.
    
    Sales Rep: Perfect. I'll send over a calendar invite right after this call. 
    I'll also include a case study from a similar company in your industry. 
    Looking forward to it, John.
    
    Customer: Thanks, Sarah. Talk to you Tuesday.
    """


@pytest.fixture
def sample_rubric_scores() -> dict:
    """Sample rubric scores for testing."""
    return {
        "pricing_discussed": True,
        "objection_handled": True,
        "next_steps_confirmed": True,
        "discovery_questions_asked": True,
        "product_demo_offered": True,
        "urgency_created": False,
        "competitor_mentioned": True,
        "overall_score": 0.85,
    }


@pytest.fixture
def sample_crm_record() -> dict:
    """Sample CRM record for testing."""
    return {
        "customer_name_mentioned": "John",
        "company_mentioned": "TechCorp",
        "pricing_discussed": True,
        "objections_raised": ["Budget is tight this quarter"],
        "next_steps_confirmed": True,
        "follow_up_date_mentioned": "Next Tuesday at 2 PM",
        "predicted_outcome": "conversion",
        "key_topics": [
            "analytics platform",
            "automated reporting",
            "Salesforce integration",
            "pricing",
        ],
    }


@pytest_asyncio.fixture
async def async_client():
    """Create an async test client for the FastAPI app.

    Patches database and cache dependencies to avoid requiring
    real PostgreSQL/Redis during unit tests.
    """
    # Import modules first so patch targets can be resolved
    import app.main as main_module
    import app.db.engine as engine_module

    # Mock the async engine's begin() context manager
    mock_conn = AsyncMock()
    mock_ctx = AsyncMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_ctx.__aexit__ = AsyncMock(return_value=False)

    mock_engine = MagicMock()
    mock_engine.begin.return_value = mock_ctx
    mock_engine.dispose = AsyncMock()

    # Patch Base.metadata.create_all to be a no-op
    mock_conn.run_sync = AsyncMock()

    with patch.object(engine_module, "get_engine", return_value=mock_engine), \
         patch.object(main_module, "get_engine", return_value=mock_engine), \
         patch.object(main_module, "close_redis", new_callable=AsyncMock):

        transport = ASGITransport(app=main_module.app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            yield client

