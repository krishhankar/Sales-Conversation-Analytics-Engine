"""Extract CRM fields node — pulls structured CRM data from a transcript.

Uses a separate Groq LLM call (two-node strategy: scoring + extraction are
decoupled for reliability) to extract customer name, objections, next steps,
predicted outcome, and other CRM-ready fields.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_groq import ChatGroq

from app.config import settings
from app.schemas import CRMRecord

logger = logging.getLogger(__name__)

EXTRACTION_PROMPT = """You are a CRM data extraction assistant for B2B SaaS sales teams.

Extract structured CRM fields from the following sales conversation transcript.
Be precise — only extract information that is explicitly stated or strongly implied.

FIELDS TO EXTRACT:
1. customer_name_mentioned: The customer's or prospect's name (null if not mentioned)
2. company_mentioned: The customer's company/organization name (null if not mentioned)
3. pricing_discussed: Whether pricing, budget, or cost was discussed (true/false)
4. objections_raised: List of specific objections or concerns the customer raised (empty list if none)
5. next_steps_confirmed: Whether concrete next steps were agreed upon (true/false)
6. follow_up_date_mentioned: Any follow-up date or timeline mentioned (null if none)
7. predicted_outcome: Based on the conversation tone and outcome signals, predict "conversion" or "no_conversion"
8. key_topics: List of main topics or themes discussed (e.g., "integration capabilities", "pricing plans", "onboarding support")

TRANSCRIPT:
{transcript}
"""


def extract_crm_fields(state: dict[str, Any]) -> dict[str, Any]:
    """Extract CRM-ready structured data from the transcript.

    Reads state["raw_transcript"], produces state["crm_record"] as a dict.
    """
    transcript = state.get("raw_transcript", "")
    if not transcript:
        logger.warning("extract_crm_fields: no transcript available")
        state["crm_record"] = None
        return state

    # Truncate very long transcripts
    max_chars = 6000
    truncated = transcript[:max_chars]
    if len(transcript) > max_chars:
        truncated += "\n\n[... transcript truncated for extraction ...]"

    try:
        llm = ChatGroq(
            api_key=settings.GROQ_API_KEY,
            model=settings.GROQ_MODEL,
            temperature=0.1,
        )
        structured_llm = llm.with_structured_output(CRMRecord)

        result: CRMRecord = structured_llm.invoke(
            EXTRACTION_PROMPT.format(transcript=truncated)
        )

        state["crm_record"] = result.model_dump()
        logger.info("Extracted CRM record: outcome=%s", result.predicted_outcome)

    except Exception:
        logger.error("Failed to extract CRM fields via Groq", exc_info=True)
        state["crm_record"] = CRMRecord(
            customer_name_mentioned=None,
            company_mentioned=None,
            pricing_discussed=False,
            objections_raised=[],
            next_steps_confirmed=False,
            follow_up_date_mentioned=None,
            predicted_outcome="no_conversion",
            key_topics=[],
        ).model_dump()

    return state
