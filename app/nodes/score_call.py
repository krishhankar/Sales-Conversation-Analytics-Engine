"""Score call node — evaluates a sales transcript against a rubric.

Uses Groq free-tier LLM (llama-3.1-8b-instant) with structured output
to produce a validated RubricScores Pydantic model. This is NOT free-text
summarization — it's rubric-based evaluation with boolean + float fields.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_groq import ChatGroq

from app.config import settings
from app.schemas import RubricScores

logger = logging.getLogger(__name__)

SCORING_PROMPT = """You are an expert sales call evaluator for enterprise B2B SaaS sales.

Analyze the following sales conversation transcript and evaluate it against each rubric criterion.
Be precise and evidence-based — only mark a criterion as true if there is clear evidence in the transcript.

RUBRIC CRITERIA:
1. pricing_discussed: Did the sales rep discuss pricing, cost, budget, or payment terms?
2. objection_handled: Did the sales rep acknowledge and address at least one customer objection or concern?
3. next_steps_confirmed: Were clear, concrete next steps agreed upon (e.g., follow-up meeting, trial, proposal)?
4. discovery_questions_asked: Did the sales rep ask qualifying or discovery questions about the customer's needs, pain points, or use case?
5. product_demo_offered: Was a product demonstration, trial, or walkthrough offered or conducted?
6. urgency_created: Did the rep create a sense of urgency or scarcity (e.g., limited-time offer, deadline)?
7. competitor_mentioned: Was a competitor product, brand, or alternative solution mentioned by either party?
8. overall_score: Rate the overall sales effectiveness from 0.0 (very poor) to 1.0 (excellent) based on how well the rep followed sales best practices.

TRANSCRIPT:
{transcript}
"""


def score_call(state: dict[str, Any]) -> dict[str, Any]:
    """Score the transcript against the sales rubric using structured LLM output.

    Reads state["raw_transcript"], produces state["rubric_scores"] as a dict.
    """
    transcript = state.get("raw_transcript", "")
    if not transcript:
        logger.warning("score_call: no transcript available")
        state["rubric_scores"] = None
        return state

    # Truncate very long transcripts to stay within token limits
    max_chars = 6000
    truncated = transcript[:max_chars]
    if len(transcript) > max_chars:
        truncated += "\n\n[... transcript truncated for scoring ...]"

    try:
        llm = ChatGroq(
            api_key=settings.GROQ_API_KEY,
            model=settings.GROQ_MODEL,
            temperature=0.1,
        )
        structured_llm = llm.with_structured_output(RubricScores)

        result: RubricScores = structured_llm.invoke(
            SCORING_PROMPT.format(transcript=truncated)
        )

        state["rubric_scores"] = result.model_dump()
        logger.info("Scored call: overall=%.2f", result.overall_score)

    except Exception:
        logger.error("Failed to score call via Groq", exc_info=True)
        # Provide a default empty score rather than failing the entire pipeline
        state["rubric_scores"] = RubricScores(
            pricing_discussed=False,
            objection_handled=False,
            next_steps_confirmed=False,
            discovery_questions_asked=False,
            product_demo_offered=False,
            urgency_created=False,
            competitor_mentioned=False,
            overall_score=0.0,
        ).model_dump()

    return state
