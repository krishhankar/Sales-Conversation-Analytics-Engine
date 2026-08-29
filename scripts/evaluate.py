#!/usr/bin/env python3
"""Evaluation script — measures pipeline accuracy against labeled dataset outcomes.

Runs the scoring pipeline on a sample of the DeepMostInnovations/saas-sales-conversations
dataset and compares the pipeline's predicted_outcome (from the CRM extraction node)
against the actual outcome label in the dataset.

Reports: accuracy, precision, recall, F1 score.

Usage:
    python scripts/evaluate.py [--sample-size 50] [--delay 3] [--output evaluation_results.json]

Requires:
    - GROQ_API_KEY in .env or environment
    - PostgreSQL NOT required (this runs the LLM nodes directly, skips DB storage)
"""

import argparse
import json
import logging
import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset
from langchain_groq import ChatGroq

from app.config import settings
from app.schemas import CRMRecord, RubricScores

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Prompts (same as the pipeline nodes) ──

SCORING_PROMPT = """You are an expert sales call evaluator for enterprise B2B SaaS sales.

Analyze the following sales conversation transcript and evaluate it against each rubric criterion.
Be precise and evidence-based — only mark a criterion as true if there is clear evidence in the transcript.

RUBRIC CRITERIA:
1. pricing_discussed: Did the sales rep discuss pricing, cost, budget, or payment terms?
2. objection_handled: Did the sales rep acknowledge and address at least one customer objection or concern?
3. next_steps_confirmed: Were clear, concrete next steps agreed upon?
4. discovery_questions_asked: Did the sales rep ask qualifying or discovery questions?
5. product_demo_offered: Was a product demonstration, trial, or walkthrough offered?
6. urgency_created: Did the rep create a sense of urgency or scarcity?
7. competitor_mentioned: Was a competitor product mentioned by either party?
8. overall_score: Rate overall sales effectiveness from 0.0 to 1.0.

TRANSCRIPT:
{transcript}
"""

EXTRACTION_PROMPT = """You are a CRM data extraction assistant for B2B SaaS sales teams.

Extract structured CRM fields from the following sales conversation transcript.

FIELDS TO EXTRACT:
1. customer_name_mentioned: The customer's name (null if not mentioned)
2. company_mentioned: The customer's company name (null if not mentioned)
3. pricing_discussed: Whether pricing was discussed (true/false)
4. objections_raised: List of specific objections (empty list if none)
5. next_steps_confirmed: Whether next steps were agreed upon (true/false)
6. follow_up_date_mentioned: Any follow-up date mentioned (null if none)
7. predicted_outcome: Predict "conversion" or "no_conversion"
8. key_topics: List of main topics discussed

TRANSCRIPT:
{transcript}
"""


def evaluate_single(
    llm: ChatGroq, transcript: str, max_retries: int = 3, base_delay: float = 2.0
) -> dict | None:
    """Run scoring + extraction on a single transcript with retry logic."""
    max_chars = 6000
    truncated = transcript[:max_chars]

    for attempt in range(max_retries):
        try:
            # Score call
            scoring_llm = llm.with_structured_output(RubricScores)
            rubric: RubricScores = scoring_llm.invoke(
                SCORING_PROMPT.format(transcript=truncated)
            )

            # Extract CRM fields
            extraction_llm = llm.with_structured_output(CRMRecord)
            crm: CRMRecord = extraction_llm.invoke(
                EXTRACTION_PROMPT.format(transcript=truncated)
            )

            return {
                "rubric_scores": rubric.model_dump(),
                "crm_record": crm.model_dump(),
                "predicted_outcome": crm.predicted_outcome,
            }

        except Exception as exc:
            error_msg = str(exc).lower()
            if "429" in error_msg or "rate" in error_msg:
                delay = base_delay * (2 ** attempt)
                logger.warning(
                    "Rate limited (attempt %d/%d), backoff %.1fs",
                    attempt + 1, max_retries, delay,
                )
                time.sleep(delay)
            else:
                logger.error("Evaluation error: %s", exc)
                return None

    logger.error("Max retries exceeded for evaluation")
    return None


def compute_metrics(actuals: list[int], predictions: list[int]) -> dict:
    """Compute accuracy, precision, recall, F1 for binary classification."""
    assert len(actuals) == len(predictions)
    n = len(actuals)

    if n == 0:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "n": 0}

    tp = sum(1 for a, p in zip(actuals, predictions) if a == 1 and p == 1)
    tn = sum(1 for a, p in zip(actuals, predictions) if a == 0 and p == 0)
    fp = sum(1 for a, p in zip(actuals, predictions) if a == 0 and p == 1)
    fn = sum(1 for a, p in zip(actuals, predictions) if a == 1 and p == 0)

    accuracy = (tp + tn) / n if n > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "n": n,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate pipeline against labeled data")
    parser.add_argument("--sample-size", type=int, default=50, help="Number of samples to evaluate")
    parser.add_argument("--delay", type=float, default=3.0, help="Delay between API calls (seconds)")
    parser.add_argument("--output", type=str, default="evaluation_results.json", help="Output file")
    args = parser.parse_args()

    if not settings.GROQ_API_KEY:
        logger.error("GROQ_API_KEY not set. Add it to .env or environment.")
        sys.exit(1)

    # Initialize LLM
    llm = ChatGroq(
        api_key=settings.GROQ_API_KEY,
        model=settings.GROQ_MODEL,
        temperature=0.1,
    )

    # Load dataset
    logger.info("Loading dataset: DeepMostInnovations/saas-sales-conversations")
    ds = load_dataset("DeepMostInnovations/saas-sales-conversations", split="train")

    # Take a balanced sample
    sample_size = min(args.sample_size, len(ds))
    sample = ds.shuffle(seed=42).select(range(sample_size))

    logger.info("Evaluating %d conversations...", sample_size)

    actuals = []
    predictions = []
    detailed_results = []

    for i, row in enumerate(sample):
        transcript = row.get("full_text", "")
        actual_outcome = int(row.get("outcome", 0))

        if not transcript or len(transcript) < 50:
            logger.warning("Skipping row %d: transcript too short", i)
            continue

        logger.info(
            "Evaluating %d/%d (%.0f%%) — actual_outcome=%d",
            i + 1, sample_size, (i + 1) / sample_size * 100, actual_outcome,
        )

        result = evaluate_single(llm, transcript)

        if result:
            predicted = result["predicted_outcome"]
            predicted_int = 1 if predicted == "conversion" else 0

            actuals.append(actual_outcome)
            predictions.append(predicted_int)

            match = "✓" if predicted_int == actual_outcome else "✗"
            logger.info(
                "  %s predicted=%s, actual=%d", match, predicted, actual_outcome
            )

            detailed_results.append({
                "index": i,
                "actual_outcome": actual_outcome,
                "predicted_outcome": predicted,
                "predicted_int": predicted_int,
                "match": predicted_int == actual_outcome,
                "rubric_scores": result["rubric_scores"],
            })
        else:
            logger.warning("  Skipped (pipeline error)")

        # Respect rate limits
        if i < sample_size - 1:
            time.sleep(args.delay)

    # Compute metrics
    metrics = compute_metrics(actuals, predictions)

    # Summary
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Samples evaluated: {metrics['n']}")
    print(f"  Accuracy:          {metrics['accuracy']:.1%}")
    print(f"  Precision:         {metrics['precision']:.1%}")
    print(f"  Recall:            {metrics['recall']:.1%}")
    print(f"  F1 Score:          {metrics['f1']:.1%}")
    print(f"  TP={metrics.get('tp', 0)} TN={metrics.get('tn', 0)} FP={metrics.get('fp', 0)} FN={metrics.get('fn', 0)}")
    print("=" * 60)

    # Save results
    output = {
        "metrics": metrics,
        "model": settings.GROQ_MODEL,
        "sample_size": sample_size,
        "evaluated": len(actuals),
        "detailed_results": detailed_results,
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    logger.info("Results saved to %s", args.output)


if __name__ == "__main__":
    main()
