#!/usr/bin/env python3
"""Seed the database with sample conversations from the HuggingFace dataset.

Loads a batch from DeepMostInnovations/saas-sales-conversations, runs each
through the scoring pipeline, and stores results in PostgreSQL + ChromaDB.

This pre-populates the system so POST /search and GET /analytics/summary
return meaningful results for demo purposes.

Usage:
    python scripts/seed_dataset.py [--count 20] [--delay 3]

Requires:
    - GROQ_API_KEY in .env or environment
    - PostgreSQL and Redis running (docker compose up postgres redis)
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset

from app.graph import pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def run_with_retry(transcript: str, max_retries: int = 3, base_delay: float = 2.0) -> dict | None:
    """Run the pipeline with exponential backoff for Groq rate limits."""
    for attempt in range(max_retries):
        try:
            result = pipeline.invoke({"raw_transcript": transcript})
            return result
        except Exception as exc:
            error_msg = str(exc).lower()
            if "429" in error_msg or "rate" in error_msg:
                delay = base_delay * (2 ** attempt)
                logger.warning(
                    "Rate limited (attempt %d/%d), waiting %.1fs...",
                    attempt + 1, max_retries, delay,
                )
                time.sleep(delay)
            else:
                logger.error("Pipeline error: %s", exc)
                return None
    logger.error("Max retries exceeded")
    return None


def main():
    parser = argparse.ArgumentParser(description="Seed database with sample data")
    parser.add_argument("--count", type=int, default=20, help="Number of conversations to seed")
    parser.add_argument("--delay", type=float, default=3.0, help="Delay between calls (seconds)")
    args = parser.parse_args()

    logger.info("Loading dataset: DeepMostInnovations/saas-sales-conversations")
    ds = load_dataset("DeepMostInnovations/saas-sales-conversations", split="train")

    # Take a sample
    sample = ds.select(range(min(args.count, len(ds))))
    logger.info("Processing %d conversations...", len(sample))

    success_count = 0
    for i, row in enumerate(sample):
        transcript = row.get("full_text", "")
        if not transcript or len(transcript) < 50:
            logger.warning("Skipping row %d: transcript too short", i)
            continue

        logger.info("Processing %d/%d (%.0f%%)", i + 1, len(sample), (i + 1) / len(sample) * 100)

        result = run_with_retry(transcript)
        if result:
            success_count += 1
            logger.info(
                "  ✓ call_id=%s, outcome=%s",
                result.get("call_id", "?"),
                (result.get("crm_record") or {}).get("predicted_outcome", "?"),
            )
        else:
            logger.warning("  ✗ Failed to process row %d", i)

        # Respect rate limits
        if i < len(sample) - 1:
            time.sleep(args.delay)

    logger.info("Seeding complete: %d/%d successful", success_count, len(sample))


if __name__ == "__main__":
    main()
