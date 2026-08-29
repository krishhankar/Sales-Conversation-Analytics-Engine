"""Detect language node — identifies the language of a text transcript.

For audio inputs, language is already detected by Whisper (this node passes through).
For text inputs, uses the lightweight `langdetect` library (local, no API call).
"""

from __future__ import annotations

import logging
from typing import Any

from langdetect import detect, DetectorFactory

logger = logging.getLogger(__name__)

# Make langdetect deterministic
DetectorFactory.seed = 0


def detect_language(state: dict[str, Any]) -> dict[str, Any]:
    """Detect or confirm the language of the transcript.

    If language was already set by Whisper (audio path), this is a no-op.
    For text inputs, runs langdetect on the raw transcript.
    """
    # If Whisper already detected language, pass through
    if state.get("language") and state.get("input_type") == "audio":
        logger.debug("Language already detected by Whisper: %s", state["language"])
        return state

    transcript = state.get("raw_transcript", "")
    if not transcript:
        state["language"] = "unknown"
        state["language_confidence"] = 0.0
        return state

    try:
        detected = detect(transcript)
        state["language"] = detected
        # langdetect doesn't provide confidence directly, set a reasonable default
        state["language_confidence"] = 0.85
        logger.info("Detected language for text input: %s", detected)
    except Exception:
        logger.warning("Language detection failed, defaulting to 'en'", exc_info=True)
        state["language"] = "en"
        state["language_confidence"] = 0.0

    return state
