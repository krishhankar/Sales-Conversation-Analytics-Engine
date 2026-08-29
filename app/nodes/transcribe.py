"""Transcribe node — uses faster-whisper (local, free) for speech-to-text.

The WhisperModel is loaded once as a singleton and reused across all requests.
This avoids repeated model loading overhead (~2-3 seconds per cold start).
"""

from __future__ import annotations

import logging
from typing import Any

from faster_whisper import WhisperModel

from app.config import settings

logger = logging.getLogger(__name__)

# ── Singleton whisper model ──
_whisper_model: WhisperModel | None = None


def _get_model() -> WhisperModel:
    """Lazy-load the Whisper model on first call."""
    global _whisper_model
    if _whisper_model is None:
        logger.info(
            "Loading Whisper model: %s (CPU, int8)", settings.WHISPER_MODEL_SIZE
        )
        _whisper_model = WhisperModel(
            settings.WHISPER_MODEL_SIZE,
            device="cpu",
            compute_type="int8",
        )
        logger.info("Whisper model loaded successfully")
    return _whisper_model


def transcribe(state: dict[str, Any]) -> dict[str, Any]:
    """Transcribe an audio file and set the transcript + language in state.

    Reads state["audio_path"], produces:
      - state["raw_transcript"]: full transcript text
      - state["language"]: detected language code (e.g. "en", "hi", "ta")
      - state["language_confidence"]: detection confidence (0-1)
    """
    audio_path = state.get("audio_path")
    if not audio_path:
        logger.warning("transcribe called without audio_path, skipping")
        return state

    model = _get_model()

    logger.info("Transcribing: %s", audio_path)
    segments, info = model.transcribe(
        audio_path,
        beam_size=5,
        vad_filter=True,  # Filter out silence for cleaner results
    )

    # Collect all segment texts
    full_text = " ".join(segment.text.strip() for segment in segments)

    state["raw_transcript"] = full_text
    state["language"] = info.language
    state["language_confidence"] = round(info.language_probability, 4)

    logger.info(
        "Transcribed %d chars, language=%s (%.2f%%)",
        len(full_text),
        info.language,
        info.language_probability * 100,
    )

    return state
