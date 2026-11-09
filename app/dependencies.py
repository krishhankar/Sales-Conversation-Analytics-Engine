"""Shared lazy singleton dependencies used across the application."""

from __future__ import annotations

import logging

from faster_whisper import WhisperModel
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import settings

logger = logging.getLogger(__name__)

_whisper_model: WhisperModel | None = None
_embedding_model: HuggingFaceEmbeddings | None = None


def get_whisper_model() -> WhisperModel:
    """Return the singleton local Whisper transcription model."""
    global _whisper_model

    if _whisper_model is None:
        logger.info("Loading Whisper model: %s (CPU, int8)", settings.WHISPER_MODEL_SIZE)
        _whisper_model = WhisperModel(
            settings.WHISPER_MODEL_SIZE,
            device="cpu",
            compute_type="int8",
        )
        logger.info("Whisper model loaded successfully")

    return _whisper_model


def get_embedding_model() -> HuggingFaceEmbeddings:
    """Return the singleton local sentence-transformer embedding model."""
    global _embedding_model

    if _embedding_model is None:
        logger.info("Loading embedding model: %s", settings.EMBEDDING_MODEL)
        _embedding_model = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        logger.info("Embedding model loaded successfully")

    return _embedding_model
