"""Call ingestion and retrieval endpoints.

POST /calls — ingest a call (audio file or text transcript), run the full
              LangGraph pipeline, return structured results.
GET  /calls/{call_id} — retrieve full detail for a processed call.
"""

from __future__ import annotations

import logging
import os
import tempfile

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app import cache
from app.db.crud import get_call_record
from app.db.engine import get_session
from app.graph import pipeline
from app.schemas import CallDetail, CallResponse, CRMRecord, RubricScores, TextIngestRequest

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/calls", tags=["calls"])
MAX_AUDIO_BYTES = 25 * 1024 * 1024


async def _run_pipeline(initial_state: dict) -> CallResponse:
    """Shared helper: run the LangGraph pipeline and build the response."""
    try:
        result = pipeline.invoke(initial_state)
    except Exception as exc:
        logger.error("Pipeline failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}") from exc
    finally:
        # Clean up temp audio file
        tmp_path = initial_state.get("audio_path")
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    # Build response
    rubric = result.get("rubric_scores") or {}
    crm = result.get("crm_record") or {}

    return CallResponse(
        call_id=result["call_id"],
        language=result.get("language", "unknown"),
        language_confidence=result.get("language_confidence", 0.0),
        rubric_scores=RubricScores(**rubric),
        crm_record=CRMRecord(**crm),
        transcript=result.get("raw_transcript", ""),
    )


@router.post("", response_model=CallResponse, status_code=201)
async def ingest_text(body: TextIngestRequest):
    """Ingest a sales call as a text transcript.

    Send a JSON body: {"transcript": "..."}
    """
    logger.info("Received text transcript: %d chars", len(body.transcript))
    return await _run_pipeline({"raw_transcript": body.transcript})


@router.post("/audio", response_model=CallResponse, status_code=201)
async def ingest_audio(audio: UploadFile = File(...)):
    """Ingest a sales call as an audio file.

    Send a multipart file upload with field name "audio".
    """
    if not audio.filename:
        raise HTTPException(status_code=400, detail="No audio file provided.")

    content = await audio.read(MAX_AUDIO_BYTES + 1)
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded audio file is empty.")
    if len(content) > MAX_AUDIO_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Audio file exceeds the {MAX_AUDIO_BYTES // (1024 * 1024)} MiB limit.",
        )

    extension = os.path.splitext(audio.filename)[1]
    suffix = extension if extension[1:].isalnum() else ".audio"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    logger.info("Received audio file: %s (%d bytes)", audio.filename, len(content))
    return await _run_pipeline({"audio_path": tmp_path})


@router.get("/{call_id}", response_model=CallDetail)
async def get_call(call_id: str, session: AsyncSession = Depends(get_session)):
    """Retrieve full transcript + scoring breakdown for a specific call.

    Checks Redis cache first, then falls back to PostgreSQL.
    """
    # Try cache first
    cached = await cache.get_cached_call(call_id)
    if cached:
        return CallDetail(
            call_id=cached["call_id"],
            language=cached.get("language", "unknown"),
            language_confidence=cached.get("language_confidence", 0.0),
            transcript=cached.get("transcript", ""),
            rubric_scores=RubricScores(**(cached.get("rubric_scores") or {})),
            crm_record=CRMRecord(**(cached.get("crm_record") or {})),
            created_at=cached.get("created_at", ""),
            source_type=cached.get("source_type", "text"),
        )

    # Fall back to database
    record = await get_call_record(session, call_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Call {call_id} not found")

    return CallDetail(
        call_id=record.id,
        language=record.language,
        language_confidence=record.language_confidence,
        transcript=record.transcript,
        rubric_scores=RubricScores(**(record.rubric_scores or {})),
        crm_record=CRMRecord(**(record.crm_record or {})),
        created_at=record.created_at.isoformat() if record.created_at else "",
        source_type=record.source_type or "text",
    )
