"""LangGraph pipeline — the core processing graph for sales conversations.

Defines a StateGraph with the flow:
  ingest → [conditional: audio?] → transcribe → detect_language → score_call
  → extract_crm_fields → embed_and_store → END

For text inputs, the transcribe step is skipped.
"""

from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from app.nodes.detect_language import detect_language
from app.nodes.embed import embed_and_store
from app.nodes.extract_crm_fields import extract_crm_fields
from app.nodes.ingest import ingest
from app.nodes.score_call import score_call
from app.nodes.transcribe import transcribe


class PipelineState(TypedDict, total=False):
    """State schema flowing through the LangGraph pipeline."""

    # Input fields
    input_type: str  # "audio" | "text"
    audio_path: str | None
    raw_transcript: str

    # Processing fields
    call_id: str
    language: str
    language_confidence: float

    # Output fields (set by LLM nodes)
    rubric_scores: dict[str, Any] | None
    crm_record: dict[str, Any] | None
    embedded: bool


def _route_after_ingest(state: PipelineState) -> str:
    """Conditional edge: route to transcribe (audio) or detect_language (text)."""
    if state.get("input_type") == "audio":
        return "transcribe"
    return "detect_language"


def build_pipeline() -> StateGraph:
    """Construct and compile the sales conversation analysis pipeline."""

    graph = StateGraph(PipelineState)

    # ── Add nodes ──
    graph.add_node("ingest", ingest)
    graph.add_node("transcribe", transcribe)
    graph.add_node("detect_language", detect_language)
    graph.add_node("score_call", score_call)
    graph.add_node("extract_crm_fields", extract_crm_fields)
    graph.add_node("embed_and_store", embed_and_store)

    # ── Define edges ──
    graph.set_entry_point("ingest")

    # Conditional routing after ingest
    graph.add_conditional_edges(
        "ingest",
        _route_after_ingest,
        {
            "transcribe": "transcribe",
            "detect_language": "detect_language",
        },
    )

    # Linear flow after transcribe
    graph.add_edge("transcribe", "detect_language")
    graph.add_edge("detect_language", "score_call")
    graph.add_edge("score_call", "extract_crm_fields")
    graph.add_edge("extract_crm_fields", "embed_and_store")
    graph.add_edge("embed_and_store", END)

    return graph.compile()


# Pre-compiled pipeline (singleton)
pipeline = build_pipeline()
