"""Ingest node — receives raw input and determines audio vs text path."""

from __future__ import annotations

import uuid
from typing import Any


def ingest(state: dict[str, Any]) -> dict[str, Any]:
    """Set up initial state with a unique call_id.

    Expects either:
      - state["audio_path"] to be set (audio file was saved)
      - state["raw_transcript"] to be set (text was provided)
    Sets input_type accordingly.
    """
    # Generate a unique call ID
    state["call_id"] = state.get("call_id") or str(uuid.uuid4())

    if state.get("audio_path"):
        state["input_type"] = "audio"
    elif state.get("raw_transcript"):
        state["input_type"] = "text"
    else:
        raise ValueError("Either audio_path or raw_transcript must be provided")

    return state
