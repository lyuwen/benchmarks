# benchmarks/scaleswe_interactive/transcript.py
"""Build the richer speaker-tagged interactive side file."""
from __future__ import annotations

from benchmarks.scaleswe_interactive.driver import SessionResult


def build_interactive_transcript(instance_id: str, mode: str, user_tools: str,
                                 coding_model: str, user_model: str,
                                 session: SessionResult) -> dict:
    return {
        "instance_id": instance_id,
        "mode": mode,
        "user_tools": user_tools,
        "coding_model": coding_model,
        "user_model": user_model,
        "turns": session.dialogue,
        "termination": {
            "reason": session.termination_reason,
            "user_turns": session.user_turns,
        },
    }
