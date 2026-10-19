# benchmarks/scaleswe_interactive/tests/test_transcript.py
from benchmarks.scaleswe_interactive.driver import SessionResult
from benchmarks.scaleswe_interactive.transcript import build_interactive_transcript


def test_transcript_shape():
    session = SessionResult(
        dialogue=[
            {"speaker": "coding", "text": "plan"},
            {"speaker": "user", "text": "approve", "injected_files": [],
             "readonly_tool_calls": []},
        ],
        termination_reason="user_finish", user_turns=1)
    out = build_interactive_transcript(
        instance_id="inst-1", mode="plan", user_tools="none",
        coding_model="model-a", user_model="model-b", session=session)
    assert out["instance_id"] == "inst-1"
    assert out["mode"] == "plan"
    assert out["coding_model"] == "model-a"
    assert out["user_model"] == "model-b"
    assert out["termination"] == {"reason": "user_finish", "user_turns": 1}
    assert out["turns"] == session.dialogue
