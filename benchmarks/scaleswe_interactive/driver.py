# benchmarks/scaleswe_interactive/driver.py
"""Driver loop alternating coding-agent runs and user-agent turns."""
from __future__ import annotations

import inspect

from pydantic import BaseModel, Field

from openhands.sdk.conversation.state import ConversationExecutionStatus
from openhands.sdk.event import MessageEvent

from benchmarks.scaleswe_interactive.user_agent import UserAgent
from benchmarks.utils.fake_user_response import _sync_events


def _event_key(event):
    """Stable identity for dedup; _do_full_sync can rebuild event objects."""
    eid = getattr(event, "id", None)
    return eid if eid is not None else id(event)


class SessionResult(BaseModel):
    dialogue: list[dict] = Field(default_factory=list)
    termination_reason: str = ""
    user_turns: int = 0


def _run_conversation(conversation, timeout):
    run = conversation.run
    if timeout is not None:
        try:
            sig = inspect.signature(run)
            if "timeout" in sig.parameters:
                run(timeout=timeout)
                return
        except (TypeError, ValueError):
            pass
    run()


def _latest_agent_text(conversation, seen_ids: set) -> str | None:
    """Return concatenated text of new agent MessageEvents since last check."""
    texts = []
    for event in conversation.state.events:
        key = _event_key(event)
        if key in seen_ids:
            continue
        if isinstance(event, MessageEvent) and event.source == "agent":
            for c in event.llm_message.content or []:
                t = getattr(c, "text", None)
                if t:
                    texts.append(t)
        seen_ids.add(key)
    return "\n".join(texts).strip() if texts else None


def run_interactive_session(conversation, user_agent: UserAgent,
                            initial_instruction: str, max_user_turns: int,
                            timeout: int | None = None) -> SessionResult:
    dialogue: list[dict] = []
    seen_ids: set = set()
    # Mark pre-existing events as seen (e.g., the injected instruction message).
    for event in conversation.state.events:
        seen_ids.add(_event_key(event))

    conversation.send_message(initial_instruction)
    user_turns = 0

    while True:
        _run_conversation(conversation, timeout)
        status = conversation.state.execution_status

        if status == ConversationExecutionStatus.ERROR:
            return SessionResult(dialogue=dialogue,
                                 termination_reason="agent_error",
                                 user_turns=user_turns)
        if status == ConversationExecutionStatus.STUCK:
            return SessionResult(dialogue=dialogue,
                                 termination_reason="agent_stuck",
                                 user_turns=user_turns)

        _sync_events(conversation)
        agent_text = _latest_agent_text(conversation, seen_ids)
        if agent_text:
            dialogue.append({"speaker": "coding", "text": agent_text})

        turn = user_agent.take_turn(dialogue=dialogue, user_turns=user_turns)

        if turn.error is not None:
            return SessionResult(dialogue=dialogue,
                                 termination_reason="user_error",
                                 user_turns=user_turns)
        if turn.finished:
            dialogue.append({"speaker": "user",
                             "text": f"[finish] {turn.finish_reason or ''}".strip()})
            return SessionResult(dialogue=dialogue,
                                 termination_reason="user_finish",
                                 user_turns=user_turns)

        user_turns += 1
        message = turn.message or "Please continue."
        dialogue.append({"speaker": "user", "text": message,
                         "injected_files": turn.injected_files,
                         "readonly_tool_calls": turn.readonly_tool_calls})

        if user_turns >= max_user_turns:
            return SessionResult(dialogue=dialogue,
                                 termination_reason="turn_cap",
                                 user_turns=user_turns)

        conversation.send_message(message)
