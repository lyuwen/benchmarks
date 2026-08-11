# benchmarks/scaleswe_interactive/tests/test_driver.py
from types import SimpleNamespace

from openhands.sdk.conversation.state import ConversationExecutionStatus
from openhands.sdk.event import MessageEvent
from openhands.sdk.llm import Message, TextContent

from benchmarks.scaleswe_interactive.driver import (
    run_interactive_session,
    SessionResult,
)
from benchmarks.scaleswe_interactive.user_agent import UserTurn


def _agent_msg(text):
    return MessageEvent(source="agent",
                        llm_message=Message(role="assistant",
                                            content=[TextContent(text=text)]))


class _FakeConversation:
    """Appends an agent message each run(); records sent user messages."""
    def __init__(self):
        self.state = SimpleNamespace(
            events=[], execution_status=ConversationExecutionStatus.FINISHED)
        self.sent = []
        self._replies = ["Here is my plan: 1. do X", "Done, tests pass."]

    def send_message(self, message):
        self.sent.append(message)

    def run(self, timeout=None):
        text = self._replies.pop(0) if self._replies else "still working"
        self.state.events.append(_agent_msg(text))


class _ScriptedUserAgent:
    def __init__(self, turns):
        self._turns = list(turns)
        self.seen = []

    def take_turn(self, dialogue, user_turns):
        self.seen.append(list(dialogue))
        return self._turns.pop(0)


def test_user_finish_terminates():
    conv = _FakeConversation()
    ua = _ScriptedUserAgent([
        UserTurn(message="I approve the plan"),
        UserTurn(finished=True, finish_reason="solved"),
    ])
    res = run_interactive_session(conv, ua, "PROBLEM", max_user_turns=20)
    assert isinstance(res, SessionResult)
    assert res.termination_reason == "user_finish"
    assert conv.sent[0] == "PROBLEM"
    assert "I approve the plan" in conv.sent
    # dialogue contains both coding and user turns
    speakers = {d["speaker"] for d in res.dialogue}
    assert speakers == {"coding", "user"}


def test_turn_cap_terminates():
    conv = _FakeConversation()
    conv._replies = ["a", "b", "c", "d", "e"]
    ua = _ScriptedUserAgent([UserTurn(message="keep going") for _ in range(10)])
    res = run_interactive_session(conv, ua, "P", max_user_turns=2)
    assert res.termination_reason == "turn_cap"
    assert res.user_turns == 2


def test_agent_error_terminates():
    conv = _FakeConversation()

    def boom(timeout=None):
        conv.state.execution_status = ConversationExecutionStatus.ERROR
    conv.run = boom
    ua = _ScriptedUserAgent([UserTurn(message="x")])
    res = run_interactive_session(conv, ua, "P", max_user_turns=20)
    assert res.termination_reason == "agent_error"


def test_user_error_terminates():
    conv = _FakeConversation()
    ua = _ScriptedUserAgent([UserTurn(error="boom")])
    res = run_interactive_session(conv, ua, "P", max_user_turns=20)
    assert res.termination_reason == "user_error"


class _SyncEventList(list):
    """Event list whose agent message only lands when _do_full_sync() runs.

    Mimics RemoteConversation's WebSocket race: run() does not append the
    agent MessageEvent to the local cache; only a forced sync pulls it in.
    """
    def __init__(self):
        super().__init__()
        self.sync_calls = 0
        self._pending = []

    def stage(self, event):
        self._pending.append(event)

    def _do_full_sync(self):
        self.sync_calls += 1
        while self._pending:
            self.append(self._pending.pop(0))


class _SyncFakeConversation:
    """Agent replies are only visible after _do_full_sync is invoked."""
    def __init__(self):
        events = _SyncEventList()
        self.state = SimpleNamespace(
            events=events, execution_status=ConversationExecutionStatus.FINISHED)
        self.sent = []
        self._replies = ["Plan ready.", "All done."]

    def send_message(self, message):
        self.sent.append(message)

    def run(self, timeout=None):
        text = self._replies.pop(0) if self._replies else "still working"
        self.state.events.stage(_agent_msg(text))  # not visible until sync


def test_driver_syncs_events_before_reading_agent_text():
    conv = _SyncFakeConversation()
    captured = []

    class _Recorder:
        def take_turn(self, dialogue, user_turns):
            captured.append(list(dialogue))
            # First turn approves, second finishes.
            if user_turns == 0:
                return UserTurn(message="approve")
            return UserTurn(finished=True, finish_reason="solved")

    res = run_interactive_session(conv, _Recorder(), "P", max_user_turns=20)
    assert res.termination_reason == "user_finish"
    assert conv.state.events.sync_calls >= 1  # sync was forced
    # The staged-only agent text still reached the dialogue via sync.
    assert any(d["speaker"] == "coding" and "Plan ready." in d["text"]
               for d in res.dialogue)


def test_latest_agent_text_dedups_by_stable_id():
    from benchmarks.scaleswe_interactive.driver import _latest_agent_text
    conv = _FakeConversation()
    ev = _agent_msg("hello")
    conv.state.events.append(ev)
    seen = set()
    assert _latest_agent_text(conv, seen) == "hello"
    # A rebuilt object with the SAME .id must be treated as already seen.
    ev2 = _agent_msg("hello")
    object.__setattr__(ev2, "id", ev.id)
    conv.state.events.append(ev2)
    assert _latest_agent_text(conv, seen) is None
