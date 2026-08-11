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
