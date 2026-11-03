"""Tests for the LLM-classified (smart) fake user response."""

from functools import partial
from types import SimpleNamespace
from typing import Literal
from unittest.mock import MagicMock

import pytest

import benchmarks.utils.fake_user_response as fake_user_response_module
from benchmarks.utils.fake_user_response import (
    _parse_yes_no,
    fake_user_response,
    run_conversation_with_smart_fake_user_response,
    smart_fake_user_response,
)
from openhands.sdk.event import MessageEvent
from openhands.sdk.llm import Message, TextContent


def _message_event(
    text: str, source: Literal["agent", "user"] = "agent"
) -> MessageEvent:
    role = "assistant" if source == "agent" else "user"
    return MessageEvent(
        source=source,
        llm_message=Message(role=role, content=[TextContent(text=text)]),
    )


class _FakeLLM:
    """Minimal stand-in for openhands.sdk LLM used as the classifier."""

    def __init__(
        self,
        answer: str = "YES",
        error: Exception | None = None,
        stream: bool = True,
    ):
        self.answer = answer
        self.error = error
        self.stream = stream
        self.calls: list[list[Message]] = []
        self.completion_streams: list[bool] = []

    def model_copy(self, *, update: dict | None = None) -> "_FakeLLM":
        copied = _FakeLLM(self.answer, self.error, self.stream)
        copied.calls = self.calls
        copied.completion_streams = self.completion_streams
        for key, value in (update or {}).items():
            setattr(copied, key, value)
        return copied

    def completion(self, messages: list[Message]) -> SimpleNamespace:
        self.completion_streams.append(self.stream)
        self.calls.append(messages)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(
            message=Message(role="assistant", content=[TextContent(text=self.answer)])
        )


def _make_conversation(
    events: list[MessageEvent], llm: _FakeLLM | None = None
) -> MagicMock:
    conversation = MagicMock()
    conversation.state.events = events
    conversation.agent.llm = llm
    return conversation


def _classifier_prompt(llm: _FakeLLM) -> str:
    """Return the user prompt sent to the classifier by the single recorded call."""
    assert len(llm.calls) == 1
    messages = llm.calls[0]
    assert [message.role for message in messages] == ["system", "user"]
    user_content = messages[1].content
    assert len(user_content) == 1
    part = user_content[0]
    assert isinstance(part, TextContent)
    return part.text


def test_asks_to_call_finish_tool_when_classifier_says_yes():
    llm = _FakeLLM("YES")
    conversation = _make_conversation(
        [_message_event("I fixed the bug and all tests pass. The task is complete.")],
        llm,
    )

    response = smart_fake_user_response(conversation)

    assert "directly invoke the finish tool" in response
    assert "Please continue working on the task" not in response
    prompt = _classifier_prompt(llm)
    assert "I fixed the bug and all tests pass." in prompt


def test_falls_back_to_regular_message_when_classifier_says_no():
    llm = _FakeLLM("NO")
    conversation = _make_conversation(
        [_message_event("I am still investigating the failing test.")], llm
    )

    assert smart_fake_user_response(conversation) == fake_user_response(conversation)
    _classifier_prompt(llm)


def test_falls_back_when_classifier_call_fails():
    llm = _FakeLLM(error=RuntimeError("provider unavailable"))
    conversation = _make_conversation(
        [_message_event("I have completed the task.")], llm
    )

    assert smart_fake_user_response(conversation) == fake_user_response(conversation)


def test_falls_back_when_classifier_answer_is_unparseable():
    llm = _FakeLLM("I am not sure what you mean.")
    conversation = _make_conversation(
        [_message_event("I have completed the task.")], llm
    )

    assert smart_fake_user_response(conversation) == fake_user_response(conversation)


def test_falls_back_without_agent_message():
    llm = _FakeLLM("YES")
    conversation = _make_conversation([_message_event("Continue", source="user")], llm)

    assert smart_fake_user_response(conversation) == fake_user_response(conversation)
    assert llm.calls == []


def test_classifies_only_the_latest_agent_message():
    llm = _FakeLLM("NO")
    conversation = _make_conversation(
        [
            _message_event("First update: still working."),
            _message_event("Please continue.", source="user"),
            _message_event("Latest update: tests are still running."),
        ],
        llm,
    )

    smart_fake_user_response(conversation)

    prompt = _classifier_prompt(llm)
    assert "Latest update: tests are still running." in prompt
    assert "First update: still working." not in prompt


def test_uses_llm_from_state_agent_when_conversation_agent_is_absent():
    llm = _FakeLLM("YES")
    conversation = _make_conversation([_message_event("The job is done.")])
    conversation.agent = None
    conversation.state.agent.llm = llm

    response = smart_fake_user_response(conversation)

    assert "directly invoke the finish tool" in response
    _classifier_prompt(llm)


def test_finish_directive_includes_encapsulation_instruction():
    llm = _FakeLLM("YES")
    conversation = _make_conversation([_message_event("The task is finished.")], llm)

    response = smart_fake_user_response(conversation, encapsulate_solution=True)

    assert "<solution>" in response


@pytest.mark.parametrize(
    "answer,expected",
    [
        ("YES", True),
        ("yes.", True),
        ("Yes, the task is done.", True),
        ("NO", False),
        ("No, still working.", False),
        ("The job is not finished.", None),
        ("yes or no?", None),
        ("maybe", None),
        ("", None),
    ],
)
def test_parse_yes_no(answer: str, expected: bool | None):
    assert _parse_yes_no(answer) is expected


def test_smart_runner_delegates_to_regular_runner(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}

    def fake_runner(conversation, fake_user_response_fn, max_fake_responses, timeout):
        captured["conversation"] = conversation
        captured["fake_user_response_fn"] = fake_user_response_fn
        captured["max_fake_responses"] = max_fake_responses
        captured["timeout"] = timeout

    monkeypatch.setattr(
        fake_user_response_module,
        "run_conversation_with_fake_user_response",
        fake_runner,
    )
    conversation = _make_conversation([_message_event("Done.")])

    run_conversation_with_smart_fake_user_response(
        conversation, max_fake_responses=3, timeout=7
    )

    assert captured["conversation"] is conversation
    assert captured["max_fake_responses"] == 3
    assert captured["timeout"] == 7
    response_fn = captured["fake_user_response_fn"]
    assert isinstance(response_fn, partial)
    assert response_fn.func is smart_fake_user_response
    assert response_fn.keywords == {"encapsulate_solution": False}


def test_classifier_call_is_not_streaming():
    llm = _FakeLLM("YES", stream=True)
    conversation = _make_conversation([_message_event("done")], llm)

    response = smart_fake_user_response(conversation)

    assert response  # the classification still ran through the copy
    assert llm.completion_streams == [False]
