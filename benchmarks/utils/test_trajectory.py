"""Tests for the model-view trajectory export."""

from collections.abc import Sequence

import pytest

from benchmarks.utils.trajectory import (
    COMPACTION,
    TRAJECTORY,
    build_trajectories,
    pruning_metadata,
    resolve_pruning_config,
)
from openhands.sdk import Event
from openhands.sdk.context.tool_result_pruner import (
    ENV_ENABLED,
    ENV_HEAD_CHARS,
    ENV_TAIL_CHARS,
    ENV_THRESHOLD_CHARS,
    PRUNE_MARKER,
    ToolResultPruneConfig,
)
from openhands.sdk.event import (
    Condensation,
    CondensationRequest,
    MessageEvent,
    SystemPromptEvent,
)
from openhands.sdk.event.llm_convertible import ActionEvent, ObservationEvent
from openhands.sdk.llm import Message, MessageToolCall, TextContent
from openhands.sdk.mcp.definition import MCPToolAction, MCPToolObservation


MARKER_LEN = len(PRUNE_MARKER)


@pytest.fixture(autouse=True)
def _clear_pruner_env(monkeypatch: pytest.MonkeyPatch):
    for name in (ENV_ENABLED, ENV_THRESHOLD_CHARS, ENV_HEAD_CHARS, ENV_TAIL_CHARS):
        monkeypatch.delenv(name, raising=False)


def _message_event(content: str) -> MessageEvent:
    return MessageEvent(
        llm_message=Message(role="user", content=[TextContent(text=content)]),
        source="user",
    )


def _system_event(pruning: dict | None = None) -> SystemPromptEvent:
    return SystemPromptEvent(
        source="agent",
        system_prompt=TextContent(text="system prompt"),
        tools=[],
        tool_result_pruning=pruning,
    )


def _action_event(tool_call_id: str = "call_1") -> ActionEvent:
    return ActionEvent(
        thought=[TextContent(text="run the tool")],
        action=MCPToolAction(data={}),
        tool_name="test_tool",
        tool_call_id=tool_call_id,
        tool_call=MessageToolCall(
            id=tool_call_id, name="test_tool", arguments="{}", origin="completion"
        ),
        llm_response_id="resp_1",
        source="agent",
    )


def _observation_event(
    tool_call_id: str = "call_1", text: str = "ok"
) -> ObservationEvent:
    return ObservationEvent(
        observation=MCPToolObservation.from_text(text=text, tool_name="test_tool"),
        tool_name="test_tool",
        tool_call_id=tool_call_id,
        action_id="action_1",
        source="environment",
    )


def _condensation(
    forgotten_ids: list[str],
    request_messages: list[Message] | None = None,
    response_message: Message | None = None,
) -> Condensation:
    return Condensation(
        forgotten_event_ids=forgotten_ids,
        summary="a summary",
        summary_offset=0,
        llm_response_id="condensation_response_1",
        request_messages=request_messages,
        response_message=response_message,
    )


def _types(trajectories: Sequence[dict]) -> list[str]:
    return [entry["type"] for entry in trajectories]


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------


def test_single_trajectory_without_compaction():
    events = [_message_event("hello"), _message_event("world")]

    trajectories = build_trajectories(events)

    assert _types(trajectories) == [TRAJECTORY]
    assert trajectories[0]["messages"] == [
        {"content": "hello", "role": "user"},
        {"content": "world", "role": "user"},
    ]


def test_compaction_splits_trajectories_and_records_input_output():
    first, second = _message_event("first"), _message_event("second")
    request = CondensationRequest()
    prompt = Message(role="user", content=[TextContent(text="summarize this")])
    summary_message = Message(role="assistant", content=[TextContent(text="a summary")])
    condensation = _condensation(
        forgotten_ids=[first.id],
        request_messages=[prompt],
        response_message=summary_message,
    )
    last = _message_event("after")

    trajectories = build_trajectories([first, second, request, condensation, last])

    assert _types(trajectories) == [TRAJECTORY, COMPACTION, TRAJECTORY]

    # Pre-compaction segment: the model view just before the condensation.
    assert [m["content"] for m in trajectories[0]["messages"]] == ["first", "second"]

    compaction = trajectories[1]
    assert compaction["trigger"] == "request"
    assert [event["kind"] for event in compaction["trigger_events"]] == [
        "CondensationRequest"
    ]
    assert compaction["request"] == [{"content": "summarize this", "role": "user"}]
    assert compaction["response"] == {"content": "a summary", "role": "assistant"}
    assert compaction["result"]["kind"] == "Condensation"
    assert compaction["result"]["summary"] == "a summary"
    assert compaction["result"]["forgotten_event_ids"] == [first.id]
    assert "request_messages" not in compaction["result"]
    assert "response_message" not in compaction["result"]

    # Post-compaction segment: summary inserted, forgotten event dropped.
    post_messages = trajectories[2]["messages"]
    assert [m["content"] for m in post_messages] == ["a summary", "second", "after"]


def test_pressure_compaction_has_no_trigger_events():
    first, second = _message_event("first"), _message_event("second")
    condensation = _condensation(forgotten_ids=[first.id])

    trajectories = build_trajectories([first, second, condensation])

    assert _types(trajectories) == [TRAJECTORY, COMPACTION, TRAJECTORY]
    compaction = trajectories[1]
    assert compaction["trigger"] == "pressure"
    assert compaction["trigger_events"] == []
    assert compaction["request"] is None
    assert compaction["response"] is None


def test_multiple_compactions_produce_alternating_entries():
    events: list[Event] = [
        _message_event("first"),
        _message_event("second"),
        _condensation(forgotten_ids=[]),
        _message_event("third"),
        _condensation(forgotten_ids=[]),
        _message_event("fourth"),
    ]

    trajectories = build_trajectories(events)

    assert _types(trajectories) == [
        TRAJECTORY,
        COMPACTION,
        TRAJECTORY,
        COMPACTION,
        TRAJECTORY,
    ]


def test_trailing_request_without_compaction_is_recorded():
    events = [_message_event("hello"), CondensationRequest()]

    trajectories = build_trajectories(events)

    assert _types(trajectories) == [TRAJECTORY, COMPACTION]
    unmatched = trajectories[1]
    assert unmatched["trigger"] == "request"
    assert [event["kind"] for event in unmatched["trigger_events"]] == [
        "CondensationRequest"
    ]
    assert unmatched["request"] is None
    assert unmatched["response"] is None
    assert unmatched["result"] is None


# ---------------------------------------------------------------------------
# Pruning
# ---------------------------------------------------------------------------


SMALL_PRUNING: dict = {
    "enabled": True,
    "threshold_chars": MARKER_LEN + 2,
    "head_chars": 1,
    "tail_chars": 1,
}
SMALL_CONFIG = ToolResultPruneConfig(
    threshold_chars=MARKER_LEN + 2, head_chars=1, tail_chars=1
)


def _enable_small_pruner(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_ENABLED, "1")
    monkeypatch.setenv(ENV_THRESHOLD_CHARS, str(MARKER_LEN + 2))
    monkeypatch.setenv(ENV_HEAD_CHARS, "1")
    monkeypatch.setenv(ENV_TAIL_CHARS, "1")


def _tool_messages(entry: dict) -> list[dict]:
    return [message for message in entry["messages"] if message["role"] == "tool"]


def test_recorded_pruning_replaces_content_and_keeps_original():
    original_output = "x" * 100
    events = [
        _system_event(SMALL_PRUNING),
        _action_event(),
        _observation_event(text=original_output),
    ]

    config = resolve_pruning_config(events)
    assert config == SMALL_CONFIG
    trajectories = build_trajectories(events, config)

    tool_messages = _tool_messages(trajectories[0])
    assert len(tool_messages) == 1
    assert tool_messages[0]["content"] == f"x{PRUNE_MARKER}x"
    assert tool_messages[0]["original_content"] == original_output

    assert pruning_metadata(config) == {
        "enabled": True,
        "threshold_chars": MARKER_LEN + 2,
        "head_chars": 1,
        "tail_chars": 1,
        "marker": PRUNE_MARKER,
    }


def test_recorded_policy_wins_over_conflicting_environment(
    monkeypatch: pytest.MonkeyPatch,
):
    # The host env says "huge threshold", but the run itself recorded the small
    # policy; the export must reproduce what the model saw.
    monkeypatch.setenv(ENV_ENABLED, "1")
    monkeypatch.setenv(ENV_THRESHOLD_CHARS, "100000")
    monkeypatch.setenv(ENV_HEAD_CHARS, "40")
    monkeypatch.setenv(ENV_TAIL_CHARS, "20")
    original_output = "x" * 100
    events = [
        _system_event(SMALL_PRUNING),
        _action_event(),
        _observation_event(text=original_output),
    ]

    config = resolve_pruning_config(events)
    trajectories = build_trajectories(events, config)

    tool_message = _tool_messages(trajectories[0])[0]
    assert tool_message["content"] == f"x{PRUNE_MARKER}x"
    assert tool_message["original_content"] == original_output


def test_recorded_disabled_policy_ignores_host_environment(
    monkeypatch: pytest.MonkeyPatch,
):
    _enable_small_pruner(monkeypatch)
    original_output = "x" * 100
    events = [
        _system_event({"enabled": False}),
        _action_event(),
        _observation_event(text=original_output),
    ]

    config = resolve_pruning_config(events)

    assert config is None
    assert pruning_metadata(config) is None
    tool_message = _tool_messages(build_trajectories(events, config)[0])[0]
    assert tool_message["content"] == original_output
    assert "original_content" not in tool_message


def test_pruning_disabled_leaves_messages_intact():
    original_output = "x" * 100
    events = [_action_event(), _observation_event(text=original_output)]

    config = resolve_pruning_config(events)
    trajectories = build_trajectories(events, config)

    tool_message = _tool_messages(trajectories[0])[0]
    assert tool_message["content"] == original_output
    assert "original_content" not in tool_message
    assert pruning_metadata(config) is None


def test_legacy_logs_without_recorded_policy_use_environment(
    monkeypatch: pytest.MonkeyPatch,
):
    _enable_small_pruner(monkeypatch)
    original_output = "x" * 100
    events = [
        SystemPromptEvent(
            source="agent",
            system_prompt=TextContent(text="old system prompt"),
            tools=[],
        ),
        _action_event(),
        _observation_event(text=original_output),
    ]

    config = resolve_pruning_config(events)
    trajectories = build_trajectories(events, config)

    assert config == SMALL_CONFIG
    tool_message = _tool_messages(trajectories[0])[0]
    assert tool_message["content"] == f"x{PRUNE_MARKER}x"
    assert tool_message["original_content"] == original_output


def test_pruning_applies_to_every_segment():
    first = _message_event("before")
    pre_events: list[Event] = [
        _system_event(SMALL_PRUNING),
        first,
        _action_event("call_1"),
        _observation_event("call_1", "y" * 100),
    ]
    condensation = _condensation(forgotten_ids=[first.id])
    post_events: list[Event] = [
        _action_event("call_2"),
        _observation_event("call_2", "z" * 100),
    ]
    events = pre_events + [condensation] + post_events

    config = resolve_pruning_config(events)
    trajectories = build_trajectories(events, config)

    assert _types(trajectories) == [TRAJECTORY, COMPACTION, TRAJECTORY]
    for segment_index in (0, 2):
        tool_messages = _tool_messages(trajectories[segment_index])
        assert tool_messages
        for tool_message in tool_messages:
            assert tool_message["content"] != tool_message["original_content"]
            assert tool_message["content"].endswith(
                PRUNE_MARKER + tool_message["original_content"][-1]
            )
