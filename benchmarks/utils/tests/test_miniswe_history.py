"""Tests for the mini transcript -> history.json normalization."""

import json

from benchmarks.utils.miniswe.history import (
    build_history_dump,
    build_trajectories,
    history_filename,
    normalize_messages,
    publish_canonical_histories,
    redact_secrets,
)
from openhands.sdk import Message
from openhands.sdk.llm.message import MessageToolCall


def _openhands_dict(**kwargs) -> dict:
    """Encode a message with the OpenHands exporter used by OpenHands runs."""
    message = Message(
        function_calling_enabled=True, send_reasoning_content=True, **kwargs
    )
    return message.to_chat_dict()


def test_assistant_tool_call_matches_openhands_encoding() -> None:
    mini_messages = [
        {
            "role": "assistant",
            "content": "Let me look.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "bash", "arguments": '{"command": "ls"}'},
                }
            ],
        }
    ]

    assert normalize_messages(mini_messages)[0] == _openhands_dict(
        role="assistant",
        content="Let me look.",
        tool_calls=[
            MessageToolCall(
                id="call_1",
                name="bash",
                arguments='{"command": "ls"}',
                origin="completion",
            )
        ],
    )


def test_tool_result_matches_openhands_encoding() -> None:
    mini_messages = [{"role": "tool", "content": "file.py", "tool_call_id": "call_1"}]

    normalized = normalize_messages(mini_messages)[0]

    assert normalized["name"] == "bash"
    assert normalized == _openhands_dict(
        role="tool",
        content="file.py",
        tool_call_id="call_1",
        name="bash",
    )


def test_reasoning_content_and_exit_records() -> None:
    normalized = normalize_messages(
        [
            {"role": "assistant", "content": "thinking", "reasoning_content": "why"},
            {"role": "exit", "content": "LimitsExceeded"},
        ]
    )

    assert len(normalized) == 1
    assert normalized[0]["reasoning_content"] == "why"


def test_parallel_tool_calls_are_preserved() -> None:
    mini_messages = [
        {
            "role": "assistant",
            "content": "checking two files",
            "tool_calls": [
                {
                    "id": "call_a",
                    "type": "function",
                    "function": {"name": "bash", "arguments": '{"command": "ls a"}'},
                },
                {
                    "id": "call_b",
                    "type": "function",
                    "function": {"name": "bash", "arguments": '{"command": "ls b"}'},
                },
            ],
        },
        {"role": "tool", "content": "a", "tool_call_id": "call_a"},
        {"role": "tool", "content": "b", "tool_call_id": "call_b"},
    ]

    normalized = normalize_messages(mini_messages)

    assert [call["id"] for call in normalized[0]["tool_calls"]] == [
        "call_a",
        "call_b",
    ]
    assert [message["tool_call_id"] for message in normalized[1:]] == [
        "call_a",
        "call_b",
    ]


def test_trajectories_have_one_segment_with_model_view() -> None:
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
        {"role": "exit", "content": "Submitted"},
    ]

    trajectories = build_trajectories(messages)

    assert len(trajectories) == 1
    assert trajectories[0]["type"] == "trajectory"
    assert [message["role"] for message in trajectories[0]["messages"]] == [
        "system",
        "user",
    ]


def test_history_dump_omits_evaluation_without_judge() -> None:
    dump = build_history_dump(
        instance_id="i-1",
        model="openai/test",
        temperature=0.0,
        top_p=1.0,
        messages=[{"role": "user", "content": "task"}],
        tools=[{"type": "function"}],
        git_patch="diff",
    )

    assert "evaluation" not in dump
    assert dump["test_result"] == {"git_patch": "diff"}
    assert "pruning" not in dump


def test_history_dump_includes_null_evaluation_for_unavailable_judge() -> None:
    dump = build_history_dump(
        instance_id="i-1",
        model="openai/test",
        temperature=0.0,
        top_p=1.0,
        messages=[],
        tools=[],
        git_patch="",
        evaluation=None,
        include_evaluation=True,
        miniswe={"exit_status": "Submitted"},
    )

    assert dump["evaluation"] is None
    assert dump["miniswe"] == {"exit_status": "Submitted"}


def test_redact_secrets_removes_credentials() -> None:
    redacted = redact_secrets(
        {
            "api_key": "sk-secret",
            "headers": {"Authorization": "Bearer token"},
            "max_completion_tokens": 128,
        }
    )

    assert redacted["api_key"] == "<redacted>"
    assert redacted["headers"]["Authorization"] == "<redacted>"
    assert redacted["max_completion_tokens"] == 128


def test_publish_canonical_histories_uses_aggregated_attempt(tmp_path) -> None:
    output_dir = tmp_path
    instance_id = "repo__issue-1"
    for attempt, marker in ((1, "first"), (2, "second")):
        path = output_dir / history_filename(instance_id, attempt)
        path.write_text(json.dumps({"instance_id": instance_id, "mark": marker}))
    output_path = output_dir / "output.jsonl"
    output_path.write_text(
        json.dumps({"instance_id": instance_id, "attempt": 2}) + "\n"
    )

    publish_canonical_histories(str(output_dir), str(output_path))

    canonical = json.loads((output_dir / history_filename(instance_id)).read_text())
    assert canonical["mark"] == "second"
