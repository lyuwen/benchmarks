"""Unit tests for the wire-format conversions.

The important invariant here is replay equality: an assistant message we record
must canonicalize to the same identity as the same turn a client replays on the
next request, in every protocol. If that breaks, the trajectory manager forks
every turn instead of assembling a trajectory.
"""

from __future__ import annotations

import json

import pytest

from rollout_manager.convert import (
    anthropic_count_tokens,
    anthropic_to_chat_request,
    arguments_to_dict,
    canonical_reply_message,
    canonicalize_message,
    canonicalize_messages,
    canonicalize_tools,
    chat_to_anthropic_response,
    chat_to_responses_response,
    flatten_content,
    message_identity,
    parse_chat_reply,
    responses_to_chat_request,
    wire_history_message,
)


def user(text: str) -> dict:
    return {"role": "user", "content": text}


def assistant(text: str = "", *, reasoning: str | None = None, tool_calls=None) -> dict:
    message: dict = {"role": "assistant", "content": text}
    if reasoning:
        message["reasoning_content"] = reasoning
    if tool_calls:
        message["tool_calls"] = tool_calls
    return message


def chat_reply(
    *,
    content: str | None = "answer",
    reasoning: str | None = None,
    tool_calls: list[dict] | None = None,
    finish: str = "stop",
    usage: dict | None = None,
    model: str = "fake-model",
) -> dict:
    message: dict = {"role": "assistant", "content": content}
    if reasoning:
        message["reasoning_content"] = reasoning
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": usage or {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def tool_call(name: str = "bash", arguments: dict | None = None, call_id: str = "call_1") -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments or {"cmd": "ls"})},
    }


# ---------------------------------------------------------------------------
# canonical primitives
# ---------------------------------------------------------------------------


def test_flatten_content():
    assert flatten_content(None) == ""
    assert flatten_content("plain") == "plain"
    assert flatten_content([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]) == "a\nb"
    assert flatten_content([{"type": "input_image", "image_url": "x"}]) == "[image omitted]"
    assert flatten_content({"type": "image", "source": {"data": "x"}}) == "[image omitted]"
    assert flatten_content({"type": "input_file", "file_id": "f"}) == "[file omitted]"
    assert flatten_content({"type": "tool_result", "content": [{"type": "text", "text": "42"}]}) == "42"
    assert flatten_content({"text": "mapping"}) == "mapping"
    assert flatten_content(7) == "7"


def test_arguments_to_dict():
    assert arguments_to_dict(None) == {}
    assert arguments_to_dict({"a": 1}) == {"a": 1}
    assert arguments_to_dict('{"a": 1}') == {"a": 1}
    assert arguments_to_dict("not json") == {"_raw_arguments": "not json"}
    assert arguments_to_dict("[1, 2]") == {"_raw_arguments": "[1, 2]"}


def test_canonicalize_message_preserves_wire_fields():
    message = {
        "role": "assistant",
        "content": None,
        "reasoning_content": "think",
        "tool_calls": [
            {
                "id": "call_abc",
                "type": "function",
                "function": {"name": "bash", "arguments": '{"b": 2, "a": 1}'},
            }
        ],
    }
    canonical = canonicalize_message(message)
    assert canonical == {
        "role": "assistant",
        "content": None,
        "reasoning_content": "think",
        "tool_calls": [
            {
                "id": "call_abc",
                "type": "function",
                "function": {"name": "bash", "arguments": '{"b": 2, "a": 1}'},
            }
        ],
    }

    tool = canonicalize_message({"role": "tool", "content": "out", "tool_call_id": "call_abc"})
    assert tool == {"role": "tool", "tool_call_id": "call_abc", "content": "out"}
    # Native Chat-Completions roles and content shapes are preserved.
    assert canonicalize_message({"role": "developer", "content": "sys"}) == {"role": "developer", "content": "sys"}
    assert canonicalize_message({"role": "user", "content": [{"type": "text", "text": "hi"}]}) == {
        "role": "user",
        "content": [{"type": "text", "text": "hi"}],
    }
    assert canonicalize_message({"role": "nonsense", "content": "x"}) is None
    assert canonicalize_message(["not a message"]) is None


def test_message_identity_ignores_reasoning_and_argument_encoding():
    recorded = canonicalize_message(assistant("ok", reasoning="why", tool_calls=[tool_call()]))
    replayed = canonicalize_message(
        {
            "role": "assistant",
            "content": "ok",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "bash", "arguments": '{"cmd":  "ls"}'},
                }
            ],
        }
    )
    assert message_identity(recorded) == message_identity(replayed)


def test_message_identity_distinguishes_tool_call_ids():
    first = canonicalize_message(assistant("", tool_calls=[tool_call(call_id="call_a")]))
    second = canonicalize_message(assistant("", tool_calls=[tool_call(call_id="call_b")]))
    assert message_identity(first) != message_identity(second)
    assert message_identity({"role": "tool", "content": "42", "tool_call_id": "call_a"}) != message_identity(
        {"role": "tool", "content": "42", "tool_call_id": "call_b"}
    )


def test_canonicalize_messages_skips_unknown():
    assert canonicalize_messages([user("hi"), {"role": "tool_use"}, assistant("a")]) == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "a"},
    ]


def test_canonicalize_tools_accepts_three_wire_shapes():
    parameters = {"type": "object", "properties": {"cmd": {"type": "string"}}}
    chat_style = [{"type": "function", "function": {"name": "bash", "description": "run", "parameters": parameters}}]
    responses_style = [{"type": "function", "name": "bash", "description": "run", "parameters": parameters}]
    anthropic_style = [{"name": "bash", "description": "run", "input_schema": parameters}]
    expected = chat_style
    assert canonicalize_tools(chat_style) == expected
    assert canonicalize_tools(responses_style) == expected
    assert canonicalize_tools(anthropic_style) == expected
    assert canonicalize_tools([{"type": "web_search"}]) is None
    assert canonicalize_tools(None) is None
    # strict is part of the OpenAI tool schema and must survive normalization.
    strict_tool = [
        {
            "type": "function",
            "function": {"name": "bash", "description": "run", "parameters": parameters, "strict": True},
        }
    ]
    assert canonicalize_tools(strict_tool) == strict_tool


def test_truncated_tool_arguments_are_preserved():
    truncated = '{"cmd": "ls'
    reply = parse_chat_reply(
        chat_reply(
            content="",
            tool_calls=[
                {"id": "call_trunc", "type": "function", "function": {"name": "bash", "arguments": truncated}}
            ],
            finish="length",
        )
    )
    message = canonical_reply_message(reply)
    assert message["tool_calls"][0]["function"]["arguments"] == truncated
    # Matching still works when the same truncated payload is replayed.
    replayed = canonicalize_message(
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "call_trunc", "type": "function", "function": {"name": "bash", "arguments": truncated}}
            ],
        }
    )
    assert message_identity(message) == message_identity(replayed)
    other = canonicalize_message(
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "call_other", "type": "function", "function": {"name": "bash", "arguments": truncated}}
            ],
        }
    )
    assert message_identity(message) != message_identity(other)


# ---------------------------------------------------------------------------
# Anthropic Messages
# ---------------------------------------------------------------------------


def test_anthropic_request_to_chat():
    body = {
        "model": "claude-like",
        "system": "you are helpful",
        "max_tokens": 128,
        "temperature": 0.7,
        "top_p": 0.9,
        "stop_sequences": ["STOP"],
        "tools": [{"name": "bash", "description": "run", "input_schema": {"type": "object", "properties": {}}}],
        "tool_choice": {"type": "tool", "name": "bash"},
        "messages": [
            {"role": "user", "content": "run it"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "hmm"},
                    {"type": "text", "text": "on it"},
                    {"type": "tool_use", "id": "toolu_1", "name": "bash", "input": {"cmd": "ls"}},
                ],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "42"}],
            },
        ],
    }
    request = anthropic_to_chat_request(body)
    assert request["messages"][0] == {"role": "system", "content": "you are helpful"}
    assert request["messages"][1] == {"role": "user", "content": "run it"}
    assert request["messages"][2] == {
        "role": "assistant",
        "content": "on it",
        "reasoning_content": "hmm",
        "tool_calls": [
            {
                "id": "toolu_1",
                "type": "function",
                "function": {"name": "bash", "arguments": json.dumps({"cmd": "ls"})},
            }
        ],
    }
    assert request["messages"][3] == {"role": "tool", "tool_call_id": "toolu_1", "content": "42"}
    assert request["max_tokens"] == 128
    assert request["temperature"] == 0.7
    assert request["top_p"] == 0.9
    assert request["stop"] == ["STOP"]
    assert request["tool_choice"] == {"type": "function", "function": {"name": "bash"}}
    assert request["tools"][0]["function"]["name"] == "bash"


def test_chat_to_anthropic_response():
    reply = parse_chat_reply(
        chat_reply(
            content="on it",
            reasoning="hmm",
            tool_calls=[tool_call()],
            finish="tool_calls",
            usage={"prompt_tokens": 10, "completion_tokens": 5, "prompt_tokens_details": {"cached_tokens": 3}},
        )
    )
    response = chat_to_anthropic_response(reply)
    assert [block["type"] for block in response["content"]] == ["thinking", "text", "tool_use"]
    assert response["content"][2]["input"] == {"cmd": "ls"}
    assert response["stop_reason"] == "tool_use"
    assert response["usage"] == {"input_tokens": 10, "output_tokens": 5, "cache_read_input_tokens": 3}

    length_reply = parse_chat_reply(chat_reply(content="cut", finish="length"))
    assert chat_to_anthropic_response(length_reply)["stop_reason"] == "max_tokens"


def test_anthropic_replay_matches_recorded_reply():
    reply = parse_chat_reply(chat_reply(content="on it", reasoning="hmm", tool_calls=[tool_call()]))
    response = chat_to_anthropic_response(reply)
    tool_use = next(block for block in response["content"] if block["type"] == "tool_use")

    # The client replays the assistant blocks plus the tool observation.
    body = {
        "messages": [
            {"role": "user", "content": "run it"},
            {"role": "assistant", "content": response["content"]},
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": tool_use["id"], "content": "42"}],
            },
        ]
    }
    canonical = canonicalize_messages(anthropic_to_chat_request(body)["messages"])
    assert message_identity(canonical[1]) == message_identity(canonical_reply_message(reply))
    assert canonical[2] == {"role": "tool", "tool_call_id": tool_use["id"], "content": "42"}


def test_anthropic_count_tokens():
    body = {
        "system": "system prompt",
        "messages": [{"role": "user", "content": "a" * 400}],
        "tools": [{"name": "bash", "input_schema": {"type": "object"}}],
    }
    assert anthropic_count_tokens(body, chars_per_token=4.0) == pytest.approx(120, abs=15)
    assert anthropic_count_tokens({}) == 0


# ---------------------------------------------------------------------------
# OpenAI Responses
# ---------------------------------------------------------------------------


def test_responses_input_to_chat_groups_function_calls():
    body = {
        "model": "gpt-like",
        "instructions": "be terse",
        "max_output_tokens": 64,
        "temperature": 0.2,
        "parallel_tool_calls": False,
        "tools": [{"type": "function", "name": "bash", "description": "run", "parameters": {}}],
        "input": [
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "call it"}]},
            {"type": "function_call", "call_id": "call_1", "name": "bash", "arguments": '{"cmd": "ls"}'},
            {"type": "function_call_output", "call_id": "call_1", "output": "42"},
            {"type": "message", "role": "user", "content": "thanks"},
        ],
    }
    request = responses_to_chat_request(body)
    assert request["messages"][0] == {"role": "system", "content": "be terse"}
    assert request["messages"][1] == {"role": "user", "content": "call it"}
    assert request["messages"][2] == {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "bash", "arguments": json.dumps({"cmd": "ls"})},
            }
        ],
    }
    assert request["messages"][3] == {"role": "tool", "tool_call_id": "call_1", "content": "42"}
    assert request["messages"][4] == {"role": "user", "content": "thanks"}
    assert request["max_tokens"] == 64
    assert request["temperature"] == 0.2
    assert request["parallel_tool_calls"] is False
    assert request["tools"][0]["function"]["name"] == "bash"


def test_responses_previous_messages_are_prepended():
    reply = parse_chat_reply(chat_reply(content="", tool_calls=[tool_call()], finish="tool_calls"))
    request = responses_to_chat_request(
        {"input": [{"type": "function_call_output", "call_id": "call_1", "output": "42"}]},
        previous_messages=[user("go"), wire_history_message(reply)],
    )
    assert [message["role"] for message in request["messages"]] == ["user", "assistant", "tool"]
    assert request["messages"][2] == {"role": "tool", "tool_call_id": "call_1", "content": "42"}


def test_chat_to_responses_response():
    reply = parse_chat_reply(
        chat_reply(
            content="done",
            reasoning="thought",
            usage={"prompt_tokens": 10, "completion_tokens": 5, "prompt_tokens_details": {"cached_tokens": 2}},
        )
    )
    response = chat_to_responses_response(reply, {"model": "gpt-like", "instructions": "be terse"})
    assert [item["type"] for item in response["output"]] == ["reasoning", "message"]
    assert response["output_text"] == "done"
    assert response["status"] == "completed"
    assert response["usage"] == {
        "input_tokens": 10,
        "output_tokens": 5,
        "total_tokens": 15,
        "input_tokens_details": {"cached_tokens": 2},
    }

    incomplete = chat_to_responses_response(
        parse_chat_reply(chat_reply(content="cut", finish="length")), {"model": "gpt-like"}
    )
    assert incomplete["status"] == "incomplete"
    assert incomplete["incomplete_details"] == {"reason": "max_output_tokens"}


def test_responses_replay_matches_recorded_reply():
    reply = parse_chat_reply(chat_reply(content="on it", reasoning="hmm", tool_calls=[tool_call()]))
    response = chat_to_responses_response(reply, {"model": "gpt-like"})
    function_call = next(item for item in response["output"] if item["type"] == "function_call")

    replay_input = [
        {"type": "message", "role": "user", "content": "run it"},
        *response["output"],
        {"type": "function_call_output", "call_id": function_call["call_id"], "output": "42"},
    ]
    request = responses_to_chat_request({"input": replay_input})
    canonical = canonicalize_messages(request["messages"])
    assert message_identity(canonical[1]) == message_identity(canonical_reply_message(reply))
    assert canonical[2] == {"role": "tool", "tool_call_id": "call_1", "content": "42"}
