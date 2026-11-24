"""End-to-end tests for the single-session rollout-capture service."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from pathlib import Path

import aiohttp
import pytest
from aiohttp import web

from rollout_manager.server import (
    Config,
    _redacted_headers,
    build_app,
    parse_request_params,
)

CHAT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "run a command",
            "parameters": {"type": "object", "properties": {}},
        },
    }
]
MAIN_HEADERS = {"x-claude-code-request-class": "main", "x-claude-code-prompt-id": "prompt-1"}
COMPACTION_HEADERS = {"x-claude-code-request-class": "compaction", "x-claude-code-compaction": "auto"}
POST_COMPACTION_HEADERS = {"x-claude-code-request-class": "main", "x-claude-code-context-compacted": "auto"}
AUXILIARY_HEADERS = {"x-claude-code-request-class": "auxiliary"}

STRICT_MESSAGE_FIELDS = {"role", "content", "name", "tool_calls", "tool_call_id"}


# ---------------------------------------------------------------------------
# fake upstream
# ---------------------------------------------------------------------------


class FakeUpstream:
    """Scripted OpenAI-compatible chat-completions backend."""

    def __init__(self) -> None:
        self.requests: list[dict] = []
        self.headers: list[dict[str, str]] = []
        self.models_headers: list[dict[str, str]] = []
        self.replies: list[dict] = []
        self.fail_status: int | None = None

    def reply(self, *, content: str = "", tool_calls=None, finish: str = "stop", reasoning: str | None = None) -> None:
        self.replies.append({"content": content, "tool_calls": tool_calls, "finish": finish, "reasoning": reasoning})

    async def handle_chat(self, request: web.Request) -> web.Response:
        body = await request.json()
        self.requests.append(body)
        self.headers.append(dict(request.headers))
        if self.fail_status is not None:
            return web.json_response({"error": {"message": "upstream exploded"}}, status=self.fail_status)
        if not self.replies:
            return web.json_response({"error": {"message": "no scripted reply"}}, status=500)
        spec = self.replies.pop(0)
        message: dict = {"role": "assistant", "content": spec["content"] or None}
        if spec["reasoning"]:
            message["reasoning_content"] = spec["reasoning"]
        if spec["tool_calls"]:
            message["tool_calls"] = [
                {
                    "id": call.get("id", f"call_{index}"),
                    "type": "function",
                    "function": {
                        "name": call["name"],
                        "arguments": (
                            call["arguments"]
                            if isinstance(call.get("arguments"), str)
                            else json.dumps(call.get("arguments") or {})
                        ),
                    },
                }
                for index, call in enumerate(spec["tool_calls"])
            ]
        payload = {
            "id": f"chatcmpl-{len(self.requests)}",
            "object": "chat.completion",
            "created": 0,
            "model": body.get("model", "fake-model"),
            "choices": [{"index": 0, "message": message, "finish_reason": spec["finish"]}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        return web.json_response(payload)

    async def handle_models(self, request: web.Request) -> web.Response:
        self.models_headers.append(dict(request.headers))
        return web.json_response({"object": "list", "data": [{"id": "fake-model", "object": "model"}]})

    def app(self) -> web.Application:
        app = web.Application()
        app.router.add_post("/v1/chat/completions", self.handle_chat)
        app.router.add_get("/v1/models", self.handle_models)
        return app


@contextlib.asynccontextmanager
async def _serve(app: web.Application):
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        await runner.cleanup()


def _config(upstream_url: str, output: Path, **overrides) -> Config:
    defaults = {"output": str(output), "session_id": "test-session"}
    defaults.update(overrides)
    return Config(upstream=f"{upstream_url}/v1", **defaults)


def _parse_sse(raw: str) -> list[tuple[str, object]]:
    events: list[tuple[str, object]] = []
    for block in raw.split("\n\n"):
        event = "message"
        data: str | None = None
        for line in block.splitlines():
            if line.startswith("event: "):
                event = line[len("event: ") :]
            elif line.startswith("data: "):
                data = line[len("data: ") :]
        if data is not None:
            events.append((event, data if data == "[DONE]" else json.loads(data)))
    return events


def _load_session(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def _roles(messages: list[dict]) -> list[str]:
    return [message["role"] for message in messages]


def _header(headers: dict[str, str], name: str) -> str | None:
    return next((value for key, value in headers.items() if key.lower() == name.lower()), None)


def _assert_strict_messages(trajectory: dict) -> None:
    """Exported messages must be valid OpenAI Chat-Completions messages, and
    every tool result must match a tool call id in the same trajectory."""
    messages = trajectory["messages"]
    assert len(trajectory["loss_mask"]) == len(messages)
    assert len(trajectory["reasoning_content"]) == len(messages)
    call_ids: set[str] = set()
    for message in messages:
        assert set(message) <= STRICT_MESSAGE_FIELDS, message
        assert "loss_mask" not in message and "reasoning_content" not in message
        assert message["role"] in {"system", "developer", "user", "assistant", "tool"}
        assert message["content"] is None or isinstance(message["content"], (str, list))
        for call in message.get("tool_calls") or []:
            assert isinstance(call["id"], str) and call["id"]
            assert call["type"] == "function"
            assert isinstance(call["function"]["arguments"], str)
            call_ids.add(call["id"])
        if message["role"] == "tool":
            assert isinstance(message["tool_call_id"], str)
            assert message["tool_call_id"] in call_ids, f"orphan tool result: {message['tool_call_id']}"


# ---------------------------------------------------------------------------
# Chat Completions
# ---------------------------------------------------------------------------


def test_chat_turn_is_captured_with_strict_messages(tmp_path):
    upstream = FakeUpstream()
    upstream.reply(content="hi A")
    output = tmp_path / "session.json"

    async def scenario():
        async with _serve(upstream.app()) as upstream_url:
            async with _serve(build_app(_config(upstream_url, output, model="forced-model"))) as proxy_url:
                async with aiohttp.ClientSession() as client:
                    response = await client.post(
                        f"{proxy_url}/v1/chat/completions",
                        json={
                            "model": "agent-model",
                            "messages": [{"role": "user", "content": "hello"}],
                            "temperature": 0.2,
                            "top_p": 0.9,
                            "tools": CHAT_TOOLS,
                            "tool_choice": "auto",
                        },
                    )
                    data = await response.json()
                    assert data["choices"][0]["message"]["content"] == "hi A"
                    assert upstream.requests[0]["stream"] is False
                    assert upstream.requests[0]["model"] == "forced-model"
                    finish = await (await client.post(f"{proxy_url}/rollout/finish", json={})).json()

        assert finish["finished"] is True
        assert finish["trajectories"] == 1
        document = _load_session(output)
        assert document["version"] == 2
        assert document["session_id"] == "test-session"
        assert document["stats"] == {
            "requests": 1,
            "turns": 1,
            "trajectories": 1,
            "compaction_requests": 0,
            "excluded_turns": 0,
            "request_classes": {},
        }
        assert len(document["request_log"]) == 1
        assert document["request_log"][0]["turn_index"] == 1
        assert document["request_log"][0]["message_count"] == 1

        trajectory = document["trajectories"][0]
        assert trajectory["kind"] == "conversation"
        assert trajectory["contains_compaction"] is False
        assert trajectory["exclude_from_training"] is False
        assert trajectory["messages"] == [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi A"},
        ]
        assert trajectory["loss_mask"] == [0, 1]
        assert trajectory["reasoning_content"] == [None, None]
        _assert_strict_messages(trajectory)

        turn = trajectory["turns"][0]
        assert turn["message_index"] == 1
        assert turn["trained"] is True
        assert turn["finish_reason"] == "stop"
        # Effective request + original response are retained for replay.
        assert turn["request"]["model"] == "forced-model"
        assert turn["request"]["temperature"] == 0.2
        assert turn["request"]["top_p"] == 0.9
        assert turn["request"]["tool_choice"] == "auto"
        assert turn["request"]["stream"] is False
        assert turn["request"]["message_count"] == 1
        # The effective prompt is retained verbatim.
        assert turn["request"]["messages"] == upstream.requests[0]["messages"]
        assert turn["response"]["content"] == "hi A"
        assert document["request_log"][0]["trajectory_id"] == trajectory["trajectory_id"]

    asyncio.run(scenario())


def test_native_roles_and_prompt_shapes_are_retained(tmp_path):
    upstream = FakeUpstream()
    upstream.reply(content="ok")
    output = tmp_path / "session.json"
    native_messages = [
        {"role": "developer", "content": "be terse"},
        {"role": "user", "content": [{"type": "text", "text": "hi"}]},
    ]

    async def scenario():
        async with _serve(upstream.app()) as upstream_url:
            async with _serve(build_app(_config(upstream_url, output))) as proxy_url:
                async with aiohttp.ClientSession() as client:
                    await client.post(
                        f"{proxy_url}/v1/chat/completions",
                        json={"model": "m", "messages": native_messages},
                    )
                    await client.post(f"{proxy_url}/rollout/finish")

        document = _load_session(output)
        trajectory = document["trajectories"][0]
        # Training messages keep the native role/content shape...
        assert trajectory["messages"][0] == {"role": "developer", "content": "be terse"}
        assert trajectory["messages"][1] == {"role": "user", "content": [{"type": "text", "text": "hi"}]}
        # ...and the effective request retains the exact upstream prompt.
        turn = trajectory["turns"][0]
        assert turn["request"]["messages"] == upstream.requests[0]["messages"] == native_messages
        _assert_strict_messages(trajectory)

    asyncio.run(scenario())


def test_chat_streaming_is_rendered_and_captured(tmp_path):
    upstream = FakeUpstream()
    upstream.reply(content="hello stream")
    output = tmp_path / "session.json"

    async def scenario():
        async with _serve(upstream.app()) as upstream_url:
            async with _serve(build_app(_config(upstream_url, output))) as proxy_url:
                async with aiohttp.ClientSession() as client:
                    response = await client.post(
                        f"{proxy_url}/v1/chat/completions",
                        json={"model": "m", "messages": [{"role": "user", "content": "hi"}], "stream": True},
                    )
                    events = _parse_sse(await response.text())
                    assert events[-1] == ("message", "[DONE]")
                    chunks = [payload for name, payload in events if payload != "[DONE]"]
                    content = "".join(chunk["choices"][0]["delta"].get("content", "") for chunk in chunks)
                    assert content == "hello stream"
                    assert upstream.requests[0]["stream"] is False
                    await client.post(f"{proxy_url}/rollout/finish")

        document = _load_session(output)
        trajectory = document["trajectories"][0]
        assert trajectory["messages"][-1] == {"role": "assistant", "content": "hello stream"}
        assert trajectory["loss_mask"] == [0, 1]

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# Anthropic Messages
# ---------------------------------------------------------------------------


def test_anthropic_tool_turns_keep_ids_and_argument_strings(tmp_path):
    upstream = FakeUpstream()
    upstream.reply(content="", tool_calls=[{"name": "bash", "arguments": {"cmd": "ls"}}], finish="tool_calls")
    upstream.reply(content="done")
    output = tmp_path / "session.json"

    async def scenario():
        async with _serve(upstream.app()) as upstream_url:
            async with _serve(build_app(_config(upstream_url, output))) as proxy_url:
                async with aiohttp.ClientSession() as client:
                    first = await client.post(
                        f"{proxy_url}/v1/messages",
                        json={
                            "model": "claude-like",
                            "max_tokens": 128,
                            "temperature": 0.3,
                            "messages": [{"role": "user", "content": "run it"}],
                        },
                    )
                    first_response = await first.json()
                    tool_use = first_response["content"][-1]
                    assert tool_use["type"] == "tool_use"

                    second = await client.post(
                        f"{proxy_url}/v1/messages",
                        json={
                            "model": "claude-like",
                            "max_tokens": 128,
                            "messages": [
                                {"role": "user", "content": "run it"},
                                {"role": "assistant", "content": first_response["content"]},
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "tool_result", "tool_use_id": tool_use["id"], "content": "42"}
                                    ],
                                },
                            ],
                        },
                    )
                    second_response = await second.json()
                    assert second_response["content"] == [{"type": "text", "text": "done"}]
                    assert upstream.requests[1]["messages"][-1] == {
                        "role": "tool",
                        "tool_call_id": tool_use["id"],
                        "content": "42",
                    }
                    await client.post(f"{proxy_url}/rollout/finish")

        document = _load_session(output)
        trajectory = document["trajectories"][0]
        assert _roles(trajectory["messages"]) == ["user", "assistant", "tool", "assistant"]
        assert trajectory["messages"][1] == {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": tool_use["id"],
                    "type": "function",
                    "function": {"name": "bash", "arguments": json.dumps({"cmd": "ls"})},
                }
            ],
        }
        assert trajectory["messages"][2] == {
            "role": "tool",
            "tool_call_id": tool_use["id"],
            "content": "42",
        }
        assert trajectory["loss_mask"] == [0, 1, 0, 1]
        _assert_strict_messages(trajectory)
        assert trajectory["turns"][0]["request"]["max_tokens"] == 128
        assert trajectory["turns"][0]["request"]["temperature"] == 0.3
        assert trajectory["turns"][0]["request"]["message_count"] == 1
        assert trajectory["turns"][0]["response"]["tool_calls"][0]["id"] == tool_use["id"]

    asyncio.run(scenario())


def test_anthropic_count_tokens_is_approximate_but_nonzero(tmp_path):
    upstream = FakeUpstream()
    output = tmp_path / "session.json"

    async def scenario():
        async with _serve(upstream.app()) as upstream_url:
            async with _serve(build_app(_config(upstream_url, output))) as proxy_url:
                async with aiohttp.ClientSession() as client:
                    response = await client.post(
                        f"{proxy_url}/v1/messages/count_tokens",
                        json={"messages": [{"role": "user", "content": "x" * 400}]},
                    )
                    assert response.status == 200
                    assert (await response.json())["input_tokens"] > 0
                    assert upstream.requests == []

    asyncio.run(scenario())


def test_anthropic_streaming_events(tmp_path):
    upstream = FakeUpstream()
    upstream.reply(
        content="streamed answer", tool_calls=[{"name": "bash", "arguments": {"cmd": "ls"}}], finish="tool_calls"
    )
    output = tmp_path / "session.json"

    async def scenario():
        async with _serve(upstream.app()) as upstream_url:
            async with _serve(build_app(_config(upstream_url, output))) as proxy_url:
                async with aiohttp.ClientSession() as client:
                    response = await client.post(
                        f"{proxy_url}/v1/messages",
                        json={
                            "model": "claude-like",
                            "max_tokens": 128,
                            "stream": True,
                            "messages": [{"role": "user", "content": "run it"}],
                        },
                    )
                    events = _parse_sse(await response.text())
                    names = [name for name, _ in events]
                    assert names[0] == "message_start"
                    assert names[-1] == "message_stop"
                    assert upstream.requests[0]["stream"] is False
                    await client.post(f"{proxy_url}/rollout/finish")

        document = _load_session(output)
        trajectory = document["trajectories"][0]
        assert trajectory["messages"][-1]["tool_calls"][0]["id"] == "call_0"
        assert trajectory["loss_mask"] == [0, 1]

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# OpenAI Responses
# ---------------------------------------------------------------------------


def test_responses_chaining_rebuilds_history_and_captures(tmp_path):
    upstream = FakeUpstream()
    upstream.reply(content="", tool_calls=[{"name": "bash", "arguments": {"cmd": "ls"}}], finish="tool_calls")
    upstream.reply(content="done")
    output = tmp_path / "session.json"

    async def scenario():
        async with _serve(upstream.app()) as upstream_url:
            async with _serve(build_app(_config(upstream_url, output))) as proxy_url:
                async with aiohttp.ClientSession() as client:
                    first = await client.post(
                        f"{proxy_url}/v1/responses",
                        json={
                            "model": "gpt-like",
                            "instructions": "be terse",
                            "input": "call it",
                            "reasoning": {"effort": "high"},
                        },
                    )
                    first_response = await first.json()
                    function_call = next(item for item in first_response["output"] if item["type"] == "function_call")
                    assert upstream.requests[0]["reasoning_effort"] == "high"

                    second = await client.post(
                        f"{proxy_url}/v1/responses",
                        json={
                            "model": "gpt-like",
                            "instructions": "be terse",
                            "previous_response_id": first_response["id"],
                            "input": [
                                {"type": "function_call_output", "call_id": function_call["call_id"], "output": "42"}
                            ],
                        },
                    )
                    second_response = await second.json()
                    assert second_response["output_text"] == "done"
                    assert second_response["previous_response_id"] == first_response["id"]
                    rebuilt = upstream.requests[1]["messages"]
                    assert _roles(rebuilt) == ["system", "user", "assistant", "tool"]
                    assert rebuilt[2]["tool_calls"][0]["id"] == function_call["call_id"]
                    assert rebuilt[3]["tool_call_id"] == function_call["call_id"]
                    await client.post(f"{proxy_url}/rollout/finish")

        document = _load_session(output)
        trajectory = document["trajectories"][0]
        assert _roles(trajectory["messages"]) == ["system", "user", "assistant", "tool", "assistant"]
        assert [turn["message_index"] for turn in trajectory["turns"]] == [2, 4]
        assert trajectory["turns"][0]["protocol"] == "responses"
        assert trajectory["messages"][2]["tool_calls"][0]["id"] == function_call["call_id"]
        assert trajectory["messages"][3]["tool_call_id"] == function_call["call_id"]
        assert trajectory["loss_mask"] == [0, 0, 1, 0, 1]
        _assert_strict_messages(trajectory)

    asyncio.run(scenario())


def test_responses_streaming_events(tmp_path):
    upstream = FakeUpstream()
    upstream.reply(content="streamed answer")
    output = tmp_path / "session.json"

    async def scenario():
        async with _serve(upstream.app()) as upstream_url:
            async with _serve(build_app(_config(upstream_url, output))) as proxy_url:
                async with aiohttp.ClientSession() as client:
                    response = await client.post(
                        f"{proxy_url}/v1/responses", json={"model": "gpt-like", "input": "hi", "stream": True}
                    )
                    events = _parse_sse(await response.text())
                    names = [name for name, _ in events]
                    assert "response.created" in names
                    assert "response.output_text.delta" in names
                    assert names[-1] == "response.completed"
                    await client.post(f"{proxy_url}/rollout/finish")

        document = _load_session(output)
        assert document["trajectories"][0]["messages"][-1]["content"] == "streamed answer"

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# transparent header forwarding
# ---------------------------------------------------------------------------


def test_request_headers_are_forwarded_transparently(tmp_path):
    upstream = FakeUpstream()
    upstream.reply(content="ok")
    output = tmp_path / "session.json"

    async def scenario():
        async with _serve(upstream.app()) as upstream_url:
            async with _serve(build_app(_config(upstream_url, output))) as proxy_url:
                async with aiohttp.ClientSession() as client:
                    await client.post(
                        f"{proxy_url}/v1/chat/completions",
                        json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
                        headers={
                            "Authorization": "Bearer agent-token",
                            "X-Custom-Trace": "trace-123",
                            "Cookie": "a=b",
                        },
                    )

        forwarded = upstream.headers[0]
        assert _header(forwarded, "authorization") == "Bearer agent-token"
        assert _header(forwarded, "x-custom-trace") == "trace-123"
        assert _header(forwarded, "cookie") == "a=b"
        assert _header(forwarded, "content-type") == "application/json"
        assert _header(forwarded, "host") == upstream_url.split("//")[1]

    asyncio.run(scenario())


def test_upstream_api_key_overrides_forwarded_authorization(tmp_path):
    upstream = FakeUpstream()
    upstream.reply(content="ok")
    output = tmp_path / "session.json"

    async def scenario():
        async with _serve(upstream.app()) as upstream_url:
            config = _config(upstream_url, output, upstream_api_key="configured-key")
            async with _serve(build_app(config)) as proxy_url:
                async with aiohttp.ClientSession() as client:
                    await client.post(
                        f"{proxy_url}/v1/chat/completions",
                        json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
                        headers={"Authorization": "Bearer agent-token", "X-Api-Key": "client-key"},
                    )

        forwarded = upstream.headers[0]
        assert _header(forwarded, "authorization") == "Bearer configured-key"
        assert _header(forwarded, "x-api-key") == "client-key"

    asyncio.run(scenario())


def test_backend_routing_headers_use_a_fresh_per_launch_key(tmp_path):
    upstream = FakeUpstream()
    upstream.reply(content="ok")
    output = tmp_path / "session.json"

    async def scenario():
        async with _serve(upstream.app()) as upstream_url:
            async with _serve(build_app(_config(upstream_url, output))) as proxy_url:
                async with aiohttp.ClientSession() as client:
                    await client.post(
                        f"{proxy_url}/v1/chat/completions",
                        json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
                        headers={"X-SMG-Routing-Key": "client-spoof", "X-Session-ID": "client-spoof"},
                    )
                    await client.get(f"{proxy_url}/v1/models")
                    status = await (await client.get(f"{proxy_url}/rollout/status")).json()

        routing_key = status["routing_key"]
        assert routing_key.startswith("rollout-")
        assert routing_key != status["session_id"]
        for headers in (upstream.headers[0], upstream.models_headers[0]):
            assert _header(headers, "x-smg-routing-key") == routing_key
            assert _header(headers, "x-session-id") == routing_key

    asyncio.run(scenario())


def test_routing_key_is_fresh_for_every_service_launch(tmp_path):
    upstream = FakeUpstream()

    async def scenario():
        keys = []
        async with _serve(upstream.app()) as upstream_url:
            for index in range(2):
                output = tmp_path / f"session-{index}.json"
                async with _serve(build_app(_config(upstream_url, output))) as proxy_url:
                    async with aiohttp.ClientSession() as client:
                        status = await (await client.get(f"{proxy_url}/rollout/status")).json()
                        keys.append(status["routing_key"])
        assert len(set(keys)) == 2

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# request body parameters
# ---------------------------------------------------------------------------


def test_chat_body_parameters_are_forwarded(tmp_path):
    upstream = FakeUpstream()
    upstream.reply(content="ok")
    output = tmp_path / "session.json"
    body = {
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0.2,
        "top_p": 0.9,
        "top_k": 20,
        "reasoning_effort": "high",
        "max_tokens": 64,
        "seed": 7,
        "presence_penalty": 0.1,
        "frequency_penalty": 0.2,
        "response_format": {"type": "json_object"},
        "extra_body": {"chat_template_kwargs": {"enable_thinking": True}, "top_k": 99},
    }

    async def scenario():
        async with _serve(upstream.app()) as upstream_url:
            async with _serve(build_app(_config(upstream_url, output))) as proxy_url:
                async with aiohttp.ClientSession() as client:
                    await client.post(f"{proxy_url}/v1/chat/completions", json=body)

        sent = upstream.requests[0]
        assert sent["temperature"] == 0.2
        assert sent["top_p"] == 0.9
        assert sent["top_k"] == 20
        assert sent["reasoning_effort"] == "high"
        assert sent["max_tokens"] == 64
        assert sent["seed"] == 7
        assert sent["presence_penalty"] == 0.1
        assert sent["frequency_penalty"] == 0.2
        assert sent["response_format"] == {"type": "json_object"}
        assert sent["chat_template_kwargs"] == {"enable_thinking": True}
        assert sent["stream"] is False
        assert "extra_body" not in sent

    asyncio.run(scenario())


def test_anthropic_sampling_params_and_extensions_are_forwarded(tmp_path):
    upstream = FakeUpstream()
    upstream.reply(content="ok")
    output = tmp_path / "session.json"

    async def scenario():
        async with _serve(upstream.app()) as upstream_url:
            async with _serve(build_app(_config(upstream_url, output))) as proxy_url:
                async with aiohttp.ClientSession() as client:
                    await client.post(
                        f"{proxy_url}/v1/messages",
                        json={
                            "model": "claude-like",
                            "max_tokens": 128,
                            "temperature": 0.3,
                            "top_p": 0.8,
                            "top_k": 10,
                            "stop_sequences": ["STOP"],
                            "messages": [{"role": "user", "content": "hi"}],
                            "chat_template_kwargs": {"enable_thinking": False},
                            "extra_body": {"min_p": 0.05},
                        },
                    )

        sent = upstream.requests[0]
        assert sent["max_tokens"] == 128
        assert sent["temperature"] == 0.3
        assert sent["top_p"] == 0.8
        assert sent["top_k"] == 10
        assert sent["stop"] == ["STOP"]
        assert sent["chat_template_kwargs"] == {"enable_thinking": False}
        assert sent["min_p"] == 0.05

    asyncio.run(scenario())


def test_responses_generation_params_are_mapped(tmp_path):
    upstream = FakeUpstream()
    upstream.reply(content="ok")
    output = tmp_path / "session.json"

    async def scenario():
        async with _serve(upstream.app()) as upstream_url:
            async with _serve(build_app(_config(upstream_url, output))) as proxy_url:
                async with aiohttp.ClientSession() as client:
                    await client.post(
                        f"{proxy_url}/v1/responses",
                        json={
                            "model": "gpt-like",
                            "input": "hi",
                            "max_output_tokens": 64,
                            "temperature": 0.1,
                            "top_p": 0.7,
                            "reasoning": {"effort": "high"},
                            "text": {"format": {"type": "json_object"}},
                            "store": True,
                            "metadata": {"k": "v"},
                            "user": "u1",
                            "service_tier": "auto",
                            "prompt_cache_key": "pk",
                            "parallel_tool_calls": False,
                        },
                    )

        sent = upstream.requests[0]
        assert sent["max_tokens"] == 64
        assert sent["reasoning_effort"] == "high"
        assert sent["response_format"] == {"type": "json_object"}
        assert sent["store"] is True
        assert sent["metadata"] == {"k": "v"}
        assert sent["user"] == "u1"
        assert sent["service_tier"] == "auto"
        assert sent["prompt_cache_key"] == "pk"
        assert sent["parallel_tool_calls"] is False

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# compaction signals and auxiliary requests
# ---------------------------------------------------------------------------


def test_explicit_compaction_without_tools_is_split(tmp_path):
    upstream = FakeUpstream()
    upstream.reply(content="work")
    upstream.reply(content="summary text")
    upstream.reply(content="post-compaction work")
    output = tmp_path / "session.json"

    async def scenario():
        async with _serve(upstream.app()) as upstream_url:
            async with _serve(build_app(_config(upstream_url, output))) as proxy_url:
                async with aiohttp.ClientSession() as client:
                    await client.post(
                        f"{proxy_url}/v1/chat/completions",
                        json={
                            "model": "m",
                            "messages": [{"role": "user", "content": "task"}],
                            "tools": CHAT_TOOLS,
                        },
                        headers=MAIN_HEADERS,
                    )
                    await client.post(
                        f"{proxy_url}/v1/chat/completions",
                        json={
                            "model": "m",
                            "messages": [
                                {"role": "user", "content": "task"},
                                {"role": "assistant", "content": "work"},
                                {"role": "user", "content": "summarize the session"},
                            ],
                        },
                        headers=COMPACTION_HEADERS,
                    )
                    await client.post(
                        f"{proxy_url}/v1/chat/completions",
                        json={
                            "model": "m",
                            "messages": [
                                {"role": "user", "content": "summary text"},
                                {"role": "user", "content": "keep going"},
                            ],
                            "tools": CHAT_TOOLS,
                        },
                        headers=POST_COMPACTION_HEADERS,
                    )
                    finish = await (await client.post(f"{proxy_url}/rollout/finish", json={})).json()

        document = _load_session(output)
        assert finish["compaction_requests"] == 1
        assert document["stats"]["trajectories"] == 3
        assert document["stats"]["compaction_requests"] == 1
        assert document["stats"]["request_classes"] == {"main": 2, "compaction": 1}
        before, compaction, after = document["trajectories"]

        assert before["kind"] == "conversation"
        assert before["loss_mask"] == [0, 1]
        assert compaction["kind"] == "compaction"
        assert compaction["contains_compaction"] is True
        assert [(m["role"], m["content"]) for m in compaction["messages"]] == [
            ("user", "task"),
            ("assistant", "work"),
            ("user", "summarize the session"),
            ("assistant", "summary text"),
        ]
        assert compaction["loss_mask"] == [0, 0, 0, 1]
        assert [turn["message_index"] for turn in compaction["turns"]] == [3]
        assert compaction["turns"][0]["compaction"] is True
        assert compaction["turns"][0]["request_class"] == "compaction"
        assert compaction["turns"][0]["compaction_trigger"] == "auto"
        assert compaction["parent_trajectory_id"] == before["trajectory_id"]
        assert after["kind"] == "conversation"
        assert after["loss_mask"] == [0, 0, 1]
        assert after["turns"][0]["context_compacted"] == "auto"
        _assert_strict_messages(compaction)

        log = document["request_log"]
        assert [entry["turn_index"] for entry in log] == [1, 2, 3]
        assert [entry.get("request_class") for entry in log] == ["main", "compaction", "main"]
        assert log[2]["context_compacted"] == "auto"
        assert log[1]["trajectory_id"] == compaction["trajectory_id"]
        assert log[0]["trajectory_id"] == before["trajectory_id"]

    asyncio.run(scenario())


def test_explicit_compaction_with_tools_is_labelled_not_split(tmp_path):
    upstream = FakeUpstream()
    upstream.reply(content="work")
    upstream.reply(content="summary text")
    upstream.reply(content="post-compaction work")
    output = tmp_path / "session.json"

    async def scenario():
        async with _serve(upstream.app()) as upstream_url:
            async with _serve(build_app(_config(upstream_url, output))) as proxy_url:
                async with aiohttp.ClientSession() as client:
                    await client.post(
                        f"{proxy_url}/v1/chat/completions",
                        json={"model": "m", "messages": [{"role": "user", "content": "task"}], "tools": CHAT_TOOLS},
                        headers=MAIN_HEADERS,
                    )
                    await client.post(
                        f"{proxy_url}/v1/chat/completions",
                        json={
                            "model": "m",
                            "messages": [
                                {"role": "user", "content": "task"},
                                {"role": "assistant", "content": "work"},
                                {"role": "user", "content": "summarize the session"},
                            ],
                            "tools": CHAT_TOOLS,
                        },
                        headers=COMPACTION_HEADERS,
                    )
                    finish = await (await client.post(f"{proxy_url}/rollout/finish", json={})).json()

        document = _load_session(output)
        assert finish["compaction_requests"] == 1
        assert document["stats"]["trajectories"] == 1
        trajectory = document["trajectories"][0]
        assert trajectory["kind"] == "conversation"
        assert trajectory["contains_compaction"] is True
        assert trajectory["loss_mask"] == [0, 1, 0, 1]
        assert [turn["compaction"] for turn in trajectory["turns"]] == [False, True]

    asyncio.run(scenario())


def test_auxiliary_request_is_not_a_compaction_and_can_be_excluded(tmp_path):
    upstream = FakeUpstream()
    upstream.reply(content="work")
    upstream.reply(content="no")
    output = tmp_path / "session.json"

    async def scenario():
        async with _serve(upstream.app()) as upstream_url:
            config = _config(upstream_url, output, exclude_request_classes=("auxiliary",))
            async with _serve(build_app(config)) as proxy_url:
                async with aiohttp.ClientSession() as client:
                    await client.post(
                        f"{proxy_url}/v1/chat/completions",
                        json={"model": "m", "messages": [{"role": "user", "content": "task"}], "tools": CHAT_TOOLS},
                        headers=MAIN_HEADERS,
                    )
                    # A tools-less side request (session title / security monitor)
                    # is not a compaction and must not split the trajectory.
                    await client.post(
                        f"{proxy_url}/v1/chat/completions",
                        json={
                            "model": "m",
                            "messages": [
                                {"role": "user", "content": "task"},
                                {"role": "assistant", "content": "work"},
                                {"role": "user", "content": "classify this session"},
                            ],
                        },
                        headers=AUXILIARY_HEADERS,
                    )
                    finish = await (await client.post(f"{proxy_url}/rollout/finish", json={})).json()

        document = _load_session(output)
        assert finish["compaction_requests"] == 0
        assert finish["excluded_turns"] == 1
        assert document["stats"]["trajectories"] == 1
        assert document["stats"]["request_classes"] == {"main": 1, "auxiliary": 1}
        trajectory = document["trajectories"][0]
        assert trajectory["loss_mask"] == [0, 1, 0, 0]
        assert trajectory["turns"][1]["excluded"] is True
        assert trajectory["turns"][1]["request_class"] == "auxiliary"
        assert trajectory["turns"][1]["trained"] is False

    asyncio.run(scenario())


def test_request_log_preserves_timeline_when_leaves_are_out_of_order(tmp_path):
    upstream = FakeUpstream()
    upstream.reply(content="A1")
    upstream.reply(content="no")
    upstream.reply(content="A2")
    output = tmp_path / "session.json"

    async def scenario():
        async with _serve(upstream.app()) as upstream_url:
            async with _serve(build_app(_config(upstream_url, output))) as proxy_url:
                async with aiohttp.ClientSession() as client:
                    await client.post(
                        f"{proxy_url}/v1/chat/completions",
                        json={"model": "m", "messages": [{"role": "user", "content": "one"}]},
                        headers=MAIN_HEADERS,
                    )
                    # Independent side conversation captured between main turns.
                    await client.post(
                        f"{proxy_url}/v1/chat/completions",
                        json={
                            "model": "m",
                            "messages": [
                                {"role": "system", "content": "classifier"},
                                {"role": "user", "content": "classify"},
                            ],
                        },
                        headers=AUXILIARY_HEADERS,
                    )
                    # Main conversation continues; its leaf now ends at turn 3.
                    await client.post(
                        f"{proxy_url}/v1/chat/completions",
                        json={
                            "model": "m",
                            "messages": [
                                {"role": "user", "content": "one"},
                                {"role": "assistant", "content": "A1"},
                                {"role": "user", "content": "two"},
                            ],
                        },
                        headers=MAIN_HEADERS,
                    )
                    await client.post(f"{proxy_url}/rollout/finish")

        document = _load_session(output)
        main, auxiliary = document["trajectories"]
        assert main["path_turn_indices"] == [1, 3]
        assert auxiliary["path_turn_indices"] == [2]
        log = document["request_log"]
        assert [entry["turn_index"] for entry in log] == [1, 2, 3]
        assert log[0]["trajectory_id"] == main["trajectory_id"]
        assert log[1]["trajectory_id"] == auxiliary["trajectory_id"]
        assert log[2]["trajectory_id"] == main["trajectory_id"]

    asyncio.run(scenario())


def test_truncated_tool_arguments_are_preserved(tmp_path):
    truncated = '{"cmd": "ls'
    upstream = FakeUpstream()
    upstream.reply(
        content="",
        tool_calls=[{"name": "bash", "arguments": truncated}],
        finish="length",
    )
    output = tmp_path / "session.json"

    async def scenario():
        async with _serve(upstream.app()) as upstream_url:
            async with _serve(build_app(_config(upstream_url, output))) as proxy_url:
                async with aiohttp.ClientSession() as client:
                    await client.post(
                        f"{proxy_url}/v1/chat/completions",
                        json={"model": "m", "messages": [{"role": "user", "content": "run"}]},
                    )
                    await client.post(f"{proxy_url}/rollout/finish")

        document = _load_session(output)
        trajectory = document["trajectories"][0]
        call = trajectory["messages"][1]["tool_calls"][0]
        assert call["function"]["arguments"] == truncated
        assert "_raw_arguments" not in json.dumps(trajectory["messages"])
        assert trajectory["turns"][0]["response"]["tool_calls"][0]["function"]["arguments"] == truncated
        assert trajectory["loss_mask"] == [0, 1]
        _assert_strict_messages(trajectory)

    asyncio.run(scenario())


def test_subagent_branch_appears_later_in_the_series(tmp_path):
    upstream = FakeUpstream()
    upstream.reply(content="main work")
    upstream.reply(content="subagent work")
    output = tmp_path / "session.json"

    async def scenario():
        async with _serve(upstream.app()) as upstream_url:
            async with _serve(build_app(_config(upstream_url, output))) as proxy_url:
                async with aiohttp.ClientSession() as client:
                    await client.post(
                        f"{proxy_url}/v1/chat/completions",
                        json={"model": "m", "messages": [{"role": "user", "content": "main task"}]},
                        headers=MAIN_HEADERS,
                    )
                    await client.post(
                        f"{proxy_url}/v1/chat/completions",
                        json={
                            "model": "m",
                            "messages": [
                                {"role": "system", "content": "you are a search sub-agent"},
                                {"role": "user", "content": "find the file"},
                            ],
                        },
                        headers={"x-claude-code-request-class": "subagent", "x-claude-code-agent-id": "agent-7"},
                    )
                    await client.post(f"{proxy_url}/rollout/finish")

        document = _load_session(output)
        assert document["stats"]["trajectories"] == 2
        main, subagent = document["trajectories"]
        assert main["parent_trajectory_id"] is None
        assert subagent["first_turn_index"] == 2
        assert subagent["messages"][0] == {"role": "system", "content": "you are a search sub-agent"}
        assert subagent["turns"][0]["request_class"] == "subagent"
        assert subagent["turns"][0]["agent_id"] == "agent-7"

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# lifecycle / errors
# ---------------------------------------------------------------------------


def test_status_and_idempotent_finish(tmp_path):
    upstream = FakeUpstream()
    upstream.reply(content="hi")
    output = tmp_path / "session.json"

    async def scenario():
        async with _serve(upstream.app()) as upstream_url:
            config = _config(upstream_url, output, metadata={"rollout_id": 7})
            async with _serve(build_app(config)) as proxy_url:
                async with aiohttp.ClientSession() as client:
                    await client.post(
                        f"{proxy_url}/v1/chat/completions",
                        json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
                    )
                    status = await (await client.get(f"{proxy_url}/rollout/status")).json()
                    assert status["finished"] is False
                    assert status["turns"] == 1
                    first = await (
                        await client.post(f"{proxy_url}/rollout/finish", json={"metadata": {"tag": "x"}})
                    ).json()
                    second = await (await client.post(f"{proxy_url}/rollout/finish")).json()
                    assert first == second
                    assert (await (await client.get(f"{proxy_url}/rollout/status")).json())["finished"] is True
                    late = await client.post(
                        f"{proxy_url}/v1/chat/completions",
                        json={"model": "m", "messages": [{"role": "user", "content": "again"}]},
                    )
                    assert late.status == 503

        document = _load_session(output)
        assert document["finished"] is True
        assert document["finish_reason"] == "explicit"
        assert document["metadata"] == {"rollout_id": 7, "tag": "x"}

    asyncio.run(scenario())


def test_shutdown_finalizes_an_unfinished_session(tmp_path):
    upstream = FakeUpstream()
    upstream.reply(content="hi")
    output = tmp_path / "session.json"

    async def scenario():
        async with _serve(upstream.app()) as upstream_url:
            async with _serve(build_app(_config(upstream_url, output))) as proxy_url:
                async with aiohttp.ClientSession() as client:
                    await client.post(
                        f"{proxy_url}/v1/chat/completions",
                        json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
                    )

        document = _load_session(output)
        assert document["finished"] is True
        assert document["finish_reason"] == "shutdown"

    asyncio.run(scenario())


def test_max_turns_cap(tmp_path):
    upstream = FakeUpstream()
    upstream.reply(content="one")
    upstream.reply(content="two")
    output = tmp_path / "session.json"

    async def scenario():
        async with _serve(upstream.app()) as upstream_url:
            async with _serve(build_app(_config(upstream_url, output, max_turns=1))) as proxy_url:
                async with aiohttp.ClientSession() as client:
                    first = await client.post(
                        f"{proxy_url}/v1/chat/completions",
                        json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
                    )
                    assert first.status == 200
                    second = await client.post(
                        f"{proxy_url}/v1/chat/completions",
                        json={"model": "m", "messages": [{"role": "user", "content": "again"}]},
                    )
                    assert second.status == 429

    asyncio.run(scenario())


def test_upstream_error_is_reported_and_not_captured(tmp_path):
    upstream = FakeUpstream()
    upstream.fail_status = 500
    output = tmp_path / "session.json"

    async def scenario():
        async with _serve(upstream.app()) as upstream_url:
            async with _serve(build_app(_config(upstream_url, output))) as proxy_url:
                async with aiohttp.ClientSession() as client:
                    response = await client.post(
                        f"{proxy_url}/v1/chat/completions",
                        json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
                    )
                    assert response.status == 502
                    status = await (await client.get(f"{proxy_url}/rollout/status")).json()
                    assert status["turns"] == 0
                    assert status["trajectories"] == 0

    asyncio.run(scenario())


def test_invalid_body_is_rejected(tmp_path):
    upstream = FakeUpstream()
    output = tmp_path / "session.json"

    async def scenario():
        async with _serve(upstream.app()) as upstream_url:
            async with _serve(build_app(_config(upstream_url, output))) as proxy_url:
                async with aiohttp.ClientSession() as client:
                    bad = await client.post(
                        f"{proxy_url}/v1/chat/completions",
                        data="not json",
                        headers={"Content-Type": "application/json"},
                    )
                    assert bad.status == 400

    asyncio.run(scenario())


def test_models_passthrough_and_health(tmp_path):
    upstream = FakeUpstream()
    output = tmp_path / "session.json"

    async def scenario():
        async with _serve(upstream.app()) as upstream_url:
            async with _serve(build_app(_config(upstream_url, output))) as proxy_url:
                async with aiohttp.ClientSession() as client:
                    models = await (await client.get(f"{proxy_url}/v1/models")).json()
                    assert models["data"][0]["id"] == "fake-model"
                    assert (await (await client.get(f"{proxy_url}/healthz")).json()) == {"ok": True}

    asyncio.run(scenario())


def test_forced_request_params_override_converted_requests(tmp_path):
    upstream = FakeUpstream()
    upstream.reply(content="ok")
    output = tmp_path / "session.json"
    forced = {
        "temperature": 0.0,
        "top_p": 0.5,
        "top_k": 12,
        "reasoning_effort": "high",
        "max_tokens": 256,
        "seed": 11,
    }

    async def scenario():
        async with _serve(upstream.app()) as upstream_url:
            config = _config(
                upstream_url,
                output,
                model="forced-model",
                request_params={**forced, "model": "param-model"},
            )
            async with _serve(build_app(config)) as proxy_url:
                async with aiohttp.ClientSession() as client:
                    await client.post(
                        f"{proxy_url}/v1/messages",
                        json={
                            "model": "claude-like",
                            "max_tokens": 64,
                            "temperature": 0.9,
                            "top_p": 0.9,
                            "messages": [{"role": "user", "content": "hi"}],
                        },
                    )
                    await (await client.post(f"{proxy_url}/rollout/finish", json={})).json()

        sent = upstream.requests[0]
        assert sent["temperature"] == 0.0
        assert sent["top_p"] == 0.5
        assert sent["top_k"] == 12
        assert sent["reasoning_effort"] == "high"
        assert sent["max_tokens"] == 256
        assert sent["seed"] == 11
        # --model wins over a request_params model entry.
        assert sent["model"] == "forced-model"

        document = _load_session(output)
        effective = document["trajectories"][0]["turns"][0]["request"]
        assert effective["temperature"] == 0.0
        assert effective["reasoning_effort"] == "high"
        assert effective["max_tokens"] == 256

    asyncio.run(scenario())


def test_chat_requests_also_receive_forced_request_params(tmp_path):
    upstream = FakeUpstream()
    upstream.reply(content="ok")
    output = tmp_path / "session.json"
    forced = {"temperature": 0.3, "reasoning_effort": "low"}

    async def scenario():
        async with _serve(upstream.app()) as upstream_url:
            config = _config(upstream_url, output, request_params=forced)
            async with _serve(build_app(config)) as proxy_url:
                async with aiohttp.ClientSession() as client:
                    await client.post(
                        f"{proxy_url}/v1/chat/completions",
                        json={
                            "model": "m",
                            "messages": [{"role": "user", "content": "hi"}],
                            "temperature": 0.9,
                        },
                    )
                    await (await client.post(f"{proxy_url}/rollout/finish", json={})).json()

        assert upstream.requests[0]["temperature"] == 0.3
        assert upstream.requests[0]["reasoning_effort"] == "low"

    asyncio.run(scenario())


def test_parse_request_params_validation():
    assert parse_request_params(None) == {}
    assert parse_request_params("") == {}
    assert parse_request_params('{"temperature": 0.2}') == {"temperature": 0.2}
    for raw in (
        "not-json",
        "[1, 2]",
        '{"messages": []}',
        '{"stream": true}',
        '{"stream_options": {}}',
        '{"tools": []}',
        '{"messages": [], "temperature": 0.2}',
    ):
        with pytest.raises(ValueError):
            parse_request_params(raw)


def test_redacted_headers_elide_credentials():
    headers = [
        ("Authorization", "Bearer super-secret"),
        ("x-api-key", "key-secret"),
        ("Content-Type", "application/json"),
        ("X-Custom", "visible"),
    ]
    assert _redacted_headers(headers) == {
        "Authorization": "<redacted>",
        "x-api-key": "<redacted>",
        "Content-Type": "application/json",
        "X-Custom": "visible",
    }


def test_log_upstream_requests_full_logs_body_and_redacts_headers(tmp_path, caplog):
    upstream = FakeUpstream()
    upstream.reply(content="ok")
    output = tmp_path / "session.json"

    async def scenario():
        async with _serve(upstream.app()) as upstream_url:
            config = _config(
                upstream_url,
                output,
                upstream_api_key="super-secret-key",
                log_upstream_requests="full",
                log_upstream_max_chars=0,
            )
            async with _serve(build_app(config)) as proxy_url:
                async with aiohttp.ClientSession() as client:
                    await client.post(
                        f"{proxy_url}/v1/chat/completions",
                        json={
                            "model": "model-x",
                            "messages": [{"role": "user", "content": "prompt-marker-abc"}],
                        },
                    )
                    await (await client.post(f"{proxy_url}/rollout/finish", json={})).json()

    with caplog.at_level(logging.INFO, logger="rollout_manager.server"):
        asyncio.run(scenario())

    text = caplog.text
    assert "upstream request: POST" in text
    assert "prompt-marker-abc" in text
    assert '"model": "model-x"' in text
    assert "<redacted>" in text
    assert "super-secret-key" not in text


def test_log_upstream_requests_summary_and_truncation(tmp_path, caplog):
    upstream = FakeUpstream()
    upstream.reply(content="ok")
    output = tmp_path / "session.json"

    async def scenario():
        async with _serve(upstream.app()) as upstream_url:
            config = _config(
                upstream_url,
                output,
                log_upstream_requests="full",
                log_upstream_max_chars=40,
            )
            async with _serve(build_app(config)) as proxy_url:
                async with aiohttp.ClientSession() as client:
                    await client.post(
                        f"{proxy_url}/v1/chat/completions",
                        json={
                            "model": "model-x",
                            "messages": [{"role": "user", "content": "x" * 500}],
                        },
                    )
                    await (await client.post(f"{proxy_url}/rollout/finish", json={})).json()

            summary_config = _config(upstream_url, output, log_upstream_requests="summary")
            async with _serve(build_app(summary_config)) as proxy_url:
                async with aiohttp.ClientSession() as client:
                    await client.post(
                        f"{proxy_url}/v1/chat/completions",
                        json={
                            "model": "model-x",
                            "messages": [{"role": "user", "content": "summary-prompt"}],
                        },
                    )
                    await (await client.post(f"{proxy_url}/rollout/finish", json={})).json()

    with caplog.at_level(logging.INFO, logger="rollout_manager.server"):
        asyncio.run(scenario())

    text = caplog.text
    assert "truncated" in text
    assert "summary-prompt" not in text  # summary does not log the body
    assert "messages=1" in text
