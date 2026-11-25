"""Tests for the rollout-manager proxy lifecycle and export helpers."""

from __future__ import annotations

import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast

import pytest

from benchmarks.utils.harnesses import rollout as rollout_module
from benchmarks.utils.harnesses.rollout import (
    RolloutManager,
    RolloutManagerConfig,
    RolloutManagerError,
    build_metrics,
    collect_tool_sets,
    embed_history_loss_mask,
    embed_history_reasoning,
    embed_loss_mask_into_messages,
    embed_reasoning_into_messages,
    export_summary,
    find_free_port,
    http_json,
    is_openai_tool_list,
    load_rollout_export,
    main_trajectory,
    normalize_upstream_base_url,
    parse_stream_json_result,
    resolve_trajectory_tools,
    resolve_upstream_model,
    rollout_manager_command,
    upstream_credentials,
)


def make_config(tmp_path: Path, **overrides: Any) -> RolloutManagerConfig:
    defaults: dict[str, Any] = dict(
        upstream_base_url="http://127.0.0.1:9999/v1",
        session_id="inst-attempt1",
        output_path=str(tmp_path / "rollout.json"),
        log_path=str(tmp_path / "rollout.log"),
        upstream_api_key="key",
        model="served-model",
        metadata={"instance_id": "inst"},
        exclude_request_classes="auxiliary",
        host="172.17.0.1",
    )
    defaults.update(overrides)
    return RolloutManagerConfig(**defaults)


def test_command_contains_required_flags(tmp_path: Path) -> None:
    argv = rollout_manager_command(make_config(tmp_path))
    assert argv[:3] == [argv[0], "-m", "rollout_manager"]
    assert "--upstream" in argv and "http://127.0.0.1:9999/v1" in argv
    assert "--output" in argv
    assert "--session-id" in argv
    assert "--upstream-api-key" in argv
    assert "--model" in argv
    assert "--metadata" in argv
    assert "--exclude-request-classes" in argv


def test_command_omits_optional_flags(tmp_path: Path) -> None:
    argv = rollout_manager_command(
        make_config(
            tmp_path,
            upstream_api_key=None,
            model=None,
            metadata={},
            exclude_request_classes="",
        )
    )
    assert "--upstream-api-key" not in argv
    assert "--model" not in argv
    assert "--metadata" not in argv
    assert "--exclude-request-classes" not in argv


def test_command_forwards_forced_request_params(tmp_path: Path) -> None:
    argv = rollout_manager_command(
        make_config(
            tmp_path,
            request_params={
                "temperature": 0.2,
                "top_p": 0.9,
                "reasoning_effort": "high",
            },
        )
    )
    index = argv.index("--request-params")
    assert json.loads(argv[index + 1]) == {
        "temperature": 0.2,
        "top_p": 0.9,
        "reasoning_effort": "high",
    }


def test_command_omits_forced_request_params_when_empty(tmp_path: Path) -> None:
    assert "--request-params" not in rollout_manager_command(make_config(tmp_path))


def test_command_forwards_upstream_logging(tmp_path: Path) -> None:
    argv = rollout_manager_command(
        make_config(
            tmp_path,
            log_upstream_requests="full",
            log_upstream_max_chars=5000,
        )
    )
    index = argv.index("--log-upstream-requests")
    assert argv[index + 1] == "full"
    index = argv.index("--log-upstream-max-chars")
    assert argv[index + 1] == "5000"
    assert "--log-upstream-requests" not in rollout_manager_command(
        make_config(tmp_path)
    )


def test_normalize_upstream_base_url() -> None:
    assert normalize_upstream_base_url("https://api.example.com") == (
        "https://api.example.com/v1"
    )
    assert normalize_upstream_base_url("https://api.example.com/") == (
        "https://api.example.com/v1"
    )
    assert normalize_upstream_base_url("https://api.example.com/v1") == (
        "https://api.example.com/v1"
    )
    assert normalize_upstream_base_url("http://127.0.0.1:30000/v1/") == (
        "http://127.0.0.1:30000/v1"
    )
    with pytest.raises(RolloutManagerError, match="no base_url"):
        normalize_upstream_base_url("")


def test_resolve_upstream_model() -> None:
    assert resolve_upstream_model("litellm_proxy/anthropic/claude-sonnet-4") == (
        "anthropic/claude-sonnet-4"
    )
    assert resolve_upstream_model("openai/deepseek-ai/DeepSeek-V4") == (
        "deepseek-ai/DeepSeek-V4"
    )
    assert resolve_upstream_model("deepseek-ai/DeepSeek-V4") == (
        "deepseek-ai/DeepSeek-V4"
    )
    assert resolve_upstream_model("gpt-4o") == "gpt-4o"


def test_find_free_port_is_bindable() -> None:
    port = find_free_port("127.0.0.1")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", port))


class FakeProcess:
    def __init__(self, returncode: int | None = None) -> None:
        self.returncode = returncode
        self.terminated = False
        self.killed = False

    def poll(self):
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True

    def wait(self, timeout=None):
        return self.returncode or 0


def test_manager_start_waits_for_health_then_finish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = make_config(tmp_path)
    process = FakeProcess(returncode=None)
    health_calls: list[str] = []
    finished: list[str] = []

    def fake_spawn(argv, *, log_path):
        Path(log_path).write_text("starting\n")
        return process, open(log_path)

    def fake_request(url, *, method="GET", payload=None, timeout=30.0):
        if url.endswith("/healthz"):
            health_calls.append(url)
            if len(health_calls) < 2:
                raise ConnectionError("refused")
            return 200, {}
        if url.endswith("/rollout/finish"):
            finished.append(url)
            return 200, {"session_id": "inst-attempt1", "turns": 3, "trajectories": 2}
        raise AssertionError(url)

    monkeypatch.setattr(rollout_module, "require_rollout_manager", lambda: None)
    manager = RolloutManager(
        config, spawn=cast(Any, fake_spawn), request=cast(Any, fake_request)
    )
    manager.start()
    port = manager.port
    assert port > 0
    assert len(health_calls) == 2
    assert f"172.17.0.1:{port}" in manager.base_url
    summary = manager.finish(timeout=10)
    assert summary["turns"] == 3
    manager.stop()
    assert process.terminated is True


def test_manager_detects_process_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = make_config(tmp_path)
    process = FakeProcess(returncode=2)

    def fake_spawn(argv, *, log_path):
        Path(log_path).write_text("fatal: cannot import rollout_manager\n")
        return process, open(log_path)

    monkeypatch.setattr(rollout_module, "require_rollout_manager", lambda: None)
    manager = RolloutManager(
        config,
        spawn=cast(Any, fake_spawn),
        request=cast(Any, lambda *a, **k: (200, {})),
    )
    with pytest.raises(RolloutManagerError, match="exited during startup"):
        manager.start()


def test_manager_finish_http_error(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    manager = RolloutManager(
        config,
        spawn=cast(Any, lambda argv, *, log_path: (FakeProcess(returncode=None), None)),
        request=cast(Any, lambda url, **kwargs: (500, {"error": "boom"})),
    )
    with pytest.raises(RolloutManagerError, match="finish request failed"):
        manager.finish(timeout=1)


def test_upstream_credentials_strips_prefixes() -> None:
    llm = type(
        "FakeLLM",
        (),
        {
            "base_url": "https://llm-proxy.example.com",
            "api_key": "secret",
            "model": "litellm_proxy/anthropic/claude-sonnet-4",
        },
    )()
    config = type(
        "Config",
        (),
        {
            "upstream_base_url": None,
            "upstream_api_key": None,
            "upstream_model": None,
            "harness_model": None,
        },
    )()
    url, key, model, harness_model = upstream_credentials(llm, config)
    assert url == "https://llm-proxy.example.com/v1"
    assert key == "secret"
    assert model == "anthropic/claude-sonnet-4"
    assert harness_model == "anthropic/claude-sonnet-4"


def write_export(path: Path) -> dict[str, Any]:
    export = {
        "type": "rollout_session",
        "version": 2,
        "session_id": "inst-attempt1",
        "finished": True,
        "finish_reason": "explicit",
        "model": "served-model",
        "stats": {
            "requests": 2,
            "turns": 2,
            "trajectories": 2,
            "compaction_requests": 1,
            "excluded_turns": 0,
            "request_classes": {"main": 1, "compaction": 1},
        },
        "request_log": [{"turn_index": 1, "request_class": "main"}],
        "trajectories": [
            {
                "trajectory_id": "s:1",
                "kind": "conversation",
                "leaf_turn_index": 1,
                "messages": [{"role": "user", "content": "hi"}],
                "tool_sets": [[{"type": "function", "function": {"name": "Bash"}}]],
                "turns": [
                    {
                        "turn_index": 1,
                        "model": "served-model",
                        "response_id": "r1",
                        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                    }
                ],
            },
            {
                "trajectory_id": "s:2",
                "kind": "compaction",
                "leaf_turn_index": 2,
                "messages": [{"role": "user", "content": "summary"}],
                "tool_sets": [[{"type": "function", "function": {"name": "Bash"}}]],
                "turns": [
                    {
                        "turn_index": 2,
                        "model": "served-model",
                        "response_id": "r2",
                        "usage": {"prompt_tokens": 20, "completion_tokens": 7},
                    }
                ],
            },
        ],
    }
    path.write_text(json.dumps(export))
    return export


def test_load_export_validates_schema(tmp_path: Path) -> None:
    path = tmp_path / "export.json"
    write_export(path)
    export = load_rollout_export(path)
    assert export["session_id"] == "inst-attempt1"
    summary = export_summary(export)
    assert summary["requests"] == 2
    assert summary["trajectories"] == 2
    assert summary["request_classes"] == {"main": 1, "compaction": 1}

    path.write_text(json.dumps({"type": "something_else", "version": 2}))
    with pytest.raises(RolloutManagerError, match="unexpected type"):
        load_rollout_export(path)
    path.write_text(json.dumps({"type": "rollout_session", "version": 1}))
    with pytest.raises(RolloutManagerError, match="unsupported schema version"):
        load_rollout_export(path)


def test_load_export_missing_file(tmp_path: Path) -> None:
    with pytest.raises(RolloutManagerError, match="not found"):
        load_rollout_export(tmp_path / "missing.json")


def test_build_metrics_sums_turns(tmp_path: Path) -> None:
    path = tmp_path / "export.json"
    export = write_export(path)
    metrics = build_metrics(export, "llm-model")
    assert metrics is not None
    assert metrics.accumulated_token_usage is not None
    assert metrics.accumulated_token_usage.prompt_tokens == 30
    assert metrics.accumulated_token_usage.completion_tokens == 12
    assert len(metrics.token_usages) == 2
    assert metrics.accumulated_cost == 0.0


def test_build_metrics_without_usage(tmp_path: Path) -> None:
    assert build_metrics({"trajectories": []}, "m") is None


def test_embed_loss_mask_into_messages() -> None:
    messages = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "Bash", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "ok"},
        {"role": "assistant", "content": "done"},
    ]
    assert embed_loss_mask_into_messages(messages, [0, 1, 0, 0]) == 2
    assert "loss_mask" not in messages[0]
    assert "loss_mask" not in messages[2]
    assert list(messages[1].keys()) == ["role", "content", "tool_calls", "loss_mask"]
    assert messages[1]["loss_mask"] == 1
    assert messages[3]["loss_mask"] == 0

    history = {
        "trajectories": [
            {
                "trajectory_id": "s:1",
                "kind": "conversation",
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "done"},
                ],
                "turns": [{"message_index": 1, "trained": True}],
                "loss_mask": [0, 1],
            }
        ]
    }
    assert embed_history_loss_mask(history) == 1
    assert history["trajectories"][0]["messages"][1]["loss_mask"] == 1
    assert history["messages"][1]["loss_mask"] == 1


def test_embed_reasoning_into_messages() -> None:
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": ""},
        {"role": "assistant", "content": "done"},
    ]
    changed = embed_reasoning_into_messages(
        messages, [None, "thought", "answer reasoning"]
    )
    assert changed == 2
    assert "reasoning_content" not in messages[0]
    assert messages[1]["reasoning_content"] == "thought"
    assert messages[2]["reasoning_content"] == "answer reasoning"
    assert list(messages[1].keys()) == ["role", "reasoning_content", "content"]
    assert list(messages[2].keys()) == ["role", "reasoning_content", "content"]

    with_calls = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "Bash", "arguments": "{}"},
                }
            ],
        }
    ]
    assert embed_reasoning_into_messages(with_calls, ["plan"]) == 1
    assert list(with_calls[0].keys()) == [
        "role",
        "reasoning_content",
        "content",
        "tool_calls",
    ]

    history = {
        "trajectories": [
            {
                "trajectory_id": "s:1",
                "kind": "conversation",
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "done"},
                ],
                "turns": [{"message_index": 1, "trained": True}],
                "reasoning_content": [None, "thought"],
            }
        ]
    }
    assert embed_history_reasoning(history) == 1
    assert history["trajectories"][0]["messages"][1]["reasoning_content"] == "thought"
    assert history["messages"][1]["reasoning_content"] == "thought"


def test_resolve_trajectory_tools() -> None:
    tools_a = [{"type": "function", "function": {"name": "Bash"}}]
    tools_b = [{"type": "function", "function": {"name": "Read"}}]
    trajectory = {
        "tool_sets": [tools_a, tools_b],
        "turns": [{"tool_set": 0}, {"tool_set": 1}],
    }
    assert resolve_trajectory_tools(trajectory, {}) == tools_b
    assert is_openai_tool_list(tools_a) is True
    assert is_openai_tool_list([]) is False

    # Explicitly tool-less requests (auto-mode classifier) must not inherit
    # another trajectory's tools.
    tool_less = {"tool_sets": [], "turns": [{"tool_set": None}]}
    assert resolve_trajectory_tools(tool_less, {"tools": [tools_a]}) == []

    # Older dumps without per-trajectory tool_sets fall back to the root view,
    # accepting both the resolved list and the legacy list-of-lists shape.
    assert resolve_trajectory_tools({}, {"tools": tools_a}) == tools_a
    assert resolve_trajectory_tools({}, {"tools": [tools_a]}) == tools_a
    assert resolve_trajectory_tools({}, {}) == []


def test_main_trajectory_skips_excluded_branches() -> None:
    tools = [{"type": "function", "function": {"name": "Bash"}}]
    export = {
        "trajectories": [
            {
                "trajectory_id": "s:1",
                "kind": "conversation",
                "exclude_from_training": False,
                "leaf_turn_index": 2,
                "messages": [{"role": "user", "content": "x"}],
                "tool_sets": [tools],
            },
            {
                "trajectory_id": "s:2",
                "kind": "conversation",
                "exclude_from_training": True,
                "leaf_turn_index": 99,
                "messages": [{"role": "user", "content": "aux"}],
                "tool_sets": [],
            },
        ]
    }
    trajectory = main_trajectory(export)
    assert trajectory is not None
    assert trajectory["trajectory_id"] == "s:1"


def test_main_trajectory_and_tool_sets(tmp_path: Path) -> None:
    path = tmp_path / "export.json"
    export = write_export(path)
    trajectory = main_trajectory(export)
    assert trajectory is not None
    assert trajectory["trajectory_id"] == "s:1"
    tool_sets = collect_tool_sets(export)
    assert len(tool_sets) == 1
    assert tool_sets[0][0]["function"]["name"] == "Bash"


def test_parse_stream_json_result(tmp_path: Path) -> None:
    log = tmp_path / "harness.jsonl"
    log.write_text(
        '{"type": "system", "subtype": "init"}\n'
        '{"type": "assistant", "message": {}}\n'
        '{"type": "result", "subtype": "success", "is_error": false, '
        '"total_cost_usd": 0.1}\n'
    )
    result = parse_stream_json_result(log)
    assert result is not None
    assert result["subtype"] == "success"
    assert result["total_cost_usd"] == 0.1


def test_parse_stream_json_result_missing(tmp_path: Path) -> None:
    assert parse_stream_json_result(tmp_path / "nope.jsonl") is None


def test_http_json_ignores_environment_proxies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            pass

        def do_GET(self) -> None:  # noqa: N802 - stdlib API
            body = json.dumps({"ok": True}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        # A blackholed proxy in the environment must not affect control traffic.
        for name in ("http_proxy", "HTTP_PROXY", "all_proxy", "ALL_PROXY"):
            monkeypatch.setenv(name, "http://127.0.0.1:1")
        monkeypatch.delenv("no_proxy", raising=False)
        monkeypatch.delenv("NO_PROXY", raising=False)
        port = server.server_address[1]
        code, payload = http_json(f"http://127.0.0.1:{port}/healthz", timeout=5)
        assert code == 200
        assert payload == {"ok": True}
    finally:
        server.shutdown()
        server.server_close()


def test_wait_ready_timeout_reports_diagnostics(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    manager = RolloutManager(
        config,
        spawn=cast(Any, lambda argv, *, log_path: (FakeProcess(returncode=None), None)),
        request=cast(
            Any, lambda *a, **k: (_ for _ in ()).throw(ConnectionError("timed out"))
        ),
    )
    manager.port = find_free_port("127.0.0.1")
    manager.process = cast(Any, FakeProcess(returncode=None))
    with pytest.raises(RolloutManagerError) as excinfo:
        manager.wait_ready(timeout=0.1)
    message = str(excinfo.value)
    assert "Control endpoint" in message
    assert "Proxy env" in message
    assert "tcp connect failed" in message


def test_start_stops_process_when_never_healthy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = make_config(tmp_path)
    config.start_timeout_sec = 0.1
    process = FakeProcess(returncode=None)

    def fake_spawn(argv, *, log_path):
        Path(log_path).write_text("Running on http://172.17.0.1:1234\n")
        return process, open(log_path)

    monkeypatch.setattr(rollout_module, "require_rollout_manager", lambda: None)
    manager = RolloutManager(
        config,
        spawn=cast(Any, fake_spawn),
        request=cast(
            Any, lambda *a, **k: (_ for _ in ()).throw(ConnectionError("timed out"))
        ),
    )
    # Bind a real free port on the configured host so the probe diagnostic works.
    manager.port = find_free_port("127.0.0.1")
    with pytest.raises(RolloutManagerError, match="did not become healthy"):
        manager.start()
    assert process.terminated is True
    assert manager.process is None
