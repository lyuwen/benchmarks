"""Tests for the rollout-manager proxy lifecycle and export helpers."""

from __future__ import annotations

import json
import socket
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
    export_summary,
    find_free_port,
    load_rollout_export,
    main_trajectory,
    normalize_upstream_base_url,
    parse_stream_json_result,
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
