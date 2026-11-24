"""Tests for the DeepSeek Harness adapter (no container, no model calls)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from benchmarks.utils.harnesses.base import InstalledHarness, create_harness
from benchmarks.utils.harnesses.config import HarnessConfig
from benchmarks.utils.harnesses.deepseek_harness import (
    API_KEY_ENV,
    CAPTURE_AUTH_TOKEN,
    DeepSeekHarness,
)
from benchmarks.utils.harnesses.distribution import HarnessDistribution
from benchmarks.utils.harnesses.rollout import (
    RolloutManagerError,
    parse_dsh_json_result,
)
from benchmarks.utils.tests.harness_fakes import FakeWorkspace, command_result


def make_installed() -> InstalledHarness:
    return InstalledHarness(
        binary_path="/opt/harness/deepseek-harness-sdk-runtime-linux-x64",
        install_dir="/opt/harness",
        work_dir="/tmp/harness",
        settings_path="/tmp/harness/dsh-patch.json",
        instruction_path="/tmp/harness/task.txt",
        version="0.2.2-rc.2",
    )


def make_distribution(tmp_path: Path) -> HarnessDistribution:
    root = tmp_path / "dist"
    (root / "linux-x64").mkdir(parents=True)
    executable = root / "deepseek-harness-sdk-runtime-linux-x64"
    executable.write_bytes(b"binary")
    (root / "deepseek-harness-sdk-runtime-linux-x64-rg").write_bytes(b"rg")
    (root / "linux-x64" / "downloads.json").write_text("{}")
    return HarnessDistribution(
        harness="deepseek-harness",
        version="0.2.2rc2",
        platform="linux-x64",
        directory=root,
        binary_path=executable,
        sha256="deadbeef",
        size_bytes=6,
        files=(
            "deepseek-harness-sdk-runtime-linux-x64",
            "deepseek-harness-sdk-runtime-linux-x64-rg",
            "linux-x64/downloads.json",
        ),
    )


def test_registry_and_defaults() -> None:
    harness = create_harness("deepseek-harness")
    assert isinstance(harness, DeepSeekHarness)
    assert harness.name == "deepseek-harness"
    config = HarnessConfig(name="deepseek-harness", version="0.2.2rc2")
    assert harness.permission_mode(config) == "danger-full-access"
    assert (
        DeepSeekHarness().permission_mode(
            HarnessConfig(
                name="deepseek-harness", version="0.2.2rc2", permission_mode="plan"
            )
        )
        == "read-only"
    )


def test_build_patch_configures_provider_and_compaction() -> None:
    config = HarnessConfig(
        name="deepseek-harness",
        version="0.2.2rc2",
        context_window=128_000,
        compaction_threshold=50,
        max_output_tokens=8192,
        custom_headers={"x-custom": "value"},
    )
    patch = DeepSeekHarness().build_patch(
        config, proxy_url="http://host.docker.internal:34567", harness_model="served"
    )
    rows = {row["id"]: row for row in patch}
    assert rows["agent-default-model"]["config"] == {
        "provider": "benchmark",
        "model": "served",
    }
    provider = rows["llm-pi-ai"]["config"]["providers"]["benchmark"]
    assert provider["api"] == "openai-completions"
    assert provider["baseURL"] == "http://host.docker.internal:34567/v1"
    assert provider["apiKeyEnv"] == API_KEY_ENV
    assert provider["models"] == [
        {"id": "served", "contextWindow": 128_000, "maxTokens": 8192}
    ]
    assert provider["compat"] == {
        "supportsStore": False,
        "supportsDeveloperRole": False,
        "supportsReasoningEffort": False,
        "maxTokensField": "max_tokens",
    }
    assert provider["headers"] == {"x-custom": "value"}
    assert rows["session-title-llm"]["disabled"] is True
    assert rows["compaction-basic"]["config"]["thresholdRatio"] == 0.5

    no_compaction = DeepSeekHarness().build_patch(
        HarnessConfig(
            name="deepseek-harness", version="0.2.2rc2", compaction_threshold=0
        ),
        proxy_url="http://proxy",
        harness_model="m",
    )
    assert "compaction-basic" not in {row["id"] for row in no_compaction}


def test_build_env_and_command() -> None:
    config = HarnessConfig(
        name="deepseek-harness",
        version="0.2.2rc2",
        extra_args=["--add-dir", "/repo/.git"],
    )
    harness = DeepSeekHarness()
    env = harness.build_env(config, proxy_url="http://proxy", harness_model="served")
    assert env["DSH_HOME"] == "/tmp/harness/dsh-home"
    assert env["HOME"] == "/tmp/harness/home"
    assert env[API_KEY_ENV] == CAPTURE_AUTH_TOKEN
    assert env["DSH_TELEMETRY_DISABLED"] == "1"
    assert "host.docker.internal" in env["NO_PROXY"]

    argv = harness.build_command(config, make_installed(), instruction="fix the bug")
    assert argv[:2] == [
        "/opt/harness/deepseek-harness-sdk-runtime-linux-x64",
        "--patch",
    ]
    assert argv[argv.index("--profile") + 1] == "headless"
    assert "--json" in argv
    assert argv[argv.index("--permission-mode") + 1] == "danger-full-access"
    assert argv[argv.index("--approval-policy") + 1] == "never"
    # extra_args stay launcher/app options ahead of the positional task.
    assert argv[-1] == "fix the bug"
    assert argv[-3:-1] == ["--add-dir", "/repo/.git"]


def test_install_copy_mode_copies_whole_payload(tmp_path: Path) -> None:
    distribution = make_distribution(tmp_path)
    workspace = FakeWorkspace(
        result_for=lambda command: command_result(stdout="0.2.2-rc.2")
        if "--version" in command
        else command_result()
    )
    config = HarnessConfig(dist_mode="copy")
    installed = DeepSeekHarness().install(cast(Any, workspace), distribution, config)
    assert installed.binary_path == (
        "/opt/harness/deepseek-harness-sdk-runtime-linux-x64"
    )
    assert installed.version == "0.2.2-rc.2"
    targets = [remote for _, remote in workspace.copied]
    assert targets == [
        "/opt/harness/deepseek-harness-sdk-runtime-linux-x64",
        "/opt/harness/deepseek-harness-sdk-runtime-linux-x64-rg",
        "/opt/harness/linux-x64/downloads.json",
    ]
    assert any("mkdir -p /opt/harness/linux-x64" in call for call in workspace.calls)
    assert any(
        "chmod 0755 /opt/harness/deepseek-harness-sdk-runtime-linux-x64-rg" in call
        for call in workspace.calls
    )
    assert workspace.written == {}


def test_install_reports_missing_mount_payload(tmp_path: Path) -> None:
    distribution = make_distribution(tmp_path)
    workspace = FakeWorkspace(
        result_for=lambda command: command_result(stdout="0.2.2-rc.2")
    )
    with __import__("pytest").raises(RolloutManagerError, match="distribution mount"):
        DeepSeekHarness().install(
            cast(Any, workspace),
            distribution,
            HarnessConfig(dist_mode="mount"),
        )


def test_install_reports_unrunnable_binary(tmp_path: Path) -> None:
    distribution = make_distribution(tmp_path)
    workspace = FakeWorkspace(
        result_for=lambda command: command_result(
            stderr="GLIBC_2.28 not found", exit_code=127
        )
        if "--version" in command
        else command_result()
    )
    with __import__("pytest").raises(RolloutManagerError, match="does not run"):
        DeepSeekHarness().install(
            cast(Any, workspace), distribution, HarnessConfig(dist_mode="copy")
        )


def test_run_writes_patch_and_parses_final_event(tmp_path: Path) -> None:
    workspace = FakeWorkspace()
    log_path = tmp_path / "harness.jsonl"
    events = [
        {"type": "session", "sessionId": "s-1", "cwd": "/testbed"},
        {
            "type": "final",
            "sessionId": "s-1",
            "reason": {"kind": "completed"},
            "text": "DONE",
        },
    ]
    log_path.write_text("\n".join(json.dumps(event) for event in events) + "\n")

    def run_streaming(argv, **kwargs):
        from benchmarks.utils.harnesses.workspace import ProcessOutcome

        return ProcessOutcome(
            exit_code=0,
            duration_seconds=1.0,
            timed_out=False,
            log_path=str(log_path),
            tail="",
        )

    workspace.run_streaming = run_streaming  # type: ignore[method-assign]
    harness = DeepSeekHarness()
    result = harness.run(
        cast(Any, workspace),
        HarnessConfig(name="deepseek-harness", version="0.2.2rc2"),
        make_installed(),
        proxy_url="http://proxy",
        harness_model="served",
        instruction="fix it",
        max_turns=10,
        repo_path="/testbed",
        log_path=str(log_path),
        timeout_sec=60,
    )
    patch = json.loads(workspace.written["/tmp/harness/dsh-patch.json"])
    assert {row["id"] for row in patch} >= {"llm-pi-ai", "agent-default-model"}
    assert workspace.written["/tmp/harness/task.txt"] == "fix it"
    assert result.exit_code == 0
    assert result.result_event is not None
    assert result.result_event["text"] == "DONE"


def test_parse_dsh_json_result_prefers_final_and_keeps_errors(tmp_path: Path) -> None:
    log = tmp_path / "run.jsonl"
    log.write_text(
        "not json\n"
        + json.dumps({"type": "session", "sessionId": "s"})
        + "\n"
        + json.dumps({"type": "final", "sessionId": "s", "text": "ok"})
        + "\n"
    )
    assert parse_dsh_json_result(log) == {
        "type": "final",
        "sessionId": "s",
        "text": "ok",
    }

    log.write_text(
        json.dumps({"type": "final", "text": "partial"})
        + "\n"
        + json.dumps({"type": "error", "message": "boom"})
        + "\n"
    )
    assert parse_dsh_json_result(log) == {"type": "final", "text": "partial"}

    log.write_text(json.dumps({"type": "error", "message": "boom"}) + "\n")
    assert parse_dsh_json_result(log) == {"type": "error", "message": "boom"}

    log.write_text("linestream only\n")
    assert parse_dsh_json_result(log) is None
