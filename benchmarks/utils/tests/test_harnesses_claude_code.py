"""Tests for the Claude Code harness adapter (no container, no model calls)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

from benchmarks.utils.harnesses.claude_code import (
    CAPTURE_AUTH_TOKEN,
    ClaudeCodeHarness,
    InstalledHarness,
    build_distribution_bind_volumes,
)
from benchmarks.utils.harnesses.config import HarnessConfig
from benchmarks.utils.harnesses.distribution import HarnessDistribution
from benchmarks.utils.harnesses.rollout import RolloutManagerError
from benchmarks.utils.tests.fakes import FakeWorkspace, command_result


def make_installed(tmp_path: Path) -> InstalledHarness:
    return InstalledHarness(
        binary_path="/opt/harness/claude",
        install_dir="/opt/harness",
        work_dir="/tmp/harness",
        settings_path="/tmp/harness/settings.json",
        instruction_path="/tmp/harness/instruction.txt",
        version="2.1.331 (Claude Code)",
    )


def make_distribution(tmp_path: Path) -> HarnessDistribution:
    binary = tmp_path / "dist" / "claude"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"binary")
    return HarnessDistribution(
        harness="claude-code",
        version="2.1.331",
        platform="linux-x64",
        directory=binary.parent,
        binary_path=binary,
        sha256="deadbeef",
        size_bytes=6,
    )


def test_build_env_points_at_proxy_and_configures_compaction() -> None:
    config = HarnessConfig(context_window=200_000, compaction_threshold=80)
    harness = ClaudeCodeHarness()
    env = harness.build_env(
        config, proxy_url="http://host.docker.internal:34567", harness_model="served"
    )
    assert env["ANTHROPIC_BASE_URL"] == "http://host.docker.internal:34567"
    assert env["ANTHROPIC_AUTH_TOKEN"] == CAPTURE_AUTH_TOKEN
    assert env["ANTHROPIC_MODEL"] == "served"
    assert env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] == "served"
    assert env["CLAUDE_CODE_GATEWAY_HINT_HEADERS"] == "1"
    assert env["IS_SANDBOX"] == "1"
    assert env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] == "200000"
    assert env["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] == "200000"
    assert env["CLAUDE_AUTOCOMPACT_PCT_OVERRIDE"] == "80"
    assert env["CLAUDE_CODE_DISABLE_1M_CONTEXT"] == "1"
    assert env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] == "1"
    assert env["DISABLE_TELEMETRY"] == "1"
    assert env["CLAUDE_CODE_USE_BEDROCK"] == ""


def test_build_env_omits_compaction_knobs_when_disabled() -> None:
    config = HarnessConfig(context_window=0, compaction_threshold=0)
    env = ClaudeCodeHarness().build_env(
        config, proxy_url="http://127.0.0.1:1", harness_model="m"
    )
    assert "CLAUDE_CODE_MAX_CONTEXT_TOKENS" not in env
    assert "CLAUDE_CODE_AUTO_COMPACT_WINDOW" not in env
    assert "CLAUDE_AUTOCOMPACT_PCT_OVERRIDE" not in env


def test_build_env_small_context_window_skips_autocompact_window() -> None:
    config = HarnessConfig(context_window=64_000, compaction_threshold=50)
    env = ClaudeCodeHarness().build_env(
        config, proxy_url="http://127.0.0.1:1", harness_model="m"
    )
    assert env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] == "64000"
    assert "CLAUDE_CODE_AUTO_COMPACT_WINDOW" not in env
    assert env["CLAUDE_AUTOCOMPACT_PCT_OVERRIDE"] == "50"


def test_build_env_custom_headers_and_extra_env() -> None:
    config = HarnessConfig(
        custom_headers={"anthropic-beta": "interleaved-thinking-2025-05-14"},
        extra_env={"EXTRA": "1"},
    )
    env = ClaudeCodeHarness().build_env(
        config, proxy_url="http://127.0.0.1:1", harness_model="m"
    )
    assert env["ANTHROPIC_CUSTOM_HEADERS"] == (
        "anthropic-beta: interleaved-thinking-2025-05-14"
    )
    assert env["EXTRA"] == "1"


def test_build_settings() -> None:
    harness = ClaudeCodeHarness()
    settings = harness.build_settings(
        HarnessConfig(skip_permissions=True), compaction_window=200_000
    )
    assert settings["autoCompactWindow"] == 200_000
    assert settings["autoCompactEnabled"] is True
    assert settings["permissions"] == {"defaultMode": "bypassPermissions"}
    assert harness.build_settings(
        HarnessConfig(skip_permissions=False), compaction_window=None
    ) == {
        "includeCoAuthoredBy": False,
        "cleanupPeriodDays": 1,
        "autoCompactEnabled": True,
    }


def test_build_command_includes_required_flags(tmp_path: Path) -> None:
    config = HarnessConfig()
    harness = ClaudeCodeHarness()
    argv = harness.build_command(
        config,
        make_installed(tmp_path),
        instruction="fix it",
        max_turns=42,
        settings_path="/tmp/harness/settings.json",
        harness_model="served",
    )
    assert argv[0] == "/opt/harness/claude"
    assert argv[1] == "-p"
    assert argv[2] == "fix it"
    assert "--output-format" in argv
    assert argv[argv.index("--output-format") + 1] == "stream-json"
    assert "--verbose" in argv
    assert argv[argv.index("--settings") + 1] == "/tmp/harness/settings.json"
    assert argv[argv.index("--model") + 1] == "served"
    assert argv[argv.index("--max-turns") + 1] == "42"
    assert "--dangerously-skip-permissions" in argv
    assert "--no-session-persistence" in argv


def test_build_command_options(tmp_path: Path) -> None:
    config = HarnessConfig(skip_permissions=False, bare=True, extra_args=["--debug"])
    argv = ClaudeCodeHarness().build_command(
        config,
        make_installed(tmp_path),
        instruction="x",
        max_turns=0,
        settings_path="/tmp/s.json",
        harness_model="m",
    )
    assert "--dangerously-skip-permissions" not in argv
    assert "--bare" in argv
    assert "--max-turns" not in argv
    assert argv[-1] == "--debug"


def test_install_copies_binary_and_verifies(tmp_path: Path) -> None:
    config = HarnessConfig(install_dir="/opt/harness", dist_mode="copy")
    workspace = FakeWorkspace(
        result_for=lambda command: (
            command_result(stdout="2.1.331 (Claude Code)\n")
            if "--version" in command
            else command_result()
        )
    )
    distribution = make_distribution(tmp_path)
    installed = ClaudeCodeHarness().install(cast(Any, workspace), distribution, config)
    assert workspace.copied == [(str(distribution.binary_path), "/opt/harness/claude")]
    assert installed.version == "2.1.331 (Claude Code)"
    assert installed.binary_path == "/opt/harness/claude"


def test_install_mount_mode_requires_binary(tmp_path: Path) -> None:
    config = HarnessConfig(install_dir="/opt/harness", dist_mode="mount")
    workspace = FakeWorkspace()
    with pytest.raises(RolloutManagerError, match="distribution mount is missing"):
        ClaudeCodeHarness().install(
            cast(Any, workspace), make_distribution(tmp_path), config
        )


def test_install_reports_non_running_binary(tmp_path: Path) -> None:
    config = HarnessConfig(dist_mode="copy")
    workspace = FakeWorkspace(
        result_for=lambda command: (
            command_result(stderr="Exec format error", exit_code=1)
            if "--version" in command
            else command_result()
        )
    )
    with pytest.raises(RolloutManagerError, match="does not run"):
        ClaudeCodeHarness().install(
            cast(Any, workspace), make_distribution(tmp_path), config
        )


def test_distribution_bind_volumes() -> None:
    config = HarnessConfig(dist_mode="mount", install_dir="/opt/harness")
    distribution = HarnessDistribution(
        harness="claude-code",
        version="2.1.331",
        platform="linux-x64",
        directory=Path("/cache/dist"),
        binary_path=Path("/cache/dist/claude"),
        sha256="x",
        size_bytes=1,
    )
    assert build_distribution_bind_volumes(config, distribution) == [
        "/cache/dist:/opt/harness:ro"
    ]
    copy_config = HarnessConfig(dist_mode="copy")
    assert build_distribution_bind_volumes(copy_config, distribution) == []
