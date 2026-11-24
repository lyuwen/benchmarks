"""Tests for the task-independent per-instance harness pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from pydantic import Field

from benchmarks.utils.execution_judge import ExecutionBasedJudge
from benchmarks.utils.harnesses import runner
from benchmarks.utils.harnesses.claude_code import HarnessRun
from benchmarks.utils.harnesses.config import HarnessConfig
from benchmarks.utils.harnesses.distribution import HarnessDistribution
from benchmarks.utils.harnesses.rollout import RolloutManagerError
from benchmarks.utils.harnesses.workspace import ProcessOutcome
from benchmarks.utils.models import EvalInstance
from benchmarks.utils.tests.fakes import (
    PATCH,
    FakeWorkspace,
    command_result,
    repo_workspace,
)
from openhands.sdk.llm import LLM


EXPORT = {
    "type": "rollout_session",
    "version": 2,
    "session_id": "i1-attempt1",
    "finished": True,
    "finish_reason": "explicit",
    "model": "served",
    "stats": {
        "requests": 2,
        "turns": 2,
        "trajectories": 1,
        "compaction_requests": 0,
        "excluded_turns": 0,
        "request_classes": {"main": 2},
    },
    "request_log": [{"turn_index": 1}, {"turn_index": 2}],
    "trajectories": [
        {
            "trajectory_id": "s:1",
            "kind": "conversation",
            "leaf_turn_index": 2,
            "messages": [
                {"role": "user", "content": "fix it"},
                {"role": "assistant", "content": "done"},
            ],
            "tool_sets": [[{"type": "function", "function": {"name": "Bash"}}]],
            "turns": [
                {
                    "turn_index": 1,
                    "model": "served",
                    "response_id": "r1",
                    "usage": {"prompt_tokens": 10, "completion_tokens": 2},
                },
                {
                    "turn_index": 2,
                    "model": "served",
                    "response_id": "r2",
                    "usage": {"prompt_tokens": 20, "completion_tokens": 3},
                },
            ],
        }
    ],
}


class FakeHarness:
    failed = False
    timed_out = False

    def __init__(self) -> None:
        self.ran: list[dict[str, Any]] = []

    def install(self, workspace, distribution, config):
        return SimpleNamespace(
            binary_path="/opt/harness/claude",
            version="2.1.331 (Claude Code)",
        )

    def run(self, workspace, config, installed, **kwargs: Any) -> HarnessRun:
        self.ran.append(kwargs)
        outcome = ProcessOutcome(
            exit_code=1 if self.failed else 0,
            duration_seconds=1.5,
            timed_out=self.timed_out,
            log_path=kwargs["log_path"],
            tail="",
        )
        return HarnessRun(
            argv=["claude"],
            env={},
            outcome=outcome,
            result_event={"type": "result", "is_error": False},
        )


class FakeManager:
    def __init__(self) -> None:
        self.port = 34567
        self.finished = 0
        self.stopped = 0
        self.finish_error: Exception | None = None

    def finish(self, timeout=None):
        self.finished += 1
        if self.finish_error is not None:
            raise self.finish_error
        return {"session_id": "i1", "turns": 2, "trajectories": 1}

    def stop(self) -> None:
        self.stopped += 1


class RecordingJudge(ExecutionBasedJudge):
    result: bool | None = True
    error: Any = None
    calls: list[dict[str, Any]] = Field(default_factory=list)

    def judge(self, instance_id, git_patch, instance_data):
        self.calls.append(
            {
                "instance_id": instance_id,
                "git_patch": git_patch,
                "instance_data": instance_data,
            }
        )
        if self.error is not None:
            raise self.error
        return self.result


def make_llm() -> LLM:
    return cast(
        LLM,
        SimpleNamespace(
            model="litellm_proxy/anthropic/claude-sonnet-4",
            base_url="http://127.0.0.1:9999",
            api_key="secret",
        ),
    )


def make_distribution(tmp_path: Path) -> HarnessDistribution:
    binary = tmp_path / "dist" / "claude"
    binary.parent.mkdir(parents=True, exist_ok=True)
    binary.write_bytes(b"binary")
    return HarnessDistribution(
        harness="claude-code",
        version="2.1.331",
        platform="linux-x64",
        directory=binary.parent,
        binary_path=binary,
        sha256="x",
        size_bytes=6,
    )


def patch_runner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    workspace: FakeWorkspace | None = None,
) -> tuple[FakeHarness, FakeManager, FakeWorkspace]:
    harness = FakeHarness()
    manager = FakeManager()
    ws = workspace or repo_workspace()
    monkeypatch.setattr(
        runner, "ensure_distribution", lambda config: make_distribution(tmp_path)
    )
    monkeypatch.setattr(runner, "create_harness", lambda name: harness)
    monkeypatch.setattr(runner, "_start_rollout_manager", lambda **kwargs: manager)
    monkeypatch.setattr(runner, "load_rollout_export", lambda path: EXPORT)
    return harness, manager, ws


def make_request(tmp_path: Path, **overrides: Any) -> runner.HarnessRunRequest:
    defaults: dict[str, Any] = dict(
        instance=EvalInstance(id="i1", data={"repo": "o/r"}),
        attempt=1,
        repo_path="/testbed",
        base_commit="base123",
        instruction="fix it",
        output_dir=str(tmp_path / "out"),
        max_iterations=50,
        conversation_timeout=600.0,
    )
    defaults.update(overrides)
    return runner.HarnessRunRequest(**defaults)


def test_env_from_setup_commands() -> None:
    env = runner.env_from_setup_commands(
        [
            "export PIP_CACHE_DIR=~/.cache/pip",
            "export FOO='bar baz'\nexport EMPTY=",
            "echo not-an-export",
            "export WEIRD=unterminated'",
        ]
    )
    assert env["PIP_CACHE_DIR"] == "~/.cache/pip"
    assert env["FOO"] == "bar baz"
    assert env["EMPTY"] == ""
    assert "not-an-export" not in env


def test_prepare_repository_resets_and_collects_baseline() -> None:
    workspace = repo_workspace(base_commit="abc123", untracked="build/out.so\n")
    commit, baseline = runner.prepare_repository(
        cast(Any, workspace), "/testbed", "abc123"
    )
    assert commit == "abc123"
    assert baseline == {"build/out.so"}
    assert any("git reset --hard" in call for call in workspace.calls)
    assert any("safe.directory" in call for call in workspace.calls)


def test_prepare_repository_rejects_unknown_base_commit() -> None:
    workspace = FakeWorkspace(
        result_for=lambda command: (
            command_result(stderr="unknown revision", exit_code=128)
            if "rev-parse" in command
            else command_result()
        )
    )
    with pytest.raises(RuntimeError, match="not present"):
        runner.prepare_repository(cast(Any, workspace), "/testbed", "deadbeef")


def test_collect_patch_filters_baseline_untracked() -> None:
    patch = (
        "diff --git a/build/out.so b/build/out.so\n"
        "Binary files a/build/out.so and b/build/out.so differ\n"
        "diff --git a/pkg/module.py b/pkg/module.py\n"
        "--- a/pkg/module.py\n"
        "+++ b/pkg/module.py\n"
        "@@ -1 +1 @@\n"
        "-x = 1\n"
        "+x = 2\n"
    )
    workspace = repo_workspace(diff=patch)
    result = runner.collect_patch(
        cast(Any, workspace),
        "/testbed",
        "base123",
        baseline_untracked={"build/out.so"},
    )
    assert "build/out.so" not in result
    assert "pkg/module.py" in result


def test_collect_patch_reports_stage_failure() -> None:
    workspace = FakeWorkspace(
        result_for=lambda command: (
            command_result(stderr="index locked", exit_code=1)
            if "git add -A" in command
            else command_result()
        )
    )
    with pytest.raises(RuntimeError, match="stage changes"):
        runner.collect_patch(cast(Any, workspace), "/testbed", "base123")


def test_run_harness_instance_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    harness, manager, workspace = patch_runner(monkeypatch, tmp_path)
    judge = RecordingJudge(result=True)
    request = make_request(tmp_path)

    result = runner.run_harness_instance(
        request,
        workspace=cast(Any, workspace),
        config=HarnessConfig(cache_dir=str(tmp_path / "cache")),
        llm=make_llm(),
        judge=judge,
    )

    assert result.git_patch.strip() == PATCH.strip()
    assert result.evaluation is True
    assert result.metrics is not None
    assert result.metrics.accumulated_token_usage is not None
    assert result.metrics.accumulated_token_usage.prompt_tokens == 30
    assert Path(result.history_path).is_file()
    history = json.loads(Path(result.history_path).read_text())
    assert history["instance_id"] == "i1"
    assert history["test_result"]["git_patch"].strip() == PATCH.strip()
    assert history["messages"][0]["role"] == "user"
    assert history["stats"]["requests"] == 2
    assert history["run"]["exit_code"] == 0
    assert history["evaluation"] is True
    assert history["repository"]["base_commit"] == "base123"
    assert judge.calls[0]["git_patch"].strip() == PATCH.strip()
    # The harness ran inside the container, pointed at the proxy.
    run_kwargs = harness.ran[0]
    assert run_kwargs["proxy_url"] == "http://host.docker.internal:34567"
    assert run_kwargs["max_turns"] == 50
    assert manager.finished == 1
    assert manager.stopped == 1


def test_run_harness_instance_warns_when_export_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    patch_runner(monkeypatch, tmp_path)

    def missing(path):
        raise RolloutManagerError("Rollout export not found")

    monkeypatch.setattr(runner, "load_rollout_export", missing)
    result = runner.run_harness_instance(
        make_request(tmp_path),
        workspace=cast(Any, repo_workspace()),
        config=HarnessConfig(cache_dir=str(tmp_path / "cache")),
        llm=make_llm(),
    )
    assert result.rollout_export_path is None
    assert result.metrics is None
    assert any("rollout export unavailable" in warning for warning in result.warnings)
    assert result.git_patch.strip() == PATCH.strip()


def test_run_harness_instance_handles_timeout_and_finish_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    harness, manager, workspace = patch_runner(monkeypatch, tmp_path)
    harness.timed_out = True
    manager.finish_error = RolloutManagerError("finish timed out")
    result = runner.run_harness_instance(
        make_request(tmp_path),
        workspace=cast(Any, workspace),
        config=HarnessConfig(cache_dir=str(tmp_path / "cache")),
        llm=make_llm(),
    )
    assert any("timed out" in warning for warning in result.warnings)
    assert any("finish failed" in warning for warning in result.warnings)
    assert manager.stopped == 1


def test_judge_exception_is_not_fatal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    patch_runner(monkeypatch, tmp_path)
    judge = RecordingJudge(error=RuntimeError("judge exploded"))
    result = runner.run_harness_instance(
        make_request(tmp_path),
        workspace=cast(Any, repo_workspace()),
        config=HarnessConfig(cache_dir=str(tmp_path / "cache")),
        llm=make_llm(),
        judge=judge,
    )
    assert result.evaluation is None


def test_publish_canonical_histories(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    output_path = output_dir / "output.jsonl"
    output_path.write_text(
        json.dumps({"instance_id": "i1", "attempt": 1})
        + "\n"
        + json.dumps({"instance_id": "i2", "attempt": 2})
        + "\n"
    )
    for instance_id, attempt in (("i1", 1), ("i2", 2)):
        runner.write_json_atomic(
            runner.history_file_path(str(output_dir), instance_id, attempt),
            {"instance_id": instance_id},
        )
    runner.publish_canonical_histories(str(output_dir), str(output_path))
    assert json.loads((output_dir / "i1.history.json").read_text()) == {
        "instance_id": "i1"
    }
    assert json.loads((output_dir / "i2.history.json").read_text()) == {
        "instance_id": "i2"
    }
