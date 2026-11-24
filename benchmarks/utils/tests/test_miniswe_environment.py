"""Tests for the mini-swe-agent ``Environment`` adapter over a workspace."""

import shlex
import time
from typing import Any, cast

import pytest

from benchmarks.utils.miniswe.environment import (
    DEFAULT_ENV,
    MAX_SINGLE_OUTPUT_CHARS,
    SUBMISSION_MARKER,
    SubmissionTruncatedError,
    WorkspaceCommandError,
    WorkspaceEnvironment,
)
from benchmarks.utils.tests.fakes import FakeWorkspace, ok_result
from openhands.sdk.workspace import RemoteWorkspace


pytest.importorskip("minisweagent")

from minisweagent.exceptions import Submitted, TimeExceeded  # noqa: E402


def _env(workspace: FakeWorkspace, **kwargs: Any) -> WorkspaceEnvironment:
    return WorkspaceEnvironment(
        cast(RemoteWorkspace, workspace), cwd="/workspace/repo", **kwargs
    )


def test_action_runs_in_task_cwd_with_interpreter_and_env(
    fake_workspace_factory,
) -> None:
    workspace = fake_workspace_factory(lambda command, cwd: ok_result(command))
    env = _env(workspace, timeout=42, bash_env_file="/root/.bashrc")

    output = env.execute({"command": "ls -la"})

    assert output == {"output": "", "returncode": 0, "exception_info": ""}
    call = workspace.calls[-1]
    assert call["cwd"] == "/workspace/repo"
    assert call["timeout"] == 42
    tokens = shlex.split(call["command"])
    assert tokens[0] == f"PAGER={DEFAULT_ENV['PAGER']}"
    assert "BASH_ENV=/root/.bashrc" in tokens
    assert tokens[-3:] == ["bash", "-lc", "ls -la"]


def test_cwd_override_is_honoured(fake_workspace_factory) -> None:
    workspace = fake_workspace_factory(lambda command, cwd: ok_result(command))
    env = _env(workspace)

    env.execute({"command": "pwd"}, cwd="/workspace/other")

    assert workspace.calls[-1]["cwd"] == "/workspace/other"


def test_multiline_command_round_trips_through_quoting(
    fake_workspace_factory,
) -> None:
    workspace = fake_workspace_factory(lambda command, cwd: ok_result(command))
    env = _env(workspace)
    command = "cat <<'EOF' > f.py\nprint(\"it's fine\")\nEOF"

    env.execute({"command": command})

    # Exactly one shell argument reaches the interpreter, unchanged.
    assert shlex.split(workspace.calls[-1]["command"])[-1] == command


def test_stdout_and_stderr_are_preserved(fake_workspace_factory) -> None:
    workspace = fake_workspace_factory(
        lambda command, cwd: ok_result(command, stdout="out", stderr="err")
    )
    env = _env(workspace)

    assert env.execute({"command": "run"})["output"] == "out\nerr"


def test_submission_marker_raises_submitted_with_payload(
    fake_workspace_factory,
) -> None:
    patch = "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n"
    workspace = fake_workspace_factory(
        lambda command, cwd: ok_result(
            command, stdout=f"{SUBMISSION_MARKER}\n{patch}", stderr="git warning"
        )
    )
    env = _env(workspace)

    with pytest.raises(Submitted) as excinfo:
        env.execute({"command": f"echo {SUBMISSION_MARKER} && cat /tmp/patch.txt"})

    extra = excinfo.value.messages[0]["extra"]
    assert extra["exit_status"] == "Submitted"
    # stderr is never part of the candidate patch.
    assert extra["submission"] == patch


def test_marker_not_on_first_line_is_not_a_submission(
    fake_workspace_factory,
) -> None:
    workspace = fake_workspace_factory(
        lambda command, cwd: ok_result(
            command, stdout=f"warning\n{SUBMISSION_MARKER}\npatch"
        )
    )
    env = _env(workspace)

    output = env.execute({"command": "run"})

    assert output["returncode"] == 0
    assert SUBMISSION_MARKER in output["output"]


def test_marker_with_nonzero_exit_is_a_command_failure(
    fake_workspace_factory,
) -> None:
    workspace = fake_workspace_factory(
        lambda command, cwd: ok_result(
            command, stdout=f"{SUBMISSION_MARKER}\npatch", exit_code=2
        )
    )
    env = _env(workspace)

    assert env.execute({"command": "run"})["returncode"] == 2


def test_oversized_submission_is_rejected_explicitly(
    fake_workspace_factory,
) -> None:
    payload = "x" * MAX_SINGLE_OUTPUT_CHARS
    workspace = fake_workspace_factory(
        lambda command, cwd: ok_result(
            command, stdout=f"{SUBMISSION_MARKER}\n{payload}"
        )
    )
    env = _env(workspace)

    with pytest.raises(SubmissionTruncatedError):
        env.execute({"command": "submit"})


def test_timeout_is_reported_as_failed_action(fake_workspace_factory) -> None:
    workspace = fake_workspace_factory(
        lambda command, cwd: ok_result(
            command, stdout="partial", timeout_occurred=True, exit_code=-1
        )
    )
    env = _env(workspace, timeout=30)

    output = env.execute({"command": "sleep 100"})

    assert output["returncode"] == -1
    assert "timed out after 30 seconds" in output["exception_info"]


def test_workspace_transport_error_is_retryable(fake_workspace_factory) -> None:
    def boom(command: str, cwd: Any):
        raise ConnectionError("lost runtime")

    workspace = fake_workspace_factory(boom)
    env = _env(workspace)

    with pytest.raises(WorkspaceCommandError, match="ConnectionError"):
        env.execute({"command": "ls"})


def test_command_timeout_is_clamped_by_rollout_deadline(
    fake_workspace_factory,
) -> None:
    workspace = fake_workspace_factory(lambda command, cwd: ok_result(command))
    env = _env(workspace, timeout=60, deadline=time.monotonic() + 5)

    env.execute({"command": "ls"})

    assert workspace.calls[-1]["timeout"] <= 5


def test_expired_deadline_stops_the_rollout(fake_workspace_factory) -> None:
    workspace = fake_workspace_factory(lambda command, cwd: ok_result(command))
    env = _env(workspace, deadline=time.monotonic() - 1)

    with pytest.raises(TimeExceeded):
        env.execute({"command": "ls"})

    assert workspace.calls == []
