"""Regression test for the extracted Scale-SWE checkout preparation helper."""

from typing import Any, cast

from benchmarks.scaleswe.run_infer import ScaleSWEEvaluation
from benchmarks.utils.models import EvalInstance
from benchmarks.utils.tests.fakes import FakeWorkspace, ok_result
from openhands.sdk.workspace import RemoteWorkspace


def _workspace() -> FakeWorkspace:
    def responder(command: str, cwd: Any):
        if "git --no-pager log" in command:
            return ok_result(command, stdout="abc123 first commit")
        return ok_result(command)

    return FakeWorkspace(responder=responder)


def test_prepare_repo_normalizes_fields_and_runs_pre_commands() -> None:
    workspace = _workspace()
    instance = EvalInstance(
        id="owner__repo-1",
        data={
            "workdir": "/workspace/repo",
            "parent_commit": "abc123",
            "user": "owner",
            "repo": "repo",
            "pre_commands": "git checkout abc123\\n",
        },
    )

    evaluator = cast(ScaleSWEEvaluation, object())
    repo_path = ScaleSWEEvaluation.prepare_repo(
        evaluator, cast(RemoteWorkspace, workspace), instance
    )

    assert repo_path == "/workspace/repo/"
    assert instance.data["repo_path"] == "/workspace/repo/"
    assert instance.data["base_commit"] == "abc123"
    assert instance.data["repo"] == "owner/repo"
    commands = workspace.commands
    assert commands[0] == "cd /workspace/repo/ && git checkout abc123"
    assert "git --no-pager log --oneline -10" in commands[1]
