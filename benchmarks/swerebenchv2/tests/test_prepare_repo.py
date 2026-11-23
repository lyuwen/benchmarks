"""Regression tests for the extracted SWE-rebench-V2 preparation helpers."""

from typing import Any, cast

from benchmarks.swerebenchv2.run_infer import SWERebenchV2Evaluation
from benchmarks.utils.models import EvalInstance
from benchmarks.utils.tests.fakes import FakeWorkspace, ok_result
from openhands.sdk.workspace import RemoteWorkspace


def _workspace(untracked: str = "") -> FakeWorkspace:
    def responder(command: str, cwd: Any):
        if "git ls-files --others" in command:
            return ok_result(command, stdout=untracked)
        return ok_result(command, stdout="abc123 commit")

    return FakeWorkspace(responder=responder)


def _instance() -> EvalInstance:
    return EvalInstance(
        id="owner__repo-1",
        data={"repo": "owner/repo", "base_commit": "abc123"},
    )


def test_prepare_repo_copies_links_and_resets() -> None:
    workspace = _workspace()

    evaluator = cast(SWERebenchV2Evaluation, object())
    repo_path = SWERebenchV2Evaluation.prepare_repo(
        evaluator, cast(RemoteWorkspace, workspace), _instance()
    )

    assert repo_path == "/workspace/repo/"
    commands = workspace.commands
    assert commands[0] == "mkdir -p /workspace/repo/ ; cp -r /repo/. /workspace/repo/"
    assert commands[1] == "rm -rf /repo ; ln -s /workspace/repo /repo"
    assert commands[2] == "cd /workspace/repo/ ; git reset --hard"


def test_prepare_repo_resets_to_requested_baseline() -> None:
    workspace = _workspace()

    evaluator = cast(SWERebenchV2Evaluation, object())
    SWERebenchV2Evaluation.prepare_repo(
        evaluator,
        cast(RemoteWorkspace, workspace),
        _instance(),
        checkout_commit="abc123",
    )

    assert workspace.commands[2] == "cd /workspace/repo/ ; git reset --hard abc123"


def test_baseline_untracked_files_are_recorded() -> None:
    workspace = _workspace(untracked="go.tar.gz\nsub/build.o\n\n")

    evaluator = cast(SWERebenchV2Evaluation, object())
    files = SWERebenchV2Evaluation.baseline_untracked_files(
        evaluator, cast(RemoteWorkspace, workspace), "/workspace/repo/"
    )

    assert files == {"go.tar.gz", "sub/build.o"}
