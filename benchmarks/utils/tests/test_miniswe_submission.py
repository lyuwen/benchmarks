"""Tests for candidate-patch validation against the task baseline."""

from typing import Any, cast

from benchmarks.utils.miniswe.submission import (
    REMOTE_PATCH_PATH,
    check_patch_applies,
    get_head_commit,
    head_matches_commit,
)
from benchmarks.utils.tests.fakes import FakeWorkspace, ok_result
from openhands.sdk.workspace import RemoteWorkspace
from openhands.sdk.workspace.models import FileOperationResult


PATCH = "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-x\n+y\n"


def _workspace(apply_exit: int = 0) -> FakeWorkspace:
    def responder(command: str, cwd: Any):
        if "git apply --check" in command:
            return ok_result(command, exit_code=apply_exit, stderr="bad patch")
        return ok_result(command)

    return FakeWorkspace(responder=responder)


def test_empty_patch_is_not_valid() -> None:
    workspace = _workspace()

    assert (
        check_patch_applies(
            workspace=cast(RemoteWorkspace, workspace),
            repo_path="/workspace/repo",
            patch="  ",
        )
        is False
    )
    assert workspace.uploads == []


def test_patch_is_uploaded_and_checked_against_baseline() -> None:
    workspace = _workspace(apply_exit=0)

    assert (
        check_patch_applies(
            workspace=cast(RemoteWorkspace, workspace),
            repo_path="/workspace/repo",
            patch=PATCH,
        )
        is True
    )

    assert workspace.uploads[-1][1] == REMOTE_PATCH_PATH
    commands = workspace.commands
    assert any("git checkout -- ." in command for command in commands)
    assert any("git apply --check" in command for command in commands)


def test_non_applying_patch_returns_false() -> None:
    assert (
        check_patch_applies(
            workspace=cast(RemoteWorkspace, _workspace(apply_exit=1)),
            repo_path="/workspace/repo",
            patch=PATCH,
        )
        is False
    )


def test_upload_failure_makes_verdict_unavailable() -> None:
    class NoUploadWorkspace(FakeWorkspace):
        def file_upload(self, source_path, destination_path):
            return FileOperationResult(
                success=False,
                source_path=str(source_path),
                destination_path=str(destination_path),
                error="file api disabled",
            )

    assert (
        check_patch_applies(
            workspace=cast(RemoteWorkspace, NoUploadWorkspace()),
            repo_path="/workspace/repo",
            patch=PATCH,
        )
        is None
    )


def test_head_commit_is_read_from_checkout() -> None:
    workspace = FakeWorkspace(
        responder=lambda command, cwd: ok_result(command, stdout="abc123\n")
    )

    assert (
        get_head_commit(
            workspace=cast(RemoteWorkspace, workspace), repo_path="/workspace/repo"
        )
        == "abc123"
    )


def test_head_commit_unavailable() -> None:
    workspace = FakeWorkspace(
        responder=lambda command, cwd: ok_result(command, exit_code=128)
    )

    assert (
        get_head_commit(
            workspace=cast(RemoteWorkspace, workspace), repo_path="/workspace/repo"
        )
        is None
    )


def _head_workspace(head: str, resolved_base: str | None) -> FakeWorkspace:
    def responder(command: str, cwd: Any):
        if "rev-parse --verify" in command:
            if resolved_base is None:
                return ok_result(command, exit_code=128, stderr="unknown revision")
            return ok_result(command, stdout=f"{resolved_base}\n")
        return ok_result(command, stdout=f"{head}\n")

    return FakeWorkspace(responder=responder)


def test_head_matches_commit_resolves_abbreviated_commit() -> None:
    full = "a" * 40

    assert (
        head_matches_commit(
            workspace=cast(RemoteWorkspace, _head_workspace(full, full)),
            repo_path="/workspace/repo",
            commit="abc1234",
        )
        is True
    )


def test_head_matches_commit_detects_mismatch() -> None:
    assert (
        head_matches_commit(
            workspace=cast(RemoteWorkspace, _head_workspace("a" * 40, "b" * 40)),
            repo_path="/workspace/repo",
            commit="abc1234",
        )
        is False
    )


def test_head_matches_commit_unavailable_when_base_cannot_resolve() -> None:
    assert (
        head_matches_commit(
            workspace=cast(RemoteWorkspace, _head_workspace("a" * 40, None)),
            repo_path="/workspace/repo",
            commit="abc1234",
        )
        is None
    )
