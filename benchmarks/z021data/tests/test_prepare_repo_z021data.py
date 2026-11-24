"""Regression test for the extracted z021data checkout preparation helper."""

from typing import Any, cast

from benchmarks.utils.models import EvalInstance
from benchmarks.utils.tests.fakes import FakeWorkspace, ok_result
from benchmarks.z021data.run_infer import Z021DataEvaluation
from openhands.sdk.workspace import RemoteWorkspace


def _workspace(checkout_exit: int = 0) -> FakeWorkspace:
    def responder(command: str, cwd: Any):
        if "git checkout" in command:
            return ok_result(command, exit_code=checkout_exit, stderr="bad commit")
        return ok_result(command, stdout="abc123 first commit")

    return FakeWorkspace(responder=responder)


def _instance() -> EvalInstance:
    return EvalInstance(
        id="owner__repo-1",
        data={
            "workdir": "/app",
            "parent_commit": "abc123",
            "user": "owner",
            "repo": "repo",
            "problem_statement": "",
            "pr_description": {"pr_title": "Fix it", "pr_body": "Body text"},
        },
    )


def test_prepare_repo_normalizes_fields_and_checks_out_parent() -> None:
    workspace = _workspace()
    instance = _instance()

    evaluator = cast(Z021DataEvaluation, object())
    repo_path = Z021DataEvaluation.prepare_repo(
        evaluator, cast(RemoteWorkspace, workspace), instance
    )

    assert repo_path == "/app/"
    assert instance.data["repo_path"] == "/app/"
    assert instance.data["base_commit"] == "abc123"
    assert instance.data["repo"] == "owner/repo"
    # Empty problem statement falls back to the PR title/body.
    assert "Fix it" in instance.data["problem_statement"]
    assert "Body text" in instance.data["problem_statement"]
    commands = workspace.commands
    assert commands[0] == "cd /app/ && git checkout abc123 -f && git checkout ."
    assert "git --no-pager log --oneline -10" in commands[1]


def test_prepare_repo_checkout_failure_is_warn_only() -> None:
    workspace = _workspace(checkout_exit=1)

    evaluator = cast(Z021DataEvaluation, object())
    # Legacy behavior is to continue; the mini runner verifies HEAD itself.
    repo_path = Z021DataEvaluation.prepare_repo(
        evaluator, cast(RemoteWorkspace, workspace), _instance()
    )

    assert repo_path == "/app/"
