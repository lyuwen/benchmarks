"""Regression test for the extracted SWE-bench checkout preparation helper."""

from typing import cast

from benchmarks.swebench.run_infer import SWEBenchEvaluation
from benchmarks.utils.models import EvalInstance
from benchmarks.utils.tests.fakes import FakeWorkspace
from openhands.sdk.workspace import RemoteWorkspace


def test_prepare_repo_copies_links_and_resets_testbed() -> None:
    workspace = FakeWorkspace()
    instance = EvalInstance(
        id="django__django-1",
        data={"repo": "django/django", "base_commit": "abc"},
    )

    evaluator = cast(SWEBenchEvaluation, object())
    repo_path = SWEBenchEvaluation.prepare_repo(
        evaluator, cast(RemoteWorkspace, workspace), instance
    )

    assert repo_path == "/workspace/django/"
    assert instance.data["repo_path"] == "/workspace/django/"
    commands = workspace.commands
    assert (
        commands[0]
        == "mkdir -p /workspace/django/ ; cp -r /testbed/. /workspace/django/"
    )
    assert (
        commands[1] == "sudo rm -rf /testbed ; sudo ln -s /workspace/django/ /testbed"
    )
    assert commands[2] == "cd /workspace/django/ ; git reset --hard"
