"""Wiring tests for the SWE-rebench-V2 mini entry point (no Docker/model)."""

import json
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import SecretStr

from benchmarks.swerebenchv2 import run_infer_miniswe
from benchmarks.swerebenchv2.run_infer_miniswe import SWERebenchV2MiniEvaluation
from benchmarks.utils.execution_judge import ExecutionBasedJudge
from benchmarks.utils.miniswe.model import MiniModelSpec
from benchmarks.utils.miniswe.rollout import MiniRolloutResult
from benchmarks.utils.models import EvalInstance, EvalMetadata, EvalOutput
from benchmarks.utils.tests.fakes import FakeWorkspace
from openhands.sdk import LLM
from openhands.sdk.critic import PassCritic
from openhands.sdk.llm import Metrics
from openhands.sdk.llm.utils.metrics import TokenUsage
from openhands.sdk.workspace import RemoteWorkspace


def _diff(path: str) -> str:
    return (
        f"diff --git a/{path} b/{path}\n"
        f"--- a/{path}\n+++ b/{path}\n@@ -1 +1 @@\n-x\n+y\n"
    )


SOURCE_PATCH = _diff("src/a.py")
FULL_PATCH = SOURCE_PATCH + _diff("artifact.bin")


class RecordingJudge(ExecutionBasedJudge):
    calls: list[dict[str, Any]] = []

    def judge(self, instance_id: str, git_patch: str, instance_data: dict) -> bool:
        self.calls.append({"git_patch": git_patch})
        return True


def _prompt_path() -> str:
    return str(
        Path(run_infer_miniswe.__file__).parent / "prompts" / "miniswe-default.j2"
    )


def _metadata(output_dir: Path) -> EvalMetadata:
    return EvalMetadata(
        llm=LLM(
            model="openai/gpt-4o-mini",
            api_key=SecretStr("sk-test"),
            base_url="http://localhost:1/v1",
        ),
        dataset="test/swerebenchv2",
        max_iterations=11,
        eval_output_dir=str(output_dir),
        details={},
        prompt_path=_prompt_path(),
        critic=PassCritic(),
        workspace_type="flex",
    )


def _rollout(submission: str, status: str = "Submitted") -> MiniRolloutResult:
    return MiniRolloutResult(
        status=status,
        submission=submission,
        messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "task"},
            {"role": "exit", "content": submission},
        ],
        tools=[{"type": "function", "function": {"name": "bash", "parameters": {}}}],
        model=MiniModelSpec(model_name="openai/gpt-4o-mini"),
        metrics=Metrics(
            model_name="openai/gpt-4o-mini",
            accumulated_cost=0.1,
            accumulated_token_usage=TokenUsage(
                model="openai/gpt-4o-mini", prompt_tokens=5, completion_tokens=2
            ),
        ),
        n_calls=1,
        cost=0.1,
    )


def _instance() -> EvalInstance:
    return EvalInstance(
        id="owner__repo-1",
        data={"repo": "owner/repo", "base_commit": "abc123"},
    )


@pytest.fixture
def monkeypatched(monkeypatch):
    baseline: set[str] = set()

    def fake_prepare_repo(self, workspace, instance, *, checkout_commit=None) -> str:
        instance.data["repo_path"] = "/workspace/repo/"
        return "/workspace/repo/"

    def fake_baseline(self, workspace, repo_path) -> set[str]:
        return baseline

    monkeypatch.setattr(SWERebenchV2MiniEvaluation, "prepare_repo", fake_prepare_repo)
    monkeypatch.setattr(
        SWERebenchV2MiniEvaluation, "baseline_untracked_files", fake_baseline
    )
    monkeypatch.setattr(run_infer_miniswe, "head_matches_commit", lambda **kwargs: True)
    monkeypatch.setattr(
        run_infer_miniswe, "get_instruction", lambda **kwargs: "fix the bug"
    )
    monkeypatch.setattr(
        run_infer_miniswe,
        "run_mini_rollout",
        lambda **kwargs: _rollout(SOURCE_PATCH),
    )
    monkeypatch.setattr(run_infer_miniswe, "check_patch_applies", lambda **kwargs: True)
    return baseline


def _evaluator(tmp_path: Path, judge: Any = None) -> SWERebenchV2MiniEvaluation:
    return SWERebenchV2MiniEvaluation(
        metadata=_metadata(tmp_path),
        num_workers=1,
        mini_bash_env_file=None,
        judge=judge,
    )


def test_evaluate_instance_writes_history_and_output(tmp_path, monkeypatched) -> None:
    evaluator = _evaluator(tmp_path)

    output = evaluator.evaluate_instance(
        _instance(), cast(RemoteWorkspace, FakeWorkspace())
    )

    assert isinstance(output, EvalOutput)
    assert output.test_result == {"git_patch": SOURCE_PATCH}
    assert output.history == []
    assert output.metrics is not None

    history = json.loads((tmp_path / "owner__repo-1.attempt1.history.json").read_text())
    assert history["test_result"] == {"git_patch": SOURCE_PATCH}
    assert history["temperature"] is None or isinstance(history["temperature"], float)
    assert history["miniswe"]["exit_status"] == "Submitted"
    assert history["miniswe"]["step_limit"] == 11
    assert history["miniswe"]["baseline_untracked_files"] == 0


def test_baseline_untracked_artifacts_are_filtered_from_the_candidate(
    tmp_path, monkeypatch, monkeypatched
) -> None:
    monkeypatched.update({"artifact.bin"})
    monkeypatch.setattr(
        run_infer_miniswe,
        "run_mini_rollout",
        lambda **kwargs: _rollout(FULL_PATCH),
    )
    evaluator = _evaluator(tmp_path)

    output = evaluator.evaluate_instance(
        _instance(), cast(RemoteWorkspace, FakeWorkspace())
    )

    assert "artifact.bin" not in output.test_result["git_patch"]
    assert "src/a.py" in output.test_result["git_patch"]
    history = json.loads((tmp_path / "owner__repo-1.attempt1.history.json").read_text())
    assert history["test_result"]["git_patch"] == output.test_result["git_patch"]
    assert history["miniswe"]["baseline_untracked_files"] == 1


def test_judge_receives_the_filtered_candidate(
    tmp_path, monkeypatch, monkeypatched
) -> None:
    monkeypatched.update({"artifact.bin"})
    monkeypatch.setattr(
        run_infer_miniswe,
        "run_mini_rollout",
        lambda **kwargs: _rollout(FULL_PATCH),
    )
    judge = RecordingJudge()
    evaluator = _evaluator(tmp_path, judge=judge)

    evaluator.evaluate_instance(_instance(), cast(RemoteWorkspace, FakeWorkspace()))

    assert "artifact.bin" not in judge.calls[0]["git_patch"]
    assert "src/a.py" in judge.calls[0]["git_patch"]


def test_prompt_template_renders_submission_protocol() -> None:
    from benchmarks.swerebenchv2.run_infer import get_instruction

    metadata = _metadata(Path("/tmp"))
    text = get_instruction(
        instance={
            "repo": "owner/repo",
            "repo_path": "/workspace/repo/",
            "language": "python",
            "problem_statement": "Fix the widget",
            "base_commit": "abc123",
        },
        metadata=metadata,
        workspace_path="/workspace",
    )

    assert "Fix the widget" in text
    assert "python" in text
    assert "/workspace/repo/" in text
    assert "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in text
