"""Wiring tests for the z021data mini entry point (no Docker or model)."""

import json
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import SecretStr

from benchmarks.utils.execution_judge import ExecutionBasedJudge
from benchmarks.utils.miniswe.model import MiniModelSpec
from benchmarks.utils.miniswe.rollout import MiniRolloutResult
from benchmarks.utils.models import EvalInstance, EvalMetadata, EvalOutput
from benchmarks.utils.tests.fakes import PATCH, FakeWorkspace
from benchmarks.z021data import run_infer_miniswe
from benchmarks.z021data.run_infer_miniswe import Z021DataMiniEvaluation
from openhands.sdk import LLM
from openhands.sdk.critic import PassCritic
from openhands.sdk.llm import Metrics
from openhands.sdk.llm.utils.metrics import TokenUsage
from openhands.sdk.workspace import RemoteWorkspace


class RecordingJudge(ExecutionBasedJudge):
    calls: list[dict[str, Any]] = []

    def judge(self, instance_id: str, git_patch: str, instance_data: dict) -> bool:
        self.calls.append({"instance_id": instance_id, "git_patch": git_patch})
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
        dataset="test/z021data",
        max_iterations=9,
        eval_output_dir=str(output_dir),
        details={},
        prompt_path=_prompt_path(),
        critic=PassCritic(),
        workspace_type="flex",
    )


def _rollout() -> MiniRolloutResult:
    return MiniRolloutResult(
        status="Submitted",
        submission=PATCH,
        messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "task"},
            {"role": "exit", "content": PATCH},
        ],
        tools=[{"type": "function", "function": {"name": "bash", "parameters": {}}}],
        model=MiniModelSpec(model_name="openai/gpt-4o-mini"),
        metrics=Metrics(
            model_name="openai/gpt-4o-mini",
            accumulated_cost=0.2,
            accumulated_token_usage=TokenUsage(
                model="openai/gpt-4o-mini", prompt_tokens=7, completion_tokens=3
            ),
        ),
        n_calls=1,
        cost=0.2,
    )


def _instance() -> EvalInstance:
    return EvalInstance(
        id="owner__repo-1",
        data={
            "workdir": "/app",
            "base_commit": "abc123",
            "repo": "owner/repo",
        },
    )


def _evaluator(tmp_path: Path, judge: Any = None) -> Z021DataMiniEvaluation:
    return Z021DataMiniEvaluation(
        metadata=_metadata(tmp_path),
        num_workers=1,
        mini_bash_env_file=None,
        judge=judge,
    )


@pytest.fixture
def monkeypatched(monkeypatch):
    def fake_prepare_repo(self, workspace, instance) -> str:
        instance.data["repo_path"] = "/app/"
        return "/app/"

    monkeypatch.setattr(Z021DataMiniEvaluation, "prepare_repo", fake_prepare_repo)
    monkeypatch.setattr(run_infer_miniswe, "head_matches_commit", lambda **kwargs: True)
    monkeypatch.setattr(
        run_infer_miniswe, "get_instruction", lambda **kwargs: "fix the bug"
    )
    monkeypatch.setattr(
        run_infer_miniswe, "run_mini_rollout", lambda **kwargs: _rollout()
    )
    monkeypatch.setattr(run_infer_miniswe, "check_patch_applies", lambda **kwargs: True)


def test_evaluate_instance_writes_history_and_output(tmp_path, monkeypatched) -> None:
    evaluator = _evaluator(tmp_path)

    output = evaluator.evaluate_instance(
        _instance(), cast(RemoteWorkspace, FakeWorkspace())
    )

    assert isinstance(output, EvalOutput)
    assert output.test_result == {"git_patch": PATCH}
    assert output.instruction == "fix the bug"
    assert output.history == []
    assert output.metrics is not None and output.metrics.accumulated_cost == 0.2

    history_path = tmp_path / "owner__repo-1.attempt1.history.json"
    history = json.loads(history_path.read_text())
    assert history["model"] == "openai/gpt-4o-mini"
    assert history["tools"][0]["function"]["name"] == "bash"
    assert "evaluation" not in history
    assert history["miniswe"]["exit_status"] == "Submitted"
    assert history["miniswe"]["step_limit"] == 9
    assert history["trajectories"][0]["type"] == "trajectory"


def test_judge_receives_the_submitted_patch(
    tmp_path, monkeypatch, monkeypatched
) -> None:
    judge = RecordingJudge()
    evaluator = _evaluator(tmp_path, judge=judge)

    evaluator.evaluate_instance(_instance(), cast(RemoteWorkspace, FakeWorkspace()))

    assert judge.calls[0]["git_patch"] == PATCH
    history = json.loads((tmp_path / "owner__repo-1.attempt1.history.json").read_text())
    assert history["evaluation"] is True


def test_prompt_template_renders_submission_protocol() -> None:
    from benchmarks.z021data.run_infer import get_instruction

    metadata = _metadata(Path("/tmp"))
    text = get_instruction(
        instance={
            "repo_path": "/app/",
            "base_commit": "abc123",
            "problem_statement": "Fix the widget",
        },
        metadata=metadata,
        workspace_path="/app",
    )

    assert "Fix the widget" in text
    assert "/app/" in text
    assert "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in text
