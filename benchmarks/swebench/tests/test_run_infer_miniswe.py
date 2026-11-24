"""Wiring test for the SWE-bench mini entry point without Docker or a model.

The heavy pieces (checkout, rollout, judge, patch validation) are stubbed so the
test exercises exactly the benchmark-owned wiring: history publication, JSONL
fields, judge activation and metric propagation.
"""

import json
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import SecretStr

from benchmarks.swebench import run_infer_miniswe
from benchmarks.swebench.run_infer_miniswe import SWEBenchMiniEvaluation
from benchmarks.utils.execution_judge import ExecutionBasedJudge
from benchmarks.utils.miniswe import publish_canonical_histories
from benchmarks.utils.miniswe.model import MiniModelSpec
from benchmarks.utils.miniswe.rollout import MiniRolloutResult
from benchmarks.utils.models import EvalInstance, EvalMetadata, EvalOutput
from benchmarks.utils.tests.fakes import FakeWorkspace
from openhands.sdk import LLM
from openhands.sdk.critic import PassCritic
from openhands.sdk.llm import Metrics
from openhands.sdk.llm.utils.metrics import TokenUsage
from openhands.sdk.workspace import RemoteWorkspace


PATCH = "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-x\n+y\n"


class RecordingJudge(ExecutionBasedJudge):
    calls: list[dict[str, Any]] = []

    def judge(self, instance_id: str, git_patch: str, instance_data: dict) -> bool:
        self.calls.append(
            {
                "instance_id": instance_id,
                "git_patch": git_patch,
                "instance_data": instance_data,
            }
        )
        return True


def _metadata(output_dir: Path) -> EvalMetadata:
    return EvalMetadata(
        llm=LLM(
            model="openai/gpt-4o-mini",
            api_key=SecretStr("sk-test"),
            base_url="http://localhost:1/v1",
        ),
        dataset="test/swebench",
        max_iterations=7,
        eval_output_dir=str(output_dir),
        details={},
        prompt_path=str(
            Path(run_infer_miniswe.__file__).parent / "prompts" / "miniswe-default.j2"
        ),
        critic=PassCritic(),
        workspace_type="docker",
    )


def _rollout() -> MiniRolloutResult:
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
        {
            "role": "assistant",
            "content": "listing",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "bash", "arguments": '{"command": "ls"}'},
                }
            ],
        },
        {"role": "tool", "content": "a.py", "tool_call_id": "call_1"},
        {"role": "exit", "content": PATCH},
    ]
    return MiniRolloutResult(
        status="Submitted",
        submission=PATCH,
        messages=messages,
        tools=[
            {
                "type": "function",
                "function": {"name": "bash", "parameters": {}},
            }
        ],
        model=MiniModelSpec(
            model_name="openai/gpt-4o-mini", temperature=0.0, top_p=1.0
        ),
        metrics=Metrics(
            model_name="openai/gpt-4o-mini",
            accumulated_cost=0.5,
            accumulated_token_usage=TokenUsage(
                model="openai/gpt-4o-mini", prompt_tokens=10, completion_tokens=5
            ),
        ),
        n_calls=2,
        cost=0.5,
        duration_seconds=1.25,
        trace_path="/tmp/trace.json",
    )


def _evaluator(tmp_path: Path, judge: Any = None) -> SWEBenchMiniEvaluation:
    return SWEBenchMiniEvaluation(
        metadata=_metadata(tmp_path),
        num_workers=1,
        mini_bash_env_file=None,
        judge=judge,
    )


def _instance() -> EvalInstance:
    return EvalInstance(
        id="test__instance-1",
        data={
            "repo": "test/repo",
            "base_commit": "abc123",
            "problem_statement": "fix it",
        },
    )


@pytest.fixture
def evaluator_calls() -> list:
    return []


@pytest.fixture
def evaluator(tmp_path, monkeypatch, evaluator_calls) -> SWEBenchMiniEvaluation:
    evaluation = _evaluator(tmp_path)
    calls = evaluator_calls

    def fake_prepare_repo(self, workspace, instance, *, checkout_commit=None) -> str:
        calls.append(checkout_commit)
        instance.data["repo_path"] = "/workspace/repo/"
        return "/workspace/repo/"

    monkeypatch.setattr(SWEBenchMiniEvaluation, "prepare_repo", fake_prepare_repo)
    monkeypatch.setattr(run_infer_miniswe, "head_matches_commit", lambda **kwargs: True)
    monkeypatch.setattr(
        run_infer_miniswe, "get_instruction", lambda **kwargs: "fix the bug"
    )
    monkeypatch.setattr(
        run_infer_miniswe, "run_mini_rollout", lambda **kwargs: _rollout()
    )
    monkeypatch.setattr(run_infer_miniswe, "check_patch_applies", lambda **kwargs: True)
    return evaluation


def _evaluate(evaluator: SWEBenchMiniEvaluation) -> EvalOutput:
    workspace = cast(RemoteWorkspace, FakeWorkspace())
    return evaluator.evaluate_instance(_instance(), workspace)


def test_evaluate_instance_writes_history_and_output(
    evaluator, evaluator_calls
) -> None:
    output = _evaluate(evaluator)

    # The rollout must start from the dataset baseline so the submitted patch and
    # the judge share one base commit.
    assert evaluator_calls == ["abc123"]

    assert output.test_result == {"git_patch": PATCH}
    assert output.instruction == "fix the bug"
    assert output.error is None
    assert output.history == []
    assert output.attempt == 1
    assert output.metrics is not None
    assert output.metrics.accumulated_cost == 0.5

    history_path = Path(evaluator.metadata.eval_output_dir) / (
        "test__instance-1.attempt1.history.json"
    )
    history = json.loads(history_path.read_text())
    assert history["test_result"] == {"git_patch": PATCH}
    assert history["model"] == "openai/gpt-4o-mini"
    assert history["tools"][0]["function"]["name"] == "bash"
    assert "evaluation" not in history
    assert history["miniswe"]["exit_status"] == "Submitted"
    assert history["miniswe"]["patch_applies_cleanly"] is True
    assert history["miniswe"]["step_limit"] == 7
    # mini's internal exit record is diagnostics-only.
    assert all(message["role"] != "exit" for message in history["messages"])
    assert history["trajectories"][0]["type"] == "trajectory"


def test_judge_receives_the_submitted_patch(tmp_path, monkeypatch) -> None:
    judge = RecordingJudge()
    evaluation = _evaluator(tmp_path, judge=judge)

    monkeypatch.setattr(
        SWEBenchMiniEvaluation,
        "prepare_repo",
        lambda self, workspace, instance, **kwargs: "/workspace/repo/",
    )
    monkeypatch.setattr(run_infer_miniswe, "head_matches_commit", lambda **kwargs: True)
    monkeypatch.setattr(
        run_infer_miniswe, "get_instruction", lambda **kwargs: "fix the bug"
    )
    monkeypatch.setattr(
        run_infer_miniswe, "run_mini_rollout", lambda **kwargs: _rollout()
    )
    monkeypatch.setattr(run_infer_miniswe, "check_patch_applies", lambda **kwargs: True)

    _evaluate(evaluation)

    assert judge.calls[0]["instance_id"] == "test__instance-1"
    assert judge.calls[0]["git_patch"] == PATCH
    history_path = Path(evaluation.metadata.eval_output_dir) / (
        "test__instance-1.attempt1.history.json"
    )
    history = json.loads(history_path.read_text())
    assert history["evaluation"] is True


def test_publish_canonical_history_selects_aggregated_attempt(evaluator) -> None:
    output_dir = Path(evaluator.metadata.eval_output_dir)
    for attempt, marker in ((1, "first"), (2, "second")):
        (output_dir / f"test__instance-1.attempt{attempt}.history.json").write_text(
            json.dumps({"mark": marker})
        )
    (output_dir / "output.jsonl").write_text(
        json.dumps({"instance_id": "test__instance-1", "attempt": 2}) + "\n"
    )

    publish_canonical_histories(str(output_dir), str(output_dir / "output.jsonl"))

    canonical = json.loads((output_dir / "test__instance-1.history.json").read_text())
    assert canonical["mark"] == "second"
