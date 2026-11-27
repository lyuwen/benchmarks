"""Lifecycle, manifest, resume, and trial-layout tests for ``harbor-infer``.

No Docker or LLM calls: the evaluation object is exercised through its
``prepare_instances`` / ``_write_run_metadata`` / ``_begin_trial`` helpers.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from benchmarks.harbor.run_infer import (
    AGENT_STATUS_TIMEOUT,
    HarborEvaluation,
    _relative_to_run_dir,
    _stop_agent_execution,
    task_from_instance_data,
)
from benchmarks.harbor.task import (
    HarborTaskError,
    HarborTaskSet,
    TaskLoadError,
    load_tasks,
)
from benchmarks.harbor.tests.conftest import write_task
from benchmarks.utils.models import EvalMetadata
from openhands.sdk import LLM
from openhands.sdk.conversation.state import ConversationExecutionStatus
from openhands.sdk.critic import PassCritic


def _metadata(output_dir: Path) -> EvalMetadata:
    return EvalMetadata(
        llm=LLM(model="gpt-4o-mini", api_key="test-key", usage_id="harbor-test"),
        dataset="unit-test",
        dataset_split="test",
        max_iterations=5,
        eval_output_dir=str(output_dir),
        details=None,
        prompt_path=None,
        eval_limit=0,
        env_setup_commands=[],
        max_attempts=1,
        critic=PassCritic(),
        selected_instances_file=None,
        max_retries=0,
        workspace_type="docker",
    )


def _evaluation(
    output_dir: Path,
    tasks_path: Path,
    *,
    verifier_enabled: bool = True,
    agent_timeout_override: float | None = None,
    verifier_timeout_override: float | None = None,
) -> HarborEvaluation:
    return HarborEvaluation(
        metadata=_metadata(output_dir),
        num_workers=1,
        tasks_path=str(tasks_path),
        docker_image_prefix=None,
        agent_plugin_image="openhands/agent-plugin",
        verifier_enabled=verifier_enabled,
        primary_reward_key="reward",
        workspace_memory_limit="14g",
        agent_timeout_override=agent_timeout_override,
        verifier_timeout_override=verifier_timeout_override,
        detach_docker_logs=False,
    )


def _make_tasks(root: Path) -> Path:
    write_task(root, "task-a")
    write_task(root, "task-b")
    return root


def test_prepare_instances_and_run_metadata(tmp_path: Path) -> None:
    tasks_path = _make_tasks(tmp_path / "dataset")
    output_dir = tmp_path / "run"
    evaluator = _evaluation(output_dir, tasks_path)

    instances = evaluator.prepare_instances()
    assert [instance.id for instance in instances] == ["task-a", "task-b"]
    assert instances[0].data["relative_path"] == "task-a"
    assert instances[0].data["config"]["environment"]["docker_image"] == (
        "example/task:latest"
    )
    # Preparing twice is cached.
    assert evaluator.prepare_instances() == instances

    evaluator._write_run_metadata()

    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    harbor = metadata["details"]["harbor"]
    assert harbor["reference_version"] == "0.9.0"
    assert harbor["task_count"] == 2
    assert harbor["manifest_file"] == "harbor_tasks.json"
    assert harbor["manifest_hash"]
    assert harbor["run_settings_hash"]
    assert harbor["run_settings"]["workspace_backend"] == "docker"

    manifest = json.loads(
        (output_dir / "harbor_tasks.json").read_text(encoding="utf-8")
    )
    assert manifest["task_count"] == 2
    assert [entry["id"] for entry in manifest["tasks"]] == ["task-a", "task-b"]
    assert all(len(entry["fingerprint"]) == 64 for entry in manifest["tasks"])


def test_resume_same_inputs_allowed(tmp_path: Path) -> None:
    tasks_path = _make_tasks(tmp_path / "dataset")
    output_dir = tmp_path / "run"

    _evaluation(output_dir, tasks_path)._write_run_metadata()
    _evaluation(output_dir, tasks_path)._write_run_metadata()


def test_resume_rejects_changed_task_content(tmp_path: Path) -> None:
    tasks_path = _make_tasks(tmp_path / "dataset")
    output_dir = tmp_path / "run"
    _evaluation(output_dir, tasks_path)._write_run_metadata()

    (tasks_path / "task-a" / "instruction.md").write_text(
        "A completely different instruction.", encoding="utf-8"
    )
    changed = _evaluation(output_dir, tasks_path)
    with pytest.raises(HarborTaskError, match="Task set/content changed"):
        changed._write_run_metadata()


def test_resume_rejects_changed_run_settings(tmp_path: Path) -> None:
    tasks_path = _make_tasks(tmp_path / "dataset")
    output_dir = tmp_path / "run"
    _evaluation(output_dir, tasks_path)._write_run_metadata()

    changed = _evaluation(output_dir, tasks_path, verifier_enabled=False)
    with pytest.raises(HarborTaskError, match="run settings changed"):
        changed._write_run_metadata()


def test_resume_rejects_added_task(tmp_path: Path) -> None:
    tasks_path = _make_tasks(tmp_path / "dataset")
    output_dir = tmp_path / "run"
    _evaluation(output_dir, tasks_path)._write_run_metadata()

    write_task(tasks_path, "task-c")
    changed = _evaluation(output_dir, tasks_path)
    with pytest.raises(HarborTaskError, match="Task set/content changed"):
        changed._write_run_metadata()


def test_begin_trial_layout(tmp_path: Path) -> None:
    tasks_path = _make_tasks(tmp_path / "dataset")
    output_dir = tmp_path / "run"
    evaluator = _evaluation(output_dir, tasks_path)
    instances = evaluator.prepare_instances()
    evaluator.current_attempt = 2

    trial_dir = Path(evaluator._begin_trial(instances[0]))

    parts = trial_dir.relative_to(output_dir).parts
    assert parts[:2] == ("trials", "task-a")
    assert parts[2] == "attempt-2"
    assert len(parts) == 4
    assert (trial_dir / "verifier").is_dir()
    assert (trial_dir / "artifacts").is_dir()
    record = json.loads((trial_dir / "trial.json").read_text(encoding="utf-8"))
    assert record["instance_id"] == "task-a"
    assert record["attempt"] == 2
    assert record["execution_id"] == parts[3]
    assert record["fingerprint"]
    assert evaluator._active_trial_dirs["task-a"] == str(trial_dir)


def test_task_from_instance_data_roundtrip(tmp_path: Path) -> None:
    tasks_path = _make_tasks(tmp_path / "dataset")
    task = load_tasks(tasks_path).tasks[0]
    rebuilt = task_from_instance_data(task.to_instance_data())
    assert rebuilt.id == task.id
    assert rebuilt.relative_path == task.relative_path
    assert rebuilt.instruction == task.instruction
    assert rebuilt.fingerprint == task.fingerprint
    assert rebuilt.config.model_dump() == task.config.model_dump()


def test_setup_commands_are_propagated_to_metadata(tmp_path: Path) -> None:
    tasks_path = _make_tasks(tmp_path / "dataset")
    output_dir = tmp_path / "run"
    evaluator = _evaluation(output_dir, tasks_path)
    evaluator.metadata.env_setup_commands = ["export A=1"]
    evaluator.prepare_instances()
    evaluator._write_run_metadata()
    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["env_setup_commands"] == ["export A=1"]


def test_timeout_overrides(tmp_path: Path) -> None:
    tasks_path = _make_tasks(tmp_path / "dataset")
    task = load_tasks(tasks_path).tasks[0]
    assert task.config.agent.timeout_sec == 60.0
    assert task.config.verifier.timeout_sec == 60.0

    evaluator = _evaluation(
        tmp_path / "run",
        tasks_path,
        agent_timeout_override=5.0,
        verifier_timeout_override=7.0,
    )
    assert evaluator._agent_timeout(task) == 5.0
    assert evaluator._verifier_timeout(task) == 7.0
    assert evaluator._agent_timeout.__self__ is evaluator


def test_relative_to_run_dir() -> None:
    assert _relative_to_run_dir("/tmp/run/trials/a", "/tmp/run") == "trials/a"
    assert _relative_to_run_dir("/elsewhere/x", "/tmp/run") == "/elsewhere/x"


class _FakeState:
    def __init__(self, status: ConversationExecutionStatus) -> None:
        self.execution_status = status


class _FakeConversation:
    def __init__(self, status: ConversationExecutionStatus) -> None:
        self.state = _FakeState(status)
        self.paused = False

    def pause(self) -> None:
        self.paused = True


def test_stop_agent_execution_returns_on_terminal_state() -> None:
    conversation = _FakeConversation(ConversationExecutionStatus.PAUSED)
    started = time.monotonic()
    _stop_agent_execution(conversation)  # type: ignore[arg-type]
    assert time.monotonic() - started < 5
    assert conversation.paused


def test_agent_status_constants() -> None:
    assert AGENT_STATUS_TIMEOUT == "timeout"


def test_load_tasks_can_be_empty_after_selection(tmp_path: Path) -> None:
    tasks_path = _make_tasks(tmp_path / "dataset")
    selection = tmp_path / "select.txt"
    selection.write_text("", encoding="utf-8")
    task_set = load_tasks(tasks_path, selected_instances_file=str(selection))
    assert isinstance(task_set, HarborTaskSet)
    assert task_set.tasks == []


def test_missing_tasks_path(tmp_path: Path) -> None:
    with pytest.raises(TaskLoadError):
        load_tasks(tmp_path / "does-not-exist")


# ---------------------------------------------------------------------------
# CLI preflight
# ---------------------------------------------------------------------------


def _cli_args(**overrides: object) -> "object":
    from types import SimpleNamespace

    values: dict[str, object] = {
        "workspace": "docker",
        "max_attempts": 1,
        "critic": "pass",
        "max_retries": 0,
        "tasks_path": "/tmp",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_validate_cli_rejects_unsupported_workspace() -> None:
    from benchmarks.harbor.run_infer import _validate_cli

    with pytest.raises(SystemExit, match="local Docker backend"):
        _validate_cli(_cli_args(workspace="flex"))


def test_validate_cli_rejects_multiple_attempts() -> None:
    from benchmarks.harbor.run_infer import _validate_cli

    with pytest.raises(SystemExit, match="max-attempts 1"):
        _validate_cli(_cli_args(max_attempts=3))


def test_validate_cli_rejects_non_pass_critic() -> None:
    from benchmarks.harbor.run_infer import _validate_cli

    with pytest.raises(SystemExit, match="pass"):
        _validate_cli(_cli_args(critic="finish_with_patch"))


def test_validate_cli_rejects_missing_tasks_path() -> None:
    from benchmarks.harbor.run_infer import _validate_cli

    with pytest.raises(SystemExit, match="tasks-path"):
        _validate_cli(_cli_args(tasks_path="/definitely/not/a/dir"))


def test_validate_cli_accepts_supported_defaults(tmp_path: Path) -> None:
    from benchmarks.harbor.run_infer import _validate_cli

    _validate_cli(_cli_args(tasks_path=str(tmp_path)))


def test_build_parser_defaults() -> None:
    from benchmarks.harbor.run_infer import build_parser

    args = build_parser().parse_args(["cfg.json", "--tasks-path", "/tmp/tasks"])
    assert args.dataset is None
    assert args.workspace == "docker"
    assert args.max_attempts == 1
    assert args.max_retries == 0
    assert not hasattr(args, "verifier_enabled")  # derived from --skip-verifier
    assert args.skip_verifier is False
    assert args.primary_reward_key == "reward"


def test_build_parser_accepts_harbor_options() -> None:
    from benchmarks.harbor.run_infer import build_parser

    args = build_parser().parse_args(
        [
            "cfg.json",
            "--tasks-path",
            "/tmp/tasks",
            "--docker-image-prefix",
            "mirror.example.org",
            "--agent-plugin-image",
            "plugin:latest",
            "--skip-verifier",
            "--primary-reward-key",
            "f2p",
            "--agent-timeout",
            "12",
            "--verifier-timeout",
            "34",
        ]
    )
    assert args.docker_image_prefix == "mirror.example.org"
    assert args.agent_plugin_image == "plugin:latest"
    assert args.skip_verifier is True
    assert args.primary_reward_key == "f2p"
    assert args.agent_timeout == 12.0
    assert args.verifier_timeout == 34.0
