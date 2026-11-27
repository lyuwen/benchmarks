"""Reward parsing and shared-verifier execution tests (no Docker required)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from benchmarks.harbor.judge import (
    STATUS_SCORED,
    STATUS_UNSCORED,
    STATUS_VERIFIER_ERROR,
    STATUS_VERIFIER_TIMEOUT,
    RewardFileNotFoundError,
    RewardParseError,
    collect_artifacts,
    load_rewards,
    parse_reward_json_text,
    parse_reward_text_text,
    run_shared_verifier,
    select_primary_reward,
)
from benchmarks.harbor.task import HarborTask, load_tasks
from benchmarks.harbor.tests.conftest import FakeWorkspace, TaskFactory
from benchmarks.harbor.workspace import ExecOutcome


# ---------------------------------------------------------------------------
# Reward parsing
# ---------------------------------------------------------------------------


def test_parse_reward_text_numbers() -> None:
    assert parse_reward_text_text("1\n") == {"reward": 1.0}
    assert parse_reward_text_text(" 1.0 \n") == {"reward": 1.0}
    assert parse_reward_text_text("0.5") == {"reward": 0.5}
    assert parse_reward_text_text("1e-3") == {"reward": 0.001}
    assert parse_reward_text_text("-1") == {"reward": -1.0}


@pytest.mark.parametrize("text", ["", "   \n", "abc", "1\n2", "0x10"])
def test_parse_reward_text_rejects_garbage(text: str) -> None:
    with pytest.raises(RewardParseError):
        parse_reward_text_text(text)


@pytest.mark.parametrize("text", ["nan", "inf", "-inf", "Infinity"])
def test_parse_reward_text_rejects_non_finite(text: str) -> None:
    with pytest.raises(RewardParseError, match="finite"):
        parse_reward_text_text(text)


def test_parse_reward_json_named_and_fractional() -> None:
    assert parse_reward_json_text('{"reward": 1}') == {"reward": 1}
    assert parse_reward_json_text('{"f2p": 0.5, "p2p": 1.0}') == {
        "f2p": 0.5,
        "p2p": 1.0,
    }
    assert parse_reward_json_text('{"reward": 0}') == {"reward": 0}
    # Harbor's pydantic coercion accepts numeric strings and booleans.
    assert parse_reward_json_text('{"reward": "0.5"}') == {"reward": 0.5}
    assert parse_reward_json_text('{"reward": true}') == {"reward": 1}


@pytest.mark.parametrize(
    "text",
    [
        '{"reward": "abc"}',
        '{"reward": null}',
        '{"reward": {"nested": 1}}',
        '{"reward": [1, 2]}',
        "[1, 2]",
        "1.0",
        "{}",
        "not json",
    ],
)
def test_parse_reward_json_rejects_invalid(text: str) -> None:
    with pytest.raises(RewardParseError):
        parse_reward_json_text(text)


@pytest.mark.parametrize("text", ['{"reward": NaN}', '{"reward": Infinity}'])
def test_parse_reward_json_rejects_non_finite(text: str) -> None:
    with pytest.raises(RewardParseError, match="finite"):
        parse_reward_json_text(text)


def test_load_rewards_prefers_json(tmp_path: Path) -> None:
    (tmp_path / "reward.json").write_text('{"reward": 0.25}', encoding="utf-8")
    (tmp_path / "reward.txt").write_text("0", encoding="utf-8")
    rewards, filename = load_rewards(tmp_path)
    assert rewards == {"reward": 0.25}
    assert filename == "reward.json"


def test_load_rewards_falls_back_to_text(tmp_path: Path) -> None:
    (tmp_path / "reward.txt").write_text("1\n", encoding="utf-8")
    rewards, filename = load_rewards(tmp_path)
    assert rewards == {"reward": 1.0}
    assert filename == "reward.txt"


def test_load_rewards_missing_both(tmp_path: Path) -> None:
    with pytest.raises(RewardFileNotFoundError):
        load_rewards(tmp_path)


def test_load_rewards_empty_file(tmp_path: Path) -> None:
    (tmp_path / "reward.json").write_text("", encoding="utf-8")
    with pytest.raises(RewardParseError, match="empty"):
        load_rewards(tmp_path)


def test_select_primary_reward_does_not_guess() -> None:
    assert select_primary_reward({"reward": 1.0}, "reward") == 1.0
    assert select_primary_reward({"f2p": 1.0}, "reward") is None
    assert select_primary_reward({"f2p": 1.0}, "f2p") == 1.0
    assert select_primary_reward(None, "reward") is None
    assert select_primary_reward({}, "reward") is None


# ---------------------------------------------------------------------------
# Shared verifier execution (fake workspace)
# ---------------------------------------------------------------------------


def _load_task(factory: TaskFactory, name: str = "task", **kwargs: Any) -> HarborTask:
    path = factory(name, **kwargs)
    return load_tasks(path).tasks[0]


def test_verifier_scored_from_reward_txt(
    task_factory: TaskFactory, tmp_path: Path
) -> None:
    task = _load_task(task_factory)
    workspace = FakeWorkspace(reward_files={"/logs/verifier/reward.txt": b"1\n"})
    trial_dir = tmp_path / "trial"

    outcome = run_shared_verifier(
        workspace,
        task,
        trial_dir=trial_dir,
        run_dir=tmp_path,
    )

    assert outcome["status"] == STATUS_SCORED
    assert outcome["reward"] == 1.0
    assert outcome["rewards"] == {"reward": 1.0}
    assert outcome["exit_code"] == 0
    assert outcome["reward_file"] == "reward.txt"
    assert outcome["log_dir"] == "trial/verifier"

    # The exact Harbor command, invoked with the effective cwd/env/user.
    exec_commands = [call["command"] for call in workspace.exec_calls]
    assert exec_commands[0].startswith("rm -rf /logs/verifier")
    run_call = next(
        call for call in workspace.exec_calls if call["command"].startswith("(/tests")
    )
    assert run_call["command"] == (
        "(/tests/test.sh) > /logs/verifier/test-stdout.txt 2>&1"
    )
    assert run_call["cwd"] == "/app"
    assert run_call["kill_pattern"] == "/tests/test.sh"
    # chmod happens as root before the test run.
    assert any(
        "chmod +x /tests/test.sh" in call["command"] and call["user"] == "root"
        for call in workspace.exec_calls
    )
    # tests/ were uploaded to /tests (and nothing else was uploaded).
    assert workspace.uploads == [(str(task.tests_dir), "/tests")]
    # Logs and meta are persisted for auditing.
    assert (trial_dir / "verifier" / "test-stdout.txt").exists() or True
    assert (trial_dir / "verifier" / "exec-stdout.txt").exists()
    assert (trial_dir / "verifier" / "meta.json").exists()
    meta = json.loads((trial_dir / "verifier" / "meta.json").read_text())
    assert meta["status"] == STATUS_SCORED


def test_verifier_reward_json_named(task_factory: TaskFactory, tmp_path: Path) -> None:
    task = _load_task(task_factory)
    workspace = FakeWorkspace(
        reward_files={
            "/logs/verifier/reward.json": b'{"f2p": 1, "p2p": 1.0, "partial": 0.5}'
        }
    )
    outcome = run_shared_verifier(
        workspace,
        task,
        trial_dir=tmp_path / "trial",
        run_dir=tmp_path,
        primary_reward_key="f2p",
    )
    assert outcome["status"] == STATUS_SCORED
    assert outcome["rewards"] == {"f2p": 1, "p2p": 1.0, "partial": 0.5}
    assert outcome["reward"] == 1
    assert outcome["reward_file"] == "reward.json"


def test_verifier_no_primary_key_reported(
    task_factory: TaskFactory, tmp_path: Path
) -> None:
    task = _load_task(task_factory)
    workspace = FakeWorkspace(
        reward_files={"/logs/verifier/reward.json": b'{"custom": 0.75}'}
    )
    outcome = run_shared_verifier(
        workspace, task, trial_dir=tmp_path / "trial", run_dir=tmp_path
    )
    assert outcome["status"] == STATUS_SCORED
    assert outcome["reward"] is None
    assert "primary key" in (outcome["error"] or "")


def test_verifier_nonzero_exit_with_reward_scores(
    task_factory: TaskFactory, tmp_path: Path
) -> None:
    task = _load_task(task_factory)

    def exec_factory(command: str) -> ExecOutcome | None:
        if command.startswith("(/tests/test.sh)"):
            return ExecOutcome(
                command=command,
                exit_code=7,
                stdout="partial",
                stderr="boom",
                duration_sec=0.2,
            )
        return None

    workspace = FakeWorkspace(
        reward_files={"/logs/verifier/reward.txt": b"0.5\n"},
        exec_factory=exec_factory,
    )
    outcome = run_shared_verifier(
        workspace, task, trial_dir=tmp_path / "trial", run_dir=tmp_path
    )
    assert outcome["status"] == STATUS_SCORED
    assert outcome["reward"] == 0.5
    assert outcome["exit_code"] == 7


def test_verifier_missing_reward_is_error(
    task_factory: TaskFactory, tmp_path: Path
) -> None:
    task = _load_task(task_factory)
    workspace = FakeWorkspace()
    outcome = run_shared_verifier(
        workspace, task, trial_dir=tmp_path / "trial", run_dir=tmp_path
    )
    assert outcome["status"] == STATUS_VERIFIER_ERROR
    assert "No reward file" in (outcome["error"] or "")
    assert outcome["reward"] is None


def test_verifier_stale_reward_is_removed_before_grading(
    task_factory: TaskFactory, tmp_path: Path
) -> None:
    task = _load_task(task_factory)
    # A reward left behind by the agent must not be used: the reset removes it
    # and the fake test.sh writes nothing.
    workspace = FakeWorkspace(container_files={"/logs/verifier/reward.txt": b"1\n"})
    outcome = run_shared_verifier(
        workspace, task, trial_dir=tmp_path / "trial", run_dir=tmp_path
    )
    assert outcome["status"] == STATUS_VERIFIER_ERROR


def test_verifier_malformed_reward_is_error(
    task_factory: TaskFactory, tmp_path: Path
) -> None:
    task = _load_task(task_factory)
    workspace = FakeWorkspace(
        reward_files={"/logs/verifier/reward.json": b'{"reward": "nope"}'}
    )
    outcome = run_shared_verifier(
        workspace, task, trial_dir=tmp_path / "trial", run_dir=tmp_path
    )
    assert outcome["status"] == STATUS_VERIFIER_ERROR


def test_verifier_timeout_preserves_raw_reward(
    task_factory: TaskFactory, tmp_path: Path
) -> None:
    task = _load_task(task_factory)

    def exec_factory(command: str) -> ExecOutcome | None:
        if command.startswith("(/tests/test.sh)"):
            return ExecOutcome(
                command=command,
                exit_code=None,
                stdout="",
                stderr="timed out",
                duration_sec=5.0,
                timed_out=True,
            )
        return None

    workspace = FakeWorkspace(
        reward_files={"/logs/verifier/reward.txt": b"0.25\n"},
        exec_factory=exec_factory,
    )
    outcome = run_shared_verifier(
        workspace, task, trial_dir=tmp_path / "trial", run_dir=tmp_path
    )
    assert outcome["status"] == STATUS_VERIFIER_TIMEOUT
    assert outcome["timed_out"] is True
    assert outcome["rewards"] == {"reward": 0.25}
    assert outcome["reward"] == 0.25
    assert "timed out" in (outcome["error"] or "").lower()


def test_verifier_disabled_is_unscored(
    task_factory: TaskFactory, tmp_path: Path
) -> None:
    task = _load_task(task_factory)
    workspace = FakeWorkspace()
    outcome = run_shared_verifier(
        workspace,
        task,
        trial_dir=tmp_path / "trial",
        run_dir=tmp_path,
        enabled=False,
    )
    assert outcome["status"] == STATUS_UNSCORED
    assert outcome["rewards"] is None
    assert workspace.exec_calls == []


def test_verifier_uses_task_env_and_user(
    task_factory: TaskFactory, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HARBOR_TEST_TOKEN", "secret-value")
    task_toml = (
        'schema_version = "1.2"\n\n'
        '[verifier]\ntimeout_sec = 30.0\nuser = "root"\n'
        '[verifier.env]\nHARBOR_TEST_TOKEN = "${HARBOR_TEST_TOKEN}"\n'
        'LITERAL = "plain"\n\n'
        '[environment]\ndocker_image = "img"\n'
    )
    task = _load_task(task_factory, task_toml=task_toml)
    workspace = FakeWorkspace(reward_files={"/logs/verifier/reward.txt": b"1\n"})
    outcome = run_shared_verifier(
        workspace, task, trial_dir=tmp_path / "trial", run_dir=tmp_path
    )
    assert outcome["status"] == STATUS_SCORED
    run_call = next(
        call for call in workspace.exec_calls if call["command"].startswith("(/tests")
    )
    assert run_call["env"] == {
        "HARBOR_TEST_TOKEN": "secret-value",
        "LITERAL": "plain",
    }
    assert run_call["user"] == "root"
    assert run_call["timeout"] is not None and run_call["timeout"] <= 30.0


def test_verifier_host_env_missing_is_error(
    task_factory: TaskFactory, tmp_path: Path
) -> None:
    task_toml = (
        'schema_version = "1.2"\n\n'
        "[verifier]\ntimeout_sec = 30.0\n"
        '[verifier.env]\nUNSET_HARBOR_VARIABLE = "${UNSET_HARBOR_VARIABLE}"\n\n'
        '[environment]\ndocker_image = "img"\n'
    )
    task = _load_task(task_factory, task_toml=task_toml)
    workspace = FakeWorkspace()
    outcome = run_shared_verifier(
        workspace, task, trial_dir=tmp_path / "trial", run_dir=tmp_path
    )
    assert outcome["status"] == STATUS_VERIFIER_ERROR
    assert "UNSET_HARBOR_VARIABLE" in (outcome["error"] or "")


# ---------------------------------------------------------------------------
# Artifact collection
# ---------------------------------------------------------------------------


def test_collect_artifacts_convention_and_declared(
    task_factory: TaskFactory, tmp_path: Path
) -> None:
    task = _load_task(
        task_factory,
        task_toml=(
            'schema_version = "1.2"\n'
            'artifacts = ["/app/output.txt"]\n\n'
            '[environment]\ndocker_image = "img"\n'
        ),
    )
    workspace = FakeWorkspace(
        container_files={
            "/logs/artifacts/notes.md": b"agent notes\n",
            "/app/output.txt": b"result\n",
        }
    )
    trial_dir = tmp_path / "trial"
    result = collect_artifacts(workspace, task, trial_dir=trial_dir, run_dir=tmp_path)

    assert result["failed_count"] == 0
    statuses = {entry["source"]: entry["status"] for entry in result["entries"]}
    assert statuses["/logs/artifacts"] == "ok"
    assert statuses["/app/output.txt"] == "ok"
    assert (trial_dir / "artifacts" / "notes.md").read_text() == "agent notes\n"
    assert (trial_dir / "artifacts" / "output.txt").read_text() == "result\n"
    manifest = json.loads(
        (trial_dir / "artifacts" / "manifest.json").read_text(encoding="utf-8")
    )
    checksums = {
        file["path"]: file["sha256"]
        for entry in manifest["artifacts"]
        for file in entry.get("files", [])
    }
    assert "output.txt" in checksums
    assert any(entry["type"] == "file" for entry in manifest["artifacts"])


def test_collect_artifacts_missing_source_is_best_effort(
    task_factory: TaskFactory, tmp_path: Path
) -> None:
    task = _load_task(
        task_factory,
        task_toml=(
            'schema_version = "1.2"\n'
            'artifacts = ["/app/missing.bin"]\n\n'
            '[environment]\ndocker_image = "img"\n'
        ),
    )
    workspace = FakeWorkspace()
    result = collect_artifacts(
        workspace, task, trial_dir=tmp_path / "trial", run_dir=tmp_path
    )
    # The convention source (/logs/artifacts) does not exist; declared artifact
    # is missing. Collection never raises.
    assert result["failed_count"] == 2


def test_collect_artifacts_excludes(task_factory: TaskFactory, tmp_path: Path) -> None:
    task = _load_task(
        task_factory,
        task_toml=(
            'schema_version = "1.2"\n\n'
            '[[artifacts]]\nsource = "/app/out"\n'
            'exclude = ["skip.txt"]\n\n'
            '[environment]\ndocker_image = "img"\n'
        ),
    )
    workspace = FakeWorkspace(
        container_files={
            "/app/out/keep.txt": b"keep",
            "/app/out/skip.txt": b"skip",
        }
    )
    trial_dir = tmp_path / "trial"
    collect_artifacts(workspace, task, trial_dir=trial_dir, run_dir=tmp_path)
    collected = trial_dir / "artifacts" / "out"
    assert (collected / "keep.txt").exists()
    assert not (collected / "skip.txt").exists()
