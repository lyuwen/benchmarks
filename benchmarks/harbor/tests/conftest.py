"""Shared fixtures for the Harbor adapter tests.

Every fixture builds a *real* Harbor task directory on disk so the loader is
exercised exactly as it is in production.  Nothing here needs Docker or an LLM.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any, Callable

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_TASK_TOML = """\
schema_version = "1.2"

[verifier]
timeout_sec = 60.0

[agent]
timeout_sec = 60.0

[environment]
docker_image = "example/task:latest"
"""

DEFAULT_TEST_SH = """\
#!/bin/bash
set -uo pipefail
mkdir -p /logs/verifier
echo 1 > /logs/verifier/reward.txt
"""

TaskFactory = Callable[..., Path]


def write_task(
    root: Path,
    name: str,
    *,
    task_toml: str | None = None,
    instruction: str | None = "Do the thing.",
    tests: dict[str, str] | None = None,
    environment_files: dict[str, str] | None = None,
    dockerfile: str | None = None,
    extra_files: dict[str, str] | None = None,
) -> Path:
    """Create one Harbor task directory and return its path."""
    task_dir = root / name
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "task.toml").write_text(
        task_toml if task_toml is not None else DEFAULT_TASK_TOML,
        encoding="utf-8",
    )
    if instruction is not None:
        (task_dir / "instruction.md").write_text(instruction, encoding="utf-8")
    tests_dir = task_dir / "tests"
    tests_dir.mkdir(exist_ok=True)
    test_files = {"test.sh": DEFAULT_TEST_SH} if tests is None else tests
    for filename, content in test_files.items():
        path = tests_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    environment_dir = task_dir / "environment"
    environment_dir.mkdir(exist_ok=True)
    if dockerfile is not None:
        (environment_dir / "Dockerfile").write_text(dockerfile, encoding="utf-8")
    for filename, content in (environment_files or {}).items():
        path = environment_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    for filename, content in (extra_files or {}).items():
        path = task_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return task_dir


@pytest.fixture
def task_factory(tmp_path: Path) -> TaskFactory:
    """Return a factory that creates tasks under a per-test temporary root."""

    def factory(name: str, **kwargs: Any) -> Path:
        return write_task(tmp_path, name, **kwargs)

    return factory


@pytest.fixture
def tasks_dir(tmp_path: Path) -> Path:
    """A directory that holds one or more task directories."""
    path = tmp_path / "tasks"
    path.mkdir()
    return path


@pytest.fixture
def dataset_root(tmp_path: Path) -> Path:
    root = tmp_path / "dataset"
    root.mkdir()
    return root


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    """A minimal persisted harbor-infer run directory."""
    path = tmp_path / "run"
    path.mkdir()
    (path / "harbor_tasks.json").write_text(
        json.dumps(
            {
                "reference_version": "0.9.0",
                "dataset_revision": "deadbeef",
                "manifest_hash": "manifest-hash",
                "task_count": 3,
                "tasks": [
                    {"id": "task-a", "relative_path": "task-a", "fingerprint": "a"},
                    {"id": "task-b", "relative_path": "task-b", "fingerprint": "b"},
                    {"id": "task-c", "relative_path": "task-c", "fingerprint": "c"},
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def scored_record(
    instance_id: str,
    *,
    attempt: int = 1,
    reward: float | int = 1.0,
    rewards: dict[str, float | int] | None = None,
) -> dict[str, Any]:
    return {
        "instance_id": instance_id,
        "attempt": attempt,
        "test_result": {
            "verifier": {
                "status": "scored",
                "reward": reward,
                "rewards": rewards if rewards is not None else {"reward": reward},
                "exit_code": 0,
                "duration_sec": 1.0,
            },
            "agent": {"status": "completed", "duration_sec": 2.0},
            "trial": {"trial_dir": f"trials/{instance_id}/attempt-{attempt}/exec"},
            "task": {"relative_path": instance_id},
        },
        "error": None,
    }


def errored_record(instance_id: str, *, attempt: int = 1) -> dict[str, Any]:
    return {
        "instance_id": instance_id,
        "attempt": attempt,
        "test_result": {},
        "error": "workspace failed to start",
    }


def verifier_error_record(instance_id: str, *, attempt: int = 1) -> dict[str, Any]:
    return {
        "instance_id": instance_id,
        "attempt": attempt,
        "test_result": {
            "verifier": {
                "status": "verifier_error",
                "reward": None,
                "rewards": None,
                "error": "No reward file found",
                "exit_code": 0,
                "duration_sec": 1.0,
            },
            "agent": {"status": "completed"},
            "trial": {"trial_dir": f"trials/{instance_id}/attempt-{attempt}/exec"},
            "task": {"relative_path": instance_id},
        },
        "error": None,
    }


def dedent(text: str) -> str:
    return textwrap.dedent(text).lstrip("\n")


class FakeWorkspace:
    """In-memory stand-in for :class:`HarborDockerWorkspace` in unit tests.

    It implements the subset of the workspace protocol used by the verifier and
    artifact adapters: ``docker_exec``, ``upload_dir``, ``download_dir``,
    ``download_file``, ``download_dir_with_exclusions``, ``is_dir`` and
    ``path_exists``.  Container paths are represented as a flat ``path -> bytes``
    mapping so tests can assert on exactly what the adapter does.
    """

    working_dir = "/app"

    def __init__(
        self,
        *,
        container_files: dict[str, bytes] | None = None,
        reward_files: dict[str, bytes] | None = None,
        exec_factory: Callable[[str], Any] | None = None,
    ) -> None:
        from benchmarks.harbor.workspace import ExecOutcome

        self.files: dict[str, bytes] = dict(container_files or {})
        self.reward_files = dict(reward_files or {})
        self.exec_factory = exec_factory
        self._default_outcome = ExecOutcome(
            command="", exit_code=0, stdout="", stderr="", duration_sec=0.05
        )
        self.exec_calls: list[dict[str, Any]] = []
        self.uploads: list[tuple[str, str]] = []
        self.downloads: list[tuple[str, str]] = []

    # -- helpers -------------------------------------------------------

    def _is_dir(self, path: str) -> bool:
        path = path.rstrip("/")
        return any(candidate.startswith(path + "/") for candidate in self.files)

    def _children(self, directory: str) -> dict[str, bytes]:
        prefix = directory.rstrip("/") + "/"
        return {
            path: data for path, data in self.files.items() if path.startswith(prefix)
        }

    def _remove(self, path: str) -> None:
        path = path.rstrip("/")
        for candidate in list(self.files):
            if candidate == path or candidate.startswith(path + "/"):
                del self.files[candidate]

    # -- workspace protocol -------------------------------------------

    def docker_exec(
        self,
        command: str,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        user: str | int | None = None,
        timeout: float | None = None,
        kill_pattern: str | None = None,
    ) -> Any:
        from benchmarks.harbor.workspace import ExecOutcome

        self.exec_calls.append(
            {
                "command": command,
                "cwd": cwd,
                "env": dict(env or {}),
                "user": user,
                "timeout": timeout,
                "kill_pattern": kill_pattern,
            }
        )
        if command.startswith("rm -rf /logs/verifier"):
            self._remove("/logs/verifier")
            self.files["/logs/verifier/.keep"] = b""
        if command.startswith("rm -rf /tests"):
            self._remove("/tests")
        if command.startswith("(/tests/test.sh)"):
            self.files.update(self.reward_files)
        if self.exec_factory is not None:
            outcome = self.exec_factory(command)
            if outcome is not None:
                return outcome
        return ExecOutcome(
            command=command,
            exit_code=self._default_outcome.exit_code,
            stdout=self._default_outcome.stdout,
            stderr=self._default_outcome.stderr,
            duration_sec=self._default_outcome.duration_sec,
            timed_out=self._default_outcome.timed_out,
        )

    def upload_dir(self, source_dir: str | Path, target_dir: str) -> None:
        source = Path(source_dir)
        self.uploads.append((str(source), target_dir))
        for path in source.rglob("*"):
            if path.is_file():
                relative = path.relative_to(source).as_posix()
                self.files[f"{target_dir.rstrip('/')}/{relative}"] = path.read_bytes()

    def download_dir(self, source_dir: str, target_dir: str | Path) -> None:
        self.downloads.append((source_dir, str(target_dir)))
        children = self._children(source_dir)
        if not children:
            raise FileNotFoundError(f"no such directory in container: {source_dir}")
        target = Path(target_dir)
        target.mkdir(parents=True, exist_ok=True)
        prefix = source_dir.rstrip("/") + "/"
        for path, data in children.items():
            destination = target / path[len(prefix) :]
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)

    def download_file(self, source_path: str, target_path: str | Path) -> None:
        self.downloads.append((source_path, str(target_path)))
        if source_path not in self.files:
            raise FileNotFoundError(f"no such file in container: {source_path}")
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(self.files[source_path])

    def download_dir_with_exclusions(
        self, source_dir: str, target_dir: str | Path, exclude: list[str]
    ) -> None:
        from fnmatch import fnmatch

        children = self._children(source_dir)
        if not children:
            raise FileNotFoundError(f"no such directory in container: {source_dir}")
        target = Path(target_dir)
        target.mkdir(parents=True, exist_ok=True)
        prefix = source_dir.rstrip("/") + "/"
        for path, data in children.items():
            relative = path[len(prefix) :]
            if any(fnmatch(relative, pattern) for pattern in exclude):
                continue
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)

    def is_dir(self, remote_path: str) -> bool:
        return self._is_dir(remote_path)

    def path_exists(self, remote_path: str) -> bool:
        path = remote_path.rstrip("/")
        return path in self.files or any(
            candidate.startswith(path + "/") for candidate in self.files
        )
