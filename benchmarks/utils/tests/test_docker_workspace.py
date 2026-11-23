"""Tests for the benchmark Docker workspace startup handling."""

import pytest

from benchmarks.utils.docker_workspace import (
    DEFAULT_STARTUP_TIMEOUT_SEC,
    ENV_STARTUP_TIMEOUT,
    BenchDockerWorkspace,
    resolve_startup_timeout,
)
from openhands.workspace import DockerWorkspace


class _StubWorkspace(BenchDockerWorkspace):
    """BenchDockerWorkspace without pydantic init or container cleanup."""

    def __del__(self) -> None:  # pragma: no cover - test stub
        pass


def _workspace_stub(**fields: object) -> BenchDockerWorkspace:
    """Build an uninitialized workspace so no Docker daemon is touched."""
    workspace = object.__new__(_StubWorkspace)
    for key, value in fields.items():
        object.__setattr__(workspace, key, value)
    return workspace


@pytest.fixture(autouse=True)
def _no_container_start(monkeypatch):
    """Keep unit tests from starting a real container through model_post_init."""
    monkeypatch.setattr(DockerWorkspace, "model_post_init", lambda self, context: None)


def test_startup_timeout_default(monkeypatch) -> None:
    monkeypatch.delenv(ENV_STARTUP_TIMEOUT, raising=False)

    assert resolve_startup_timeout() == DEFAULT_STARTUP_TIMEOUT_SEC


def test_startup_timeout_env_override(monkeypatch) -> None:
    monkeypatch.setenv(ENV_STARTUP_TIMEOUT, "240")

    assert resolve_startup_timeout() == 240.0


def test_startup_timeout_explicit_value_wins(monkeypatch) -> None:
    monkeypatch.setenv(ENV_STARTUP_TIMEOUT, "240")

    assert resolve_startup_timeout(90.0) == 90.0


@pytest.mark.parametrize("raw", ["soon", "0", "-5"])
def test_invalid_startup_timeout_falls_back(monkeypatch, raw: str) -> None:
    monkeypatch.setenv(ENV_STARTUP_TIMEOUT, raw)

    assert resolve_startup_timeout() == DEFAULT_STARTUP_TIMEOUT_SEC


def test_health_wait_uses_configured_budget(monkeypatch) -> None:
    recorded: list[float] = []

    def fake_wait(self, timeout: float = 120.0) -> None:
        recorded.append(timeout)

    monkeypatch.setattr(DockerWorkspace, "_wait_for_health", fake_wait)
    workspace = _workspace_stub(startup_timeout_sec=42.0)

    # The SDK calls this without an argument (startup) and with 120s (resume);
    # both must use the benchmark budget.
    workspace._wait_for_health()
    workspace._wait_for_health(timeout=120.0)

    assert recorded == [42.0, 42.0]


def test_failed_start_cleans_up_and_includes_logs(monkeypatch) -> None:
    def boom(self, image, context) -> None:
        raise RuntimeError("container stopped unexpectedly")

    cleaned: list[bool] = []
    monkeypatch.setattr(DockerWorkspace, "_start_container", boom)
    monkeypatch.setattr(
        BenchDockerWorkspace,
        "_container_log_tail",
        staticmethod(lambda container_id, lines=20: "uvicorn: import failed"),
    )
    monkeypatch.setattr(
        BenchDockerWorkspace, "cleanup", lambda self: cleaned.append(True)
    )
    workspace = _workspace_stub(startup_timeout_sec=1.0)

    with pytest.raises(RuntimeError) as excinfo:
        workspace._start_container("image:tag", None)

    message = str(excinfo.value)
    assert "Agent-server container failed to start" in message
    assert "image:tag" in message
    assert "uvicorn: import failed" in message
    assert cleaned == [True]


def test_container_log_tail(monkeypatch) -> None:
    class Result:
        stdout = "line one\nline two"
        stderr = "stderr line"
        returncode = 0

    monkeypatch.setattr(
        "benchmarks.utils.docker_workspace.execute_command", lambda command: Result()
    )

    tail = BenchDockerWorkspace._container_log_tail("container-id", lines=5)

    assert "line two" in tail
    assert "stderr line" in tail
    assert BenchDockerWorkspace._container_log_tail(None) == ""
