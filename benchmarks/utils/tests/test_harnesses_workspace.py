"""Tests for the docker-CLI task-container workspace (no Docker needed)."""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.utils.harnesses import workspace as workspace_module
from benchmarks.utils.harnesses.workspace import (
    CommandResult,
    DockerContainerWorkspace,
    WorkspaceError,
)


class RecordedRun:
    def __init__(self) -> None:
        self.returncode = 0
        self.stdout = ""
        self.stderr = ""

    @property
    def ok(self) -> bool:
        return self.returncode == 0


def make_workspace(**overrides) -> DockerContainerWorkspace:
    defaults = dict(
        image="example/image:tag",
        container_name="harness-abc",
        container_id="container123",
        working_dir="/testbed",
        host_alias="host.docker.internal",
        gateway_ip="172.17.0.1",
        docker_bin="docker",
    )
    defaults.update(overrides)
    return DockerContainerWorkspace(**defaults)


def test_execute_command_builds_docker_exec_argv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded: list[list[str]] = []

    def fake_run(argv, **kwargs):
        recorded.append(argv)
        result = RecordedRun()
        result.stdout = "out"
        return result

    monkeypatch.setattr(workspace_module, "_run", fake_run)
    ws = make_workspace()
    result = ws.execute_command("pytest -q", cwd="/testbed")
    assert result.stdout == "out"
    assert recorded[0] == [
        "docker",
        "exec",
        "-w",
        "/testbed",
        "container123",
        "bash",
        "-lc",
        "pytest -q",
    ]


def test_execute_argv_passes_env(monkeypatch: pytest.MonkeyPatch) -> None:
    recorded: list[list[str]] = []

    def fake_run(argv, **kwargs):
        recorded.append(argv)
        return RecordedRun()

    monkeypatch.setattr(workspace_module, "_run", fake_run)
    ws = make_workspace()
    ws.execute_argv(["claude", "--version"], env={"A": "b", "C": "d e"})
    assert recorded[0] == [
        "docker",
        "exec",
        "-w",
        "/testbed",
        "-e",
        "A=b",
        "-e",
        "C=d e",
        "container123",
        "claude",
        "--version",
    ]


def test_start_pulls_missing_image_and_waits(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def fake_run(argv, **kwargs):
        calls.append(argv)
        result = RecordedRun()
        if argv[:2] == ["docker", "run"]:
            result.stdout = "container123\n"
        return result

    monkeypatch.setattr(workspace_module, "_run", fake_run)
    monkeypatch.setattr(
        workspace_module, "docker_image_exists", lambda image, **kw: False
    )
    monkeypatch.setattr(workspace_module, "docker_pull", lambda image, **kw: None)
    monkeypatch.setattr(workspace_module, "_is_running", lambda cid, **kw: True)
    monkeypatch.setattr(
        workspace_module, "docker_network_gateway", lambda **kw: "172.17.0.1"
    )
    resolved = []
    monkeypatch.setattr(
        DockerContainerWorkspace,
        "resolve_host_alias",
        lambda self: resolved.append(True) or "host.docker.internal",
    )

    ws = DockerContainerWorkspace.start(
        image="example/image:tag",
        working_dir="/testbed",
        bind_volumes=["/cache/dist:/opt/harness:ro"],
    )
    assert ws.container_id == "container123"
    assert resolved == [True]
    run_argv = next(argv for argv in calls if argv[:2] == ["docker", "run"])
    assert "--add-host" in run_argv
    assert "host.docker.internal:host-gateway" in run_argv
    assert "-v" in run_argv and "/cache/dist:/opt/harness:ro" in run_argv
    entrypoint = run_argv.index("--entrypoint")
    assert run_argv[entrypoint + 1 :] == [
        "sleep",
        "example/image:tag",
        "infinity",
    ]


def test_start_fails_when_image_missing_and_pull_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        workspace_module, "docker_image_exists", lambda image, **kw: False
    )
    with pytest.raises(WorkspaceError, match="skip-image-pull"):
        DockerContainerWorkspace.start(image="missing/image:tag", pull_missing=False)


def test_start_reports_container_startup_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(argv, **kwargs):
        result = RecordedRun()
        if argv[:2] == ["docker", "run"]:
            result.stdout = "container123\n"
        return result

    cleaned: list[str] = []
    monkeypatch.setattr(workspace_module, "_run", fake_run)
    monkeypatch.setattr(
        workspace_module, "docker_image_exists", lambda image, **kw: True
    )
    monkeypatch.setattr(workspace_module, "_is_running", lambda cid, **kw: False)
    monkeypatch.setattr(workspace_module, "_container_logs", lambda cid, **kw: "boom")
    monkeypatch.setattr(
        workspace_module, "_remove_container", lambda cid, **kw: cleaned.append(cid)
    )
    with pytest.raises(WorkspaceError, match="exited during startup"):
        DockerContainerWorkspace.start(image="example/image:tag", start_timeout=0.1)
    assert cleaned == ["container123"]


def test_resolve_host_alias_falls_back_to_gateway(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ws = make_workspace(host_alias="172.17.0.1")
    monkeypatch.setattr(
        DockerContainerWorkspace,
        "execute_command",
        lambda self, command, **kw: CommandResult(stdout="unknown\n"),
    )
    assert ws.resolve_host_alias() == "172.17.0.1"


def test_write_text_and_copy_in(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    recorded: list[list[str]] = []

    def fake_run(argv, **kwargs):
        recorded.append(argv)
        return RecordedRun()

    monkeypatch.setattr(workspace_module, "_run", fake_run)
    monkeypatch.setattr(
        DockerContainerWorkspace,
        "execute_command",
        lambda self, command, **kw: CommandResult(),
    )
    ws = make_workspace()
    ws.write_text("hello", "/tmp/harness/settings.json")
    ws.copy_in(tmp_path / "claude", "/opt/harness/claude")
    assert recorded[0][-3:] == ["sh", "-c", "cat > /tmp/harness/settings.json"]
    assert recorded[1] == [
        "docker",
        "cp",
        str(tmp_path / "claude"),
        "container123:/opt/harness/claude",
    ]


def test_cleanup_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    removed: list[str] = []
    monkeypatch.setattr(
        workspace_module, "_remove_container", lambda cid, **kw: removed.append(cid)
    )
    ws = make_workspace()
    ws.cleanup()
    ws.cleanup()
    assert removed == ["container123"]


def test_run_streaming_flushes_each_line_for_live_tailing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    log_path = tmp_path / "harness.jsonl"
    observed: list[str] = []

    class FakePopen:
        def __init__(self, argv, **kwargs):
            self.stdout = self._lines()
            self.returncode = 0

        def _lines(self):
            yield "first event\n"
            # Runs after run_streaming wrote the first line but before the
            # process exits: the file must already contain it.
            observed.append(log_path.read_text())
            yield "second event\n"

        def wait(self, timeout=None):
            return self.returncode

        def kill(self) -> None:
            pass

        def poll(self):
            return self.returncode

    monkeypatch.setattr(
        workspace_module.subprocess,
        "Popen",
        lambda argv, **kwargs: FakePopen(argv, **kwargs),
    )
    ws = make_workspace()
    ws.run_streaming(
        ["/opt/harness/claude", "-p", "hi"],
        cwd="/testbed",
        log_path=log_path,
        timeout_sec=30,
    )
    assert observed == ["first event\n"]
    assert log_path.read_text() == "first event\nsecond event\n"


def test_run_streaming_streams_to_log_and_wraps_timeout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}

    class FakePopen:
        def __init__(self, argv, **kwargs):
            captured["argv"] = argv
            captured["popen"] = True
            self.stdout = iter(["line one\n", "line two\n"])
            self.returncode = 0

        def wait(self, timeout=None):
            captured["wait_timeout"] = timeout
            return self.returncode

        def kill(self) -> None:
            captured["killed"] = True

        def poll(self):
            return self.returncode

    monkeypatch.setattr(
        workspace_module.subprocess,
        "Popen",
        lambda argv, **kwargs: FakePopen(argv, **kwargs),
    )
    ws = make_workspace()
    log_path = tmp_path / "harness.jsonl"
    outcome = ws.run_streaming(
        ["/opt/harness/claude", "-p", "hi"],
        cwd="/testbed",
        env={"A": "b"},
        log_path=log_path,
        timeout_sec=30,
    )
    argv = captured["argv"]
    assert isinstance(argv, list)
    assert argv[argv.index("container123") + 1 :][:4] == [
        "timeout",
        "-k",
        "10",
        "30",
    ]
    assert captured["wait_timeout"] == 150
    assert log_path.read_text() == "line one\nline two\n"
    assert outcome.exit_code == 0
    assert outcome.timed_out is False
    assert "line one" in outcome.tail
