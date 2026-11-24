"""A plain Docker container workspace driven by the docker CLI.

The benchmark's OpenHands/SDK workspaces talk to an agent server inside the
image. A blackbox harness needs the opposite: the *task* container itself, with
the official upstream image (no agent server) and the harness executing inside
it. This module provides the small amount of Docker plumbing that requires:
start/stop a long-lived container, run commands with an environment, stream
output to a host file, and copy files in and out.

It intentionally mirrors the subset of the SDK ``RemoteWorkspace`` surface that
the evaluation orchestrator touches (``execute_command``, ``file_download``,
``__exit__``) so the generic retry/resume machinery keeps working.
"""

from __future__ import annotations

import logging
import shlex
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path


logger = logging.getLogger(__name__)

DEFAULT_DOCKER_BIN = "docker"
DEFAULT_LINUX_BRIDGE_GATEWAY = "172.17.0.1"
CONTAINER_START_TIMEOUT_SEC = 180.0
LOG_TAIL_CHARS = 8000
STREAM_TAIL_LINES = 300


class WorkspaceError(RuntimeError):
    """The task container could not be created or driven."""


@dataclass
class CommandResult:
    """Result of a command executed inside the task container."""

    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    timeout_occurred: bool = False

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timeout_occurred


@dataclass
class ProcessOutcome:
    """Result of a long-running, streamed harness process."""

    exit_code: int
    duration_seconds: float
    timed_out: bool
    log_path: str
    tail: str


def _run(
    argv: list[str],
    *,
    timeout: float | None = None,
    input_text: str | None = None,
    check: bool = False,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            argv,
            capture_output=True,
            text=True,
            input=input_text,
            timeout=timeout,
            check=check,
        )
    except subprocess.TimeoutExpired as exc:  # pragma: no cover - host backstop
        raise WorkspaceError(f"docker command timed out: {' '.join(argv)}") from exc
    except FileNotFoundError as exc:
        raise WorkspaceError(
            f"Docker CLI {argv[0]!r} is not available; harness runs need Docker"
        ) from exc


def docker_image_exists(image: str, *, docker_bin: str = DEFAULT_DOCKER_BIN) -> bool:
    result = _run([docker_bin, "image", "inspect", image, "--format", "{{.Id}}"])
    return result.returncode == 0


def docker_pull(image: str, *, docker_bin: str = DEFAULT_DOCKER_BIN) -> None:
    logger.info("Pulling task image %s (missing locally)", image)
    result = _run([docker_bin, "pull", image], timeout=3600)
    if result.returncode != 0:
        raise WorkspaceError(
            f"Failed to pull task image {image}: {(result.stderr or '').strip()[:2000]}"
        )


def docker_network_gateway(
    network: str = "bridge", *, docker_bin: str = DEFAULT_DOCKER_BIN
) -> str | None:
    result = _run(
        [
            docker_bin,
            "network",
            "inspect",
            network,
            "--format",
            "{{(index .IPAM.Config 0).Gateway}}",
        ]
    )
    if result.returncode != 0:
        return None
    gateway = (result.stdout or "").strip()
    return gateway or None


class DockerContainerWorkspace:
    """A running task container used to execute the harness inside the image."""

    def __init__(
        self,
        *,
        image: str,
        container_name: str,
        container_id: str,
        working_dir: str,
        host_alias: str,
        gateway_ip: str | None,
        docker_bin: str = DEFAULT_DOCKER_BIN,
    ) -> None:
        self.image = image
        self.container_name = container_name
        self.container_id = container_id
        self.working_dir = working_dir
        self.host_alias = host_alias
        self.gateway_ip = gateway_ip
        self.docker_bin = docker_bin
        self._cleaned_up = False

    # -------------------- lifecycle ---------------------------------------

    @classmethod
    def start(
        cls,
        *,
        image: str,
        working_dir: str = "/",
        pull_missing: bool = True,
        bind_volumes: list[str] | None = None,
        docker_bin: str = DEFAULT_DOCKER_BIN,
        start_timeout: float = CONTAINER_START_TIMEOUT_SEC,
    ) -> "DockerContainerWorkspace":
        """Create and start a long-lived container from ``image``."""
        if not docker_image_exists(image, docker_bin=docker_bin):
            if not pull_missing:
                raise WorkspaceError(
                    f"Task image {image} is not available locally and "
                    "--skip-image-pull was set"
                )
            docker_pull(image, docker_bin=docker_bin)

        name = f"harness-{uuid.uuid4().hex[:12]}"
        argv = [
            docker_bin,
            "run",
            "-d",
            "--name",
            name,
            "--add-host",
            "host.docker.internal:host-gateway",
        ]
        for volume in bind_volumes or []:
            argv.extend(["-v", volume])
        argv.extend(["--entrypoint", "sleep", image, "infinity"])

        logger.info("Starting task container %s from %s", name, image)
        result = _run(argv, timeout=300)
        if result.returncode != 0:
            raise WorkspaceError(
                f"Failed to start task container from {image}: "
                f"{(result.stderr or '').strip()[:2000]}"
            )
        container_id = (result.stdout or "").strip()
        if not container_id:
            raise WorkspaceError(f"Docker did not return a container id for {image}")

        deadline = time.monotonic() + start_timeout
        running = False
        while time.monotonic() < deadline:
            running = _is_running(container_id, docker_bin=docker_bin)
            if running:
                break
            time.sleep(0.2)
        if not running:
            logs = _container_logs(container_id, docker_bin=docker_bin)
            _remove_container(container_id, docker_bin=docker_bin)
            raise WorkspaceError(
                f"Task container from {image} exited during startup.\n"
                f"Last container log lines:\n{logs}"
            )

        gateway = docker_network_gateway(docker_bin=docker_bin)
        workspace = cls(
            image=image,
            container_name=name,
            container_id=container_id,
            working_dir=working_dir,
            host_alias=gateway or DEFAULT_LINUX_BRIDGE_GATEWAY,
            gateway_ip=gateway,
            docker_bin=docker_bin,
        )
        workspace.host_alias = workspace.resolve_host_alias()
        logger.info(
            "Task container %s (%s) ready; host alias %s",
            name,
            container_id[:12],
            workspace.host_alias,
        )
        return workspace

    def resolve_host_alias(self) -> str:
        """Name/address the container uses to reach services on the host."""
        result = self.execute_command(
            "getent hosts host.docker.internal >/dev/null 2>&1 && "
            "echo host.docker.internal || echo unknown",
            timeout=30,
        )
        alias = (result.stdout or "").strip()
        if result.exit_code == 0 and alias and alias != "unknown":
            return "host.docker.internal"
        if self.gateway_ip:
            return self.gateway_ip
        raise WorkspaceError(
            "Cannot determine how the task container reaches the host; pass "
            "--rollout-bind-host and use a local Docker daemon"
        )

    def cleanup(self) -> None:
        if self._cleaned_up:
            return
        self._cleaned_up = True
        _remove_container(self.container_id, docker_bin=self.docker_bin)

    def __enter__(self) -> "DockerContainerWorkspace":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.cleanup()

    # -------------------- command execution --------------------------------

    def _exec_argv(
        self,
        argv: list[str],
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> list[str]:
        command = [self.docker_bin, "exec"]
        if cwd:
            command.extend(["-w", cwd])
        for key, value in (env or {}).items():
            command.extend(["-e", f"{key}={value}"])
        command.append(self.container_id)
        command.extend(argv)
        return command

    def execute_command(
        self,
        command: str,
        cwd: str | Path | None = None,
        timeout: float = 60.0,
    ) -> CommandResult:
        """Run a shell command inside the container (like ``RemoteWorkspace``)."""
        argv = self._exec_argv(
            ["bash", "-lc", command], cwd=str(cwd) if cwd else self.working_dir
        )
        result = _run(argv, timeout=timeout)
        return CommandResult(
            stdout=result.stdout or "",
            stderr=result.stderr or "",
            exit_code=result.returncode,
        )

    def execute_argv(
        self,
        argv: list[str],
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout: float = 60.0,
    ) -> CommandResult:
        """Run a program directly (no shell) inside the container."""
        result = _run(
            self._exec_argv(argv, cwd=cwd or self.working_dir, env=env),
            timeout=timeout,
        )
        return CommandResult(
            stdout=result.stdout or "",
            stderr=result.stderr or "",
            exit_code=result.returncode,
        )

    def run_streaming(
        self,
        argv: list[str],
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        log_path: str | Path,
        timeout_sec: float | None = None,
        container_timeout: bool = True,
        deadline_grace_sec: float = 120.0,
    ) -> ProcessOutcome:
        """Run a long-lived process, streaming its output to a host file.

        The process is wrapped in the container's ``timeout`` when a wall-clock
        budget is configured, so it cannot outlive the benchmark's conversation
        timeout. A host-side backstop kills the exec client if Docker itself
        hangs.
        """
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        exec_argv = list(argv)
        host_timeout: float | None = None
        if timeout_sec is not None and timeout_sec > 0:
            host_timeout = timeout_sec + deadline_grace_sec
            if container_timeout:
                exec_argv = [
                    "timeout",
                    "-k",
                    "10",
                    str(int(timeout_sec)),
                    *exec_argv,
                ]
        docker_argv = self._exec_argv(exec_argv, cwd=cwd, env=env)

        started = time.monotonic()
        timed_out = False
        tail: list[str] = []
        process: subprocess.Popen[str] | None = None
        # Line-buffered so the file can be tailed live while the harness runs;
        # otherwise Python's block buffering would delay events until the buffer
        # fills or the process exits.
        with open(path, "w", encoding="utf-8", buffering=1) as log:
            try:
                process = subprocess.Popen(  # noqa: S603 - argv list, no shell
                    docker_argv,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                assert process.stdout is not None
                for line in process.stdout:
                    log.write(line)
                    tail.append(line)
                    if len(tail) > STREAM_TAIL_LINES:
                        tail.pop(0)
                exit_code = process.wait(timeout=host_timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                if process is not None:
                    process.kill()
                    process.wait()
                exit_code = 124
                self._kill_container_process(argv[0])
            except FileNotFoundError as exc:
                raise WorkspaceError(
                    f"Docker CLI {docker_argv[0]!r} is not available"
                ) from exc
        duration = time.monotonic() - started
        if exit_code == 124 and timeout_sec is not None:
            timed_out = True
        outcome = ProcessOutcome(
            exit_code=exit_code,
            duration_seconds=duration,
            timed_out=timed_out,
            log_path=str(path),
            tail="".join(tail),
        )
        logger.info(
            "Harness process finished exit=%s duration=%.1fs timed_out=%s log=%s",
            outcome.exit_code,
            outcome.duration_seconds,
            outcome.timed_out,
            outcome.log_path,
        )
        return outcome

    def _kill_container_process(self, program: str) -> None:
        try:
            self.execute_command(f"pkill -9 -f {shlex.quote(program)}", timeout=30)
        except WorkspaceError:  # pragma: no cover - best effort
            pass

    # -------------------- file transfer -----------------------------------

    def write_text(self, content: str, remote_path: str) -> None:
        """Write a UTF-8 string to a file inside the container."""
        parent = str(Path(remote_path).parent)
        mkdir = self.execute_command(f"mkdir -p {shlex.quote(parent)}", timeout=30)
        if not mkdir.ok:
            raise WorkspaceError(
                f"Cannot create {parent} in the container: {mkdir.stderr.strip()}"
            )
        result = _run(
            [
                self.docker_bin,
                "exec",
                "-i",
                self.container_id,
                "sh",
                "-c",
                f"cat > {shlex.quote(remote_path)}",
            ],
            input_text=content,
        )
        if result.returncode != 0:
            raise WorkspaceError(
                f"Cannot write {remote_path} in the container: "
                f"{(result.stderr or '').strip()[:1000]}"
            )

    def copy_in(self, local_path: str | Path, remote_path: str) -> None:
        """Copy a host file/directory into the container."""
        parent = str(Path(remote_path).parent)
        mkdir = self.execute_command(f"mkdir -p {shlex.quote(parent)}", timeout=30)
        if not mkdir.ok:
            raise WorkspaceError(
                f"Cannot create {parent} in the container: {mkdir.stderr.strip()}"
            )
        result = _run(
            [
                self.docker_bin,
                "cp",
                str(local_path),
                f"{self.container_id}:{remote_path}",
            ],
            timeout=3600,
        )
        if result.returncode != 0:
            raise WorkspaceError(
                f"Cannot copy {local_path} into the container: "
                f"{(result.stderr or '').strip()[:1000]}"
            )

    def copy_out(self, remote_path: str, local_path: str | Path) -> None:
        """Copy a file from the container to the host."""
        destination = Path(local_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        result = _run(
            [
                self.docker_bin,
                "cp",
                f"{self.container_id}:{remote_path}",
                str(destination),
            ],
            timeout=3600,
        )
        if result.returncode != 0:
            raise WorkspaceError(
                f"Cannot copy {remote_path} out of the container: "
                f"{(result.stderr or '').strip()[:1000]}"
            )

    def container_logs(self, tail: int = 100) -> str:
        return _container_logs(self.container_id, tail=tail, docker_bin=self.docker_bin)

    # -------------------- convenience -------------------------------------

    def path_exists(self, remote_path: str) -> bool:
        result = self.execute_command(
            f"test -e {shlex.quote(remote_path)} && echo yes || echo no", timeout=30
        )
        return (result.stdout or "").strip() == "yes"

    def read_text(self, remote_path: str) -> str:
        with tempfile.TemporaryDirectory() as tmp:
            local = Path(tmp) / "file"
            self.copy_out(remote_path, local)
            return local.read_text(encoding="utf-8", errors="replace")


def _is_running(container_id: str, *, docker_bin: str = DEFAULT_DOCKER_BIN) -> bool:
    result = _run(
        [docker_bin, "inspect", "--format", "{{.State.Running}}", container_id]
    )
    return result.returncode == 0 and (result.stdout or "").strip() == "true"


def _container_logs(
    container_id: str, *, tail: int = 100, docker_bin: str = DEFAULT_DOCKER_BIN
) -> str:
    result = _run([docker_bin, "logs", "--tail", str(tail), container_id])
    parts = [part for part in (result.stdout, result.stderr) if part]
    return "\n".join(parts).strip()[-LOG_TAIL_CHARS:]


def _remove_container(
    container_id: str, *, docker_bin: str = DEFAULT_DOCKER_BIN
) -> None:
    if not container_id:
        return
    try:
        _run([docker_bin, "rm", "-f", container_id], timeout=120)
    except WorkspaceError:  # pragma: no cover - best effort cleanup
        logger.warning("Failed to remove harness container %s", container_id)
