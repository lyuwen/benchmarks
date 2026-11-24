"""Local Docker workspace adapter for Harbor tasks.

Harbor's Docker environment starts the task image through ``docker compose`` and
runs commands with ``docker compose exec``.  This adapter keeps the same
execution semantics but starts the container directly so it can inject the
OpenHands agent server (via the ``openhands-agent-plugin`` volume pattern used
by :class:`~openhands.workspace.FlexWorkspace`) while preserving Harbor's
container settings.

The class owns the full lifecycle of two containers:

* the *agent server* container: the task image with the plugin volume mounted
  and an overridden entrypoint running ``openhands.agent_server``;
* a *plugin data* container whose volumes supply the self-contained agent
  server Python environment.

It exposes two execution paths:

* :meth:`execute_command`, inherited from :class:`RemoteWorkspace`, which talks
  to the in-container agent server (used by the agent harness);
* :meth:`docker_exec`, which runs a command directly with ``docker exec`` using
  Harbor's effective working directory, environment, and user (used for the
  verifier, healthcheck, and artifact collection).

Pinned reference: Harbor v0.9.0 ``environments/docker/docker.py``.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import tarfile
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.request import urlopen

from pydantic import Field, PrivateAttr

from benchmarks.harbor.task import HealthcheckConfig
from openhands.sdk.logger import get_logger
from openhands.sdk.workspace import RemoteWorkspace
from openhands.workspace.docker.workspace import (
    check_port_available,
    find_available_tcp_port,
)


logger = get_logger(__name__)

AGENT_SERVER_PORT = 8000
_AGENT_SERVER_PYTHON = "/agent-server/.venv/bin/python"
_AGENT_SERVER_BIN = "/agent-server/bin"
_AGENT_SERVER_LIB = "/agent-server/lib"
_DEFAULT_SYSTEM_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
_LOG_DIRS = ("/logs/agent", "/logs/verifier", "/logs/artifacts")
# Keep the agent-server's persisted conversations/bash events out of the task
# filesystem (Harbor tasks may use /workspace for their own files).  These are
# the OH_* environment variables the agent server's config reads.
CONVERSATIONS_DIR = "/tmp/harbor/conversations"
BASH_EVENTS_DIR = "/tmp/harbor/bash_events"


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested without Docker)
# ---------------------------------------------------------------------------


def has_registry_host(image_reference: str) -> bool:
    """True when the reference already carries a registry host component.

    Docker's rule: the first ``/``-separated component is a registry host when
    it contains ``.`` or ``:`` or equals ``localhost``.
    """
    head, separator, _ = image_reference.partition("/")
    if not separator:
        return False
    return "." in head or ":" in head or head == "localhost"


def resolve_image_reference(image_reference: str, prefix: str | None) -> str:
    """Prepend an optional registry prefix to a bare ``namespace/repo:tag``.

    An already-qualified reference (e.g. ``myregistry.com/ns/repo:tag`` or a
    digest-pinned reference) is returned unchanged apart from lowercasing, as
    published image names are lowercase.  The prefix is never hard-coded: it is
    supplied at runtime via ``--docker-image-prefix``.
    """
    reference = image_reference.strip()
    if not reference:
        raise ValueError("Empty docker image reference")
    if not prefix or has_registry_host(reference):
        return reference.lower()
    return f"{prefix.rstrip('/')}/{reference}".lower()


def resolve_effective_workdir(
    config_workdir: str | None, image_workdir: str | None
) -> str:
    """Resolve Harbor's effective command cwd.

    Explicit ``[environment].workdir`` wins, then the image's ``WORKDIR`` (the
    container default), then ``/`` (Docker's default when no WORKDIR is set).
    """
    for candidate in (config_workdir, image_workdir):
        if candidate and candidate.strip():
            normalized = candidate.strip()
            return normalized if normalized.startswith("/") else f"/{normalized}"
    return "/"


def docker_run_flags(
    *,
    name: str,
    plugin_container_name: str,
    host_port: int,
    platform: str,
    env: dict[str, str],
    cpus: int | None,
    memory_mb: int | None,
    memory_limit: str | None,
) -> list[str]:
    """Build the ``docker run`` flags shared by every Harbor agent container."""
    flags: list[str] = [
        "-d",
        "--rm",
        "--name",
        name,
        "--platform",
        platform,
        "--volumes-from",
        plugin_container_name,
    ]
    for key, value in env.items():
        flags += ["-e", f"{key}={value}"]
    flags += ["-p", f"{host_port}:{AGENT_SERVER_PORT}"]
    if cpus:
        flags += ["--cpus", str(cpus)]
    if memory_mb:
        flags += ["--memory", f"{memory_mb}m"]
    elif memory_limit:
        flags += ["--memory", memory_limit]
    # The agent server must start inside the image root; tool commands get the
    # task's effective working directory through RemoteWorkspace.working_dir.
    flags += ["-w", "/"]
    return flags


@dataclass(frozen=True)
class ExecOutcome:
    """Result of a direct ``docker exec`` command."""

    command: str
    exit_code: int | None
    stdout: str
    stderr: str
    duration_sec: float
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out


def build_agent_server_env(
    base_path: str,
    variant_paths: tuple[str | None, str | None],
    extra_env: dict[str, str] | None = None,
) -> dict[str, str]:
    """Compose the environment for the injected agent server.

    Mirrors FlexWorkspace: prepend the plugin's ``bin``/``lib`` (or the detected
    glibc variant) to the base image's ``PATH`` and ``LD_LIBRARY_PATH``.
    """
    variant_bin, variant_lib = variant_paths
    path_parts = [variant_bin or _AGENT_SERVER_BIN]
    if variant_bin:
        path_parts.append(_AGENT_SERVER_BIN)
    path_parts.append(base_path)
    lib_parts = [variant_lib or _AGENT_SERVER_LIB]
    if variant_lib:
        lib_parts.append(_AGENT_SERVER_LIB)
    env = {
        "PATH": ":".join(path_parts),
        "LD_LIBRARY_PATH": ":".join(lib_parts),
        "UV_PYTHON_INSTALL_DIR": "/agent-server/uv-managed-python",
    }
    if extra_env:
        env.update(extra_env)
    return env


def _sanitize_name_suffix(value: str) -> str:
    sanitized = "".join(
        char if char.isalnum() or char in "._-" else "-" for char in value.lower()
    )
    return sanitized[:32] or "container"


# ---------------------------------------------------------------------------
# Workspace
# ---------------------------------------------------------------------------


class HarborDockerWorkspace(RemoteWorkspace):
    """Remote workspace whose container is a Harbor task environment."""

    base_image: str = Field(description="Resolved task image reference.")
    agent_plugin_image: str = Field(
        default="openhands/agent-plugin",
        description="Image providing the /agent-server volume.",
    )
    task_env: dict[str, str] = Field(
        default_factory=dict,
        description="Resolved [environment].env variables injected into the container.",
    )
    cpus: int | None = Field(default=None, description="CPU limit for the container.")
    memory_mb: int | None = Field(
        default=None, description="Memory limit in MiB from [environment].memory_mb."
    )
    memory_limit: str | None = Field(
        default_factory=lambda: os.getenv("HARBOR_WORKSPACE_MEMORY_LIMIT", "14g"),
        description=(
            "Fallback Docker memory limit when the task does not declare "
            "memory_mb (the pinned SDK uses 14g for the agent server)."
        ),
    )
    network_mode: str = Field(
        default="public", description="Only 'public' is supported in Phase 1."
    )
    platform: str = Field(default="linux/amd64")
    host_port: int | None = Field(
        default=None, description="Host port for the agent server (auto by default)."
    )
    forward_env: list[str] = Field(
        default_factory=list, description="Host env vars forwarded into the container."
    )
    detach_logs: bool = Field(
        default=True, description="Stream container logs in a background thread."
    )
    image_pull_timeout_sec: float = Field(
        default=1800.0, description="Timeout for pulling a missing task image."
    )
    startup_timeout_sec: float = Field(
        default_factory=lambda: float(
            os.getenv("HARBOR_WORKSPACE_STARTUP_TIMEOUT", "180")
        ),
        description=(
            "Timeout for the injected agent server to become healthy "
            "(HARBOR_WORKSPACE_STARTUP_TIMEOUT)."
        ),
    )
    healthcheck: HealthcheckConfig | None = Field(default=None)
    upload_environment_dir: str | None = Field(
        default=None,
        description=(
            "Local environment/ directory to copy into the working directory "
            "after startup (Harbor's prebuilt-image post-start upload)."
        ),
    )
    host: str = Field(default="", description="Set automatically after startup.")

    _container_id: str | None = PrivateAttr(default=None)
    _plugin_container_name: str | None = PrivateAttr(default=None)
    _logs_thread: threading.Thread | None = PrivateAttr(default=None)
    _stop_logs: threading.Event = PrivateAttr(default_factory=threading.Event)
    _image_digest_value: str | None = PrivateAttr(default=None)
    _started_at: float = PrivateAttr(default=0.0)

    # -- lifecycle ---------------------------------------------------------

    def model_post_init(self, context: Any) -> None:
        self._started_at = time.monotonic()
        try:
            if self.network_mode.lower() != "public":
                raise ValueError(
                    f"Unsupported workspace network_mode={self.network_mode!r}; "
                    "only 'public' is supported."
                )
            self._prepare_image()
            self._start_container(context)
            self._prepare_container_layout()
            if self.upload_environment_dir is not None:
                self.upload_dir(self.upload_environment_dir, self.working_dir)
            if self.healthcheck is not None:
                self.run_healthcheck(self.healthcheck)
        except BaseException:
            # `prepare_workspace` has no workspace handle when startup raises, so
            # any partially created containers must be removed here.
            self.cleanup()
            raise

    @property
    def container_id(self) -> str | None:
        return self._container_id

    @property
    def image_digest(self) -> str | None:
        return self._image_digest_value

    @property
    def conversations_dir(self) -> str:
        return CONVERSATIONS_DIR

    @property
    def startup_duration_sec(self) -> float:
        return max(0.0, time.monotonic() - self._started_at)

    @property
    def effective_memory_limit(self) -> str | None:
        if self.memory_mb:
            return f"{self.memory_mb}m"
        return self.memory_limit

    def __enter__(self) -> "HarborDockerWorkspace":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.cleanup()

    def __del__(self) -> None:
        try:
            self.cleanup()
        except Exception:  # pragma: no cover - interpreter teardown
            pass

    def cleanup(self) -> None:
        """Remove the agent-server and plugin containers (idempotent)."""
        self._stop_logs.set()
        thread = self._logs_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2)
        self._logs_thread = None

        container_id = self._container_id
        self._container_id = None
        if container_id:
            logger.info("Removing Harbor agent container %s", container_id)
            self._run_docker(["rm", "-f", container_id], timeout=120)

        plugin_name = self._plugin_container_name
        self._plugin_container_name = None
        if plugin_name:
            self._run_docker(["rm", "-f", plugin_name], timeout=60)

        self._stop_logs = threading.Event()
        try:
            self.reset_client()
        except Exception:  # pragma: no cover - best effort
            pass

    # -- docker plumbing ---------------------------------------------------

    def _run_docker(
        self,
        args: list[str],
        *,
        timeout: float | None = None,
        check: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        logger.debug("$ docker %s", " ".join(args))
        try:
            completed = subprocess.run(
                ["docker", *args],
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            if check:
                raise RuntimeError(
                    f"docker {' '.join(args)} timed out after {timeout}s"
                ) from exc
            completed = subprocess.CompletedProcess(
                ["docker", *args], returncode=124, stdout="", stderr=str(exc)
            )
        if check and completed.returncode != 0:
            raise RuntimeError(
                f"docker {' '.join(args)} failed (exit {completed.returncode}): "
                f"{completed.stderr.strip()[:2000]}"
            )
        return completed

    def _prepare_image(self) -> None:
        inspect = self._run_docker(
            ["image", "inspect", "--format", "{{.Id}}", self.base_image], timeout=60
        )
        if inspect.returncode != 0:
            logger.info("Pulling task image %s", self.base_image)
            pull = self._run_docker(
                ["pull", "--platform", self.platform, self.base_image],
                timeout=self.image_pull_timeout_sec,
            )
            if pull.returncode != 0:
                raise RuntimeError(
                    f"Failed to pull task image {self.base_image}: "
                    f"{pull.stderr.strip()[:2000]}"
                )
        digest = self._run_docker(
            [
                "image",
                "inspect",
                "--format",
                "{{if .RepoDigests}}{{index .RepoDigests 0}}{{else}}{{.Id}}{{end}}",
                self.base_image,
            ],
            timeout=60,
        )
        self._image_digest_value = digest.stdout.strip() or None

    def _start_container(self, context: Any) -> None:
        if self.host_port is None:
            self.host_port = find_available_tcp_port()
        else:
            self.host_port = int(self.host_port)
        if self.host_port < 0 or not check_port_available(self.host_port):
            raise RuntimeError(f"Host port {self.host_port} is not available")

        docker_version = self._run_docker(["version"], timeout=30)
        if docker_version.returncode != 0:
            raise RuntimeError(
                "Docker is not available. Please install and start Docker."
            )

        plugin_name = f"harbor-plugin-{uuid.uuid4().hex[:12]}"
        created = self._run_docker(
            ["create", "--name", plugin_name, self.agent_plugin_image, "true"],
            timeout=600,
        )
        if created.returncode != 0:
            raise RuntimeError(
                "Failed to create the agent-plugin data container: "
                f"{created.stderr.strip()[:2000]}"
            )
        self._plugin_container_name = plugin_name

        variant_paths = self._detect_variant_paths()
        env = build_agent_server_env(
            self._base_image_path(),
            variant_paths,
            extra_env={
                "OH_CONVERSATIONS_PATH": CONVERSATIONS_DIR,
                "OH_BASH_EVENTS_DIR": BASH_EVENTS_DIR,
                **self.task_env,
            },
        )
        for key in self.forward_env:
            if key in os.environ:
                env.setdefault(key, os.environ[key])
        for key, value in os.environ.items():
            if key.startswith("OH_"):
                env.setdefault(key, value)

        container_name = f"harbor-agent-{uuid.uuid4().hex[:12]}"
        flags = docker_run_flags(
            name=container_name,
            plugin_container_name=plugin_name,
            host_port=self.host_port,
            platform=self.platform,
            env=env,
            cpus=self.cpus,
            memory_mb=self.memory_mb,
            memory_limit=self.memory_limit,
        )
        run = self._run_docker(
            [
                "run",
                *flags,
                "--entrypoint",
                _AGENT_SERVER_PYTHON,
                self.base_image,
                "-E",
                "-m",
                "openhands.agent_server",
                "--host",
                "0.0.0.0",
                "--port",
                str(AGENT_SERVER_PORT),
            ],
            timeout=300,
        )
        if run.returncode != 0:
            raise RuntimeError(
                "Failed to start the Harbor agent container: "
                f"{run.stderr.strip()[:2000]}"
            )
        self._container_id = run.stdout.strip()
        logger.info(
            "Started Harbor agent container %s for image %s",
            self._container_id,
            self.base_image,
        )

        if self.detach_logs:
            self._logs_thread = threading.Thread(
                target=self._stream_docker_logs, daemon=True
            )
            self._logs_thread.start()

        object.__setattr__(self, "host", f"http://localhost:{self.host_port}")
        object.__setattr__(self, "api_key", None)
        self._wait_for_health(timeout=self.startup_timeout_sec)
        RemoteWorkspace.model_post_init(self, context)
        logger.info("Harbor workspace ready at %s", self.host)

    def _detect_variant_paths(self) -> tuple[str | None, str | None]:
        """Select the plugin glibc variant directory for the base image."""
        try:
            from openhands.workspace.docker.flex_workspace import (
                _detect_glibc_version,
                _select_variant,
            )
        except Exception:  # pragma: no cover - pinned SDK always provides these
            logger.debug("FlexWorkspace glibc helpers unavailable", exc_info=True)
            return None, None
        glibc_version = _detect_glibc_version(self.base_image, self.platform)
        variant = _select_variant(glibc_version)
        if not variant:
            return None, None
        return (
            f"/agent-server/variants/{variant}/bin",
            f"/agent-server/variants/{variant}/lib",
        )

    def _base_image_path(self) -> str:
        try:
            from openhands.workspace.docker.flex_workspace import (
                _get_base_image_path,
            )
        except Exception:  # pragma: no cover - pinned SDK always provides these
            return _DEFAULT_SYSTEM_PATH
        return _get_base_image_path(self.base_image)

    def _prepare_container_layout(self) -> None:
        dirs = " ".join(shlex.quote(path) for path in _LOG_DIRS)
        outcome = self.docker_exec(
            f"mkdir -p {dirs} && chmod -R 777 /logs", user="root", timeout=60
        )
        if not outcome.ok:
            raise RuntimeError(
                "Failed to prepare /logs directories in the container: "
                f"{outcome.stderr.strip()[:2000]}"
            )

    def _stream_docker_logs(self) -> None:
        if not self._container_id:
            return
        try:
            process = subprocess.Popen(
                ["docker", "logs", "-f", self._container_id],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except OSError as exc:  # pragma: no cover - docker log streaming
            logger.debug("Failed to stream docker logs: %s", exc)
            return
        if process.stdout is None:  # pragma: no cover - defensive
            return
        for line in iter(process.stdout.readline, ""):
            if self._stop_logs.is_set():
                break
            if line:
                logger.debug("[docker] %s", line.rstrip("\n"))
        try:
            process.terminate()
        except OSError:  # pragma: no cover
            pass

    def _wait_for_health(self, timeout: float = 180.0) -> None:
        start = time.monotonic()
        health_url = f"http://127.0.0.1:{self.host_port}/health"
        while time.monotonic() - start < timeout:
            try:
                with urlopen(health_url, timeout=1.0) as response:
                    if 200 <= getattr(response, "status", 200) < 300:
                        return
            except Exception:
                pass
            if self._container_id:
                state = self._run_docker(
                    [
                        "inspect",
                        "-f",
                        "{{.State.Running}}",
                        self._container_id,
                    ],
                    timeout=30,
                )
                if state.stdout.strip() != "true":
                    logs = self._run_docker(["logs", self._container_id], timeout=60)
                    raise RuntimeError(
                        "Harbor agent container stopped unexpectedly. Logs:\n"
                        f"{logs.stdout}\n{logs.stderr}"
                    )
            time.sleep(1)
        raise RuntimeError(
            f"Harbor agent server did not become healthy within {timeout}s"
        )

    # -- direct container execution ---------------------------------------

    def docker_exec(
        self,
        command: str,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        user: str | int | None = None,
        timeout: float | None = None,
        kill_pattern: str | None = None,
    ) -> ExecOutcome:
        """Run ``command`` with ``docker exec`` inside the task container."""
        if not self._container_id:
            raise RuntimeError("Harbor workspace is not running")
        args = ["exec"]
        if cwd:
            args += ["-w", cwd]
        for key, value in (env or {}).items():
            args += ["-e", f"{key}={value}"]
        if user is not None:
            args += ["-u", str(user)]
        args += [self._container_id, "bash", "-c", command]

        started = time.monotonic()
        timed_out = False
        try:
            completed = subprocess.run(
                ["docker", *args],
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            stdout, stderr, exit_code = (
                completed.stdout,
                completed.stderr,
                completed.returncode,
            )
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            stdout = _as_text(exc.stdout)
            stderr = _as_text(exc.stderr)
            exit_code = None
            stderr += f"\n[docker exec timed out after {timeout}s]"
            if kill_pattern:
                self._kill_matching_processes(kill_pattern)
        duration = time.monotonic() - started
        return ExecOutcome(
            command=command,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            duration_sec=duration,
            timed_out=timed_out,
        )

    def _kill_matching_processes(self, pattern: str) -> None:
        """Best-effort kill of timed-out verifier processes in the container."""
        if not self._container_id:
            return
        try:
            subprocess.run(
                [
                    "docker",
                    "exec",
                    "-u",
                    "root",
                    self._container_id,
                    "pkill",
                    "-9",
                    "-f",
                    pattern,
                ],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            logger.debug("Failed to kill timed-out process %r", pattern)

    def run_healthcheck(self, healthcheck: HealthcheckConfig) -> None:
        """Run Harbor's post-start healthcheck/one-time-setup command."""
        started = time.monotonic()
        failures = 0
        while True:
            outcome = self.docker_exec(
                healthcheck.command, user="root", timeout=healthcheck.timeout_sec
            )
            if outcome.ok:
                logger.info(
                    "Healthcheck passed after %.1fs",
                    time.monotonic() - started,
                )
                return
            elapsed = time.monotonic() - started
            if elapsed >= healthcheck.start_period_sec:
                failures += 1
                logger.warning(
                    "Healthcheck failed (%d/%d): exit=%s stderr=%s",
                    failures,
                    healthcheck.retries,
                    outcome.exit_code,
                    outcome.stderr.strip()[:500],
                )
                if failures >= healthcheck.retries:
                    raise RuntimeError(
                        "Harbor environment healthcheck failed after "
                        f"{failures} consecutive failure(s): "
                        f"{outcome.stderr.strip()[:2000]}"
                    )
            interval = (
                healthcheck.start_interval_sec
                if elapsed < healthcheck.start_period_sec
                else healthcheck.interval_sec
            )
            time.sleep(max(0.0, interval))

    # -- file transfer -----------------------------------------------------

    def _chown_to_host_user(self, remote_path: str) -> None:
        if not self._container_id or not hasattr(os, "getuid"):
            return
        quoted = shlex.quote(remote_path)
        self.docker_exec(
            f"chown -R {os.getuid()}:{os.getgid()} {quoted}",
            user="root",
            timeout=120,
        )

    def upload_dir(self, source_dir: str | Path, target_dir: str) -> None:
        if not self._container_id:
            raise RuntimeError("Harbor workspace is not running")
        quoted = shlex.quote(target_dir)
        mkdir = self.docker_exec(f"mkdir -p {quoted}", user="root", timeout=60)
        if not mkdir.ok:
            raise RuntimeError(
                f"Failed to create {target_dir} in container: {mkdir.stderr.strip()}"
            )
        completed = self._run_docker(
            ["cp", f"{source_dir}/.", f"{self._container_id}:{target_dir}"],
            timeout=600,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"docker cp {source_dir} -> {target_dir} failed: "
                f"{completed.stderr.strip()[:2000]}"
            )

    def upload_file(self, source_path: str | Path, target_path: str) -> None:
        if not self._container_id:
            raise RuntimeError("Harbor workspace is not running")
        completed = self._run_docker(
            ["cp", str(source_path), f"{self._container_id}:{target_path}"],
            timeout=600,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"docker cp {source_path} -> {target_path} failed: "
                f"{completed.stderr.strip()[:2000]}"
            )

    def download_dir(self, source_dir: str, target_dir: str | Path) -> None:
        if not self._container_id:
            raise RuntimeError("Harbor workspace is not running")
        self._chown_to_host_user(source_dir)
        target = Path(target_dir)
        target.mkdir(parents=True, exist_ok=True)
        completed = self._run_docker(
            ["cp", f"{self._container_id}:{source_dir}/.", str(target)],
            timeout=600,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"docker cp {source_dir} -> {target_dir} failed: "
                f"{completed.stderr.strip()[:2000]}"
            )

    def download_file(self, source_path: str, target_path: str | Path) -> None:
        if not self._container_id:
            raise RuntimeError("Harbor workspace is not running")
        self._chown_to_host_user(source_path)
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        completed = self._run_docker(
            ["cp", f"{self._container_id}:{source_path}", str(target)],
            timeout=600,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"docker cp {source_path} -> {target_path} failed: "
                f"{completed.stderr.strip()[:2000]}"
            )

    def download_dir_with_exclusions(
        self,
        source_dir: str,
        target_dir: str | Path,
        exclude: list[str],
    ) -> None:
        """Download a directory applying Harbor's ``tar --exclude`` semantics."""
        if not self._container_id:
            raise RuntimeError("Harbor workspace is not running")
        remote_tar = f"/tmp/harbor-artifact-{uuid.uuid4().hex[:8]}.tar.gz"
        exclude_flags = " ".join(
            f"--exclude={shlex.quote(pattern)}" for pattern in exclude
        )
        quoted_source = shlex.quote(source_dir)
        outcome = self.docker_exec(
            f"tar -czf {remote_tar} -C {quoted_source} {exclude_flags} .",
            user="root",
            timeout=600,
        )
        if not outcome.ok:
            raise RuntimeError(
                f"Failed to archive {source_dir} with exclusions: "
                f"{outcome.stderr.strip()[:2000]}"
            )
        local_tar = Path(target_dir).parent / f"{Path(remote_tar).name}"
        local_tar.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.download_file(remote_tar, local_tar)
            target = Path(target_dir)
            target.mkdir(parents=True, exist_ok=True)
            with tarfile.open(local_tar, "r:gz") as archive:
                for member in archive.getmembers():
                    member_path = Path(member.name)
                    if member_path.is_absolute() or ".." in member_path.parts:
                        continue
                    archive.extract(member, path=target)
        finally:
            local_tar.unlink(missing_ok=True)
            self.docker_exec(
                f"rm -f {shlex.quote(remote_tar)}", user="root", timeout=60
            )

    def is_dir(self, remote_path: str) -> bool:
        outcome = self.docker_exec(
            f"test -d {shlex.quote(remote_path)}", user="root", timeout=60
        )
        return outcome.exit_code == 0

    def path_exists(self, remote_path: str) -> bool:
        outcome = self.docker_exec(
            f"test -e {shlex.quote(remote_path)}", user="root", timeout=60
        )
        return outcome.exit_code == 0


def _as_text(value: bytes | str | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def ensure_local_image(
    image_reference: str,
    *,
    platform: str = "linux/amd64",
    timeout_sec: float = 1800.0,
) -> None:
    """Pull an image if it is not present locally.

    Harbor's compose ``up`` pulls a missing prebuilt image; the adapter needs
    the image *before* container start to read its ``WORKDIR`` (the effective
    working directory when the task does not set one).
    """
    inspect = subprocess.run(
        ["docker", "image", "inspect", image_reference],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if inspect.returncode == 0:
        return
    logger.info("Pulling task image %s", image_reference)
    try:
        pull = subprocess.run(
            ["docker", "pull", "--platform", platform, image_reference],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"Pulling task image {image_reference} timed out after {timeout_sec}s"
        ) from exc
    if pull.returncode != 0:
        raise RuntimeError(
            f"Failed to pull task image {image_reference}: {pull.stderr.strip()[:2000]}"
        )


def read_local_digest(image_reference: str) -> str | None:
    """Return the local image digest/id for provenance, if the image exists."""
    try:
        completed = subprocess.run(
            [
                "docker",
                "image",
                "inspect",
                "--format",
                "{{if .RepoDigests}}{{index .RepoDigests 0}}{{else}}{{.Id}}{{end}}",
                image_reference,
            ],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


__all__ = [
    "AGENT_SERVER_PORT",
    "BASH_EVENTS_DIR",
    "CONVERSATIONS_DIR",
    "ensure_local_image",
    "ExecOutcome",
    "HarborDockerWorkspace",
    "build_agent_server_env",
    "docker_run_flags",
    "has_registry_host",
    "read_local_digest",
    "resolve_effective_workdir",
    "resolve_image_reference",
]
