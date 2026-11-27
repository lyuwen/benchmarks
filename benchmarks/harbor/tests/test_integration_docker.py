"""Docker integration tests for the Harbor workspace + verifier adapter.

These are opt-in (``HARBOR_DOCKER_TEST=1``) because they start real containers.
They never call an LLM: the agent harness is exercised through the injected
agent server's HTTP API and the verifier through ``tests/test.sh``.

Configuration:
    HARBOR_DOCKER_TEST=1
    HARBOR_TEST_AGENT_PLUGIN_IMAGE=<image with /agent-server volume>
    HARBOR_TEST_BASE_IMAGE=<small Linux base for the fixture build>
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks.harbor.judge import STATUS_SCORED, run_shared_verifier
from benchmarks.harbor.task import load_tasks
from benchmarks.harbor.tests.conftest import write_task
from benchmarks.harbor.workspace import (
    HarborDockerWorkspace,
    resolve_effective_workdir,
)


PLUGIN_IMAGE = os.getenv("HARBOR_TEST_AGENT_PLUGIN_IMAGE", "openhands/agent-plugin")
BASE_IMAGE = os.getenv("HARBOR_TEST_BASE_IMAGE", "debian:bookworm-slim")

pytestmark = pytest.mark.skipif(
    os.getenv("HARBOR_DOCKER_TEST") != "1",
    reason="set HARBOR_DOCKER_TEST=1 to run Docker integration tests",
)


def _docker_available() -> bool:
    return shutil.which("docker") is not None


def _image_exists(image: str) -> bool:
    if not _docker_available():
        return False
    completed = subprocess.run(
        ["docker", "image", "inspect", image],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    return completed.returncode == 0


def _container_exists(container_id: str | None) -> bool:
    if not container_id:
        return False
    completed = subprocess.run(
        ["docker", "inspect", container_id],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    return completed.returncode == 0


def _require_docker() -> None:
    if not _docker_available():
        pytest.skip("docker is not available")
    if not _image_exists(PLUGIN_IMAGE):
        pytest.skip(
            f"agent plugin image {PLUGIN_IMAGE!r} is not available locally; "
            "set HARBOR_TEST_AGENT_PLUGIN_IMAGE"
        )
    if not _image_exists(BASE_IMAGE):
        pytest.skip(f"base test image {BASE_IMAGE!r} is not available locally")


HEALTHCHECK_SEEN = "/app/.healthcheck-ran"


def test_dockerfile_task_workspace_and_verifier(tmp_path: Path) -> None:
    _require_docker()
    task_toml = (
        'schema_version = "1.2"\n\n'
        "[verifier]\ntimeout_sec = 120.0\n\n"
        "[agent]\ntimeout_sec = 120.0\n\n"
        "[environment]\n"
        "build_timeout_sec = 600.0\n"
        'workdir = "/app"\n'
        "cpus = 1\n"
        "memory_mb = 1024\n"
        "[environment.healthcheck]\n"
        f'command = "touch {HEALTHCHECK_SEEN}"\n'
        "retries = 1\n"
        "timeout_sec = 30.0\n"
    )
    test_sh = (
        "#!/bin/bash\n"
        "set -uo pipefail\n"
        "mkdir -p /logs/verifier\n"
        "pwd > /logs/verifier/pwd.txt\n"
        f"test -f {HEALTHCHECK_SEEN} && echo healthcheck-ok "
        "> /logs/verifier/healthcheck.txt\n"
        "echo 0.5 > /logs/verifier/reward.txt\n"
    )
    task_dir = write_task(
        tmp_path,
        "smoke",
        task_toml=task_toml,
        dockerfile=f"FROM {BASE_IMAGE}\nWORKDIR /app\nRUN mkdir -p /app\n",
        tests={"test.sh": test_sh},
    )
    task = load_tasks(task_dir).tasks[0]
    assert task.config.environment.workdir == "/app"

    trial_dir = tmp_path / "trial"
    workspace = HarborDockerWorkspace(
        base_image=_build_fixture_image(task, tmp_path),
        working_dir="/app",
        agent_plugin_image=PLUGIN_IMAGE,
        cpus=1,
        memory_mb=1024,
        healthcheck=task.config.environment.healthcheck,
        detach_logs=False,
    )
    container_id = workspace.container_id
    try:
        # The agent server is reachable and executes commands in the container.
        executed = workspace.execute_command("echo agent-server-ok")
        assert "agent-server-ok" in executed.stdout

        # Direct docker exec uses Harbor's effective cwd.
        direct = workspace.docker_exec("pwd", cwd=workspace.working_dir)
        assert direct.ok
        assert direct.stdout.strip() == "/app"

        # Healthcheck ran after startup, before verification.
        assert workspace.docker_exec(f"test -f {HEALTHCHECK_SEEN}").ok

        outcome = run_shared_verifier(
            workspace,
            task,
            trial_dir=trial_dir,
            run_dir=tmp_path,
        )
        assert outcome["status"] == STATUS_SCORED
        assert outcome["reward"] == 0.5
        assert (trial_dir / "verifier" / "pwd.txt").read_text(
            encoding="utf-8"
        ).strip() == "/app"
        assert (trial_dir / "verifier" / "healthcheck.txt").read_text(
            encoding="utf-8"
        ).strip() == "healthcheck-ok"
    finally:
        workspace.cleanup()

    assert not _container_exists(container_id)


def test_prebuilt_image_workdir_resolution(tmp_path: Path) -> None:
    _require_docker()
    assert _image_exists(BASE_IMAGE)
    task_toml = (
        'schema_version = "1.2"\n\n'
        "[verifier]\ntimeout_sec = 60.0\n\n"
        "[environment]\n"
        f'docker_image = "{BASE_IMAGE}"\n'
    )
    task_dir = write_task(
        tmp_path,
        "prebuilt",
        task_toml=task_toml,
        tests={
            "test.sh": (
                "#!/bin/bash\nset -uo pipefail\nmkdir -p /logs/verifier\n"
                "echo 1 > /logs/verifier/reward.txt\n"
            )
        },
    )
    task = load_tasks(task_dir).tasks[0]
    image = task.config.environment.docker_image
    assert image is not None

    from benchmarks.harbor.run_infer import image_workdir

    workdir = resolve_effective_workdir(
        task.config.environment.workdir, image_workdir(image)
    )
    assert workdir == "/"

    workspace = HarborDockerWorkspace(
        base_image=image,
        working_dir=workdir,
        agent_plugin_image=PLUGIN_IMAGE,
        detach_logs=False,
    )
    container_id = workspace.container_id
    try:
        outcome = run_shared_verifier(
            workspace,
            task,
            trial_dir=tmp_path / "trial",
            run_dir=tmp_path,
        )
        assert outcome["status"] == STATUS_SCORED
        assert outcome["reward"] == 1.0
    finally:
        workspace.cleanup()
    assert not _container_exists(container_id)


def _build_fixture_image(task, tmp_path: Path) -> str:
    from benchmarks.harbor.run_infer import resolve_task_image

    image = resolve_task_image(task, None, trial_dir=tmp_path / "build")
    assert image.built
    return image.reference


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-v"]))
