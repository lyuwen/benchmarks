"""Unit tests for pure workspace helpers (no Docker required)."""

from __future__ import annotations

import pytest

from benchmarks.harbor.workspace import (
    ExecOutcome,
    build_agent_server_env,
    docker_run_flags,
    has_registry_host,
    resolve_effective_workdir,
    resolve_image_reference,
)


@pytest.mark.parametrize(
    ("reference", "expected"),
    [
        ("aweaiteam/calibforge:tag", False),
        ("docker.io/library/python:3.12", True),
        ("localhost/foo:tag", True),
        ("registry:5000/foo", True),
        ("busybox", False),
    ],
)
def test_has_registry_host(reference: str, expected: bool) -> None:
    assert has_registry_host(reference) is expected


def test_resolve_image_reference_prepends_prefix() -> None:
    assert resolve_image_reference("aweaiteam/calibforge:Tag", None) == (
        "aweaiteam/calibforge:tag"
    )
    assert resolve_image_reference(
        "aweaiteam/calibforge:tag", "mirror.example.org"
    ) == ("mirror.example.org/aweaiteam/calibforge:tag")
    assert resolve_image_reference(
        "aweaiteam/calibforge:tag", "mirror.example.org/"
    ) == ("mirror.example.org/aweaiteam/calibforge:tag")


def test_resolve_image_reference_keeps_qualified_names() -> None:
    qualified = "docker.io/xiaomimimo/mimo@sha256:abc"
    assert resolve_image_reference(qualified, "mirror.example.org") == qualified
    # No prefix: unchanged apart from lowercasing.
    assert resolve_image_reference("Registry.Example/Repo:Tag", None) == (
        "registry.example/repo:tag"
    )


def test_resolve_image_reference_rejects_empty() -> None:
    with pytest.raises(ValueError):
        resolve_image_reference("   ", None)


def test_resolve_effective_workdir_precedence() -> None:
    assert resolve_effective_workdir("/app", "/image") == "/app"
    assert resolve_effective_workdir(None, "/image") == "/image"
    assert resolve_effective_workdir(None, None) == "/"
    assert resolve_effective_workdir(None, "") == "/"
    assert resolve_effective_workdir("relative", None) == "/relative"


def test_docker_run_flags_include_task_settings() -> None:
    flags = docker_run_flags(
        name="harbor-agent-x",
        plugin_container_name="harbor-plugin-x",
        host_port=34567,
        platform="linux/amd64",
        env={"PATH": "/agent-server/bin:/usr/bin", "FOO": "bar"},
        cpus=2,
        memory_mb=4096,
        memory_limit="14g",
    )
    joined = " ".join(flags)
    assert "--volumes-from harbor-plugin-x" in joined
    assert "-p 34567:8000" in joined
    assert "-e FOO=bar" in joined
    assert "--cpus 2" in joined
    assert "--memory 4096m" in joined
    # Task memory wins over the fallback limit.
    assert "--memory 14g" not in joined
    assert "--platform linux/amd64" in joined
    assert ["-w", "/"] == flags[-2:]


def test_docker_run_flags_fall_back_to_memory_limit() -> None:
    flags = docker_run_flags(
        name="n",
        plugin_container_name="p",
        host_port=1,
        platform="linux/amd64",
        env={},
        cpus=None,
        memory_mb=None,
        memory_limit="14g",
    )
    assert "--memory 14g" in " ".join(flags)
    assert "--cpus" not in flags


def test_build_agent_server_env_with_variant() -> None:
    env = build_agent_server_env(
        "/usr/bin:/bin",
        ("/agent-server/variants/buster/bin", "/agent-server/variants/buster/lib"),
        extra_env={"TASK_VAR": "1"},
    )
    assert env["PATH"] == (
        "/agent-server/variants/buster/bin:/agent-server/bin:/usr/bin:/bin"
    )
    assert env["LD_LIBRARY_PATH"] == (
        "/agent-server/variants/buster/lib:/agent-server/lib"
    )
    assert env["UV_PYTHON_INSTALL_DIR"] == "/agent-server/uv-managed-python"
    assert env["TASK_VAR"] == "1"


def test_build_agent_server_env_without_variant() -> None:
    env = build_agent_server_env(
        "/usr/bin", (None, None), extra_env={"OH_CONVERSATIONS_PATH": "/tmp/c"}
    )
    assert env["PATH"] == "/agent-server/bin:/usr/bin"
    assert env["LD_LIBRARY_PATH"] == "/agent-server/lib"
    assert env["OH_CONVERSATIONS_PATH"] == "/tmp/c"


def test_exec_outcome_ok() -> None:
    assert ExecOutcome("c", 0, "", "", 0.1).ok
    assert not ExecOutcome("c", 1, "", "", 0.1).ok
    assert not ExecOutcome("c", None, "", "", 0.1, timed_out=True).ok
