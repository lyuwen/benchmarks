"""Tests for the SWE-bench blackbox-harness evaluation entry point."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from benchmarks.swebench import run_infer_harnesses as harness_runner
from benchmarks.utils.harnesses.config import HarnessConfig
from benchmarks.utils.harnesses.distribution import HarnessDistribution
from benchmarks.utils.harnesses.runner import HarnessRunResult
from benchmarks.utils.harnesses.workspace import DockerContainerWorkspace
from benchmarks.utils.models import EvalInstance, EvalMetadata
from benchmarks.utils.tests.fakes import PATCH, FakeWorkspace
from openhands.sdk.critic import PassCritic
from openhands.sdk.llm import LLM


def make_llm() -> LLM:
    return LLM(
        model="litellm_proxy/anthropic/claude-sonnet-4",
        base_url="http://127.0.0.1:9999",
        api_key="secret",
    )


def make_metadata(tmp_path: Path) -> EvalMetadata:
    prompt_path = (
        Path(harness_runner.__file__).parent / "prompts" / "harness-default.j2"
    )
    return EvalMetadata(
        llm=make_llm(),
        dataset="princeton-nlp/SWE-bench_Verified",
        dataset_split="test",
        max_iterations=30,
        eval_output_dir=str(tmp_path / "out"),
        details={"harness": {}},
        prompt_path=str(prompt_path.resolve()),
        env_setup_commands=["export PIP_CACHE_DIR=~/.cache/pip"],
        conversation_timeout=600,
        critic=PassCritic(),
        condenser="none",
    )


def make_instance() -> EvalInstance:
    return EvalInstance(
        id="django__django-1234",
        data={
            "repo": "django/django",
            "base_commit": "abc123def",
            "problem_statement": "Fix the ORM crash when filtering.",
        },
    )


def make_evaluation(tmp_path: Path, **kwargs: Any) -> Any:
    defaults: dict[str, Any] = dict(
        metadata=make_metadata(tmp_path),
        num_workers=1,
        harness_config=HarnessConfig(cache_dir=str(tmp_path / "cache")),
    )
    defaults.update(kwargs)
    return harness_runner.SWEBenchHarnessEvaluation(**defaults)


def test_capture_conversation_archive_is_noop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evaluation = make_evaluation(tmp_path)
    workspace = FakeWorkspace()
    evaluation._capture_conversation_archive(cast(Any, workspace), make_instance())
    assert workspace.calls == []


def test_prepare_workspace_uses_official_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    started: list[dict[str, Any]] = []
    distribution = HarnessDistribution(
        harness="claude-code",
        version="2.1.331",
        platform="linux-x64",
        directory=tmp_path / "dist",
        binary_path=tmp_path / "dist" / "claude",
        sha256="x",
        size_bytes=1,
    )
    monkeypatch.setattr(
        harness_runner, "ensure_distribution", lambda config: distribution
    )

    def fake_start(**kwargs: Any) -> FakeWorkspace:
        started.append(kwargs)
        return FakeWorkspace()

    monkeypatch.setattr(
        harness_runner.DockerContainerWorkspace, "start", staticmethod(fake_start)
    )
    config = HarnessConfig(
        cache_dir=str(tmp_path / "cache"), dist_mode="mount", install_dir="/opt/h"
    )
    evaluation = make_evaluation(tmp_path, harness_config=config)
    manifest = evaluation.metadata.model_copy(update={"workspace_type": "docker"})
    evaluation.metadata = manifest
    evaluation.prepare_workspace(make_instance())

    assert started[0]["image"] == (
        "docker.io/swebench/sweb.eval.x86_64.django_1776_django-1234:latest"
    )
    assert started[0]["pull_missing"] is True
    assert "/opt/h:ro" in started[0]["bind_volumes"][0]


def test_prepare_workspace_requires_docker(tmp_path: Path) -> None:
    evaluation = make_evaluation(tmp_path)
    evaluation.metadata = evaluation.metadata.model_copy(
        update={"workspace_type": "flex"}
    )
    with pytest.raises(ValueError, match="workspace docker"):
        evaluation.prepare_workspace(make_instance())


def test_evaluate_instance_returns_patch_and_instruction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evaluation = make_evaluation(tmp_path)
    captured: list[Any] = []

    def fake_run(request: Any, **kwargs: Any) -> HarnessRunResult:
        captured.append((request, kwargs))
        return HarnessRunResult(
            git_patch=PATCH,
            instruction=request.instruction,
            history={},
            history_path=str(tmp_path / "out" / "i1.history.json"),
            evaluation=None,
            metrics=None,
            rollout_export_path=None,
        )

    monkeypatch.setattr(harness_runner, "run_harness_instance", fake_run)
    workspace = DockerContainerWorkspace(
        image="example/image:tag",
        container_name="harness-test",
        container_id="container123",
        working_dir="/",
        host_alias="host.docker.internal",
        gateway_ip="172.17.0.1",
    )
    instance = make_instance()
    output = evaluation.evaluate_instance(instance, cast(Any, workspace))

    assert output.test_result == {"git_patch": PATCH}
    assert output.error is None
    assert output.history == []
    assert output.instance_id == "django__django-1234"
    assert instance.data["repo_path"] == "/testbed"
    request, kwargs = captured[0]
    assert request.repo_path == "/testbed"
    assert request.base_commit == "abc123def"
    assert request.conversation_timeout == 600.0
    assert request.extra_env == {"PIP_CACHE_DIR": "~/.cache/pip"}
    # Prompt rendering includes the issue text and the harness note.
    assert "Fix the ORM crash when filtering." in request.instruction
    assert "working-tree changes are captured automatically" in request.instruction
    assert kwargs["judge"] is None


def test_prompt_renders_harness_instructions() -> None:
    from benchmarks.swebench.run_infer import get_instruction

    metadata = SimpleNamespace(
        prompt_path=str(
            (
                Path(harness_runner.__file__).parent / "prompts" / "harness-default.j2"
            ).resolve()
        ),
        details={},
    )
    instance = {
        "repo": "django/django",
        "base_commit": "abc123",
        "problem_statement": "Fix the ORM crash.",
        "repo_path": "/testbed",
    }
    instruction = get_instruction(cast(Any, instance), cast(Any, metadata), "/")
    assert "Fix the ORM crash." in instruction
    assert "/testbed" in instruction
    assert "do NOT create a patch file" in instruction


def test_prepare_workspace_prefers_instance_image_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    started: list[dict[str, Any]] = []
    distribution = HarnessDistribution(
        harness="claude-code",
        version="2.1.331",
        platform="linux-x64",
        directory=tmp_path / "dist",
        binary_path=tmp_path / "dist" / "claude",
        sha256="x",
        size_bytes=1,
    )
    monkeypatch.setattr(
        harness_runner, "ensure_distribution", lambda config: distribution
    )
    monkeypatch.setattr(
        harness_runner.DockerContainerWorkspace,
        "start",
        staticmethod(lambda **kwargs: started.append(kwargs) or FakeWorkspace()),
    )
    instance = make_instance()
    instance.data["image_name"] = "docker.io/custom/sweb.eval.x86_64.repo-1:local"
    evaluation = make_evaluation(tmp_path)
    evaluation.metadata = evaluation.metadata.model_copy(
        update={"workspace_type": "docker"}
    )
    evaluation.prepare_workspace(instance)
    assert started[0]["image"] == "docker.io/custom/sweb.eval.x86_64.repo-1:local"
