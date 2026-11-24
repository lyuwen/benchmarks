"""Tests for the SWE-rebench-V2 blackbox-harness evaluation entry point."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

from benchmarks.swerebenchv2 import run_infer_harnesses as harness_runner
from benchmarks.utils.harnesses.config import HarnessConfig
from benchmarks.utils.harnesses.distribution import HarnessDistribution
from benchmarks.utils.harnesses.runner import HarnessRunResult
from benchmarks.utils.harnesses.workspace import DockerContainerWorkspace
from benchmarks.utils.models import EvalInstance, EvalMetadata
from benchmarks.utils.tests.harness_fakes import PATCH, FakeWorkspace
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
        dataset="thirdparty/SWE-rebench-V2/SWE-rebench-V2-py.jsonl",
        dataset_split="train",
        max_iterations=30,
        eval_output_dir=str(tmp_path / "out"),
        details={"harness": {}},
        prompt_path=str(prompt_path.resolve()),
        env_setup_commands=["export PIP_CACHE_DIR=~/.cache/pip"],
        conversation_timeout=900,
        critic=PassCritic(),
        condenser="none",
    )


def make_instance() -> EvalInstance:
    return EvalInstance(
        id="pandas-dev__pandas-59636",
        data={
            "repo": "pandas-dev/pandas",
            "base_commit": "abc123def",
            "problem_statement": "Fix the groupby crash.",
            "image_name": "docker.io/swerebenchv2/pandas-dev-pandas:59636-abc123",
            "language": "python",
        },
    )


def make_evaluation(tmp_path: Path, **kwargs: Any) -> Any:
    defaults: dict[str, Any] = dict(
        metadata=make_metadata(tmp_path),
        num_workers=1,
        harness_config=HarnessConfig(cache_dir=str(tmp_path / "cache")),
    )
    defaults.update(kwargs)
    return harness_runner.SWERebenchV2HarnessEvaluation(**defaults)


def test_prepare_workspace_uses_dataset_image(
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
    evaluation = make_evaluation(tmp_path)
    evaluation.metadata = evaluation.metadata.model_copy(
        update={"workspace_type": "docker"}
    )
    evaluation.prepare_workspace(make_instance())
    assert started[0]["image"] == (
        "docker.io/swerebenchv2/pandas-dev-pandas:59636-abc123"
    )


def test_prepare_workspace_applies_image_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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
    started: list[dict[str, Any]] = []
    monkeypatch.setattr(
        harness_runner.DockerContainerWorkspace,
        "start",
        staticmethod(lambda **kwargs: started.append(kwargs) or FakeWorkspace()),
    )
    evaluation = make_evaluation(
        tmp_path, docker_image_prefix="registry.example.com/bench"
    )
    evaluation.metadata = evaluation.metadata.model_copy(
        update={"workspace_type": "docker"}
    )
    evaluation.prepare_workspace(make_instance())
    assert started[0]["image"] == (
        "registry.example.com/bench/pandas-dev-pandas:59636-abc123"
    )


def test_evaluate_instance_uses_repo_path_under_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evaluation = make_evaluation(tmp_path)
    captured: list[Any] = []

    def fake_run(request: Any, **kwargs: Any) -> HarnessRunResult:
        captured.append(request)
        return HarnessRunResult(
            git_patch=PATCH,
            instruction=request.instruction,
            history={},
            history_path=str(tmp_path / "out" / "i1.history.json"),
            evaluation=None,
            metrics=None,
        )

    monkeypatch.setattr(harness_runner, "run_harness_instance", fake_run)
    instance = make_instance()
    workspace = DockerContainerWorkspace(
        image="example/image:tag",
        container_name="harness-test",
        container_id="container123",
        working_dir="/",
        host_alias="host.docker.internal",
        gateway_ip="172.17.0.1",
    )
    output = evaluation.evaluate_instance(instance, cast(Any, workspace))
    assert output.test_result == {"git_patch": PATCH}
    request = captured[0]
    assert request.repo_path == "/pandas"
    assert instance.data["repo_path"] == "/pandas"
    assert "Fix the groupby crash." in request.instruction
    assert "/pandas" in request.instruction


def test_prepare_workspace_requires_docker(tmp_path: Path) -> None:
    evaluation = make_evaluation(tmp_path)
    evaluation.metadata = evaluation.metadata.model_copy(
        update={"workspace_type": "flex"}
    )
    with pytest.raises(ValueError, match="workspace docker"):
        evaluation.prepare_workspace(make_instance())
