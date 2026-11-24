"""Tests for harness config helpers (request parameter parsing/merging)."""

from __future__ import annotations

from argparse import ArgumentParser
from types import SimpleNamespace
from typing import Any

import pytest

from benchmarks.utils.harnesses.config import (
    HarnessConfig,
    merge_request_params,
    parse_request_param_values,
    parse_request_params_json,
    validate_harness_args,
)


def test_parse_request_params_json() -> None:
    assert parse_request_params_json(None) == {}
    assert parse_request_params_json("") == {}
    assert parse_request_params_json('{"temperature": 0.7, "top_p": 0.95}') == {
        "temperature": 0.7,
        "top_p": 0.95,
    }
    with pytest.raises(ValueError, match="not valid JSON"):
        parse_request_params_json("temperature=0.7")
    with pytest.raises(ValueError, match="must be a JSON object"):
        parse_request_params_json("[0.7]")
    with pytest.raises(ValueError, match="structural fields"):
        parse_request_params_json('{"messages": []}')


def test_parse_request_param_values() -> None:
    assert parse_request_param_values([]) == {}
    assert parse_request_param_values(
        ["temperature=0.2", "top_k=20", "reasoning_effort=high", "flag=true"]
    ) == {
        "temperature": 0.2,
        "top_k": 20,
        "reasoning_effort": "high",
        "flag": True,
    }
    with pytest.raises(ValueError, match="expects KEY=VALUE"):
        parse_request_param_values(["temperature"])
    with pytest.raises(ValueError, match="structural fields"):
        parse_request_param_values(["stream=true"])


def test_merge_request_params_overrides_and_validates() -> None:
    assert merge_request_params({"temperature": 0.2}, {"temperature": 0.5}) == {
        "temperature": 0.5
    }
    assert merge_request_params({"temperature": 0.2}, {}) == {"temperature": 0.2}
    with pytest.raises(ValueError, match="tools"):
        merge_request_params({"temperature": 0.2}, {"tools": []})


def test_harness_config_defaults_forward_llm_request_params() -> None:
    config = HarnessConfig()
    assert config.request_params == {}
    assert config.use_llm_request_params is True


def validate_namespace(**overrides: Any) -> SimpleNamespace:
    values: dict[str, Any] = {
        "workspace": "docker",
        "max_attempts": 1,
        "harness": "claude-code",
        "harness_compaction_threshold": 80,
        "harness_context_window": 200_000,
        "rollout_log_upstream_max_chars": 20_000,
        "harness_platform": None,
        "harness_permission_mode": None,
        "harness_bare": False,
        "harness_auto_mode_model": None,
        "harness_max_turns": None,
        "harness_env": [],
        "harness_header": [],
        "request_params": None,
        "request_param": [],
        "harness_include_auxiliary": False,
        "exclude_request_classes": "auxiliary",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_validate_deepseek_harness_platform_and_permission() -> None:
    parser = ArgumentParser()
    with pytest.raises(SystemExit):
        validate_harness_args(
            parser,
            validate_namespace(
                harness="deepseek-harness", harness_platform="linux-x64-musl"
            ),
        )
    with pytest.raises(SystemExit):
        validate_harness_args(
            parser,
            validate_namespace(
                harness="deepseek-harness", harness_permission_mode="auto"
            ),
        )
    validate_harness_args(
        parser,
        validate_namespace(
            harness="deepseek-harness",
            harness_platform="linux-x64",
            harness_permission_mode="danger-full-access",
        ),
    )


def test_validate_deepseek_harness_rejects_claude_only_flags() -> None:
    parser = ArgumentParser()
    for overrides in (
        {"harness_bare": True},
        {"harness_auto_mode_model": "claude-opus-4-6"},
        {"harness_max_turns": 20},
    ):
        with pytest.raises(SystemExit):
            validate_harness_args(
                parser, validate_namespace(harness="deepseek-harness", **overrides)
            )
