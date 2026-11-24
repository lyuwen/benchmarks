"""Tests for loading the benchmark LLM config for rollout capture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from benchmarks.utils.harnesses.config import add_harness_args
from benchmarks.utils.harnesses.llm_config import (
    build_harness_config,
    extract_llm_request_params,
    load_llm_config,
)


def write_config(tmp_path: Path, config: dict[str, Any]) -> SimpleNamespace:
    path = tmp_path / "llm.json"
    path.write_text(json.dumps(config))
    return SimpleNamespace(llm_config_path=str(path))


def test_load_llm_config_forwards_non_transport_headers(tmp_path: Path) -> None:
    args = write_config(
        tmp_path,
        {
            "model": "litellm_proxy/anthropic/claude-sonnet-4",
            "base_url": "http://127.0.0.1:9999",
            "api_key": "secret",
            "litellm_params": {
                "extra_headers": {
                    "anthropic-beta": "interleaved-thinking-2025-05-14",
                    "Authorization": "Bearer leaked",
                    "X-SMG-Routing-Key": "client-key",
                    "x-api-key": "leaked",
                }
            },
        },
    )
    llm, headers, request_params = load_llm_config(cast(Any, args))
    assert llm.model == "litellm_proxy/anthropic/claude-sonnet-4"
    assert headers == {"anthropic-beta": "interleaved-thinking-2025-05-14"}
    assert request_params == {}


def test_load_llm_config_missing_file(tmp_path: Path) -> None:
    args = SimpleNamespace(llm_config_path=str(tmp_path / "missing.json"))
    with pytest.raises(ValueError, match="does not exist"):
        load_llm_config(cast(Any, args))


def test_extract_llm_request_params() -> None:
    params = extract_llm_request_params(
        {
            "model": "openai/deepseek-ai/DeepSeek-V4",
            "temperature": 0.2,
            "top_p": 0.9,
            "top_k": 20,
            "max_output_tokens": 4096,
            "reasoning_effort": "high",
            "seed": 7,
            "litellm_params": {
                "reasoning_effort": "low",
                "presence_penalty": 0.1,
                "extra_body": {
                    "thinking": {"type": "enabled"},
                    "top_k": 99,
                    "tools": [],
                },
            },
            "extra_body": {"min_p": 0.05},
        }
    )
    assert params["temperature"] == 0.2
    assert params["top_p"] == 0.9
    # Top-level values win over litellm_params / extra_body entries.
    assert params["top_k"] == 20
    assert params["reasoning_effort"] == "high"
    assert params["max_tokens"] == 4096
    assert "max_output_tokens" not in params
    assert params["seed"] == 7
    assert params["presence_penalty"] == 0.1
    assert params["thinking"] == {"type": "enabled"}
    assert params["min_p"] == 0.05
    # Structural fields are never forwarded as sampling parameters.
    assert "tools" not in params


def make_args(tmp_path: Path, extra: list[str]) -> Any:
    path = tmp_path / "llm.json"
    path.write_text(
        json.dumps(
            {
                "model": "litellm_proxy/anthropic/claude-sonnet-4",
                "base_url": "http://127.0.0.1:9999",
                "api_key": "secret",
                "temperature": 0.3,
                "litellm_params": {
                    "reasoning_effort": "high",
                    "extra_headers": {"anthropic-beta": "beta-flag"},
                    "extra_body": {"thinking": {"type": "enabled"}},
                },
            }
        )
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("llm_config_path")
    add_harness_args(parser)
    return parser.parse_args([str(path), *extra])


def test_build_harness_config_merges_llm_and_cli_params(tmp_path: Path) -> None:
    args = make_args(
        tmp_path,
        [
            "--request-param",
            "top_p=0.5",
            "--request-param",
            "seed=42",
            "--harness-header",
            "x-extra: 1",
        ],
    )
    llm, config = build_harness_config(args)
    assert llm.model == "litellm_proxy/anthropic/claude-sonnet-4"
    assert config.request_params == {
        "temperature": 0.3,
        "reasoning_effort": "high",
        "thinking": {"type": "enabled"},
        "top_p": 0.5,
        "seed": 42,
    }
    assert config.custom_headers == {"anthropic-beta": "beta-flag", "x-extra": "1"}


def test_build_harness_config_permission_mode(tmp_path: Path) -> None:
    args = make_args(
        tmp_path,
        [
            "--harness-permission-mode",
            "auto",
            "--harness-auto-mode-model",
            "claude-opus-4-6",
        ],
    )
    _, config = build_harness_config(args)
    assert config.permission_mode == "auto"
    assert config.auto_mode_model == "claude-opus-4-6"


def test_build_harness_config_permission_mode_default_model(tmp_path: Path) -> None:
    args = make_args(tmp_path, ["--harness-permission-mode", "auto"])
    _, config = build_harness_config(args)
    assert config.permission_mode == "auto"
    assert config.auto_mode_model is None


def test_build_harness_config_no_llm_params(tmp_path: Path) -> None:
    args = make_args(
        tmp_path, ["--no-llm-request-params", "--request-param", "seed=42"]
    )
    _, config = build_harness_config(args)
    assert config.request_params == {"seed": 42}


def test_build_harness_config_rejects_structural_overrides(
    tmp_path: Path,
) -> None:
    args = make_args(tmp_path, ["--request-param", "messages=[]"])
    with pytest.raises(ValueError, match="structural fields"):
        build_harness_config(args)
