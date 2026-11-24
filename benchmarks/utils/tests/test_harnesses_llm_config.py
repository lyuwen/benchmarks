"""Tests for loading the benchmark LLM config for rollout capture."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from benchmarks.utils.harnesses.llm_config import load_llm_config


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
    llm, headers = load_llm_config(cast(Any, args))
    assert llm.model == "litellm_proxy/anthropic/claude-sonnet-4"
    assert headers == {"anthropic-beta": "interleaved-thinking-2025-05-14"}


def test_load_llm_config_missing_file(tmp_path: Path) -> None:
    args = SimpleNamespace(llm_config_path=str(tmp_path / "missing.json"))
    with pytest.raises(ValueError, match="does not exist"):
        load_llm_config(cast(Any, args))
