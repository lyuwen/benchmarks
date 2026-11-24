"""Tests for harness config helpers (request parameter parsing/merging)."""

from __future__ import annotations

import pytest

from benchmarks.utils.harnesses.config import (
    HarnessConfig,
    merge_request_params,
    parse_request_param_values,
    parse_request_params_json,
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
