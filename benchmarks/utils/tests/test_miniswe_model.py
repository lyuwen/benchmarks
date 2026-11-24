"""Tests for the OpenHands LLM -> mini LiteLLM model translation."""

import json

import pytest

from benchmarks.utils.miniswe.model import (
    ROLLOUT_ID_HEADERS,
    MiniModelSpec,
    build_mini_model,
    litellm_model_kwargs,
)
from openhands.sdk import LLM


pytest.importorskip("minisweagent")


def _llm(**overrides) -> LLM:
    config = {
        "model": "openai/gpt-4o-mini",
        "api_key": "sk-test",
        "base_url": "http://localhost:1/v1",
        "temperature": 0.2,
        "top_p": 0.8,
        "max_output_tokens": 123,
        "extra_headers": {"X-SMG-Routing-Key": "stale", "X-Test": "1"},
    }
    config.update(overrides)
    return LLM.model_validate(config)


def test_request_kwargs_mirror_openhands_llm_config() -> None:
    kwargs = litellm_model_kwargs(llm=_llm(), rollout_key="rollout-1", timeout=42)

    assert kwargs["api_base"] == "http://localhost:1/v1"
    assert kwargs["api_key"] == "sk-test"
    assert kwargs["temperature"] == 0.2
    assert kwargs["top_p"] == 0.8
    assert kwargs["max_completion_tokens"] == 123
    assert kwargs["timeout"] == 42
    assert kwargs["drop_params"] is True


def test_rollout_headers_are_refreshed_together() -> None:
    kwargs = litellm_model_kwargs(
        llm=_llm(
            extra_headers={
                "X-SMG-Routing-Key": "stale",
                "X-Session-ID": "stale-session",
                "X-Test": "1",
            }
        ),
        rollout_key="rollout-1",
    )

    assert kwargs["extra_headers"] == {
        "X-SMG-Routing-Key": "rollout-1",
        "X-Session-ID": "rollout-1",
        "X-Test": "1",
    }


def test_session_id_alone_is_refreshed_per_rollout() -> None:
    kwargs = litellm_model_kwargs(
        llm=_llm(extra_headers={"X-Session-ID": "stale-session"}),
        rollout_key="rollout-2",
    )

    assert kwargs["extra_headers"] == {"X-Session-ID": "rollout-2"}


def test_rollout_headers_are_not_injected_when_absent() -> None:
    unchanged = litellm_model_kwargs(
        llm=_llm(extra_headers={"X-Test": "1"}), rollout_key="rollout-1"
    )
    assert unchanged["extra_headers"] == {"X-Test": "1"}

    no_headers = litellm_model_kwargs(
        llm=_llm(extra_headers=None), rollout_key="rollout-1"
    )
    assert "extra_headers" not in no_headers


def test_rollout_id_headers_constant() -> None:
    assert ROLLOUT_ID_HEADERS == ("X-SMG-Routing-Key", "X-Session-ID")


def test_unregistered_models_do_not_abort_on_cost_lookup(monkeypatch) -> None:
    monkeypatch.delenv("MSWEA_COST_TRACKING", raising=False)

    model, _ = build_mini_model(llm=_llm())
    assert model.config.cost_tracking == "ignore_errors"
    assert "cost_tracking" not in litellm_model_kwargs(llm=_llm())

    monkeypatch.setenv("MSWEA_COST_TRACKING", "default")
    model, _ = build_mini_model(llm=_llm())
    assert model.config.cost_tracking == "default"


def test_timeout_defaults_to_llm_timeout() -> None:
    kwargs = litellm_model_kwargs(llm=_llm(timeout=17))

    assert kwargs["timeout"] == 17


def test_reasoning_models_drop_sampling_controls() -> None:
    spec_kwargs = litellm_model_kwargs(llm=_llm(model="openai/o1"))

    assert "temperature" not in spec_kwargs
    assert "top_p" not in spec_kwargs


def test_built_model_and_spec_redact_credentials() -> None:
    model, spec = build_mini_model(llm=_llm(), rollout_key="rollout-1")

    assert model.config.model_name == "openai/gpt-4o-mini"
    assert isinstance(spec, MiniModelSpec)
    assert spec.model_kwargs["api_key"] == "<redacted>"
    assert "sk-test" not in json.dumps(spec.model_kwargs)
    assert spec.temperature == 0.2
    assert spec.top_p == 0.8


def test_native_tool_calling_required() -> None:
    with pytest.raises(ValueError, match="native tool calling"):
        build_mini_model(llm=_llm(native_tool_calling=False))
