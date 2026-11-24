"""LLM config loading shared by the harness inference entry points.

The rollout manager proxies Anthropic Messages traffic to an *OpenAI-compatible*
upstream. The benchmark LLM config describes the same backend, so it supplies
the upstream URL, key and model. Provider headers configured in
``litellm_params.extra_headers`` are forwarded through the harness instead (the
proxy passes client headers to the upstream).
"""

from __future__ import annotations

import os
from typing import Any, cast

from omegaconf import OmegaConf

from benchmarks.utils.harnesses.config import (
    REQUEST_PARAM_FORBIDDEN_KEYS,
    HarnessConfig,
    harness_config_from_args,
    merge_request_params,
)
from openhands.sdk import LLM, get_logger


logger = get_logger(__name__)

# Headers the proxy owns or that must never be copied from a config.
# Request parameters taken from the benchmark LLM config and forced onto the
# harness traffic by the rollout manager. Only sampling/generation parameters:
# structural fields are excluded via REQUEST_PARAM_FORBIDDEN_KEYS.
_REQUEST_PARAM_KEYS = (
    "temperature",
    "top_p",
    "top_k",
    "seed",
    "reasoning_effort",
    "max_tokens",
    "max_output_tokens",
    "presence_penalty",
    "frequency_penalty",
    "repetition_penalty",
)

_PROXY_OWNED_HEADERS = frozenset(
    {
        "authorization",
        "x-api-key",
        "content-type",
        "host",
        "x-smg-routing-key",
        "x-session-id",
    }
)


def extract_llm_request_params(llm_config: dict[str, Any]) -> dict[str, Any]:
    """Extract sampling/generation params to force onto harness traffic.

    Explicit top-level values win over ``litellm_params`` values, which win over
    provider ``extra_body`` entries. ``max_output_tokens`` is exposed as the
    upstream ``max_tokens`` name.
    """
    params: dict[str, Any] = {}
    for key in _REQUEST_PARAM_KEYS:
        value = llm_config.get(key)
        if value is not None:
            params[key] = value
    litellm_params = llm_config.get("litellm_params")
    if isinstance(litellm_params, dict):
        for key in _REQUEST_PARAM_KEYS:
            value = litellm_params.get(key)
            if value is not None:
                params.setdefault(key, value)
        extra_body = litellm_params.get("extra_body")
        if isinstance(extra_body, dict):
            for key, value in extra_body.items():
                params.setdefault(str(key), value)
    extra_body = llm_config.get("extra_body")
    if isinstance(extra_body, dict):
        for key, value in extra_body.items():
            params.setdefault(str(key), value)

    if "max_output_tokens" in params and "max_tokens" not in params:
        params["max_tokens"] = params.pop("max_output_tokens")
    for key in REQUEST_PARAM_FORBIDDEN_KEYS:
        params.pop(key, None)
    return params


def load_llm_config(
    args: Any,
) -> tuple[LLM, dict[str, str], dict[str, Any]]:
    """Load the LLM config file.

    Returns the LLM, forwardable provider headers, and the sampling request
    parameters to force through the rollout manager.
    """
    llm_config_path = args.llm_config_path
    if not os.path.isfile(llm_config_path):
        raise ValueError(f"LLM config file {llm_config_path} does not exist")
    llm_config = OmegaConf.to_container(OmegaConf.load(llm_config_path), resolve=True)
    assert isinstance(llm_config, dict), (
        f"LLM config {llm_config_path} must resolve to a JSON object"
    )
    config_data = cast(dict[str, Any], llm_config)
    litellm_params = llm_config.get("litellm_params") or {}
    extra_headers = dict(litellm_params.get("extra_headers") or {})
    forwardable = {
        str(name): str(value)
        for name, value in extra_headers.items()
        if name.lower() not in _PROXY_OWNED_HEADERS
    }
    llm = LLM.model_validate(llm_config)
    request_params = extract_llm_request_params(config_data)
    logger.info("Using LLM config: %s", llm.model_dump_json(indent=2))
    return llm, forwardable, request_params


def build_harness_config(args: Any) -> tuple[LLM, HarnessConfig]:
    """Load the LLM config and apply its headers/sampling params to the harness."""
    llm, config_headers, llm_request_params = load_llm_config(args)
    config = harness_config_from_args(args)
    updates: dict[str, Any] = {}
    if config_headers:
        updates["custom_headers"] = {**config_headers, **config.custom_headers}
    if config.use_llm_request_params:
        updates["request_params"] = merge_request_params(
            llm_request_params, config.request_params
        )
    if updates:
        config = config.model_copy(update=updates)
    return llm, config
