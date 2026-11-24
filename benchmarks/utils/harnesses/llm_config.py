"""LLM config loading shared by the harness inference entry points.

The rollout manager proxies Anthropic Messages traffic to an *OpenAI-compatible*
upstream. The benchmark LLM config describes the same backend, so it supplies
the upstream URL, key and model. Provider headers configured in
``litellm_params.extra_headers`` are forwarded through the harness instead (the
proxy passes client headers to the upstream).
"""

from __future__ import annotations

import os
from typing import Any

from omegaconf import OmegaConf

from openhands.sdk import LLM, get_logger


logger = get_logger(__name__)

# Headers the proxy owns or that must never be copied from a config.
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


def load_llm_config(args: Any) -> tuple[LLM, dict[str, str]]:
    """Load the LLM config file and return the LLM plus forwardable headers."""
    llm_config_path = args.llm_config_path
    if not os.path.isfile(llm_config_path):
        raise ValueError(f"LLM config file {llm_config_path} does not exist")
    llm_config = OmegaConf.to_container(OmegaConf.load(llm_config_path), resolve=True)
    assert isinstance(llm_config, dict), (
        f"LLM config {llm_config_path} must resolve to a JSON object"
    )
    litellm_params = llm_config.get("litellm_params") or {}
    extra_headers = dict(litellm_params.get("extra_headers") or {})
    forwardable = {
        str(name): str(value)
        for name, value in extra_headers.items()
        if name.lower() not in _PROXY_OWNED_HEADERS
    }
    llm = LLM.model_validate(llm_config)
    logger.info("Using LLM config: %s", llm.model_dump_json(indent=2))
    return llm, forwardable
