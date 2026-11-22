"""Translate an OpenHands ``LLM`` config into a mini-swe-agent LiteLLM model.

The goal is parity with the OpenHands client so that a model config behaves the
same under either harness: same model alias, API base, credentials, sampling
controls, token limits, extra headers/body and request kwargs. The SDK's own
``select_chat_options`` helper is reused instead of re-deriving provider quirks
(e.g. reasoning models drop temperature, Azure renames the token limit), so the
two harnesses cannot drift apart silently.
"""

from __future__ import annotations

import os
from typing import Any

from pydantic import BaseModel, Field, SecretStr

from benchmarks.utils.miniswe.availability import require_miniswe
from benchmarks.utils.miniswe.history import redact_secrets
from openhands.sdk import LLM, get_logger


logger = get_logger(__name__)

ROUTING_KEY_HEADER = "X-SMG-Routing-Key"
"""Header used by the eval gateway for router affinity (sglang/llm-proxy)."""


class MiniModelSpec(BaseModel):
    """Redacted description of the model actually used for one rollout."""

    model_name: str
    model_kwargs: dict[str, Any] = Field(
        default_factory=dict, description="Redacted LiteLLM request kwargs"
    )
    temperature: float | None = None
    top_p: float | None = None


def build_mini_model(
    *,
    llm: LLM,
    rollout_key: str | None = None,
    timeout: int | None = None,
) -> tuple[Any, MiniModelSpec]:
    """Build mini's ``LitellmModel`` for one rollout.

    Args:
        llm: The benchmark's OpenHands LLM config. Read at rollout time so the
            per-attempt temperature adjustment made by ``Evaluation`` applies.
        rollout_key: Fresh identifier for router affinity. When the config
            declares ``X-SMG-Routing-Key``, its value is replaced so all requests
            of this rollout stick to one backend.
        timeout: Hard per-request timeout in seconds (rollout budget), defaulting
            to the LLM config's own timeout.

    Returns:
        ``(model, spec)`` where the model implements mini's ``Model`` protocol and
        the spec is safe to serialize.
    """
    require_miniswe()
    from minisweagent.models.litellm_model import LitellmModel

    if not llm.native_tool_calling:
        raise ValueError(
            "mini-swe-agent requires native tool calling: the agent only supports "
            "the `bash` tool. Set `native_tool_calling: true` in the LLM config."
        )

    kwargs = litellm_model_kwargs(llm=llm, rollout_key=rollout_key, timeout=timeout)
    # Benchmark gateways serve aliases/local models that LiteLLM has no price
    # entry for; mini raises when cost lookup fails, which would abort every
    # rollout. Missing prices become cost=0 instead (cost metrics and cost limits
    # are then unavailable for that model). Respect an explicit operator setting.
    cost_tracking = os.getenv("MSWEA_COST_TRACKING", "ignore_errors")
    model = LitellmModel(
        model_name=llm.model, model_kwargs=kwargs, cost_tracking=cost_tracking
    )
    spec = MiniModelSpec(
        model_name=llm.model,
        model_kwargs=redact_secrets(kwargs),
        temperature=_optional_float(kwargs.get("temperature")),
        top_p=_optional_float(kwargs.get("top_p")),
    )
    return model, spec


def litellm_model_kwargs(
    *,
    llm: LLM,
    rollout_key: str | None = None,
    timeout: int | None = None,
) -> dict[str, Any]:
    """Return the LiteLLM request kwargs equivalent to ``llm``."""
    # Imported lazily: keeps `benchmarks.utils.miniswe` importable without the
    # extra, and matches how the harness imports SDK internals on demand.
    from openhands.sdk.llm.options.chat_options import select_chat_options

    kwargs: dict[str, Any] = dict(select_chat_options(llm, {}, has_tools=True))
    kwargs["drop_params"] = llm.drop_params
    if llm.seed is not None:
        kwargs.setdefault("seed", llm.seed)

    if llm.api_key is not None:
        kwargs["api_key"] = (
            llm.api_key.get_secret_value()
            if isinstance(llm.api_key, SecretStr)
            else llm.api_key
        )
    if llm.base_url:
        kwargs["api_base"] = llm.base_url
    if llm.api_version:
        kwargs["api_version"] = llm.api_version

    effective_timeout = timeout if timeout is not None else llm.timeout
    if effective_timeout is not None:
        kwargs["timeout"] = int(effective_timeout)

    _apply_routing_key(kwargs, rollout_key)
    return kwargs


def _apply_routing_key(kwargs: dict[str, Any], rollout_key: str | None) -> None:
    """Reuse one routing key across a rollout's requests when configured."""
    if not rollout_key:
        return
    headers = kwargs.get("extra_headers")
    if not isinstance(headers, dict):
        return
    if ROUTING_KEY_HEADER not in headers:
        return
    headers = dict(headers)
    headers[ROUTING_KEY_HEADER] = rollout_key
    kwargs["extra_headers"] = headers


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return None
