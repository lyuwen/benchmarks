"""Tests for building condenser backends for benchmark runs."""

import importlib
from typing import cast

import pytest
from pydantic import SecretStr

from benchmarks.utils.condenser import (
    CONDENSER_LLM_USAGE_ID,
    CondenserKind,
    get_condenser,
)
from openhands.sdk import LLM


def _llm(stream: bool) -> LLM:
    return LLM(model="gpt-4o", api_key=SecretStr("test-key"), stream=stream)


def _replay_available() -> bool:
    module = importlib.import_module("openhands.sdk.context.condenser")
    return hasattr(module, "ReplaySummarizingCondenser")


@pytest.mark.parametrize("kind", ["legacy", "replay"])
def test_condenser_llm_does_not_inherit_streaming(kind: str):
    if kind == "replay" and not _replay_available():
        pytest.skip("The running SDK does not provide the replay condenser")

    agent_llm = _llm(stream=True)

    condenser = get_condenser(kind, agent_llm)

    assert condenser is not None
    # Condenser completions have no token callback, so streaming must be off.
    assert condenser.llm.stream is False
    assert condenser.llm.usage_id == CONDENSER_LLM_USAGE_ID
    # The agent LLM itself is untouched.
    assert agent_llm.stream is True


def test_none_condenser_is_disabled():
    assert get_condenser("none", _llm(stream=False)) is None


def test_unknown_condenser_kind_is_rejected():
    with pytest.raises(ValueError, match="Unknown condenser kind"):
        get_condenser(cast(CondenserKind, "bogus"), _llm(stream=False))
