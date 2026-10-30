"""Tests for the configurable condenser backends used by benchmark runs."""

import argparse
from typing import Any, cast

import pytest
from pydantic import ValidationError

from benchmarks.utils.args_parser import get_parser
from benchmarks.utils.condenser import (
    CONDENSER_KINDS,
    CONDENSER_LLM_USAGE_ID,
    get_condenser,
    parse_condenser_config,
)
from benchmarks.utils.models import EvalMetadata
from openhands.sdk import LLM, Agent
from openhands.sdk.context.condenser import LLMSummarizingCondenser
from openhands.sdk.critic import PassCritic


def make_llm() -> LLM:
    return LLM(model="test-model", usage_id="main-agent")


def make_metadata(**overrides: Any) -> EvalMetadata:
    kwargs: dict[str, Any] = {
        "llm": make_llm(),
        "dataset": "princeton-nlp/SWE-bench_Verified",
        "max_iterations": 10,
        "eval_output_dir": "/tmp/eval",
        "critic": PassCritic(),
    }
    kwargs.update(overrides)
    return EvalMetadata(**kwargs)


def test_none_disables_condensation() -> None:
    assert get_condenser("none", make_llm()) is None


def test_legacy_builds_event_count_condenser_with_own_llm() -> None:
    base_llm = make_llm()
    condenser = get_condenser("legacy", base_llm)

    assert isinstance(condenser, LLMSummarizingCondenser)
    assert condenser.llm is not base_llm
    assert condenser.llm.usage_id == CONDENSER_LLM_USAGE_ID
    assert (condenser.max_size, condenser.keep_first) == (80, 4)


def test_legacy_config_overrides_defaults() -> None:
    condenser = get_condenser("legacy", make_llm(), {"max_size": 40, "keep_first": 2})

    assert isinstance(condenser, LLMSummarizingCondenser)
    assert (condenser.max_size, condenser.keep_first) == (40, 2)


def test_condenser_llm_is_discovered_for_metrics() -> None:
    agent = Agent(
        llm=make_llm(),
        tools=[],
        condenser=get_condenser("legacy", make_llm()),
    )

    usage_ids = {llm.usage_id for llm in agent.get_all_llms()}
    assert usage_ids == {"main-agent", CONDENSER_LLM_USAGE_ID}


def test_replay_backend_is_resolved_from_the_running_sdk(monkeypatch) -> None:
    calls: dict[str, Any] = {}

    class FakeReplayCondenser:
        def __init__(self, **kwargs: Any) -> None:
            calls.update(kwargs)

    monkeypatch.setattr(
        "openhands.sdk.context.condenser.ReplaySummarizingCondenser",
        FakeReplayCondenser,
        raising=False,
    )

    condenser = get_condenser("replay", make_llm(), {"threshold_ratio": 0.7})

    assert isinstance(condenser, FakeReplayCondenser)
    assert calls["threshold_ratio"] == 0.7
    assert calls["llm"].usage_id == CONDENSER_LLM_USAGE_ID


def test_replay_backend_reports_missing_sdk_support(monkeypatch) -> None:
    monkeypatch.setattr(
        "openhands.sdk.context.condenser.ReplaySummarizingCondenser",
        None,
        raising=False,
    )

    with pytest.raises(RuntimeError, match="ReplaySummarizingCondenser"):
        get_condenser("replay", make_llm())


def test_unknown_kind_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown condenser kind"):
        get_condenser(cast(Any, "bogus"), make_llm())


def test_parse_condenser_config_accepts_json_object() -> None:
    assert parse_condenser_config('{"threshold_ratio": 0.7}') == {
        "threshold_ratio": 0.7
    }
    assert parse_condenser_config("{}") == {}


@pytest.mark.parametrize("value", ["not json", "[1, 2]", '"text"'])
def test_parse_condenser_config_rejects_invalid_payloads(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        parse_condenser_config(value)


def test_parser_defaults_to_no_condenser() -> None:
    args = get_parser(add_llm_config=False).parse_args([])

    assert args.condenser == "none"
    assert args.condenser_config is None


@pytest.mark.parametrize("kind", CONDENSER_KINDS)
def test_parser_accepts_all_condenser_kinds(kind: str) -> None:
    args = get_parser(add_llm_config=False).parse_args(["--condenser", kind])

    assert args.condenser == kind


def test_parser_accepts_condenser_config() -> None:
    args = get_parser(add_llm_config=False).parse_args(
        ["--condenser", "replay", "--condenser-config", '{"keep_first": 1}']
    )

    assert args.condenser_config == {"keep_first": 1}


def test_parser_rejects_unknown_condenser() -> None:
    with pytest.raises(SystemExit):
        get_parser(add_llm_config=False).parse_args(["--condenser", "bogus"])


def test_metadata_defaults_preserve_historical_behavior() -> None:
    metadata = make_metadata()

    assert metadata.condenser == "none"
    assert metadata.condenser_config is None


def test_metadata_round_trips_condenser_settings() -> None:
    metadata = make_metadata(
        condenser="replay", condenser_config={"threshold_ratio": 0.7}
    )

    restored = EvalMetadata.model_validate_json(metadata.model_dump_json())
    assert restored.condenser == "replay"
    assert restored.condenser_config == {"threshold_ratio": 0.7}


def test_metadata_rejects_unknown_condenser() -> None:
    with pytest.raises(ValidationError):
        make_metadata(condenser="bogus")
