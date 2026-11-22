"""Tests for the mini entry points' option validation."""

from argparse import ArgumentParser, Namespace

import pytest

from benchmarks.scaleswe import run_infer_miniswe as scaleswe_miniswe
from benchmarks.swebench import run_infer_miniswe as swebench_miniswe


@pytest.mark.parametrize(
    "validate",
    [swebench_miniswe._validate_mini_args, scaleswe_miniswe._validate_mini_args],
)
@pytest.mark.parametrize("critic", ["finish_with_patch", "agent_finished"])
def test_event_dependent_critic_is_rejected(validate, critic: str) -> None:
    parser = ArgumentParser()
    args = Namespace(critic=critic, condenser="none")

    with pytest.raises(SystemExit):
        validate(parser, args)


@pytest.mark.parametrize(
    "validate",
    [swebench_miniswe._validate_mini_args, scaleswe_miniswe._validate_mini_args],
)
def test_condenser_is_rejected(validate) -> None:
    parser = ArgumentParser()
    args = Namespace(critic="pass", condenser="legacy")

    with pytest.raises(SystemExit):
        validate(parser, args)


@pytest.mark.parametrize(
    "validate",
    [swebench_miniswe._validate_mini_args, scaleswe_miniswe._validate_mini_args],
)
def test_supported_critic_passes(validate) -> None:
    parser = ArgumentParser()
    args = Namespace(critic="empty_patch_critic", condenser="none")

    validate(parser, args)
