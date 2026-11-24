"""Tests for the mini entry points' shared CLI helpers."""

from argparse import ArgumentParser, Namespace

import pytest

from benchmarks.scaleswe import run_infer_miniswe as scaleswe_miniswe
from benchmarks.swebench import run_infer_miniswe as swebench_miniswe
from benchmarks.swerebenchv2 import run_infer_miniswe as swerebenchv2_miniswe
from benchmarks.utils.miniswe import (
    add_mini_common_args,
    mini_progress_console,
    mini_progress_path,
    resolve_bash_env_file,
    validate_mini_args,
)
from benchmarks.z021data import run_infer_miniswe as z021data_miniswe


MINI_RUNNERS = [
    swebench_miniswe,
    scaleswe_miniswe,
    z021data_miniswe,
    swerebenchv2_miniswe,
]


@pytest.mark.parametrize("module", MINI_RUNNERS)
def test_event_dependent_critic_is_rejected(module) -> None:
    parser = ArgumentParser()
    args = Namespace(critic="finish_with_patch", condenser="none")

    with pytest.raises(SystemExit):
        module.validate_mini_args(parser, args)


@pytest.mark.parametrize("module", MINI_RUNNERS)
def test_condenser_is_rejected(module) -> None:
    parser = ArgumentParser()
    args = Namespace(critic="pass", condenser="legacy")

    with pytest.raises(SystemExit):
        module.validate_mini_args(parser, args)


@pytest.mark.parametrize("module", MINI_RUNNERS)
def test_supported_critic_passes(module) -> None:
    parser = ArgumentParser()
    args = Namespace(critic="empty_patch_critic", condenser="none")

    module.validate_mini_args(parser, args)


def test_validate_mini_args_matches_shared_helper() -> None:
    parser = ArgumentParser()
    args = Namespace(critic="pass", condenser="none")

    assert validate_mini_args(parser, args) is None


def test_common_args_expose_expected_defaults() -> None:
    parser = ArgumentParser()
    add_mini_common_args(parser)
    args = parser.parse_args([])

    assert args.mini_progress == "console"
    assert args.mini_cost_limit == 0.0
    assert args.mini_command_timeout == 60
    assert args.mini_bash_env_file is None


def test_progress_helpers() -> None:
    assert mini_progress_path("console", "/out", "i-1") == (
        "/out/logs/instance_i-1.mini.log"
    )
    assert mini_progress_path("none", "/out", "i-1") is None
    assert mini_progress_console("console") is True
    assert mini_progress_console("file") is False


def test_resolve_bash_env_file() -> None:
    assert resolve_bash_env_file(None) is None
    assert resolve_bash_env_file("none") is None
    assert resolve_bash_env_file("") is None
    assert resolve_bash_env_file("/home/openhands/.bashrc") == "/home/openhands/.bashrc"
