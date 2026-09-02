"""Tests for signal-aware worker shutdown."""

import logging
import signal
import time

import pytest


def test_handler_converts_sigterm_to_exception():
    from benchmarks.utils.evaluation import install_worker_signal_handlers

    original = signal.getsignal(signal.SIGTERM)
    try:
        install_worker_signal_handlers()
        handler = signal.getsignal(signal.SIGTERM)
        assert callable(handler)
        assert handler not in (signal.SIG_DFL, signal.SIG_IGN)
        with pytest.raises(KeyboardInterrupt):
            handler(signal.SIGTERM, None)
    finally:
        signal.signal(signal.SIGTERM, original)


def test_handler_installed_for_sigint_too():
    from benchmarks.utils.evaluation import install_worker_signal_handlers

    originals = (signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGINT))
    try:
        install_worker_signal_handlers()
        handler = signal.getsignal(signal.SIGINT)
        assert callable(handler)
        with pytest.raises(KeyboardInterrupt):
            handler(signal.SIGINT, None)
    finally:
        signal.signal(signal.SIGTERM, originals[0])
        signal.signal(signal.SIGINT, originals[1])


def test_cleanup_pool_escalates_to_kill_after_grace(monkeypatch):
    """SIGTERM first so cleanup can run; SIGKILL only after the deadline."""
    from unittest.mock import Mock

    from benchmarks.utils.evaluation import Evaluation

    proc = Mock()
    proc.is_alive.return_value = True  # never exits on its own
    pool = Mock()
    pool._processes = {1: proc}

    Evaluation._cleanup_pool(Mock(), pool, futures=[], wait=False, grace_seconds=0.2)

    assert proc.terminate.called, "must SIGTERM first so cleanup can run"
    assert proc.kill.called, "must escalate to SIGKILL after the deadline"


def test_grace_seconds_defaults_to_30_when_env_unset(monkeypatch):
    """The production default must stay 30.0 with no env var set."""
    from benchmarks.utils.evaluation import resolve_shutdown_grace_seconds

    monkeypatch.delenv("OH_SHUTDOWN_GRACE_SECONDS", raising=False)
    assert resolve_shutdown_grace_seconds() == 30.0


def test_grace_seconds_honours_valid_env_value(monkeypatch):
    from benchmarks.utils.evaluation import resolve_shutdown_grace_seconds

    monkeypatch.setenv("OH_SHUTDOWN_GRACE_SECONDS", "4.5")
    assert resolve_shutdown_grace_seconds() == 4.5


@pytest.mark.parametrize("bad", ["abc", "0", "-5", "", "nan", "inf"])
def test_grace_seconds_rejects_invalid_env_value(monkeypatch, bad: str):
    """Malformed or non-positive values fall back to 30.0 instead of crashing."""
    from benchmarks.utils.evaluation import resolve_shutdown_grace_seconds

    monkeypatch.setenv("OH_SHUTDOWN_GRACE_SECONDS", bad)
    assert resolve_shutdown_grace_seconds() == 30.0


def test_grace_seconds_warns_on_invalid_env_value(monkeypatch, caplog):
    from benchmarks.utils.evaluation import resolve_shutdown_grace_seconds

    monkeypatch.setenv("OH_SHUTDOWN_GRACE_SECONDS", "abc")
    with caplog.at_level(logging.WARNING):
        assert resolve_shutdown_grace_seconds() == 30.0
    assert "OH_SHUTDOWN_GRACE_SECONDS" in caplog.text


def test_cleanup_pool_reads_grace_from_env(monkeypatch):
    """With no explicit argument the env var drives the deadline."""
    from unittest.mock import Mock

    from benchmarks.utils.evaluation import Evaluation

    monkeypatch.setenv("OH_SHUTDOWN_GRACE_SECONDS", "0.2")
    proc = Mock()
    proc.is_alive.return_value = True
    pool = Mock()
    pool._processes = {1: proc}

    start = time.monotonic()
    Evaluation._cleanup_pool(Mock(), pool, futures=[], wait=False)
    elapsed = time.monotonic() - start

    assert proc.kill.called
    assert elapsed < 5.0, "env grace period was ignored"


def test_explicit_grace_argument_wins_over_env(monkeypatch):
    from unittest.mock import Mock

    from benchmarks.utils.evaluation import Evaluation

    monkeypatch.setenv("OH_SHUTDOWN_GRACE_SECONDS", "60")
    proc = Mock()
    proc.is_alive.return_value = True
    pool = Mock()
    pool._processes = {1: proc}

    start = time.monotonic()
    Evaluation._cleanup_pool(Mock(), pool, futures=[], wait=False, grace_seconds=0.2)
    elapsed = time.monotonic() - start

    assert proc.kill.called
    assert elapsed < 5.0, "explicit grace_seconds must override the env var"
