"""Tests for signal-aware worker shutdown."""

import signal

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
