"""Lightweight progress reporting for the harness JSONL tools.

The flattening/streamlining commands can process multi-hundred-MB histories, so
they print a periodic status line to stderr while working. The reporter is
enabled by default only on an interactive stderr (or explicitly with
``--progress``); ``--no-progress`` disables it. No external progress-bar
dependency is needed.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from collections.abc import Callable
from typing import TextIO


class ProgressReporter:
    """Periodically print ``[label] <elapsed> key=value ...`` to a stream."""

    def __init__(
        self,
        label: str,
        *,
        enabled: bool,
        interval_sec: float = 5.0,
        stream: TextIO | None = None,
        provider: Callable[[], dict[str, object]] | None = None,
    ) -> None:
        self.label = label
        self.enabled = enabled
        self.interval_sec = max(0.1, float(interval_sec))
        self.stream = stream if stream is not None else sys.stderr
        self._counters: dict[str, object] = {}
        self._counter_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._provider = provider
        self._started = 0.0

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self._started

    def update(self, **counters: object) -> None:
        with self._counter_lock:
            self._counters.update(counters)

    def _line(self) -> str:
        counters = dict(self._provider()) if self._provider is not None else {}
        with self._counter_lock:
            counters.update(self._counters)
        parts = [f"{key}={value}" for key, value in counters.items()]
        suffix = " ".join(parts)
        return f"[{self.label}] {self.elapsed:.1f}s {suffix}".rstrip()

    def emit(self) -> None:
        if not self.enabled:
            return
        print(self._line(), file=self.stream, flush=True)

    def _run(self) -> None:
        while not self._stop.wait(self.interval_sec):
            self.emit()

    def start(self) -> None:
        self._started = time.monotonic()
        if not self.enabled:
            return
        # Print an immediate line so even one huge input shows it is working.
        self.emit()
        self._thread = threading.Thread(
            target=self._run, name=f"{self.label}-progress", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def __enter__(self) -> ProgressReporter:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()


def add_progress_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the shared ``--progress/--no-progress/--progress-interval`` flags."""
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--progress",
        dest="progress",
        action="store_true",
        default=None,
        help="Print periodic progress lines to stderr (default: only on a TTY)",
    )
    group.add_argument(
        "--no-progress",
        dest="progress",
        action="store_false",
        help="Disable periodic progress lines",
    )
    parser.add_argument(
        "--progress-interval",
        type=float,
        default=5.0,
        help="Seconds between progress lines (default: 5)",
    )


def reporter_from_args(
    args: object,
    label: str,
    *,
    is_tty: Callable[[], bool] | None = None,
    provider: Callable[[], dict[str, object]] | None = None,
) -> ProgressReporter:
    """Build a reporter from parsed CLI args (auto -> TTY detection)."""
    progress = getattr(args, "progress", None)
    if progress is None:
        progress = (is_tty or sys.stderr.isatty)()
    return ProgressReporter(
        label,
        enabled=bool(progress),
        interval_sec=float(getattr(args, "progress_interval", 5.0)),
        provider=provider,
    )
