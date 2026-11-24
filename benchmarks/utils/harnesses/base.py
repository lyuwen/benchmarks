"""Harness registry.

Each harness is an external agent CLI that can be installed into a task
container and driven headless. Only Claude Code is implemented today; new
harnesses plug in by adding a class and an entry in ``_FACTORIES``.
"""

from __future__ import annotations

from benchmarks.utils.harnesses.claude_code import ClaudeCodeHarness
from benchmarks.utils.harnesses.config import SUPPORTED_HARNESSES


Harness = ClaudeCodeHarness

_FACTORIES = {
    "claude-code": ClaudeCodeHarness,
}


def create_harness(name: str) -> Harness:
    """Return the harness implementation registered under ``name``."""
    try:
        factory = _FACTORIES[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown harness {name!r}; supported harnesses: "
            f"{', '.join(SUPPORTED_HARNESSES)}"
        ) from exc
    return factory()
