"""Harness registry and shared run types.

Each harness is an external agent CLI that can be installed into a task container
and driven headless. Harnesses plug in by adding a class and an entry in
``_FACTORY_PATHS``; the factory import is lazy so adapters can import the shared
types below without an import cycle.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import TYPE_CHECKING, Any, Protocol

from benchmarks.utils.harnesses.config import SUPPORTED_HARNESSES
from benchmarks.utils.harnesses.workspace import ProcessOutcome


if TYPE_CHECKING:
    from benchmarks.utils.harnesses.config import HarnessConfig
    from benchmarks.utils.harnesses.distribution import HarnessDistribution
    from benchmarks.utils.harnesses.workspace import DockerContainerWorkspace


@dataclass(frozen=True)
class InstalledHarness:
    """Where the harness binary lives inside a task container."""

    binary_path: str
    install_dir: str
    work_dir: str
    settings_path: str
    instruction_path: str
    version: str


@dataclass(frozen=True)
class HarnessRun:
    """Outcome of one headless harness invocation inside the container."""

    argv: list[str]
    env: dict[str, str]
    outcome: ProcessOutcome
    result_event: dict[str, Any] | None

    @property
    def exit_code(self) -> int:
        return self.outcome.exit_code

    @property
    def timed_out(self) -> bool:
        return self.outcome.timed_out


class Harness(Protocol):
    """The interface the per-instance pipeline drives."""

    name: str

    def install(
        self,
        workspace: DockerContainerWorkspace,
        distribution: HarnessDistribution,
        config: HarnessConfig,
    ) -> InstalledHarness: ...

    def run(
        self,
        workspace: DockerContainerWorkspace,
        config: HarnessConfig,
        installed: InstalledHarness,
        *,
        proxy_url: str,
        harness_model: str,
        instruction: str,
        max_turns: int,
        repo_path: str,
        log_path: str,
        timeout_sec: float | None,
        extra_env: dict[str, str] | None = None,
    ) -> HarnessRun: ...


_FACTORY_PATHS = {
    "claude-code": ("benchmarks.utils.harnesses.claude_code", "ClaudeCodeHarness"),
    "deepseek-harness": (
        "benchmarks.utils.harnesses.deepseek_harness",
        "DeepSeekHarness",
    ),
}


def create_harness(name: str) -> Harness:
    """Return the harness implementation registered under ``name``."""
    try:
        module_name, class_name = _FACTORY_PATHS[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown harness {name!r}; supported harnesses: "
            f"{', '.join(SUPPORTED_HARNESSES)}"
        ) from exc
    factory = getattr(import_module(module_name), class_name)
    return factory()
