"""Docker workspace tuned for benchmark batch runs.

The SDK's ``DockerWorkspace`` waits a hardcoded 120 s for the agent server to
answer ``/health`` and, when that wait fails, leaves the container running: the
log-streaming thread keeps a reference to the workspace, so ``__del__`` never
runs and the container is never cleaned up. With large task images (a
SWE-bench agent-server image takes minutes to import and start on a busy host)
this turns one slow start into a cascade: every retry fails at the same 120 s
mark and leaks one more container, which makes each following start slower.

This subclass keeps the benchmark retry policy intact and fixes the two
problems:

- the health wait budget comes from ``OH_WORKSPACE_STARTUP_TIMEOUT`` (seconds,
  default 600) or the ``startup_timeout_sec`` field;
- a failed start stops/removes its container before re-raising, and the failure
  message includes the tail of the container logs so a genuinely broken image
  (crash loop) is distinguishable from a slow start.

It is a drop-in replacement; ``run_infer`` paths that select the ``docker``
workspace use it, while ``flex``/``remote`` workspaces are untouched.
"""

from __future__ import annotations

import os
from typing import Any

from pydantic import Field

from openhands.sdk import get_logger
from openhands.sdk.utils.command import execute_command
from openhands.workspace import DockerWorkspace


logger = get_logger(__name__)

DEFAULT_STARTUP_TIMEOUT_SEC = 600.0
"""Default agent-server health wait. Large images need several minutes."""

ENV_STARTUP_TIMEOUT = "OH_WORKSPACE_STARTUP_TIMEOUT"
"""Environment override for the startup wait, in seconds."""

LOG_TAIL_CHARS = 4000


def resolve_startup_timeout(value: float | None = None) -> float:
    """Resolve the agent-server health wait budget in seconds.

    Precedence: explicit field value, then ``OH_WORKSPACE_STARTUP_TIMEOUT``, then
    :data:`DEFAULT_STARTUP_TIMEOUT_SEC`. Non-positive or unparsable values are
    ignored with a warning.
    """
    if value is not None and value > 0:
        return float(value)
    raw = os.getenv(ENV_STARTUP_TIMEOUT)
    if raw:
        try:
            parsed = float(raw)
        except ValueError:
            logger.warning(
                "Ignoring invalid %s=%r (expected seconds); using %.0fs",
                ENV_STARTUP_TIMEOUT,
                raw,
                DEFAULT_STARTUP_TIMEOUT_SEC,
            )
        else:
            if parsed > 0:
                return parsed
            logger.warning(
                "Ignoring non-positive %s=%r; using %.0fs",
                ENV_STARTUP_TIMEOUT,
                raw,
                DEFAULT_STARTUP_TIMEOUT_SEC,
            )
    return DEFAULT_STARTUP_TIMEOUT_SEC


class BenchDockerWorkspace(DockerWorkspace):
    """``DockerWorkspace`` with a configurable startup wait and leak-free failures."""

    startup_timeout_sec: float | None = Field(
        default=None,
        description=(
            "Seconds to wait for the agent server health endpoint "
            f"(default: ${ENV_STARTUP_TIMEOUT} or "
            f"{DEFAULT_STARTUP_TIMEOUT_SEC:.0f}s)"
        ),
    )

    def _start_container(self, image: str, context: Any) -> None:
        timeout = resolve_startup_timeout(self.startup_timeout_sec)
        logger.info(
            "Starting agent-server container from %s (health wait budget %.0fs)",
            image,
            timeout,
        )
        try:
            super()._start_container(image, context)
        except Exception as exc:
            container_id = getattr(self, "_container_id", None)
            log_tail = self._container_log_tail(container_id)
            self.cleanup()
            message = f"Agent-server container failed to start from {image}: {exc}"
            if log_tail:
                message = f"{message}\nLast container log lines:\n{log_tail}"
            raise RuntimeError(message) from exc

    def _wait_for_health(self, timeout: float | None = None) -> None:
        # The SDK calls this with a hardcoded 120s for startup and for
        # resume-after-pause. Both use the configured benchmark budget instead.
        super()._wait_for_health(
            timeout=resolve_startup_timeout(self.startup_timeout_sec)
        )

    @staticmethod
    def _container_log_tail(container_id: str | None, lines: int = 20) -> str:
        """Return the last container log lines, for failure diagnostics."""
        if not container_id:
            return ""
        try:
            result = execute_command(
                ["docker", "logs", "--tail", str(lines), container_id]
            )
        except Exception as exc:  # pragma: no cover - docker log retrieval
            logger.debug("Could not read container logs: %s", exc)
            return ""
        parts = [part for part in (result.stdout, result.stderr) if part]
        return "\n".join(parts).strip()[-LOG_TAIL_CHARS:]
