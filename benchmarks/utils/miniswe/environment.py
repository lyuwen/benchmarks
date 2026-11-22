"""mini-swe-agent ``Environment`` protocol over an OpenHands workspace.

The benchmark ``Evaluation`` still owns workspace allocation and release; this
adapter only borrows the workspace and executes the agent's bash actions inside
the task container. Mini's own environments spawn a subprocess or container per
run, which would duplicate the benchmark's provisioning logic.

Two details are intentionally different from mini's bundled environments:

- stdout and stderr are collected separately by the agent server; they are
  concatenated for the observation ("accounting for how the backend collects
  them") and only stdout is treated as submission payload.
- a submission payload that could cross an agent-server output chunk boundary is
  refused instead of being forwarded as a corrupt patch. The remote workspace
  polls at most 100 bash events per command, so multi-chunk ordering is not
  guaranteed.
"""

from __future__ import annotations

import os
import shlex
import time
from typing import Any

from benchmarks.utils.miniswe.availability import require_miniswe
from openhands.sdk import get_logger
from openhands.sdk.workspace import RemoteWorkspace


logger = get_logger(__name__)

SUBMISSION_MARKER = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
"""First-line marker mini-swe-agent v2 uses to signal a successful submission."""

MAX_SINGLE_OUTPUT_CHARS = 1024 * 1024
"""Largest single ``BashOutput`` chunk an agent server returns (1 MiB)."""

DEFAULT_ENV = {
    "PAGER": "cat",
    "MANPAGER": "cat",
    "LESS": "-R",
    "PIP_PROGRESS_BAR": "off",
    "TQDM_DISABLE": "1",
}
"""Non-interactive shell defaults copied from mini's bundled configs."""


class SubmissionTruncatedError(RuntimeError):
    """Raised when a submission marker is present but the payload may be cut off."""


class WorkspaceCommandError(RuntimeError):
    """Raised when a bash action cannot be delivered to the workspace.

    Unlike a command that runs and fails, a transport error means the workspace
    is unusable; re-raising lets ``Evaluation`` retry the rollout with its own
    workspace-level retry policy instead of letting the agent burn steps against
    a dead container.
    """


class WorkspaceEnvironmentConfig:
    """Lightweight typed config for :class:`WorkspaceEnvironment`.

    Kept as a plain class (not pydantic) so the adapter does not depend on
    mini's config classes, which would force a mini import at module load.
    """

    def __init__(
        self,
        *,
        cwd: str = "/",
        timeout: int = 60,
        interpreter: list[str] | None = None,
        env: dict[str, str] | None = None,
        forward_env: list[str] | None = None,
        bash_env_file: str | None = None,
        deadline: float | None = None,
    ) -> None:
        self.cwd = cwd
        self.timeout = timeout
        self.interpreter = list(interpreter or ["bash", "-lc"])
        self.env = dict(env if env is not None else DEFAULT_ENV)
        self.forward_env = list(forward_env or [])
        self.bash_env_file = bash_env_file
        self.deadline = deadline

    def model_dump(self) -> dict[str, Any]:
        """Return a JSON-serializable dump (mirrors mini's config classes)."""
        return {
            "cwd": self.cwd,
            "timeout": self.timeout,
            "interpreter": list(self.interpreter),
            "env": dict(self.env),
            "forward_env": list(self.forward_env),
            "bash_env_file": self.bash_env_file,
        }


class WorkspaceEnvironment:
    """Execute mini bash actions through ``workspace.execute_command``."""

    def __init__(
        self,
        workspace: RemoteWorkspace,
        *,
        config_class: type[WorkspaceEnvironmentConfig] = WorkspaceEnvironmentConfig,
        **kwargs: Any,
    ) -> None:
        self.workspace = workspace
        self.config = config_class(**kwargs)
        self.logger = logger

    # -- mini Environment protocol ------------------------------------------------

    def get_template_vars(self, **kwargs: Any) -> dict[str, Any]:
        return {**self.config.model_dump(), **kwargs}

    def serialize(self) -> dict[str, Any]:
        return {
            "info": {
                "config": {
                    "environment": self.config.model_dump(),
                    "environment_type": (
                        f"{self.__class__.__module__}.{self.__class__.__name__}"
                    ),
                }
            }
        }

    def execute(
        self,
        action: dict[str, Any],
        cwd: str = "",
        *,
        timeout: int | None = None,
    ) -> dict[str, Any]:
        """Run one bash action and return mini's ``{"output", "returncode", ...}``."""
        require_miniswe()
        from minisweagent.exceptions import TimeExceeded

        command = action.get("command", "")
        effective_cwd = cwd or self.config.cwd
        budget = self._resolve_timeout(timeout)
        if budget <= 0:
            raise TimeExceeded(
                {
                    "role": "exit",
                    "content": "TimeExceeded",
                    "extra": {"exit_status": "TimeExceeded", "submission": ""},
                }
            )

        wrapped = self._wrap_command(command)
        self.logger.debug("mini action cwd=%s command=%r", effective_cwd, command)
        try:
            result = self.workspace.execute_command(
                wrapped, cwd=effective_cwd, timeout=budget
            )
        except Exception as exc:  # transport failure: let Evaluation retry
            self.logger.warning("mini action failed to reach the workspace: %s", exc)
            raise WorkspaceCommandError(
                f"Failed to execute a mini action in the workspace: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        if result.timeout_occurred:
            return {
                "output": self._merge_streams(result.stdout, result.stderr),
                "returncode": -1,
                "exception_info": (
                    f"Command timed out after {budget:g} seconds. The command was "
                    "killed; partial output is shown above."
                ),
            }

        # Raises mini's ``Submitted`` (or SubmissionTruncatedError) on success.
        self._check_finished(result.stdout, result.exit_code)

        output = self._merge_streams(result.stdout, result.stderr)
        if result.exit_code != 0 and result.stderr and not result.stdout:
            output = result.stderr
        return {"output": output, "returncode": result.exit_code, "exception_info": ""}

    # -- helpers -----------------------------------------------------------------

    def deadline_exceeded(self) -> bool:
        return self.config.deadline is not None and (
            time.monotonic() >= self.config.deadline
        )

    def _resolve_timeout(self, timeout: int | None = None) -> float:
        """Per-command budget, clamped by the remaining rollout deadline."""
        budget = float(self.config.timeout if timeout is None else timeout)
        if self.config.deadline is not None:
            remaining = self.config.deadline - time.monotonic()
            budget = min(budget, remaining)
        return budget

    def _wrap_command(self, command: str) -> str:
        """Build the shell line executed by the workspace.

        The workspace runs commands with a POSIX shell; env assignments and the
        configured interpreter are applied here so quoting (including multiline
        commands) is handled by ``shlex.quote`` instead of string concatenation.
        """
        parts: list[str] = []
        env = dict(self.config.env)
        if self.config.bash_env_file:
            env.setdefault("BASH_ENV", self.config.bash_env_file)
        for key in self.config.forward_env:
            value = os.getenv(key)
            if value is not None:
                env.setdefault(key, value)
        for key, value in env.items():
            parts.append(f"{key}={shlex.quote(value)}")
        parts.extend(shlex.quote(arg) for arg in self.config.interpreter)
        parts.append(shlex.quote(command))
        return " ".join(parts)

    @staticmethod
    def _merge_streams(stdout: str, stderr: str) -> str:
        if not stderr:
            return stdout
        if not stdout:
            return stderr
        separator = "" if stdout.endswith("\n") else "\n"
        return f"{stdout}{separator}{stderr}"

    def _check_finished(self, stdout: str, returncode: int) -> None:
        """Raise mini's ``Submitted`` when the submission marker starts stdout."""
        if returncode != 0:
            return
        lines = stdout.lstrip().splitlines(keepends=True)
        if not lines or lines[0].strip() != SUBMISSION_MARKER:
            return
        if len(stdout) >= MAX_SINGLE_OUTPUT_CHARS:
            raise SubmissionTruncatedError(
                "Submission output is at least the agent-server output chunk limit "
                f"({MAX_SINGLE_OUTPUT_CHARS} chars). The transport may truncate or "
                "reorder the payload, so this submission is rejected; write a "
                "smaller patch or split the diff."
            )
        submission = "".join(lines[1:])
        from minisweagent.exceptions import Submitted

        raise Submitted(
            {
                "role": "exit",
                "content": submission,
                "extra": {"exit_status": "Submitted", "submission": submission},
            }
        )
