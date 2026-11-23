"""Human-readable live progress for mini-swe-agent rollouts.

OpenHands runs stream conversation events to the console while a rollout is in
flight. mini has no event stream, so this module wraps its model/environment
protocols and emits one compact line per interesting event:

- ``rollout_start`` / ``rollout_end`` with the effective limits
- ``model_call`` with latency, content/tool-call counts, tokens and cost
- ``action`` with the bash command (truncated, one line)
- ``observation`` with the return code, output size and a preview
- ``interrupt`` for mini's control-flow exceptions (format errors, submissions,
  limits, deadline)
- ``error`` when the rollout aborts with an uncaught exception

Lines go to three places, matching the existing logging setup:

1. the per-instance framework log (``logs/instance_<id>.log``) via the root
   logger's file handler;
2. a dedicated tail-able file (``logs/instance_<id>.mini.log``);
3. optionally the real console (``sys.__stdout__``, escaping the worker's
   stdout redirection), like the OpenHands conversation-event handler.

The wrappers are transparent: they delegate every other protocol method to the
wrapped object, so rollout behavior is unchanged.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from typing import Any

from openhands.sdk import get_logger


logger = get_logger(__name__)

PROGRESS_LOGGER_NAME = "benchmarks.utils.miniswe.progress"
PREVIEW_CHARS = 320
"""Per-field preview budget so each event stays on one readable line."""


class MiniProgress:
    """Emit one-line mini rollout events to file, framework log and console."""

    def __init__(
        self,
        *,
        path: str | None = None,
        console: bool = True,
        label: str = "",
        preview_chars: int = PREVIEW_CHARS,
    ) -> None:
        self.label = label
        self.preview_chars = preview_chars
        self.step = 0
        self._logger = logging.getLogger(PROGRESS_LOGGER_NAME)
        self._logger.setLevel(logging.INFO)
        self._handlers: list[logging.Handler] = []

        if path:
            directory = os.path.dirname(path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            file_handler = logging.FileHandler(path, encoding="utf-8")
            file_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
            self._logger.addHandler(file_handler)
            self._handlers.append(file_handler)

        if console:
            console_handler = logging.StreamHandler(sys.__stdout__)
            console_handler.setFormatter(logging.Formatter("%(message)s"))
            self._logger.addHandler(console_handler)
            self._handlers.append(console_handler)

    @property
    def enabled(self) -> bool:
        return bool(self._handlers)

    def close(self) -> None:
        for handler in self._handlers:
            self._logger.removeHandler(handler)
            handler.flush()
            if isinstance(handler, logging.FileHandler):
                handler.close()
        self._handlers.clear()

    # -- events -----------------------------------------------------------------

    def rollout_start(
        self,
        *,
        model: str,
        cwd: str,
        step_limit: int,
        command_timeout: int,
        timeout_seconds: int | None,
    ) -> None:
        self._emit(
            "rollout_start",
            model=model,
            cwd=cwd,
            step_limit=step_limit or "unlimited",
            command_timeout=f"{command_timeout}s",
            rollout_timeout=(f"{timeout_seconds}s" if timeout_seconds else "unlimited"),
        )

    def model_call(
        self,
        *,
        duration_seconds: float,
        content: str | None,
        tool_calls: int,
        usage: dict[str, Any] | None,
        cost: float | None,
    ) -> None:
        prompt = _as_int(usage.get("prompt_tokens")) if usage else None
        completion = _as_int(usage.get("completion_tokens")) if usage else None
        self._emit(
            "model_call",
            n=self.step,
            chars=len(content or ""),
            tools=tool_calls,
            dt=f"{duration_seconds:.2f}s",
            tokens=f"{prompt}+{completion}" if usage else "n/a",
            cost=f"${cost:.4f}" if cost is not None else "n/a",
            preview=_preview(content, self.preview_chars),
        )

    def model_error(self, exc: Exception, *, duration_seconds: float) -> None:
        self._emit(
            "model_error",
            n=self.step,
            type=type(exc).__name__,
            dt=f"{duration_seconds:.2f}s",
            detail=_preview(str(exc), self.preview_chars),
        )

    def action(self, command: str) -> None:
        self._emit("action", n=self.step, command=_preview(command, self.preview_chars))

    def observation(
        self,
        *,
        returncode: int | None,
        output: str,
        duration_seconds: float,
    ) -> None:
        self._emit(
            "observation",
            n=self.step,
            rc=returncode,
            chars=len(output),
            dt=f"{duration_seconds:.2f}s",
            preview=_preview(output, self.preview_chars),
        )

    def interrupt(self, exc: Exception, *, duration_seconds: float) -> None:
        self._emit(
            "interrupt",
            n=self.step,
            type=type(exc).__name__,
            dt=f"{duration_seconds:.2f}s",
            detail=_preview(str(exc), self.preview_chars),
        )

    def generation(self) -> None:
        """Advance the logical model-call/step counter."""
        self.step += 1

    def rollout_end(
        self,
        *,
        status: str,
        n_calls: int,
        cost: float,
        duration_seconds: float,
        submission_chars: int,
    ) -> None:
        self._emit(
            "rollout_end",
            status=status,
            calls=n_calls,
            cost=f"${cost:.4f}",
            dt=f"{duration_seconds:.2f}s",
            submission_chars=submission_chars,
        )

    def rollout_error(self, exc: Exception, *, duration_seconds: float) -> None:
        self._emit(
            "rollout_error",
            type=type(exc).__name__,
            dt=f"{duration_seconds:.2f}s",
            detail=_preview(str(exc), self.preview_chars),
        )

    # -- internals --------------------------------------------------------------

    def _emit(self, event: str, **fields: Any) -> None:
        if not self._handlers:
            return
        parts = (
            [f"[miniswe {self.label}]", event] if self.label else ["[miniswe]", event]
        )
        parts.extend(
            f"{key}={value}" for key, value in fields.items() if value is not None
        )
        self._logger.info(" ".join(parts))


class LoggingModel:
    """Mini ``Model`` protocol decorator that logs each model call."""

    def __init__(self, model: Any, progress: MiniProgress) -> None:
        self._model = model
        self._progress = progress

    @property
    def config(self) -> Any:
        return self._model.config

    def query(self, messages: list[dict], **kwargs: Any) -> dict:
        self._progress.generation()
        started = time.monotonic()
        try:
            message = self._model.query(messages, **kwargs)
        except Exception as exc:
            self._progress.model_error(exc, duration_seconds=time.monotonic() - started)
            raise
        extra = message.get("extra") or {}
        response = extra.get("response") or {}
        self._progress.model_call(
            duration_seconds=time.monotonic() - started,
            content=message.get("content"),
            tool_calls=len(extra.get("actions") or []),
            usage=response.get("usage") if isinstance(response, dict) else None,
            cost=extra.get("cost"),
        )
        return message

    def format_message(self, **kwargs: Any) -> dict:
        return self._model.format_message(**kwargs)

    def format_observation_messages(
        self, message: dict, outputs: list[dict], template_vars: dict | None = None
    ) -> list[dict]:
        return self._model.format_observation_messages(message, outputs, template_vars)

    def get_template_vars(self, **kwargs: Any) -> dict[str, Any]:
        return self._model.get_template_vars(**kwargs)

    def serialize(self) -> dict:
        return self._model.serialize()


class LoggingEnvironment:
    """Mini ``Environment`` protocol decorator that logs actions/observations."""

    def __init__(self, environment: Any, progress: MiniProgress) -> None:
        self._environment = environment
        self._progress = progress

    @property
    def config(self) -> Any:
        return self._environment.config

    def execute(self, action: dict, cwd: str = "", **kwargs: Any) -> dict[str, Any]:
        command = str(action.get("command", ""))
        self._progress.action(command)
        started = time.monotonic()
        try:
            output = self._environment.execute(action, cwd, **kwargs)
        except Exception as exc:
            self._progress.interrupt(exc, duration_seconds=time.monotonic() - started)
            raise
        self._progress.observation(
            returncode=output.get("returncode"),
            output=str(output.get("output", "")),
            duration_seconds=time.monotonic() - started,
        )
        return output

    def get_template_vars(self, **kwargs: Any) -> dict[str, Any]:
        return self._environment.get_template_vars(**kwargs)

    def serialize(self) -> dict:
        return self._environment.serialize()


def _preview(value: Any, budget: int) -> str:
    text = str(value).replace("\r", " ").replace("\n", "\\n")
    if len(text) > budget:
        return f"{text[:budget]}…(+{len(text) - budget} chars)"
    return text


def _as_int(value: Any) -> int:
    if isinstance(value, (int, float)):
        return int(value)
    return 0
