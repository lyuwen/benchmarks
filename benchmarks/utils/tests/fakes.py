"""Shared fakes for mini-swe-agent adapter tests."""

import json
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from benchmarks.utils.miniswe.environment import SUBMISSION_MARKER
from openhands.sdk.workspace.models import CommandResult, FileOperationResult


def ok_result(command: str, **overrides: Any) -> CommandResult:
    """Build a successful ``CommandResult`` with optional overrides."""
    payload: dict[str, Any] = {
        "command": command,
        "exit_code": 0,
        "stdout": "",
        "stderr": "",
        "timeout_occurred": False,
    }
    payload.update(overrides)
    return CommandResult(**payload)


class FakeWorkspace:
    """Minimal stand-in for ``RemoteWorkspace`` recording executed commands."""

    def __init__(
        self,
        responder: Callable[[str, Any], CommandResult] | None = None,
        working_dir: str = "/workspace",
    ) -> None:
        self.working_dir = working_dir
        self.calls: list[dict[str, Any]] = []
        self.uploads: list[tuple[str, str]] = []
        self._responder = responder or (lambda command, cwd: ok_result(command))

    def execute_command(
        self,
        command: str,
        cwd: str | Path | None = None,
        timeout: float = 30.0,
    ) -> CommandResult:
        self.calls.append({"command": command, "cwd": cwd, "timeout": timeout})
        return self._responder(command, cwd)

    def file_upload(
        self, source_path: str | Path, destination_path: str | Path
    ) -> FileOperationResult:
        self.uploads.append((str(source_path), str(destination_path)))
        return FileOperationResult(
            success=True,
            source_path=str(source_path),
            destination_path=str(destination_path),
        )

    @property
    def commands(self) -> list[str]:
        return [call["command"] for call in self.calls]


PATCH = "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-x\n+y\n"


def submission_workspace(normal_output: str = "ok") -> FakeWorkspace:
    """Workspace that returns a submitted patch when the marker is observed."""

    def responder(command: str, cwd: Any) -> CommandResult:
        if SUBMISSION_MARKER in command:
            return ok_result(command, stdout=f"{SUBMISSION_MARKER}\n{PATCH}")
        return ok_result(command, stdout=normal_output)

    return FakeWorkspace(responder=responder)


class ScriptedModel:
    """Minimal mini ``Model`` protocol implementation driven by commands."""

    def __init__(self, commands: list[str], *, fail: bool = False) -> None:
        self.config = SimpleNamespace(model_name="scripted/test")
        self._commands = list(commands)
        self._calls = 0
        self._fail = fail

    def query(self, messages: list[dict], **kwargs: Any) -> dict:
        if self._fail:
            raise RuntimeError("provider exploded")
        index = min(self._calls, len(self._commands) - 1)
        command = self._commands[index]
        self._calls += 1
        call_id = f"call_{self._calls}"
        return {
            "role": "assistant",
            "content": f"step {self._calls}",
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "arguments": json.dumps({"command": command}),
                    },
                }
            ],
            "extra": {
                "actions": [{"command": command, "tool_call_id": call_id}],
                "response": {"usage": {"prompt_tokens": 10, "completion_tokens": 5}},
                "cost": 0.001,
            },
        }

    def format_message(self, **kwargs: Any) -> dict:
        return dict(kwargs)

    def format_observation_messages(
        self, message: dict, outputs: list[dict], template_vars: dict | None = None
    ) -> list[dict]:
        actions = message.get("extra", {}).get("actions", [])
        return [
            {
                "role": "tool",
                "content": output.get("output", ""),
                "tool_call_id": action["tool_call_id"],
            }
            for action, output in zip(actions, outputs)
        ]

    def get_template_vars(self, **kwargs: Any) -> dict:
        return {}

    def serialize(self) -> dict:
        return {"info": {"config": {"model": {"model_name": self.config.model_name}}}}
