"""Shared fakes for mini-swe-agent adapter tests."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

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
