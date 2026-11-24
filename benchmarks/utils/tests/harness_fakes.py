"""Shared fakes for harness tests (no Docker, no network)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable

from benchmarks.utils.harnesses.workspace import CommandResult


PATCH = """diff --git a/pkg/module.py b/pkg/module.py
--- a/pkg/module.py
+++ b/pkg/module.py
@@ -1,2 +1,2 @@
-x = 1
+x = 2
"""


@dataclass
class FakeWorkspace:
    """Records docker-side calls and answers with scripted results."""

    working_dir: str = "/"
    host_alias: str = "host.docker.internal"
    gateway_ip: str | None = "172.17.0.1"
    dist_mount: str | None = None
    result_for: Callable[[str], CommandResult] | None = None
    calls: list[str] = field(default_factory=list)
    copied: list[tuple[str, str]] = field(default_factory=list)
    written: dict[str, str] = field(default_factory=dict)
    streaming_runs: list[dict[str, Any]] = field(default_factory=list)

    def execute_command(
        self, command: str, cwd: str | None = None, timeout: float = 60.0
    ) -> CommandResult:
        self.calls.append(command)
        if self.result_for is not None:
            return self.result_for(command)
        return CommandResult(stdout="", stderr="", exit_code=0)

    def copy_in(self, local_path: str, remote_path: str) -> None:
        self.copied.append((str(local_path), remote_path))

    def write_text(self, content: str, remote_path: str) -> None:
        self.written[remote_path] = content

    def path_exists(self, remote_path: str) -> bool:
        return remote_path in self.written or bool(self.copied)

    def container_logs(self, tail: int = 100) -> str:
        return "container logs"

    def cleanup(self) -> None:
        pass


def command_result(
    stdout: str = "", stderr: str = "", exit_code: int = 0
) -> CommandResult:
    return CommandResult(stdout=stdout, stderr=stderr, exit_code=exit_code)


def repo_workspace(
    *,
    base_commit: str = "base123",
    untracked: str = "",
    diff: str = PATCH,
) -> FakeWorkspace:
    """A workspace whose git commands behave like a clean task container."""

    def result_for(command: str) -> CommandResult:
        if "rev-parse" in command and "^{commit}" in command:
            return command_result(stdout=f"{base_commit}\n")
        if "git reset --hard" in command:
            return command_result()
        if "git ls-files --others" in command:
            return command_result(stdout=untracked)
        if "git add -A" in command:
            return command_result()
        if "git commit" in command:
            return command_result(exit_code=1)
        if "git --no-pager diff" in command:
            return command_result(stdout=diff)
        return command_result()

    return FakeWorkspace(result_for=result_for)


def assert_matches(pattern: str, command: str) -> re.Match[str]:
    match = re.search(pattern, command)
    assert match is not None, f"{pattern!r} not found in {command!r}"
    return match
