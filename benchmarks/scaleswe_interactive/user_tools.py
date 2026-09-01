"""Read-only tools offered to the user agent in --user-tools readonly mode.

The tools are real SDK ``ToolDefinition`` objects (not raw OpenAI dicts) so
that ``LLM.completion`` can serialize them via ``to_openai_tool()``. The user
agent executes them manually (see ``execute_readonly_tool``); the tools do not
carry executors and their ``.call()`` path is never invoked.
"""
from __future__ import annotations

import shlex
from collections.abc import Sequence

from pydantic import Field

from openhands.sdk.tool import Action, ToolAnnotations, ToolDefinition

from benchmarks.scaleswe_interactive.readonly_guard import is_readonly_command

FINISH_TOOL_NAME = "finish"

_MAX_TOOL_OUTPUT = 10000

_READONLY_ANNOTATIONS = ToolAnnotations(
    readOnlyHint=True, destructiveHint=False, idempotentHint=True,
    openWorldHint=False)


class UserReadFileAction(Action):
    path: str = Field(description="Repo-relative path.")


class UserGrepAction(Action):
    pattern: str = Field(description="Pattern to search for.")
    path: str | None = Field(
        default=None, description="Repo-relative dir/file.")


class UserGlobAction(Action):
    pattern: str = Field(description="Glob pattern under the repo.")


class UserRunReadonlyBashAction(Action):
    command: str = Field(description="Read-only shell command to run.")


class UserFinishAction(Action):
    reason: str = Field(description="Why the session can end.")


class UserReadFileTool(ToolDefinition[UserReadFileAction, None]):
    name = "read_file"

    @classmethod
    def create(cls, *args, **kwargs) -> Sequence["UserReadFileTool"]:
        return [cls(description="Read a file from the repository (read-only).",
                    action_type=UserReadFileAction,
                    annotations=_READONLY_ANNOTATIONS)]


class UserGrepTool(ToolDefinition[UserGrepAction, None]):
    name = "grep"

    @classmethod
    def create(cls, *args, **kwargs) -> Sequence["UserGrepTool"]:
        return [cls(description="Search file contents with grep (read-only).",
                    action_type=UserGrepAction,
                    annotations=_READONLY_ANNOTATIONS)]


class UserGlobTool(ToolDefinition[UserGlobAction, None]):
    name = "glob"

    @classmethod
    def create(cls, *args, **kwargs) -> Sequence["UserGlobTool"]:
        return [cls(description="List files matching a glob under the repo "
                                "(read-only).",
                    action_type=UserGlobAction,
                    annotations=_READONLY_ANNOTATIONS)]


class UserRunReadonlyBashTool(ToolDefinition[UserRunReadonlyBashAction, None]):
    name = "run_readonly_bash"

    @classmethod
    def create(cls, *args, **kwargs) -> Sequence["UserRunReadonlyBashTool"]:
        return [cls(description="Run a strictly read-only shell command in the "
                                "repo. Write/mutating commands are rejected.",
                    action_type=UserRunReadonlyBashAction,
                    annotations=_READONLY_ANNOTATIONS)]


class UserFinishTool(ToolDefinition[UserFinishAction, None]):
    name = FINISH_TOOL_NAME

    @classmethod
    def create(cls, *args, **kwargs) -> Sequence["UserFinishTool"]:
        return [cls(description="End the session because the problem is solved "
                                "(user only).",
                    action_type=UserFinishAction,
                    annotations=_READONLY_ANNOTATIONS)]


FINISH_TOOL: ToolDefinition = UserFinishTool.create()[0]

USER_READONLY_TOOLS: list[ToolDefinition] = [
    UserReadFileTool.create()[0],
    UserGrepTool.create()[0],
    UserGlobTool.create()[0],
    UserRunReadonlyBashTool.create()[0],
    FINISH_TOOL,
]


def _truncate(s: str) -> str:
    return s if len(s) <= _MAX_TOOL_OUTPUT else s[:_MAX_TOOL_OUTPUT] + "\n...[truncated]"


def execute_readonly_tool(workspace, repo_path: str, name: str, arguments: dict) -> str:
    base = repo_path.rstrip("/")
    if name == "read_file":
        path = arguments.get("path")
        if path is None:
            return f"ERROR: missing required argument 'path' for tool '{name}'"
        p = f"{base}/{path}"
        res = workspace.execute_command(f"cat -- {shlex.quote(p)}")
        return _truncate(res.stdout if res.exit_code == 0 else res.stderr)
    if name == "grep":
        pattern = arguments.get("pattern")
        if pattern is None:
            return f"ERROR: missing required argument 'pattern' for tool '{name}'"
        path = arguments.get("path", ".")
        p = f"{base}/{path}"
        cmd = f"grep -rn -- {shlex.quote(pattern)} {shlex.quote(p)}"
        res = workspace.execute_command(cmd)
        return _truncate(res.stdout or res.stderr or "(no matches)")
    if name == "glob":
        pattern = arguments.get("pattern")
        if pattern is None:
            return f"ERROR: missing required argument 'pattern' for tool '{name}'"
        cmd = (f"cd {shlex.quote(base)} && "
               f"find . -path {shlex.quote('./' + pattern)}")
        res = workspace.execute_command(cmd)
        return _truncate(res.stdout or "(no matches)")
    if name == "run_readonly_bash":
        command = arguments.get("command")
        if command is None:
            return f"ERROR: missing required argument 'command' for tool '{name}'"
        if not is_readonly_command(command):
            return ("ERROR: command rejected — only read-only commands are "
                    "allowed for the user agent.")
        res = workspace.execute_command(f"cd {shlex.quote(base)} && {command}")
        return _truncate(res.stdout or res.stderr or "(no output)")
    return f"ERROR: unknown tool {name}"
