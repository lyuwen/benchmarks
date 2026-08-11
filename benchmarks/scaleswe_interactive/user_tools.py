"""Read-only tools offered to the user agent in --user-tools readonly mode."""
from __future__ import annotations

import shlex

from benchmarks.scaleswe_interactive.readonly_guard import is_readonly_command

FINISH_TOOL_NAME = "finish"

_MAX_TOOL_OUTPUT = 10000


def _tool(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


USER_READONLY_TOOLS: list[dict] = [
    _tool("read_file", "Read a file from the repository (read-only).",
          {"path": {"type": "string", "description": "Repo-relative path."}},
          ["path"]),
    _tool("grep", "Search file contents with grep (read-only).",
          {"pattern": {"type": "string"},
           "path": {"type": "string", "description": "Repo-relative dir/file."}},
          ["pattern"]),
    _tool("glob", "List files matching a glob under the repo (read-only).",
          {"pattern": {"type": "string"}}, ["pattern"]),
    _tool("run_readonly_bash",
          "Run a strictly read-only shell command in the repo. "
          "Write/mutating commands are rejected.",
          {"command": {"type": "string"}}, ["command"]),
    _tool(FINISH_TOOL_NAME,
          "End the session because the problem is solved (user only).",
          {"reason": {"type": "string"}}, ["reason"]),
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
