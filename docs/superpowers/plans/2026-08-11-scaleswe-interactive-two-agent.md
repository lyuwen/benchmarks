# Scale-SWE Interactive: Two-Agent Collaborative Mode — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a new evaluation task `benchmarks/scaleswe-interactive/` where a read-only LLM "user agent" drives a coding agent through explore→plan→fix→test in one shared workspace, producing a multi-turn trajectory.

**Architecture:** Reuse scaleswe's instance/workspace/git machinery unchanged. Keep a single persisted `Conversation` owned by the coding agent. Replace `run_conversation_with_fake_user_response` with a driver loop that, on each coding-agent yield, invokes a stateless-per-turn `UserAgent` (its own LLM) to produce the next user message or a finish signal. The user agent inspects the repo read-only via file-content injection (always) and an optional read-only tool loop.

**Tech Stack:** Python 3.12, OpenHands SDK (`openhands.sdk`: `LLM`, `Conversation`, `Message`, `TextContent`, `MessageEvent`, `ConversationExecutionStatus`), Jinja2, pydantic, pytest.

## Global Constraints

- Reference implementation to mirror for structure/patterns: `benchmarks/scaleswe/run_infer.py`.
- Reuse scaleswe's `prepare_instances` and `prepare_workspace` **verbatim** (subclass or import); only the agent-execution step differs.
- Preserve the `Conversation(...)` → first `send_message(...)` gap (repo prep between them) — RemoteConversation event-cache timing workaround from `CLAUDE.local.md`.
- The user agent performs **no writes** to the workspace. Read-only enforcement is a bash command allowlist (best-effort, documented).
- Only the **user agent** can finish the session. The coding agent yielding a message is a turn hand-off, never a session end.
- CLI defaults: `--mode plan`, `--user-tools none`, `--max-user-turns 20`. If `--user-llm-config-path` is omitted, the user LLM = coding LLM config.
- SDK types (verified signatures):
  - `llm.completion(messages: list[Message], tools: Sequence[ToolDefinition] | None = None, **kwargs) -> LLMResponse`; `LLMResponse.message: Message`.
  - `Message(role: Literal["user","system","assistant","tool"], content: Sequence[TextContent|ImageContent], tool_calls: list[MessageToolCall] | None, name: str | None)`; `Message.to_chat_dict()`.
  - `MessageToolCall` has `.name` and `.arguments` (JSON string).
  - `MessageEvent(source: SourceType, llm_message: Message)`; `source` is one of `"user"|"agent"|...`.
  - `conversation.state.events`, `conversation.state.execution_status`; `ConversationExecutionStatus.{FINISHED,ERROR,STUCK,RUNNING,IDLE,PAUSED}`.
  - `workspace.execute_command(cmd) -> CommandResult(exit_code:int, stdout:str, stderr:str)`.
- Output: keep scaleswe's `{id}.history.json` + `output.jsonl` shape; add `{id}.interactive.json` side file.

---

## File Structure

- `benchmarks/scaleswe_interactive/__init__.py` — package marker.
- `benchmarks/scaleswe_interactive/readonly_guard.py` — `is_readonly_command(cmd) -> bool`.
- `benchmarks/scaleswe_interactive/file_reference.py` — `extract_file_paths(text) -> list[str]`, `inject_files(workspace, repo_path, paths, max_files, max_bytes) -> list[InjectedFile]`.
- `benchmarks/scaleswe_interactive/user_tools.py` — read-only tool schemas (OpenAI tool dicts) + `execute_readonly_tool(workspace, repo_path, name, args) -> str`.
- `benchmarks/scaleswe_interactive/user_agent.py` — `UserAgent` (stateless-per-turn), `UserTurn` result type.
- `benchmarks/scaleswe_interactive/driver.py` — `run_interactive_session(...) -> SessionResult`.
- `benchmarks/scaleswe_interactive/transcript.py` — `build_interactive_transcript(...) -> dict`.
- `benchmarks/scaleswe_interactive/run_infer.py` — `ScaleSWEInteractiveEvaluation`, `main()`.
- `benchmarks/scaleswe_interactive/prompts/user_system.j2`, `coding_system_plan.j2`, `coding_system_auto.j2`, `initial_instruction.j2`.
- Tests under `benchmarks/scaleswe_interactive/tests/`.

> Note: use underscore package name `scaleswe_interactive` (Python-importable), consistent with `benchmarks/*` packages that need imports. The dataset default still points at the scaleswe dataset.

---

## Task 1: Package skeleton + read-only command guard

**Files:**
- Create: `benchmarks/scaleswe_interactive/__init__.py`
- Create: `benchmarks/scaleswe_interactive/readonly_guard.py`
- Create: `benchmarks/scaleswe_interactive/tests/__init__.py`
- Test: `benchmarks/scaleswe_interactive/tests/test_readonly_guard.py`

**Interfaces:**
- Produces: `is_readonly_command(cmd: str) -> bool` — True only if every segment of a (possibly chained) shell command is on the read-only allowlist.

- [ ] **Step 1: Write the failing test**

```python
# benchmarks/scaleswe_interactive/tests/test_readonly_guard.py
import pytest
from benchmarks.scaleswe_interactive.readonly_guard import is_readonly_command


@pytest.mark.parametrize("cmd", [
    "cat foo.py",
    "grep -rn needle src/",
    "ls -la",
    "git --no-pager log --oneline -10",
    "git --no-pager diff HEAD~1",
    "find . -name '*.py'",
    "head -50 a.py",
    "sed -n '1,80p' a.py",          # read-only sed (no -i)
    "cat a.py | grep x | head -5",  # all segments read-only
])
def test_allows_readonly(cmd):
    assert is_readonly_command(cmd) is True


@pytest.mark.parametrize("cmd", [
    "rm -rf /",
    "echo hi > f",
    "cat a >> b",
    "sed -i 's/a/b/' f",
    "git commit -m x",
    "git apply p.patch",
    "python -c 'open(1)'",
    "tee f",
    "mv a b",
    "cp a b",
    "cat a && rm b",       # chained: one writer taints all
    "grep x f; rm y",      # semicolon chain with writer
    "truncate -s0 f",
])
def test_rejects_writers(cmd):
    assert is_readonly_command(cmd) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/lfu/git-projects/benchmarks-main/.claude/worktrees/scaleswe-interactive && python -m pytest benchmarks/scaleswe_interactive/tests/test_readonly_guard.py -q`
Expected: FAIL — `ModuleNotFoundError: ... readonly_guard`.

- [ ] **Step 3: Write minimal implementation**

```python
# benchmarks/scaleswe_interactive/__init__.py
```

```python
# benchmarks/scaleswe_interactive/tests/__init__.py
```

```python
# benchmarks/scaleswe_interactive/readonly_guard.py
"""Best-effort allowlist guard: only permit read-only shell commands."""
from __future__ import annotations

import re
import shlex

# Command executables that only read.
_READONLY_CMDS = {
    "cat", "ls", "grep", "egrep", "fgrep", "rg", "find", "head", "tail",
    "wc", "less", "more", "file", "stat", "tree", "pwd", "echo", "which",
    "sed", "awk", "cut", "sort", "uniq", "diff", "nl", "od", "xxd", "basename",
    "dirname", "realpath", "readlink", "env", "true", "test", "python3", "python",
}
# Redirection / write operators that taint any segment.
_WRITE_OPERATORS = re.compile(r"(>>|>|\btee\b)")
# Git subcommands that are read-only.
_GIT_READONLY_SUB = {"log", "diff", "show", "status", "blame", "ls-files",
                     "cat-file", "rev-parse", "branch", "describe", "grep"}
# sed/awk in-place or program flags that write.
_SED_WRITE = re.compile(r"-i\b|--in-place")


def _segment_is_readonly(seg: str) -> bool:
    seg = seg.strip()
    if not seg:
        return True
    if _WRITE_OPERATORS.search(seg):
        return False
    try:
        tokens = shlex.split(seg)
    except ValueError:
        return False
    if not tokens:
        return True
    exe = tokens[0]
    # python -c is arbitrary code -> reject.
    if exe in {"python", "python3"} and any(t == "-c" for t in tokens):
        return False
    if exe == "git":
        sub = next((t for t in tokens[1:] if not t.startswith("-")), None)
        return sub in _GIT_READONLY_SUB
    if exe in {"sed"}:
        return not _SED_WRITE.search(seg)
    return exe in _READONLY_CMDS


def is_readonly_command(cmd: str) -> bool:
    """Return True only if every chained segment is read-only."""
    # Split on chaining/pipe operators; every part must be read-only.
    parts = re.split(r"&&|\|\||;|\|", cmd)
    return all(_segment_is_readonly(p) for p in parts)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/lfu/git-projects/benchmarks-main/.claude/worktrees/scaleswe-interactive && python -m pytest benchmarks/scaleswe_interactive/tests/test_readonly_guard.py -q`
Expected: PASS (all parametrized cases).

- [ ] **Step 5: Commit**

```bash
git add benchmarks/scaleswe_interactive/__init__.py benchmarks/scaleswe_interactive/readonly_guard.py benchmarks/scaleswe_interactive/tests/
git commit -m "feat(scaleswe-interactive): read-only command guard + package skeleton"
```

---

## Task 2: File-reference extraction and injection

**Files:**
- Create: `benchmarks/scaleswe_interactive/file_reference.py`
- Test: `benchmarks/scaleswe_interactive/tests/test_file_reference.py`

**Interfaces:**
- Consumes: `workspace.execute_command(cmd) -> CommandResult` (duck-typed in tests via a fake).
- Produces:
  - `extract_file_paths(text: str) -> list[str]` — de-duplicated, order-preserving repo-relative-looking paths.
  - `InjectedFile` = `TypedDict("InjectedFile", {"path": str, "bytes": int, "content": str, "skipped": str | None})`.
  - `inject_files(workspace, repo_path: str, paths: list[str], max_files: int = 5, max_bytes: int = 20000) -> list[InjectedFile]`.

- [ ] **Step 1: Write the failing test**

```python
# benchmarks/scaleswe_interactive/tests/test_file_reference.py
from benchmarks.scaleswe_interactive.file_reference import (
    extract_file_paths,
    inject_files,
)


def test_extract_paths_from_prose_and_backticks_and_lineref():
    text = (
        "The bug is in src/pkg/mod.py and also `lib/util.js`.\n"
        "See tests/test_x.py:42 for the failing case. Ignore http://x.com/a.py"
    )
    paths = extract_file_paths(text)
    assert "src/pkg/mod.py" in paths
    assert "lib/util.js" in paths
    assert "tests/test_x.py" in paths          # line suffix stripped
    assert all(not p.startswith("http") for p in paths)
    # order-preserving, de-duplicated
    assert paths == list(dict.fromkeys(paths))


class _FakeWS:
    def __init__(self, files):
        self._files = files

    def execute_command(self, cmd):
        # cmd looks like: cat -- '<repo>/<path>'
        class R:  # minimal CommandResult stand-in
            pass
        r = R()
        for path, content in self._files.items():
            if path in cmd:
                r.exit_code, r.stdout, r.stderr = 0, content, ""
                return r
        r.exit_code, r.stdout, r.stderr = 1, "", "No such file"
        return r


def test_inject_files_respects_caps_and_marks_missing():
    ws = _FakeWS({"a.py": "print(1)\n", "big.py": "x" * 100})
    out = inject_files(ws, "/repo", ["a.py", "big.py", "missing.py"],
                       max_files=5, max_bytes=10)
    by_path = {o["path"]: o for o in out}
    assert by_path["a.py"]["content"] == "print(1)\n"
    assert by_path["a.py"]["skipped"] is None
    assert by_path["big.py"]["skipped"] == "too_large"
    assert by_path["missing.py"]["skipped"] == "not_found"


def test_inject_files_caps_file_count():
    ws = _FakeWS({f"f{i}.py": "y" for i in range(10)})
    out = inject_files(ws, "/repo", [f"f{i}.py" for i in range(10)],
                       max_files=3, max_bytes=1000)
    injected = [o for o in out if o["skipped"] is None]
    assert len(injected) == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/lfu/git-projects/benchmarks-main/.claude/worktrees/scaleswe-interactive && python -m pytest benchmarks/scaleswe_interactive/tests/test_file_reference.py -q`
Expected: FAIL — `ModuleNotFoundError: ... file_reference`.

- [ ] **Step 3: Write minimal implementation**

```python
# benchmarks/scaleswe_interactive/file_reference.py
"""Extract file paths referenced in text and inject their contents (read-only)."""
from __future__ import annotations

import re
import shlex
from typing import TypedDict


class InjectedFile(TypedDict):
    path: str
    bytes: int
    content: str
    skipped: str | None  # None | "too_large" | "not_found" | "count_cap"


# path-like tokens: a/b/c.ext optionally followed by :line
_PATH_RE = re.compile(
    r"(?<![\w/])"                     # not preceded by word char or slash
    r"([A-Za-z0-9_.\-]+(?:/[A-Za-z0-9_.\-]+)+"  # has at least one slash
    r"\.[A-Za-z0-9]+)"               # ends with an extension
    r"(?::\d+)?"                      # optional :line
)


def extract_file_paths(text: str) -> list[str]:
    if not text:
        return []
    out: list[str] = []
    for m in _PATH_RE.finditer(text):
        path = m.group(1)
        # skip URLs
        start = m.start()
        prefix = text[max(0, start - 8):start]
        if "://" in prefix or path.startswith(("http", "www.")):
            continue
        out.append(path)
    return list(dict.fromkeys(out))  # de-dupe, keep order


def inject_files(
    workspace,
    repo_path: str,
    paths: list[str],
    max_files: int = 5,
    max_bytes: int = 20000,
) -> list[InjectedFile]:
    results: list[InjectedFile] = []
    injected = 0
    base = repo_path.rstrip("/")
    for path in paths:
        if injected >= max_files:
            results.append(InjectedFile(path=path, bytes=0, content="",
                                        skipped="count_cap"))
            continue
        full = f"{base}/{path}"
        res = workspace.execute_command(f"cat -- {shlex.quote(full)}")
        if getattr(res, "exit_code", 1) != 0:
            results.append(InjectedFile(path=path, bytes=0, content="",
                                        skipped="not_found"))
            continue
        content = res.stdout or ""
        nbytes = len(content.encode("utf-8", errors="ignore"))
        if nbytes > max_bytes:
            results.append(InjectedFile(path=path, bytes=nbytes, content="",
                                        skipped="too_large"))
            continue
        results.append(InjectedFile(path=path, bytes=nbytes, content=content,
                                    skipped=None))
        injected += 1
    return results
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/lfu/git-projects/benchmarks-main/.claude/worktrees/scaleswe-interactive && python -m pytest benchmarks/scaleswe_interactive/tests/test_file_reference.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/scaleswe_interactive/file_reference.py benchmarks/scaleswe_interactive/tests/test_file_reference.py
git commit -m "feat(scaleswe-interactive): file reference extraction + read-only injection"
```

---

## Task 3: Read-only user tools (schema + executor)

**Files:**
- Create: `benchmarks/scaleswe_interactive/user_tools.py`
- Test: `benchmarks/scaleswe_interactive/tests/test_user_tools.py`

**Interfaces:**
- Consumes: `is_readonly_command` (Task 1); `workspace.execute_command`.
- Produces:
  - `USER_READONLY_TOOLS: list[dict]` — OpenAI-format tool schemas: `read_file(path)`, `grep(pattern, path)`, `glob(pattern)`, `run_readonly_bash(command)`, and `finish(reason)`.
  - `FINISH_TOOL_NAME = "finish"`.
  - `execute_readonly_tool(workspace, repo_path: str, name: str, arguments: dict) -> str` — runs the tool; for `run_readonly_bash`, rejects non-read-only commands with an error string (never executes them).

- [ ] **Step 1: Write the failing test**

```python
# benchmarks/scaleswe_interactive/tests/test_user_tools.py
from benchmarks.scaleswe_interactive.user_tools import (
    USER_READONLY_TOOLS,
    FINISH_TOOL_NAME,
    execute_readonly_tool,
)


class _WS:
    def __init__(self):
        self.ran = []

    def execute_command(self, cmd):
        self.ran.append(cmd)

        class R:
            exit_code, stdout, stderr = 0, "ok-output", ""
        return R()


def test_finish_tool_present():
    names = {t["function"]["name"] for t in USER_READONLY_TOOLS}
    assert FINISH_TOOL_NAME in names
    assert {"read_file", "grep", "glob", "run_readonly_bash"} <= names


def test_run_readonly_bash_rejects_writer_without_executing():
    ws = _WS()
    out = execute_readonly_tool(ws, "/repo", "run_readonly_bash",
                                {"command": "rm -rf /"})
    assert "read-only" in out.lower()
    assert ws.ran == []  # never executed


def test_run_readonly_bash_allows_reader():
    ws = _WS()
    out = execute_readonly_tool(ws, "/repo", "run_readonly_bash",
                                {"command": "grep -n x a.py"})
    assert out == "ok-output"
    assert ws.ran and "grep -n x a.py" in ws.ran[0]


def test_read_file_uses_cat_within_repo():
    ws = _WS()
    execute_readonly_tool(ws, "/repo", "read_file", {"path": "pkg/m.py"})
    assert ws.ran and "/repo/pkg/m.py" in ws.ran[0] and ws.ran[0].startswith("cat")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/lfu/git-projects/benchmarks-main/.claude/worktrees/scaleswe-interactive && python -m pytest benchmarks/scaleswe_interactive/tests/test_user_tools.py -q`
Expected: FAIL — `ModuleNotFoundError: ... user_tools`.

- [ ] **Step 3: Write minimal implementation**

```python
# benchmarks/scaleswe_interactive/user_tools.py
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
        p = f"{base}/{arguments['path']}"
        res = workspace.execute_command(f"cat -- {shlex.quote(p)}")
        return _truncate(res.stdout if res.exit_code == 0 else res.stderr)
    if name == "grep":
        path = arguments.get("path", ".")
        p = f"{base}/{path}"
        cmd = f"grep -rn -- {shlex.quote(arguments['pattern'])} {shlex.quote(p)}"
        res = workspace.execute_command(cmd)
        return _truncate(res.stdout or res.stderr or "(no matches)")
    if name == "glob":
        cmd = (f"cd {shlex.quote(base)} && "
               f"find . -path {shlex.quote('./' + arguments['pattern'])}")
        res = workspace.execute_command(cmd)
        return _truncate(res.stdout or "(no matches)")
    if name == "run_readonly_bash":
        command = arguments["command"]
        if not is_readonly_command(command):
            return ("ERROR: command rejected — only read-only commands are "
                    "allowed for the user agent.")
        res = workspace.execute_command(f"cd {shlex.quote(base)} && {command}")
        return _truncate(res.stdout or res.stderr or "(no output)")
    return f"ERROR: unknown tool {name}"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/lfu/git-projects/benchmarks-main/.claude/worktrees/scaleswe-interactive && python -m pytest benchmarks/scaleswe_interactive/tests/test_user_tools.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/scaleswe_interactive/user_tools.py benchmarks/scaleswe_interactive/tests/test_user_tools.py
git commit -m "feat(scaleswe-interactive): read-only user tools with guarded bash"
```

---

## Task 4: Prompt templates

**Files:**
- Create: `benchmarks/scaleswe_interactive/prompts/user_system.j2`
- Create: `benchmarks/scaleswe_interactive/prompts/coding_system_plan.j2`
- Create: `benchmarks/scaleswe_interactive/prompts/coding_system_auto.j2`
- Create: `benchmarks/scaleswe_interactive/prompts/initial_instruction.j2`
- Test: `benchmarks/scaleswe_interactive/tests/test_prompts.py`

**Interfaces:**
- Produces: four Jinja2 templates. `user_system.j2` renders with `{problem_statement, mode, user_turns, max_user_turns}`. `initial_instruction.j2` renders with `{instance, workspace_dir_name, actual_workspace_path, mode}`. Coding templates render with `{}` (static text per mode).

- [ ] **Step 1: Write the failing test**

```python
# benchmarks/scaleswe_interactive/tests/test_prompts.py
import os
from jinja2 import Environment, FileSystemLoader

PROMPTS = os.path.join(os.path.dirname(__file__), "..", "prompts")


def _render(name, **ctx):
    env = Environment(loader=FileSystemLoader(PROMPTS))
    return env.get_template(name).render(**ctx)


def test_user_system_plan_mentions_approval_and_finish():
    txt = _render("user_system.j2", problem_statement="P", mode="plan",
                  user_turns=0, max_user_turns=20)
    assert "plan" in txt.lower()
    assert "approve" in txt.lower()
    assert "finish" in txt.lower()
    assert "read-only" in txt.lower()


def test_user_system_auto_mentions_autonomy():
    txt = _render("user_system.j2", problem_statement="P", mode="auto",
                  user_turns=0, max_user_turns=20)
    assert "finish" in txt.lower()


def test_coding_plan_requires_plan_before_edits():
    txt = _render("coding_system_plan.j2")
    assert "plan" in txt.lower()
    assert "approv" in txt.lower()


def test_initial_instruction_includes_problem():
    txt = _render("initial_instruction.j2",
                  instance={"problem_statement": "FIX THE BUG X"},
                  workspace_dir_name="repo",
                  actual_workspace_path="/workspace", mode="plan")
    assert "FIX THE BUG X" in txt
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/lfu/git-projects/benchmarks-main/.claude/worktrees/scaleswe-interactive && python -m pytest benchmarks/scaleswe_interactive/tests/test_prompts.py -q`
Expected: FAIL — `TemplateNotFound`.

- [ ] **Step 3: Write minimal implementation**

```jinja
{# benchmarks/scaleswe_interactive/prompts/user_system.j2 #}
You are role-playing a HUMAN USER who needs a software problem solved by an AI coding agent. You are NOT the coder. You collaborate with a coding agent that shares your workspace.

Your responsibilities:
- Read the problem statement below and drive the coding agent: ask it to explore the repo, propose a plan, implement fixes, and run tests.
- You have READ-ONLY access to the code. You may read files but must NEVER ask to be given write access or perform edits yourself.
- When the problem statement or the coding agent references a specific file, its contents may be injected for you; read them to stay grounded.
{% if mode == "plan" -%}
- PLAN MODE: Before the coding agent makes edits, require it to present a concrete plan. Review it, then either APPROVE it explicitly (say "I approve the plan") or request specific modifications. Do not let it edit before you approve.
{% else -%}
- AUTO MODE: Let the coding agent proceed on its own. It will return to you when it needs input. Read what it did, redirect if it goes off track.
{% endif -%}
- When you judge the problem fully solved (fix implemented AND verified by tests), call the `finish` tool with a short reason. Only you can end the session.

Turn budget: {{ user_turns }}/{{ max_user_turns }} user turns used. If you are near the limit and the problem is solved, finish.

PROBLEM STATEMENT:
{{ problem_statement }}
```

```jinja
{# benchmarks/scaleswe_interactive/prompts/coding_system_plan.j2 #}
You are an expert software engineer working with a human user in a shared repository. Work incrementally and communicate clearly.

IMPORTANT — PLAN MODE: Before making ANY file edits, first explore the repository as needed, then present a concrete, numbered plan of the changes you intend to make and STOP to let the user approve it. Do not edit files until the user has approved your plan. If the user requests modifications, revise the plan and re-present it. After approval, implement the plan and run tests. Report results back to the user rather than ending the session yourself.
```

```jinja
{# benchmarks/scaleswe_interactive/prompts/coding_system_auto.j2 #}
You are an expert software engineer working with a human user in a shared repository. Explore the repository, implement the fix, and run tests to verify it. Proceed autonomously; only return a message to the user when you genuinely need input or when you have finished a coherent chunk of work and want feedback. Do not try to end the session yourself — the user decides when the task is complete.
```

```jinja
{# benchmarks/scaleswe_interactive/prompts/initial_instruction.j2 #}
I need help solving a problem in the repository located at {{ actual_workspace_path }} (repo directory: {{ workspace_dir_name }}).

Here is the problem statement:

{{ instance.problem_statement }}

{% if mode == "plan" -%}
Please start by exploring the repo to understand the issue, then present a plan for my approval before making changes.
{% else -%}
Please explore the repo, implement a fix, and verify it with tests. Check in with me if you need input.
{% endif -%}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/lfu/git-projects/benchmarks-main/.claude/worktrees/scaleswe-interactive && python -m pytest benchmarks/scaleswe_interactive/tests/test_prompts.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/scaleswe_interactive/prompts/ benchmarks/scaleswe_interactive/tests/test_prompts.py
git commit -m "feat(scaleswe-interactive): prompt templates for user + coding agents"
```

---

## Task 5: UserAgent (stateless-per-turn)

**Files:**
- Create: `benchmarks/scaleswe_interactive/user_agent.py`
- Test: `benchmarks/scaleswe_interactive/tests/test_user_agent.py`

**Interfaces:**
- Consumes: `LLM.completion(messages, tools=...) -> LLMResponse` (`.message.content`, `.message.tool_calls`); `MessageToolCall(.name, .arguments)`; `extract_file_paths`, `inject_files` (Task 2); `USER_READONLY_TOOLS`, `execute_readonly_tool`, `FINISH_TOOL_NAME` (Task 3); prompts (Task 4).
- Produces:
  - `UserTurn` = pydantic model: `message: str | None`, `finished: bool`, `finish_reason: str | None`, `injected_files: list[dict]`, `readonly_tool_calls: list[dict]`, `error: str | None`.
  - `UserAgent(user_llm, workspace, repo_path, mode, user_tools, prompts_dir, problem_statement, max_user_turns, max_tool_iters=5, max_retries=2)`.
  - `UserAgent.take_turn(dialogue: list[dict], user_turns: int) -> UserTurn` where `dialogue` is a list of `{"speaker": "coding"|"user", "text": str}`.

- [ ] **Step 1: Write the failing test**

```python
# benchmarks/scaleswe_interactive/tests/test_user_agent.py
from types import SimpleNamespace

from benchmarks.scaleswe_interactive.user_agent import UserAgent, UserTurn


class _Msg:
    def __init__(self, text=None, tool_calls=None):
        from openhands.sdk.llm import TextContent
        self.content = [TextContent(text=text)] if text is not None else []
        self.tool_calls = tool_calls


class _Resp:
    def __init__(self, message):
        self.message = message


class _ToolCall:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments  # JSON string


class _StubLLM:
    """Returns queued responses in order."""
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def completion(self, messages, tools=None, **kwargs):
        self.calls.append((messages, tools))
        return _Resp(self._responses.pop(0))


class _WS:
    def execute_command(self, cmd):
        return SimpleNamespace(exit_code=0, stdout="file-body", stderr="")


def _agent(llm, mode="plan", user_tools="none"):
    import os
    prompts = os.path.join(os.path.dirname(__file__), "..", "prompts")
    return UserAgent(user_llm=llm, workspace=_WS(), repo_path="/repo",
                     mode=mode, user_tools=user_tools, prompts_dir=prompts,
                     problem_statement="Fix bug in a/b.py", max_user_turns=20)


def test_plain_message_turn():
    llm = _StubLLM([_Msg(text="Please show me your plan.")])
    turn = _agent(llm).take_turn(dialogue=[{"speaker": "coding", "text": "hi"}],
                                 user_turns=0)
    assert isinstance(turn, UserTurn)
    assert turn.finished is False
    assert turn.message == "Please show me your plan."


def test_finish_tool_ends_session():
    llm = _StubLLM([_Msg(tool_calls=[_ToolCall("finish", '{"reason": "done"}')])])
    turn = _agent(llm).take_turn(dialogue=[{"speaker": "coding", "text": "done?"}],
                                 user_turns=3)
    assert turn.finished is True
    assert turn.finish_reason == "done"


def test_readonly_tool_loop_then_message():
    # First response calls read_file; second returns a plain message.
    llm = _StubLLM([
        _Msg(tool_calls=[_ToolCall("read_file", '{"path": "a/b.py"}')]),
        _Msg(text="Looks right, proceed."),
    ])
    turn = _agent(llm, user_tools="readonly").take_turn(
        dialogue=[{"speaker": "coding", "text": "see a/b.py:10"}], user_turns=0)
    assert turn.message == "Looks right, proceed."
    assert any(c["name"] == "read_file" for c in turn.readonly_tool_calls)


def test_file_injection_happens_in_none_mode():
    llm = _StubLLM([_Msg(text="ok")])
    turn = _agent(llm, user_tools="none").take_turn(
        dialogue=[{"speaker": "coding", "text": "the file a/b.py is broken"}],
        user_turns=0)
    assert any(f["path"] == "a/b.py" for f in turn.injected_files)


def test_llm_error_returns_error_turn():
    class _BoomLLM:
        def completion(self, messages, tools=None, **kwargs):
            raise RuntimeError("boom")
    turn = _agent(_BoomLLM()).take_turn(dialogue=[{"speaker": "coding", "text": "x"}],
                                        user_turns=0)
    assert turn.error is not None
    assert turn.finished is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/lfu/git-projects/benchmarks-main/.claude/worktrees/scaleswe-interactive && python -m pytest benchmarks/scaleswe_interactive/tests/test_user_agent.py -q`
Expected: FAIL — `ModuleNotFoundError: ... user_agent`.

- [ ] **Step 3: Write minimal implementation**

```python
# benchmarks/scaleswe_interactive/user_agent.py
"""Stateless-per-turn user agent that drives the coding agent (read-only)."""
from __future__ import annotations

import json
import os

from jinja2 import Environment, FileSystemLoader
from pydantic import BaseModel, Field

from openhands.sdk.llm import Message, TextContent

from benchmarks.scaleswe_interactive.file_reference import (
    extract_file_paths,
    inject_files,
)
from benchmarks.scaleswe_interactive.user_tools import (
    FINISH_TOOL_NAME,
    USER_READONLY_TOOLS,
    execute_readonly_tool,
)


class UserTurn(BaseModel):
    message: str | None = None
    finished: bool = False
    finish_reason: str | None = None
    injected_files: list[dict] = Field(default_factory=list)
    readonly_tool_calls: list[dict] = Field(default_factory=list)
    error: str | None = None


def _content_text(message) -> str:
    parts = []
    for c in getattr(message, "content", []) or []:
        text = getattr(c, "text", None)
        if text:
            parts.append(text)
    return "\n".join(parts).strip()


class UserAgent:
    def __init__(self, user_llm, workspace, repo_path: str, mode: str,
                 user_tools: str, prompts_dir: str, problem_statement: str,
                 max_user_turns: int, max_tool_iters: int = 5,
                 max_retries: int = 2, inject_max_files: int = 5,
                 inject_max_bytes: int = 20000):
        self.llm = user_llm
        self.workspace = workspace
        self.repo_path = repo_path
        self.mode = mode
        self.user_tools = user_tools
        self.problem_statement = problem_statement
        self.max_user_turns = max_user_turns
        self.max_tool_iters = max_tool_iters
        self.max_retries = max_retries
        self.inject_max_files = inject_max_files
        self.inject_max_bytes = inject_max_bytes
        self._env = Environment(loader=FileSystemLoader(prompts_dir))

    def _system_message(self, user_turns: int) -> Message:
        txt = self._env.get_template("user_system.j2").render(
            problem_statement=self.problem_statement, mode=self.mode,
            user_turns=user_turns, max_user_turns=self.max_user_turns)
        return Message(role="system", content=[TextContent(text=txt)])

    def _dialogue_messages(self, dialogue: list[dict]) -> list[Message]:
        msgs: list[Message] = []
        for turn in dialogue:
            # From the user agent's POV, the coding agent is the "assistant"
            # counterpart and the user's own prior lines are "user".
            role = "user" if turn["speaker"] == "user" else "assistant"
            msgs.append(Message(role=role,
                                content=[TextContent(text=turn["text"])]))
        return msgs

    def _injection_message(self, dialogue: list[dict]) -> tuple[Message | None, list[dict]]:
        text_sources = [self.problem_statement]
        for turn in dialogue:
            if turn["speaker"] == "coding":
                text_sources.append(turn["text"])
        paths: list[str] = []
        for src in text_sources:
            for p in extract_file_paths(src):
                if p not in paths:
                    paths.append(p)
        if not paths:
            return None, []
        injected = inject_files(self.workspace, self.repo_path, paths,
                                max_files=self.inject_max_files,
                                max_bytes=self.inject_max_bytes)
        shown = [f for f in injected if f["skipped"] is None]
        if not shown:
            return None, injected
        blocks = [f"--- {f['path']} ---\n{f['content']}" for f in shown]
        msg = Message(role="user", content=[TextContent(
            text="Referenced file contents (read-only):\n\n"
                 + "\n\n".join(blocks))])
        return msg, injected

    def take_turn(self, dialogue: list[dict], user_turns: int) -> UserTurn:
        try:
            return self._take_turn_inner(dialogue, user_turns)
        except Exception as e:  # noqa: BLE001 - surface as error turn
            return UserTurn(error=f"{type(e).__name__}: {e}")

    def _take_turn_inner(self, dialogue: list[dict], user_turns: int) -> UserTurn:
        messages = [self._system_message(user_turns)]
        messages += self._dialogue_messages(dialogue)
        inject_msg, injected = self._injection_message(dialogue)
        if inject_msg is not None:
            messages.append(inject_msg)

        tools = USER_READONLY_TOOLS if self.user_tools == "readonly" else \
            [t for t in USER_READONLY_TOOLS if t["function"]["name"] == FINISH_TOOL_NAME]

        tool_traces: list[dict] = []
        for _ in range(self.max_tool_iters + 1):
            resp = self._complete_with_retry(messages, tools)
            msg = resp.message
            tool_calls = getattr(msg, "tool_calls", None) or []

            if not tool_calls:
                return UserTurn(message=_content_text(msg) or "(no message)",
                                injected_files=injected,
                                readonly_tool_calls=tool_traces)

            # Handle first tool call (sequential handling keeps it simple).
            call = tool_calls[0]
            name = call.name
            try:
                args = json.loads(call.arguments) if call.arguments else {}
            except (json.JSONDecodeError, TypeError):
                args = {}

            if name == FINISH_TOOL_NAME:
                return UserTurn(finished=True,
                                finish_reason=args.get("reason", ""),
                                injected_files=injected,
                                readonly_tool_calls=tool_traces)

            if self.user_tools != "readonly":
                # Tool offered but not enabled: coerce to a message.
                return UserTurn(message=_content_text(msg) or "(no message)",
                                injected_files=injected,
                                readonly_tool_calls=tool_traces)

            result = execute_readonly_tool(self.workspace, self.repo_path,
                                           name, args)
            tool_traces.append({"name": name, "arguments": args,
                                "result": result[:2000]})
            # Feed the tool result back and let the model continue.
            messages.append(Message(role="assistant",
                                    content=[TextContent(
                                        text=f"[called {name}({args})]")]))
            messages.append(Message(role="user",
                                    content=[TextContent(
                                        text=f"Tool result:\n{result}")]))

        # Ran out of tool iterations: ask the model for a final message.
        messages.append(Message(role="user", content=[TextContent(
            text="Please respond to the coding agent now with a message.")]))
        resp = self._complete_with_retry(messages, tools=[])
        return UserTurn(message=_content_text(resp.message) or "(no message)",
                        injected_files=injected, readonly_tool_calls=tool_traces)

    def _complete_with_retry(self, messages, tools):
        last_exc = None
        for attempt in range(self.max_retries + 1):
            try:
                return self.llm.completion(messages=messages, tools=tools or None)
            except Exception as e:  # noqa: BLE001
                last_exc = e
        raise last_exc
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/lfu/git-projects/benchmarks-main/.claude/worktrees/scaleswe-interactive && python -m pytest benchmarks/scaleswe_interactive/tests/test_user_agent.py -q`
Expected: PASS.

> Note: the stub in the test passes `tools=None` acceptably; `completion` is called with keyword `messages=` and `tools=`. If `openhands.sdk.llm` import path for `TextContent`/`Message` differs, correct the import in the test to match `user_agent.py` (both must use `from openhands.sdk.llm import Message, TextContent`).

- [ ] **Step 5: Commit**

```bash
git add benchmarks/scaleswe_interactive/user_agent.py benchmarks/scaleswe_interactive/tests/test_user_agent.py
git commit -m "feat(scaleswe-interactive): stateless per-turn user agent"
```

---

## Task 6: Driver loop

**Files:**
- Create: `benchmarks/scaleswe_interactive/driver.py`
- Test: `benchmarks/scaleswe_interactive/tests/test_driver.py`

**Interfaces:**
- Consumes: a `conversation` object exposing `send_message(str)`, `run()` (or `run(timeout=...)`), `state.events`, `state.execution_status`; `UserAgent.take_turn` (Task 5); `MessageEvent`, `ConversationExecutionStatus` from SDK.
- Produces:
  - `SessionResult` = pydantic model: `dialogue: list[dict]`, `termination_reason: str`, `user_turns: int`.
  - `run_interactive_session(conversation, user_agent: UserAgent, initial_instruction: str, max_user_turns: int, timeout: int | None = None) -> SessionResult`.
  - Termination reasons: `"user_finish"`, `"turn_cap"`, `"agent_error"`, `"agent_stuck"`, `"user_error"`.

- [ ] **Step 1: Write the failing test**

```python
# benchmarks/scaleswe_interactive/tests/test_driver.py
from types import SimpleNamespace

from openhands.sdk.conversation.state import ConversationExecutionStatus
from openhands.sdk.event import MessageEvent
from openhands.sdk.llm import Message, TextContent

from benchmarks.scaleswe_interactive.driver import (
    run_interactive_session,
    SessionResult,
)
from benchmarks.scaleswe_interactive.user_agent import UserTurn


def _agent_msg(text):
    return MessageEvent(source="agent",
                        llm_message=Message(role="assistant",
                                            content=[TextContent(text=text)]))


class _FakeConversation:
    """Appends an agent message each run(); records sent user messages."""
    def __init__(self):
        self.state = SimpleNamespace(
            events=[], execution_status=ConversationExecutionStatus.FINISHED)
        self.sent = []
        self._replies = ["Here is my plan: 1. do X", "Done, tests pass."]

    def send_message(self, message):
        self.sent.append(message)

    def run(self, timeout=None):
        text = self._replies.pop(0) if self._replies else "still working"
        self.state.events.append(_agent_msg(text))


class _ScriptedUserAgent:
    def __init__(self, turns):
        self._turns = list(turns)
        self.seen = []

    def take_turn(self, dialogue, user_turns):
        self.seen.append(list(dialogue))
        return self._turns.pop(0)


def test_user_finish_terminates():
    conv = _FakeConversation()
    ua = _ScriptedUserAgent([
        UserTurn(message="I approve the plan"),
        UserTurn(finished=True, finish_reason="solved"),
    ])
    res = run_interactive_session(conv, ua, "PROBLEM", max_user_turns=20)
    assert isinstance(res, SessionResult)
    assert res.termination_reason == "user_finish"
    assert conv.sent[0] == "PROBLEM"
    assert "I approve the plan" in conv.sent
    # dialogue contains both coding and user turns
    speakers = {d["speaker"] for d in res.dialogue}
    assert speakers == {"coding", "user"}


def test_turn_cap_terminates():
    conv = _FakeConversation()
    conv._replies = ["a", "b", "c", "d", "e"]
    ua = _ScriptedUserAgent([UserTurn(message="keep going") for _ in range(10)])
    res = run_interactive_session(conv, ua, "P", max_user_turns=2)
    assert res.termination_reason == "turn_cap"
    assert res.user_turns == 2


def test_agent_error_terminates():
    conv = _FakeConversation()

    def boom(timeout=None):
        conv.state.execution_status = ConversationExecutionStatus.ERROR
    conv.run = boom
    ua = _ScriptedUserAgent([UserTurn(message="x")])
    res = run_interactive_session(conv, ua, "P", max_user_turns=20)
    assert res.termination_reason == "agent_error"


def test_user_error_terminates():
    conv = _FakeConversation()
    ua = _ScriptedUserAgent([UserTurn(error="boom")])
    res = run_interactive_session(conv, ua, "P", max_user_turns=20)
    assert res.termination_reason == "user_error"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/lfu/git-projects/benchmarks-main/.claude/worktrees/scaleswe-interactive && python -m pytest benchmarks/scaleswe_interactive/tests/test_driver.py -q`
Expected: FAIL — `ModuleNotFoundError: ... driver`.

- [ ] **Step 3: Write minimal implementation**

```python
# benchmarks/scaleswe_interactive/driver.py
"""Driver loop alternating coding-agent runs and user-agent turns."""
from __future__ import annotations

import inspect

from pydantic import BaseModel, Field

from openhands.sdk.conversation.state import ConversationExecutionStatus
from openhands.sdk.event import MessageEvent

from benchmarks.scaleswe_interactive.user_agent import UserAgent


class SessionResult(BaseModel):
    dialogue: list[dict] = Field(default_factory=list)
    termination_reason: str = ""
    user_turns: int = 0


def _run_conversation(conversation, timeout):
    run = conversation.run
    if timeout is not None:
        try:
            sig = inspect.signature(run)
            if "timeout" in sig.parameters:
                run(timeout=timeout)
                return
        except (TypeError, ValueError):
            pass
    run()


def _latest_agent_text(conversation, seen_ids: set) -> str | None:
    """Return concatenated text of new agent MessageEvents since last check."""
    texts = []
    for event in conversation.state.events:
        if id(event) in seen_ids:
            continue
        if isinstance(event, MessageEvent) and event.source == "agent":
            for c in event.llm_message.content or []:
                t = getattr(c, "text", None)
                if t:
                    texts.append(t)
        seen_ids.add(id(event))
    return "\n".join(texts).strip() if texts else None


def run_interactive_session(conversation, user_agent: UserAgent,
                            initial_instruction: str, max_user_turns: int,
                            timeout: int | None = None) -> SessionResult:
    dialogue: list[dict] = []
    seen_ids: set = set()
    # Mark pre-existing events as seen (e.g., the injected instruction message).
    for event in conversation.state.events:
        seen_ids.add(id(event))

    conversation.send_message(initial_instruction)
    user_turns = 0

    while True:
        _run_conversation(conversation, timeout)
        status = conversation.state.execution_status

        if status == ConversationExecutionStatus.ERROR:
            return SessionResult(dialogue=dialogue,
                                 termination_reason="agent_error",
                                 user_turns=user_turns)
        if status == ConversationExecutionStatus.STUCK:
            return SessionResult(dialogue=dialogue,
                                 termination_reason="agent_stuck",
                                 user_turns=user_turns)

        agent_text = _latest_agent_text(conversation, seen_ids)
        if agent_text:
            dialogue.append({"speaker": "coding", "text": agent_text})

        turn = user_agent.take_turn(dialogue=dialogue, user_turns=user_turns)

        if turn.error is not None:
            return SessionResult(dialogue=dialogue,
                                 termination_reason="user_error",
                                 user_turns=user_turns)
        if turn.finished:
            dialogue.append({"speaker": "user",
                             "text": f"[finish] {turn.finish_reason or ''}".strip()})
            return SessionResult(dialogue=dialogue,
                                 termination_reason="user_finish",
                                 user_turns=user_turns)

        user_turns += 1
        message = turn.message or "Please continue."
        dialogue.append({"speaker": "user", "text": message,
                         "injected_files": turn.injected_files,
                         "readonly_tool_calls": turn.readonly_tool_calls})

        if user_turns >= max_user_turns:
            return SessionResult(dialogue=dialogue,
                                 termination_reason="turn_cap",
                                 user_turns=user_turns)

        conversation.send_message(message)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/lfu/git-projects/benchmarks-main/.claude/worktrees/scaleswe-interactive && python -m pytest benchmarks/scaleswe_interactive/tests/test_driver.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/scaleswe_interactive/driver.py benchmarks/scaleswe_interactive/tests/test_driver.py
git commit -m "feat(scaleswe-interactive): driver loop with termination handling"
```

---

## Task 7: Transcript side-file builder

**Files:**
- Create: `benchmarks/scaleswe_interactive/transcript.py`
- Test: `benchmarks/scaleswe_interactive/tests/test_transcript.py`

**Interfaces:**
- Consumes: `SessionResult` (Task 6).
- Produces: `build_interactive_transcript(instance_id: str, mode: str, user_tools: str, coding_model: str, user_model: str, session: SessionResult) -> dict` with keys: `instance_id, mode, user_tools, coding_model, user_model, turns, termination`.

- [ ] **Step 1: Write the failing test**

```python
# benchmarks/scaleswe_interactive/tests/test_transcript.py
from benchmarks.scaleswe_interactive.driver import SessionResult
from benchmarks.scaleswe_interactive.transcript import build_interactive_transcript


def test_transcript_shape():
    session = SessionResult(
        dialogue=[
            {"speaker": "coding", "text": "plan"},
            {"speaker": "user", "text": "approve", "injected_files": [],
             "readonly_tool_calls": []},
        ],
        termination_reason="user_finish", user_turns=1)
    out = build_interactive_transcript(
        instance_id="inst-1", mode="plan", user_tools="none",
        coding_model="model-a", user_model="model-b", session=session)
    assert out["instance_id"] == "inst-1"
    assert out["mode"] == "plan"
    assert out["coding_model"] == "model-a"
    assert out["user_model"] == "model-b"
    assert out["termination"] == {"reason": "user_finish", "user_turns": 1}
    assert out["turns"] == session.dialogue
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/lfu/git-projects/benchmarks-main/.claude/worktrees/scaleswe-interactive && python -m pytest benchmarks/scaleswe_interactive/tests/test_transcript.py -q`
Expected: FAIL — `ModuleNotFoundError: ... transcript`.

- [ ] **Step 3: Write minimal implementation**

```python
# benchmarks/scaleswe_interactive/transcript.py
"""Build the richer speaker-tagged interactive side file."""
from __future__ import annotations

from benchmarks.scaleswe_interactive.driver import SessionResult


def build_interactive_transcript(instance_id: str, mode: str, user_tools: str,
                                 coding_model: str, user_model: str,
                                 session: SessionResult) -> dict:
    return {
        "instance_id": instance_id,
        "mode": mode,
        "user_tools": user_tools,
        "coding_model": coding_model,
        "user_model": user_model,
        "turns": session.dialogue,
        "termination": {
            "reason": session.termination_reason,
            "user_turns": session.user_turns,
        },
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/lfu/git-projects/benchmarks-main/.claude/worktrees/scaleswe-interactive && python -m pytest benchmarks/scaleswe_interactive/tests/test_transcript.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/scaleswe_interactive/transcript.py benchmarks/scaleswe_interactive/tests/test_transcript.py
git commit -m "feat(scaleswe-interactive): interactive transcript builder"
```

---

## Task 8: run_infer.py wiring + CLI

**Files:**
- Create: `benchmarks/scaleswe_interactive/run_infer.py`
- Test: `benchmarks/scaleswe_interactive/tests/test_run_infer_wiring.py`

**Interfaces:**
- Consumes: everything above; scaleswe's `ScaleSWEEvaluation` (subclassed for `prepare_instances`/`prepare_workspace`), `EvalMetadata`, `EvalOutput`, `EvalInstance`, `LLM`, `Agent`, `Conversation`, `get_default_tools`, `build_event_persistence_callback`.
- Produces: `ScaleSWEInteractiveEvaluation(ScaleSWEEvaluation)` overriding `evaluate_instance`; module-level helper `load_llm_from_config(path: str) -> LLM`; `build_arg_parser()` returning the configured parser; `main()`.

- [ ] **Step 1: Write the failing test** (wiring-only; no container)

```python
# benchmarks/scaleswe_interactive/tests/test_run_infer_wiring.py
import json

from benchmarks.scaleswe_interactive.run_infer import (
    build_arg_parser,
    load_llm_from_config,
)


def test_parser_has_interactive_args():
    parser = build_arg_parser()
    ns = parser.parse_args([
        "--llm-config-path", "x.json",
    ])
    assert ns.mode == "plan"                 # default
    assert ns.user_tools == "none"           # default
    assert ns.max_user_turns == 20           # default
    assert hasattr(ns, "user_llm_config_path")


def test_parser_accepts_overrides():
    parser = build_arg_parser()
    ns = parser.parse_args([
        "--llm-config-path", "x.json",
        "--user-llm-config-path", "y.json",
        "--mode", "auto",
        "--user-tools", "readonly",
        "--max-user-turns", "5",
    ])
    assert ns.mode == "auto"
    assert ns.user_tools == "readonly"
    assert ns.max_user_turns == 5
    assert ns.user_llm_config_path == "y.json"


def test_load_llm_from_config(tmp_path):
    cfg = tmp_path / "llm.json"
    cfg.write_text(json.dumps({"model": "litellm_proxy/x", "api_key": "sk-test"}))
    llm = load_llm_from_config(str(cfg))
    assert llm.model == "litellm_proxy/x"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/lfu/git-projects/benchmarks-main/.claude/worktrees/scaleswe-interactive && python -m pytest benchmarks/scaleswe_interactive/tests/test_run_infer_wiring.py -q`
Expected: FAIL — `ModuleNotFoundError: ... run_infer`.

- [ ] **Step 3: Write minimal implementation**

```python
# benchmarks/scaleswe_interactive/run_infer.py
"""Two-agent interactive inference for Scale-SWE (trajectory generation)."""
import json
import os
from pathlib import Path

from omegaconf import OmegaConf

from benchmarks.scaleswe.run_infer import ScaleSWEEvaluation, get_instruction
from benchmarks.scaleswe_interactive.driver import run_interactive_session
from benchmarks.scaleswe_interactive.transcript import build_interactive_transcript
from benchmarks.scaleswe_interactive.user_agent import UserAgent
from benchmarks.utils.args_parser import get_parser
from benchmarks.utils.conversation import build_event_persistence_callback
from benchmarks.utils.evaluation_utils import (
    construct_eval_output_dir,
    get_default_on_result_writer,
)
from benchmarks.utils.mirror_config import get_mirror_env_commands
from benchmarks.utils.models import EvalInstance, EvalMetadata, EvalOutput
from openhands.sdk import LLM, Agent, Conversation, get_logger
from openhands.sdk.event.base import LLMConvertibleEvent
from openhands.sdk.event.llm_convertible.system import SystemPromptEvent
from openhands.sdk.tool.tool import ToolDefinition
from openhands.sdk.workspace import RemoteWorkspace
from openhands.tools.preset.default import get_default_tools

logger = get_logger(__name__)

PROMPTS_DIR = str((Path(__file__).parent / "prompts").resolve())


def load_llm_from_config(path: str) -> LLM:
    if not os.path.isfile(path):
        raise ValueError(f"LLM config file {path} does not exist")
    cfg = json.dumps(OmegaConf.to_container(OmegaConf.load(path), resolve=True))
    return LLM.model_validate_json(cfg)


class ScaleSWEInteractiveEvaluation(ScaleSWEEvaluation):
    """Reuses scaleswe instance/workspace setup; swaps in the two-agent loop."""

    def evaluate_instance(self, instance: EvalInstance,
                          workspace: RemoteWorkspace) -> EvalOutput:
        details = self.metadata.details or {}
        mode = details.get("mode", "plan")
        user_tools = details.get("user_tools", "none")
        max_user_turns = details.get("max_user_turns", 20)
        user_llm: LLM = details.get("user_llm") or self.metadata.llm

        coding_template = ("coding_system_plan.j2" if mode == "plan"
                           else "coding_system_auto.j2")
        from jinja2 import Environment, FileSystemLoader
        env = Environment(loader=FileSystemLoader(PROMPTS_DIR))
        coding_system_suffix = env.get_template(coding_template).render()

        tools = get_default_tools(enable_browser=False)
        agent = Agent(
            llm=self.metadata.llm,
            tools=tools,
            system_prompt_kwargs={"cli_mode": True},
        )

        assert isinstance(workspace, RemoteWorkspace)

        persist_callback = build_event_persistence_callback(
            run_id=self.metadata.eval_output_dir,
            instance_id=instance.id,
            attempt=self.current_attempt,
        )
        conversation = Conversation(
            agent=agent,
            workspace=workspace,
            callbacks=[persist_callback],
            max_iteration_per_run=self.metadata.max_iterations,
        )

        # --- Repo prep (also provides the Conversation->send_message gap) ---
        repo_path = instance.data.get("workdir", "/workspace")
        if not repo_path.endswith("/"):
            repo_path += "/"
        instance.data["repo_path"] = repo_path
        instance.data["base_commit"] = instance.data.get(
            "parent_commit", instance.data.get("base_commit", ""))
        if "repo" not in instance.data or "/" not in instance.data.get("repo", ""):
            user = instance.data.get("user", "")
            repo = instance.data.get("repo", "")
            if user and repo:
                instance.data["repo"] = f"{user}/{repo}"
        pre_commands = instance.data.get("pre_commands", "")
        if pre_commands and pre_commands.strip():
            pre_cmd = pre_commands.strip().removesuffix("\\n")
            workspace.execute_command(f"cd {repo_path} && {pre_cmd}")

        problem_statement = instance.data.get("problem_statement", "")
        instruction = get_instruction(
            instance=instance.data, metadata=self.metadata,
            workspace_path=workspace.working_dir)
        # Append the mode-specific coding-agent guidance to the instruction.
        instruction = f"{instruction}\n\n{coding_system_suffix}"

        user_agent = UserAgent(
            user_llm=user_llm, workspace=workspace, repo_path=repo_path,
            mode=mode, user_tools=user_tools, prompts_dir=PROMPTS_DIR,
            problem_statement=problem_statement, max_user_turns=max_user_turns)

        session = run_interactive_session(
            conversation, user_agent, initial_instruction=instruction,
            max_user_turns=max_user_turns,
            timeout=self.metadata.conversation_timeout)

        # --- git add/commit/diff (same as scaleswe) ---
        workspace.execute_command(f"cd {repo_path} ; git add -A")
        workspace.execute_command(
            f"cd {repo_path} && "
            "git config --global user.email 'evaluation@openhands.dev' && "
            "git config --global user.name 'OpenHands Evaluation' && "
            "git commit -m 'patch'")
        base_commit = instance.data["base_commit"]
        git_patch_result = workspace.execute_command(
            f"cd {repo_path} ; git --no-pager diff --no-color {base_commit} HEAD")
        git_patch = git_patch_result.stdout

        # --- dump scaleswe-compatible history.json ---
        messages = []
        tools_list = []
        convertible = [e for e in conversation.state.events
                       if isinstance(e, LLMConvertibleEvent)]
        for msg in LLMConvertibleEvent.events_to_messages(convertible):
            messages.append(
                msg.model_copy(update={"send_reasoning_content": True}).to_chat_dict())
        for event in conversation.state.events:
            if isinstance(event, SystemPromptEvent):
                for tool in event.tools:
                    if isinstance(tool, ToolDefinition):
                        tools_list.append(tool.to_openai_tool())
        if not tools_list and tools:
            for tool in tools:
                if isinstance(tool, ToolDefinition):
                    tools_list.append(tool.to_openai_tool())

        dump_data = {
            "instance_id": instance.id,
            "messages": messages,
            "model": self.metadata.llm.model,
            "tools": tools_list,
            "temperature": self.metadata.llm.temperature,
            "top_p": self.metadata.llm.top_p,
            "test_result": {"git_patch": git_patch},
        }
        history_file = os.path.join(
            self.metadata.eval_output_dir, f"{instance.id}.history.json")
        with open(history_file, "w") as f:
            json.dump(dump_data, f, indent=2)

        # --- dump richer interactive side file ---
        transcript = build_interactive_transcript(
            instance_id=instance.id, mode=mode, user_tools=user_tools,
            coding_model=self.metadata.llm.model, user_model=user_llm.model,
            session=session)
        interactive_file = os.path.join(
            self.metadata.eval_output_dir, f"{instance.id}.interactive.json")
        with open(interactive_file, "w") as f:
            json.dump(transcript, f, indent=2)
        logger.info("Dumped interactive transcript to %s (termination=%s)",
                    interactive_file, session.termination_reason)

        return EvalOutput(
            instance_id=instance.id,
            attempt=self.current_attempt,
            test_result={"git_patch": git_patch},
            instruction=instruction,
            error=None,
            history=list(conversation.state.events),
            metrics=conversation.conversation_stats.get_combined_metrics(),
        )


def build_arg_parser():
    prompt_dir = (Path(__file__).parent.parent / "scaleswe" / "prompts").resolve()
    default_prompt_path = prompt_dir / "default.j2"
    parser = get_parser()
    parser.set_defaults(
        dataset="thirdparty/Scale-SWE/scale-swe-batch1.jsonl",
        workspace="flex",
    )
    parser.add_argument("--prompt-path", type=str,
                        default=str(default_prompt_path),
                        help="Coding-agent instruction template")
    parser.add_argument("--user-llm-config-path", type=str, default=None,
                        help="LLM config for the user agent (defaults to coding LLM)")
    parser.add_argument("--mode", choices=["plan", "auto"], default="plan",
                        help="Collaboration mode")
    parser.add_argument("--user-tools", choices=["none", "readonly"],
                        default="none", help="User agent repo access")
    parser.add_argument("--max-user-turns", type=int, default=20,
                        help="Cap on total user turns")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    llm = load_llm_from_config(args.llm_config_path)
    user_llm = (load_llm_from_config(args.user_llm_config_path)
                if args.user_llm_config_path else llm)
    logger.info("Coding LLM: %s | User LLM: %s", llm.model, user_llm.model)

    dataset_description = (
        args.dataset.replace("/", "__") + "-" + args.split.replace("/", "__"))
    structured_output_dir = construct_eval_output_dir(
        base_dir=args.output_dir, dataset_name=dataset_description,
        model_name=llm.model, max_iterations=args.max_iterations,
        eval_note=args.note)

    metadata = EvalMetadata(
        llm=llm,
        dataset=args.dataset,
        dataset_split=args.split,
        max_iterations=args.max_iterations,
        eval_output_dir=structured_output_dir,
        details={
            "mode": args.mode,
            "user_tools": args.user_tools,
            "max_user_turns": args.max_user_turns,
            "user_llm": user_llm,
        },
        prompt_path=args.prompt_path,
        eval_limit=args.n_limit,
        env_setup_commands=get_mirror_env_commands()
        + ["export PIP_CACHE_DIR=~/.cache/pip"],
        max_attempts=args.max_attempts,
        selected_instances_file=args.select,
        max_retries=args.max_retries,
        workspace_type=args.workspace,
        conversation_timeout=args.conversation_timeout,
    )

    evaluator = ScaleSWEInteractiveEvaluation(
        metadata=metadata, num_workers=args.num_workers)
    evaluator.run(on_result=get_default_on_result_writer(evaluator.output_path))
    logger.info("Interactive evaluation completed!")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/lfu/git-projects/benchmarks-main/.claude/worktrees/scaleswe-interactive && python -m pytest benchmarks/scaleswe_interactive/tests/test_run_infer_wiring.py -q`
Expected: PASS.

> If `EvalMetadata.details` is a strict pydantic model that rejects arbitrary keys or a non-serializable `LLM`, fall back to storing the interactive settings as attributes on the evaluator instance (constructor kwargs) instead of in `details`. Verify by reading `benchmarks/utils/models.py::EvalMetadata` before implementing; adjust the two `details.get(...)` sites and the `details={...}` block accordingly.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/scaleswe_interactive/run_infer.py benchmarks/scaleswe_interactive/tests/test_run_infer_wiring.py
git commit -m "feat(scaleswe-interactive): run_infer wiring + CLI for two-agent mode"
```

---

## Task 9: Full unit-suite run + docs

**Files:**
- Create: `benchmarks/scaleswe_interactive/README.md`
- Modify: none (verification task).

**Interfaces:** none.

- [ ] **Step 1: Run the full interactive test suite**

Run: `cd /home/lfu/git-projects/benchmarks-main/.claude/worktrees/scaleswe-interactive && python -m pytest benchmarks/scaleswe_interactive/tests/ -q`
Expected: PASS — all tests from Tasks 1–8 green.

- [ ] **Step 2: Write README**

```markdown
# Scale-SWE Interactive (two-agent mode)

Two LLM agents collaborate in one shared workspace to produce a multi-turn
trajectory:

- **coding agent** (`--llm-config-path`): full tools, does the work.
- **user agent** (`--user-llm-config-path`, defaults to coding LLM): read-only,
  drives the coding agent, approves plans, and calls `finish` to end.

## Usage

    python -m benchmarks.scaleswe_interactive.run_infer \
      --llm-config-path configs/coder.json \
      --user-llm-config-path configs/user.json \
      --mode plan \
      --user-tools readonly \
      --max-user-turns 20 \
      --n-limit 1

## Modes
- `--mode plan` (default): coding agent must present a plan and wait for user
  approval before editing (prompt-enforced).
- `--mode auto`: coding agent proceeds autonomously, returning to the user only
  when it wants input.

## User repo access
- `--user-tools none` (default): user sees only injected contents of files
  referenced in the problem statement / coding agent messages.
- `--user-tools readonly`: user additionally gets read-only tools
  (`read_file`, `grep`, `glob`, guarded `run_readonly_bash`).

## Outputs (in the eval output dir)
- `{id}.history.json` — scaleswe-compatible (git_patch, messages, tools).
- `{id}.interactive.json` — speaker-tagged turns, mode, models, injected files,
  read-only tool traces, termination reason.

Read-only enforcement is a best-effort command allowlist, not an OS sandbox.
```

- [ ] **Step 3: Commit**

```bash
git add benchmarks/scaleswe_interactive/README.md
git commit -m "docs(scaleswe-interactive): usage README"
```

- [ ] **Step 4 (manual, optional): single-instance smoke run**

Run (requires configured LLM configs + docker/flex workspace):
`cd /home/lfu/git-projects/benchmarks-main/.claude/worktrees/scaleswe-interactive && python -m benchmarks.scaleswe_interactive.run_infer --llm-config-path <coder.json> --user-llm-config-path <user.json> --mode plan --user-tools readonly --n-limit 1 --workspace docker`
Expected: produces `{id}.history.json` and `{id}.interactive.json`; the interactive file has ≥2 user turns or a `user_finish`/`turn_cap` termination.

---

## Self-Review

**Spec coverage:**
- New task dir branched from scaleswe → Tasks 1–9 (dir `scaleswe_interactive`, reuse via subclass in Task 8). ✓
- Two LLM backends, second config path w/ fallback → Task 8 (`load_llm_from_config`, `--user-llm-config-path`). ✓
- One persisted Conversation + LLM fake-user (Approach A) → Tasks 5, 6, 8. ✓
- `--user-tools {none|readonly}` + file injection in both modes → Tasks 2, 3, 5. ✓
- User-only finish → Tasks 3 (finish tool), 5 (finish detection), 6 (only user ends). ✓
- `--mode {plan|auto}` (plan prompt-enforced) → Tasks 4, 8. ✓
- `--max-user-turns` cap (default 20) → Tasks 6, 8. ✓
- Reuse scaleswe output + richer side file → Tasks 7, 8. ✓
- Error handling (user LLM retry→`user_error`, injection skips, guard rejects, agent error/stuck) → Tasks 1, 2, 5, 6. ✓
- Preserve Conversation→send_message gap → Task 8 (repo prep between). ✓
- Testing (guard, extractor, driver, transcript, smoke) → Tasks 1–9. ✓

**Placeholder scan:** No TBD/TODO; every code step shows full code. The two "If ... verify/adjust" notes (Task 5 import path, Task 8 `EvalMetadata.details`) are explicit fallback instructions with concrete actions, not placeholders.

**Type consistency:** `UserTurn` fields (`message/finished/finish_reason/injected_files/readonly_tool_calls/error`) are produced in Task 5 and consumed identically in Task 6. `SessionResult` (`dialogue/termination_reason/user_turns`) produced in Task 6, consumed in Task 7. `is_readonly_command` (Task 1) consumed in Task 3. `extract_file_paths`/`inject_files` (Task 2) consumed in Task 5. `InjectedFile["skipped"]` sentinel values consistent between Task 2 impl and Task 5 filter. Termination-reason strings identical across Task 6 impl/tests and Task 7. ✓
