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
