# benchmarks/scaleswe_interactive/readonly_guard.py
"""Best-effort allowlist guard: only permit read-only shell commands."""
from __future__ import annotations

import re
import shlex

# Command executables that only read.
# NOTE: python/python3 (arbitrary code), env (exec vector), and awk
# (has system()) are deliberately NOT in this allowlist.
_READONLY_CMDS = {
    "cat", "ls", "grep", "egrep", "fgrep", "rg", "find", "head", "tail",
    "wc", "less", "more", "file", "stat", "tree", "pwd", "echo", "which",
    "sed", "cut", "sort", "uniq", "diff", "nl", "od", "xxd", "basename",
    "dirname", "realpath", "readlink", "true", "test",
}
# Redirection / write operators that taint any segment.
_WRITE_OPERATORS = re.compile(r"(>>|>|\btee\b)")
# Git subcommands that are read-only.
_GIT_READONLY_SUB = {"log", "diff", "show", "status", "blame", "ls-files",
                     "cat-file", "rev-parse", "branch", "describe", "grep"}
# sed/awk in-place or program flags that write.
_SED_WRITE = re.compile(r"-i\b|--in-place")
# Command substitution / process substitution / backtick / var-expansion
# constructs that can smuggle arbitrary execution into an otherwise
# read-looking segment.
_SUBSTITUTION = re.compile(r"\$\(|\$\{|`|<\(|>\(")
# find action flags that execute or write files.
_FIND_ACTIONS = {
    "-delete", "-exec", "-execdir", "-ok", "-okdir",
    "-fprintf", "-fprint", "-fprint0", "-fls",
}
# git branch write flags (anything that mutates refs).
_GIT_BRANCH_WRITE_FLAGS = {
    "-d", "-D", "-m", "-M", "-c", "-C",
    "--delete", "--move", "--copy",
    "--set-upstream-to", "--edit-description", "--force", "-f", "--unset-upstream",
}


def _git_branch_is_readonly(tokens: list[str]) -> bool:
    """Allow `git branch` only when it lists (no branch name, no write flag)."""
    # tokens are everything after "branch" (options/args, ignoring leading
    # git global options which are handled by the caller).
    for tok in tokens:
        if not tok.startswith("-"):
            # A non-option argument to `git branch` is a branch name
            # (create/rename/delete target) -> mutating.
            return False
        # Split "--set-upstream-to=..." style flags on '='.
        flag = tok.split("=", 1)[0]
        if flag in _GIT_BRANCH_WRITE_FLAGS:
            return False
    return True


def _segment_is_readonly(seg: str) -> bool:
    seg = seg.strip()
    if not seg:
        return True
    if _WRITE_OPERATORS.search(seg):
        return False
    # Reject command/process substitution and variable expansion outright:
    # these can execute arbitrary commands regardless of the visible exe.
    if _SUBSTITUTION.search(seg):
        return False
    try:
        tokens = shlex.split(seg)
    except ValueError:
        return False
    if not tokens:
        return True
    exe = tokens[0]
    if exe == "find":
        # Reject any find action flag that executes or writes files.
        if any(t in _FIND_ACTIONS for t in tokens[1:]):
            return False
        return True
    if exe == "git":
        # Skip git global options (e.g. --no-pager) to find the subcommand.
        rest = tokens[1:]
        sub_idx = next((i for i, t in enumerate(rest) if not t.startswith("-")), None)
        if sub_idx is None:
            return False
        sub = rest[sub_idx]
        if sub not in _GIT_READONLY_SUB:
            return False
        if sub == "branch":
            return _git_branch_is_readonly(rest[sub_idx + 1:])
        return True
    if exe == "sed":
        return not _SED_WRITE.search(seg)
    return exe in _READONLY_CMDS


def is_readonly_command(cmd: str) -> bool:
    """Return True only if every chained segment is read-only."""
    # Split on chaining/pipe operators and newlines; every part must be
    # read-only. Newlines act as command separators in shells too.
    parts = re.split(r"&&|\|\||;|\||\n", cmd)
    return all(_segment_is_readonly(p) for p in parts)
