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


# scheme://host/path... URL spans, stripped before path extraction so that no
# token inside a URL (regardless of domain length) is ever emitted as a path.
_URL_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9+.-]*://\S+")

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
    # Remove URL spans structurally first, replacing each with a space so
    # adjacent tokens don't fuse. Paths inside URLs then can't be matched.
    cleaned = _URL_RE.sub(" ", text)
    out: list[str] = []
    for m in _PATH_RE.finditer(cleaned):
        path = m.group(1)
        if path.startswith(("http", "www.")):
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
