"""Shared file-naming and JSON I/O for harness history dumps.

Harness runs produce four history families: the raw per-attempt history, the
streamlined per-attempt history, the published (canonical) raw history and the
published streamlined history. All four are stored gzipped; the canonical
``*.history.streamlined-lite.json`` (trajectory ``turns`` omitted) stays plain
text so downstream tooling can stream/iterate it without decompressing.

Readers accept both plain and gzipped documents, so legacy dumps remain usable.
"""

from __future__ import annotations

import contextlib
import gzip
import io
import json
import os
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any


HISTORY_SUFFIX = ".history.json"
STREAMLINED_SUFFIX = ".history.streamlined.json"
STREAMLINED_LITE_SUFFIX = ".history.streamlined-lite.json"
GZIP_SUFFIX = ".gz"
ATTEMPTS_DIRNAME = "attempts"

# gzip stream metadata: a fixed mtime keeps recompression byte-stable.
_GZIP_COMPRESS_LEVEL = 6

# Raw/streamlined histories are stored compressed; lite stays plain.
HISTORY_GZ_SUFFIX = HISTORY_SUFFIX + GZIP_SUFFIX
STREAMLINED_GZ_SUFFIX = STREAMLINED_SUFFIX + GZIP_SUFFIX


def is_compressed(path: str | os.PathLike[str]) -> bool:
    return str(path).endswith(GZIP_SUFFIX)


@contextlib.contextmanager
def open_binary_auto(path: str | os.PathLike[str]) -> Iterator[Any]:
    """Open ``path`` as binary, transparently decompressing ``.gz`` files."""
    if is_compressed(path):
        with gzip.open(path, "rb") as handle:
            yield handle
    else:
        with open(path, "rb") as handle:
            yield handle


@contextlib.contextmanager
def open_text_auto(
    path: str | os.PathLike[str], encoding: str = "utf-8"
) -> Iterator[Any]:
    """Open ``path`` as text, transparently decompressing ``.gz`` files."""
    if is_compressed(path):
        with gzip.open(path, "rt", encoding=encoding) as handle:
            yield handle
    else:
        with open(path, encoding=encoding) as handle:
            yield handle


def read_json(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load one JSON document from a plain or gzipped file."""
    with open_text_auto(path) as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object, got {type(data).__name__}")
    return data


def read_text(path: str | os.PathLike[str]) -> str:
    """Read a whole plain or gzipped UTF-8 file."""
    with open_text_auto(path) as handle:
        return handle.read()


def write_json_atomic(path: str | os.PathLike[str], payload: dict[str, Any]) -> None:
    """Write one compact JSON line atomically, gzipping when the name ends in .gz.

    One document per line (+ trailing newline) keeps ``zcat *.history.json.gz``
    directly valid JSONL for the flattening/streamlining tools.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="wb",
        dir=str(destination.parent),
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            if is_compressed(destination):
                gzip_file = gzip.GzipFile(
                    fileobj=handle,
                    mode="wb",
                    compresslevel=_GZIP_COMPRESS_LEVEL,
                    mtime=0,
                )
                text = io.TextIOWrapper(gzip_file, encoding="utf-8")
                try:
                    json.dump(
                        payload,
                        text,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    text.write("\n")
                    text.flush()
                finally:
                    text.close()
            else:
                text = io.TextIOWrapper(handle, encoding="utf-8")
                try:
                    json.dump(
                        payload,
                        text,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    text.write("\n")
                    text.flush()
                finally:
                    text.detach()
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def tail_text(path: str | os.PathLike[str], max_bytes: int = 65536) -> str:
    """Return roughly the last ``max_bytes`` of decoded text (works with .gz)."""
    if not is_compressed(path):
        size = Path(path).stat().st_size
        with open(path, "rb") as handle:
            handle.seek(max(0, size - max_bytes))
            return handle.read().decode("utf-8", errors="ignore")
    tail = b""
    with gzip.open(path, "rb") as handle:
        while True:
            chunk = handle.read(max_bytes)
            if not chunk:
                break
            tail = (tail + chunk)[-max_bytes:]
    return tail.decode("utf-8", errors="ignore")
