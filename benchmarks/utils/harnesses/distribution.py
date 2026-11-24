"""Portable harness distributions for in-container execution.

Task containers (SWE-bench, SWE-rebench-V2, ...) do not have Node.js, and the
harness must not be installed from the network at run time. We therefore build
a portable *distribution* on the host once, cache it, and install it into the
task container.

Claude Code ships since v2.1.113 as a self-contained native binary (published
per platform both on npm and on ``downloads.claude.ai``). A distribution is
therefore a single binary plus a manifest: no Node.js runtime is needed inside
the container.

The build is intentionally offline-friendly: once a distribution is cached the
pipeline never touches the network again. Tests inject ``fetch_json`` /
``download`` so they can exercise the logic without network access.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform as platform_module
import shutil
import subprocess
import time
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)

RELEASE_BASE_URL = "https://downloads.claude.ai/claude-code-releases"
MANIFEST_FILENAME = "manifest.json"
BINARY_NAME = "claude"
MIN_BINARY_BYTES = 1_000_000

PLATFORM_BY_MACHINE = {
    ("linux", "x86_64"): "linux-x64",
    ("linux", "amd64"): "linux-x64",
    ("linux", "aarch64"): "linux-arm64",
    ("linux", "arm64"): "linux-arm64",
}

FetchJson = Callable[[str], dict[str, Any]]
Download = Callable[[str, Path], None]


class DistributionError(RuntimeError):
    """Portable distribution could not be built or is not usable."""


@dataclass(frozen=True)
class HarnessDistribution:
    """One cached, host-local portable harness distribution."""

    harness: str
    version: str
    platform: str
    directory: Path
    binary_path: Path
    sha256: str
    size_bytes: int

    def manifest(self) -> dict[str, Any]:
        return {
            "harness": self.harness,
            "version": self.version,
            "platform": self.platform,
            "binary": self.binary_path.name,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


def detect_platform(override: str | None = None) -> str:
    """Return the container platform key for this host."""
    if override:
        return override
    machine = platform_module.machine().lower()
    detected = PLATFORM_BY_MACHINE.get((platform_module.system().lower(), machine))
    if detected is None:
        raise DistributionError(
            "Cannot detect a supported container platform for the harness "
            f"distribution (host is {platform_module.system()} {machine}). "
            "Pass --harness-platform explicitly (linux-x64, linux-arm64, "
            "linux-x64-musl or linux-arm64-musl)."
        )
    return detected


def distribution_dir(
    cache_dir: str | os.PathLike[str], harness: str, version: str, platform_key: str
) -> Path:
    return Path(cache_dir).expanduser() / harness / version / platform_key


def release_binary_url(
    version: str, platform_key: str, binary: str = BINARY_NAME
) -> str:
    return f"{RELEASE_BASE_URL}/{version}/{platform_key}/{binary}"


def release_manifest_url(version: str) -> str:
    return f"{RELEASE_BASE_URL}/{version}/manifest.json"


def fetch_json(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310
        data = json.load(response)
    if not isinstance(data, dict):
        raise DistributionError(f"Unexpected JSON payload from {url}")
    return data


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.part")
    with urllib.request.urlopen(url, timeout=600) as response:  # noqa: S310
        with open(temporary, "wb") as handle:
            shutil.copyfileobj(response, handle)
    os.replace(temporary, destination)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(directory: Path) -> dict[str, Any] | None:
    path = directory / MANIFEST_FILENAME
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        logger.warning("Ignoring unreadable harness manifest at %s", path)
        return None
    return data if isinstance(data, dict) else None


def _load_cached(
    directory: Path, *, harness: str, version: str, platform_key: str
) -> HarnessDistribution | None:
    manifest = read_manifest(directory)
    if not manifest:
        return None
    if (
        manifest.get("harness") != harness
        or manifest.get("version") != version
        or manifest.get("platform") != platform_key
    ):
        return None
    binary_name = str(manifest.get("binary") or BINARY_NAME)
    binary_path = directory / binary_name
    if not binary_path.is_file():
        return None
    size = binary_path.stat().st_size
    expected_size = int(manifest.get("size_bytes") or 0)
    if size < MIN_BINARY_BYTES or (expected_size and size != expected_size):
        logger.warning(
            "Ignoring incomplete harness distribution at %s (%d != %d bytes)",
            directory,
            size,
            expected_size,
        )
        return None
    return HarnessDistribution(
        harness=harness,
        version=version,
        platform=platform_key,
        directory=directory,
        binary_path=binary_path,
        sha256=str(manifest.get("sha256") or ""),
        size_bytes=size,
    )


def _platform_manifest_entry(
    manifest: dict[str, Any], platform_key: str
) -> dict[str, Any]:
    platforms = manifest.get("platforms")
    if not isinstance(platforms, dict):
        raise DistributionError("Harness release manifest has no platforms table")
    entry = platforms.get(platform_key)
    if not isinstance(entry, dict):
        available = ", ".join(sorted(platforms)) or "none"
        raise DistributionError(
            "Harness release manifest has no binary for platform "
            f"{platform_key!r}. Available: {available}"
        )
    return entry


def _download_release_binary(
    version: str,
    platform_key: str,
    directory: Path,
    *,
    fetch: FetchJson,
    download_file: Download,
) -> tuple[Path, str, int]:
    manifest = fetch(release_manifest_url(version))
    entry = _platform_manifest_entry(manifest, platform_key)
    checksum = str(entry.get("checksum") or "")
    expected_size = int(entry.get("size") or 0)
    binary = str(entry.get("binary") or BINARY_NAME)

    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / binary
    logger.info(
        "Downloading harness distribution %s@%s for %s (%.1f MiB)",
        "claude-code",
        version,
        platform_key,
        expected_size / (1024 * 1024),
    )
    download_file(release_binary_url(version, platform_key, binary), destination)
    destination.chmod(0o755)

    actual_size = destination.stat().st_size
    if expected_size and actual_size != expected_size:
        destination.unlink(missing_ok=True)
        raise DistributionError(
            f"Downloaded harness binary has {actual_size} bytes, expected "
            f"{expected_size}; the download was truncated"
        )
    actual_checksum = sha256_file(destination)
    if checksum and actual_checksum != checksum:
        destination.unlink(missing_ok=True)
        raise DistributionError(
            "Downloaded harness binary failed checksum verification: "
            f"{actual_checksum} != {checksum}"
        )
    return destination, actual_checksum, actual_size


def ensure_distribution(
    config: Any,
    *,
    fetch: FetchJson = fetch_json,
    download_file: Download = download,
) -> HarnessDistribution:
    """Return the cached portable distribution, building it when missing."""
    harness = str(config.name)
    version = str(config.version)
    platform_key = detect_platform(config.platform)
    directory = distribution_dir(
        config.expanded_cache_dir, harness, version, platform_key
    )

    cached = _load_cached(
        directory, harness=harness, version=version, platform_key=platform_key
    )
    if cached is not None:
        logger.debug(
            "Using cached harness distribution %s@%s (%s) at %s",
            harness,
            version,
            platform_key,
            cached.binary_path,
        )
        return cached

    if harness != "claude-code":
        raise DistributionError(f"Unsupported harness distribution: {harness!r}")

    binary_path, checksum, size = _download_release_binary(
        version,
        platform_key,
        directory,
        fetch=fetch,
        download_file=download_file,
    )
    if size < MIN_BINARY_BYTES:
        raise DistributionError(
            f"Harness binary {binary_path} is unexpectedly small ({size} bytes)"
        )
    if _host_can_run(platform_key):
        verify_binary(binary_path)

    distribution = HarnessDistribution(
        harness=harness,
        version=version,
        platform=platform_key,
        directory=directory,
        binary_path=binary_path,
        sha256=checksum,
        size_bytes=size,
    )
    (directory / MANIFEST_FILENAME).write_text(
        json.dumps({**distribution.manifest(), "created_at": time.time()}, indent=2)
        + "\n"
    )
    logger.info(
        "Cached harness distribution %s@%s (%s, %.1f MiB) at %s",
        harness,
        version,
        platform_key,
        size / (1024 * 1024),
        directory,
    )
    return distribution


def _host_can_run(platform_key: str) -> bool:
    """Whether the host itself can execute the extracted binary (for checks)."""
    try:
        return detect_platform() == platform_key
    except DistributionError:
        return False


def verify_binary(binary_path: Path, timeout: float = 120.0) -> str:
    """Smoke-check a native harness binary by asking for its version."""
    try:
        result = subprocess.run(
            [str(binary_path), "--version"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise DistributionError(
            f"Portable harness binary {binary_path} is not executable: {exc}"
        ) from exc
    output = (result.stdout or result.stderr or "").strip()
    if result.returncode != 0:
        raise DistributionError(
            f"Portable harness binary {binary_path} failed its version check: "
            f"{output[:500]}"
        )
    logger.info("Verified portable harness binary: %s", output.splitlines()[0])
    return output
