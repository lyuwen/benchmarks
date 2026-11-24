"""Portable harness distributions for in-container execution.

Task containers (SWE-bench, SWE-rebench-V2, ...) do not have Node.js, and the
harness must not be installed from the network at run time. We therefore build
a portable *distribution* on the host once, cache it, and install it into the
task container.

Claude Code ships since v2.1.113 as a self-contained native binary (published
per platform both on npm and on ``downloads.claude.ai``): its distribution is a
single binary plus a manifest, no Node.js runtime is needed inside the
container.

DeepSeek Harness publishes a single native executable in the
``deepseek-harness-runtime-bin`` PyPI wheel (the Python SDK runtime carrier,
built as a Node single-executable application). Its distribution is that wheel
unpacked into the cache directory: the executable, the ``-rg`` ripgrep sidecar
and the target ``downloads.json`` resource metadata. The whole payload is copied
(or bind-mounted) into the container.

The build is intentionally offline-friendly: once a distribution is cached the
pipeline never touches the network again. Tests inject ``fetch_json`` /
``download`` so they can exercise the logic without network access.
"""

from __future__ import annotations

import contextlib
import errno
import hashlib
import json
import logging
import os
import platform as platform_module
import shutil
import subprocess
import time
import urllib.request
import uuid
import zipfile
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any


try:  # POSIX only; the cache lock is advisory and optional elsewhere.
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX
    fcntl = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)

RELEASE_BASE_URL = "https://downloads.claude.ai/claude-code-releases"

CLAUDE_CODE_HARNESS = "claude-code"
DEEPSEEK_HARNESS = "deepseek-harness"

# Pinned published runtime used by the DeepSeek Harness bundle. The repository
# checkout may be ahead of the last release; the evaluator consumes published
# runtime wheels so runs stay reproducible.
DSH_RUNTIME_VERSION = "0.2.2rc2"
DSH_RUNTIME_EXECUTABLES = {
    "linux-x64": "deepseek-harness-sdk-runtime-linux-x64",
    "linux-arm64": "deepseek-harness-sdk-runtime-linux-arm64",
}
DSH_RUNTIME_WHEELS: dict[str, dict[str, Any]] = {
    "linux-x64": {
        "filename": "deepseek_harness_runtime_bin-0.2.2rc2-py3-none-manylinux_2_28_x86_64.whl",
        "url": (
            "https://files.pythonhosted.org/packages/95/37/"
            "ba779ea8e90daa84f9f1ea9982521e9e1f6d99223857efc6f9cf739b4b85/"
            "deepseek_harness_runtime_bin-0.2.2rc2-py3-none-manylinux_2_28_x86_64.whl"
        ),
        "sha256": "3c8f15a7a92b12857032995ec5dbcc105f192c36218e9b6bb2c5b54b4ee7db17",
        "size": 97364370,
    },
    "linux-arm64": {
        "filename": "deepseek_harness_runtime_bin-0.2.2rc2-py3-none-manylinux_2_28_aarch64.whl",
        "url": (
            "https://files.pythonhosted.org/packages/52/7d/"
            "2e3718bd653d61f244d77ded9122efd924d45e03c5c74a035af26adb9285/"
            "deepseek_harness_runtime_bin-0.2.2rc2-py3-none-manylinux_2_28_aarch64.whl"
        ),
        "sha256": "dc8f6e2c463c9c174b8fbafa394ada62351ff16b9febde4b90dca6b57f1b4e84",
        "size": 99239965,
    },
}
DSH_WHEEL_RUNTIME_PREFIX = "deepseek_harness_runtime/runtime/"

# Harnesses whose portable distribution is a directory payload rather than a
# single binary (the manifest lists every payload file relative to the cache
# directory).
DIRECTORY_PAYLOAD_HARNESSES = frozenset({DEEPSEEK_HARNESS})
MANIFEST_FILENAME = "manifest.json"
BINARY_NAME = "claude"
MIN_BINARY_BYTES = 1_000_000
LOCK_FILENAME = ".distribution.lock"
LOCK_WAIT_TIMEOUT_SEC = 900.0

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
    # Payload files relative to ``directory``. Empty means just the binary.
    files: tuple[str, ...] = ()

    def payload_files(self) -> list[str]:
        return list(self.files) if self.files else [self.binary_path.name]

    def manifest(self) -> dict[str, Any]:
        return {
            "harness": self.harness,
            "version": self.version,
            "platform": self.platform,
            "binary": self.binary_path.name,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "files": self.payload_files(),
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


def _temporary_path(destination: Path, suffix: str) -> Path:
    """Unique sibling temp path so concurrent hosts never share a partial file."""
    return destination.with_name(
        f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.{suffix}"
    )


def download(url: str, destination: Path) -> None:
    """Download to a unique temp file, then atomically publish it.

    The temp name includes the pid and a random token: two hosts (or two
    workers) racing on a shared cache never write the same partial file, and the
    final ``os.replace`` is atomic within the cache filesystem.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(destination, "part")
    try:
        with urllib.request.urlopen(url, timeout=600) as response:  # noqa: S310
            with open(temporary, "wb") as handle:
                shutil.copyfileobj(response, handle)
                handle.flush()
                os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_manifest(directory: Path, payload: dict[str, Any]) -> Path:
    """Atomically publish ``manifest.json`` in the cache directory."""
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / MANIFEST_FILENAME
    temporary = _temporary_path(destination, "tmp")
    data = json.dumps(payload, indent=2) + "\n"
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


@contextlib.contextmanager
def distribution_lock(
    directory: Path, *, timeout: float = LOCK_WAIT_TIMEOUT_SEC
) -> Iterator[None]:
    """Advisory inter-process/host lock for building one distribution entry.

    Serializes first-time downloads so a shared (e.g. NFS) cache is fetched
    once instead of once per node/worker. The lock is *advisory*: filesystems
    that do not support ``flock`` (or a read-only cache directory) fall back to
    lock-free building, which stays correct because downloads and manifests are
    published with unique temp files + atomic renames.
    """
    if fcntl is None:  # pragma: no cover - non-POSIX
        yield
        return
    try:
        directory.mkdir(parents=True, exist_ok=True)
        handle = open(directory / LOCK_FILENAME, "a+")
    except OSError as exc:
        logger.debug(
            "Cannot open harness cache lock in %s (%s); continuing", directory, exc
        )
        yield
        return

    locked = False
    try:
        deadline = time.monotonic() + timeout
        while True:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                locked = True
                break
            except OSError as exc:
                if exc.errno not in (errno.EACCES, errno.EAGAIN):
                    logger.warning(
                        "Harness cache lock not supported in %s (%s); "
                        "continuing without serializing the download",
                        directory,
                        exc,
                    )
                    break
                if time.monotonic() >= deadline:
                    logger.warning(
                        "Timed out waiting for the harness cache lock in %s; "
                        "continuing without it",
                        directory,
                    )
                    break
                time.sleep(0.5)
        yield
    finally:
        if locked:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except OSError:  # pragma: no cover - defensive
                pass
        handle.close()


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


def load_cached_distribution(
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
    raw_files = manifest.get("files")
    files = (
        tuple(str(name) for name in raw_files if isinstance(name, str) and name)
        if isinstance(raw_files, list)
        else ()
    )
    if not files:
        files = (binary_name,)
    for relative in files:
        if ".." in Path(relative).parts or Path(relative).is_absolute():
            logger.warning("Ignoring unsafe harness payload path %r", relative)
            return None
        if not (directory / relative).is_file():
            return None
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
        files=files,
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


def _extract_deepseek_runtime(
    wheel_path: Path, directory: Path
) -> tuple[Path, tuple[str, ...]]:
    """Unpack a DeepSeek Harness runtime wheel's executable payload."""
    # The executable name identifies the platform unambiguously.
    executables = set(DSH_RUNTIME_EXECUTABLES.values())
    staging = directory / f".payload.{os.getpid()}.{uuid.uuid4().hex}"
    staging.mkdir(parents=True, exist_ok=False)
    extracted: list[str] = []
    try:
        with zipfile.ZipFile(wheel_path) as archive:
            for info in archive.infolist():
                if info.is_dir():
                    continue
                name = info.filename
                if DSH_WHEEL_RUNTIME_PREFIX not in name:
                    continue
                relative = name.split(DSH_WHEEL_RUNTIME_PREFIX, 1)[1]
                parts = Path(relative).parts
                if not relative or Path(relative).is_absolute() or ".." in parts:
                    raise DistributionError(
                        f"Refusing unsafe runtime wheel member {name!r}"
                    )
                destination = staging / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info) as source, open(destination, "wb") as output:
                    shutil.copyfileobj(source, output)
                executable = bool((info.external_attr >> 16) & 0o111)
                destination.chmod(0o755 if executable else 0o644)
                extracted.append(relative)
        binary_names = [rel for rel in extracted if Path(rel).name in executables]
        if len(binary_names) != 1:
            raise DistributionError(
                "Runtime wheel does not contain exactly one DeepSeek Harness "
                f"executable (found {binary_names})"
            )
        binary_name = binary_names[0]
        executable_name = Path(binary_name).name
        sidecar = f"{executable_name}-rg"
        if f"{binary_name}-rg" not in extracted:
            raise DistributionError(
                f"Runtime wheel is missing the {sidecar!r} ripgrep sidecar"
            )
        # Make the runner and sidecar executable regardless of zip mode bits.
        for relative in (binary_name, f"{binary_name}-rg"):
            (staging / relative).chmod(0o755)
        for relative in extracted:
            destination = directory / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staging / relative, destination)
        platform_key = None
        for key, name in DSH_RUNTIME_EXECUTABLES.items():
            if executable_name == name:
                platform_key = key
        assert platform_key is not None
        binary = directory / binary_name
        files = tuple(sorted(extracted))
        return binary, files
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _verify_dsh_wheel(wheel_path: Path, platform_key: str) -> dict[str, Any]:
    spec = DSH_RUNTIME_WHEELS.get(platform_key)
    if spec is None:
        supported = ", ".join(sorted(DSH_RUNTIME_WHEELS)) or "none"
        raise DistributionError(
            "DeepSeek Harness publishes no runtime wheel for platform "
            f"{platform_key!r}. Supported: {supported}. (The runtime is glibc/"
            "manylinux only; musl platforms need a separately built payload "
            "seeded with --wheel.)"
        )
    size = wheel_path.stat().st_size
    expected_size = int(spec["size"])
    if expected_size and size != expected_size:
        raise DistributionError(
            f"Downloaded DeepSeek Harness wheel has {size} bytes, expected "
            f"{expected_size}; the download was truncated"
        )
    checksum = sha256_file(wheel_path)
    if checksum != spec["sha256"]:
        raise DistributionError(
            "Downloaded DeepSeek Harness wheel failed checksum verification: "
            f"{checksum} != {spec['sha256']}"
        )
    return spec


def _ensure_deepseek_harness_distribution(
    directory: Path,
    *,
    version: str,
    platform_key: str,
    download_file: Download,
) -> HarnessDistribution:
    if version != DSH_RUNTIME_VERSION:
        raise DistributionError(
            f"No pinned DeepSeek Harness runtime wheel for version {version!r} "
            f"(pinned: {DSH_RUNTIME_VERSION}). Pass --wheel or --binary to "
            "harness-prepare to seed another build."
        )
    spec = DSH_RUNTIME_WHEELS.get(platform_key)
    if spec is None:
        raise DistributionError(
            "DeepSeek Harness publishes no runtime wheel for platform "
            f"{platform_key!r}. Supported: "
            f"{', '.join(sorted(DSH_RUNTIME_WHEELS)) or 'none'}."
        )
    directory.mkdir(parents=True, exist_ok=True)
    wheel_path = directory / spec["filename"]
    logger.info(
        "Downloading DeepSeek Harness runtime %s for %s (%.1f MiB)",
        version,
        platform_key,
        int(spec["size"]) / (1024 * 1024),
    )
    download_file(spec["url"], wheel_path)
    try:
        _verify_dsh_wheel(wheel_path, platform_key)
        binary_path, files = _extract_deepseek_runtime(wheel_path, directory)
    finally:
        wheel_path.unlink(missing_ok=True)
    size = binary_path.stat().st_size
    if size < MIN_BINARY_BYTES:
        raise DistributionError(
            f"DeepSeek Harness runtime {binary_path} is unexpectedly small "
            f"({size} bytes)"
        )
    if host_can_run(platform_key):
        verify_binary(binary_path)
    checksum = sha256_file(binary_path)
    distribution = HarnessDistribution(
        harness=DEEPSEEK_HARNESS,
        version=version,
        platform=platform_key,
        directory=directory,
        binary_path=binary_path,
        sha256=checksum,
        size_bytes=size,
        files=files,
    )
    write_manifest(directory, {**distribution.manifest(), "created_at": time.time()})
    logger.info(
        "Cached DeepSeek Harness runtime %s@%s (%s, %.1f MiB, %d payload files) at %s",
        DEEPSEEK_HARNESS,
        version,
        platform_key,
        size / (1024 * 1024),
        len(files),
        directory,
    )
    return distribution


def _build_claude_code_distribution(
    directory: Path,
    *,
    harness: str,
    version: str,
    platform_key: str,
    fetch: FetchJson,
    download_file: Download,
) -> HarnessDistribution:
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
    if host_can_run(platform_key):
        verify_binary(binary_path)
    distribution = HarnessDistribution(
        harness=harness,
        version=version,
        platform=platform_key,
        directory=directory,
        binary_path=binary_path,
        sha256=checksum,
        size_bytes=size,
        files=(binary_path.name,),
    )
    write_manifest(directory, {**distribution.manifest(), "created_at": time.time()})
    logger.info(
        "Cached harness distribution %s@%s (%s, %.1f MiB) at %s",
        harness,
        version,
        platform_key,
        size / (1024 * 1024),
        directory,
    )
    return distribution


def payload_paths(distribution: HarnessDistribution) -> list[Path]:
    """Absolute cache paths of every payload file in ``distribution``."""
    return [
        distribution.directory / relative for relative in distribution.payload_files()
    ]


def build_distribution_bind_volumes(
    config: Any, distribution: HarnessDistribution
) -> list[str]:
    """Bind mounts needed to expose a portable distribution in the container."""
    if getattr(config, "dist_mode", "copy") != "mount":
        return []
    install_dir = str(getattr(config, "install_dir", "")).rstrip("/") or "/opt/harness"
    return [f"{distribution.directory}:{install_dir}:ro"]


def _publish_deepseek_payload(
    directory: Path,
    *,
    version: str,
    platform_key: str,
    binary_source: Path,
    sidecar_source: Path,
    resources_source: Path | None,
) -> HarnessDistribution:
    """Copy an extracted DeepSeek runtime payload into the cache atomically."""
    executable_name = DSH_RUNTIME_EXECUTABLES[platform_key]
    if binary_source.name != executable_name:
        raise DistributionError(
            f"DeepSeek Harness executable must be named {executable_name!r}, "
            f"got {binary_source.name!r}"
        )
    if not binary_source.is_file():
        raise DistributionError(f"DeepSeek Harness runtime not found: {binary_source}")
    if not sidecar_source.is_file():
        raise DistributionError(
            "DeepSeek Harness runtime sidecar not found next to the executable: "
            f"{sidecar_source}"
        )
    size = binary_source.stat().st_size
    if size < MIN_BINARY_BYTES:
        raise DistributionError(
            f"DeepSeek Harness runtime {binary_source} is unexpectedly small "
            f"({size} bytes)"
        )
    directory.mkdir(parents=True, exist_ok=True)
    payload: list[tuple[Path, str]] = [
        (binary_source, executable_name),
        (sidecar_source, f"{executable_name}-rg"),
    ]
    if resources_source is not None:
        payload.append((resources_source, f"{platform_key}/downloads.json"))
    files: list[str] = []
    for source, relative in payload:
        destination = directory / relative
        temporary = _temporary_path(destination, "part")
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(source, temporary)
            temporary.chmod(
                0o755
                if relative.endswith("-rg") or relative == executable_name
                else 0o644
            )
            os.replace(temporary, destination)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        files.append(relative)
    binary_path = directory / executable_name
    if host_can_run(platform_key):
        verify_binary(binary_path)
    checksum = sha256_file(binary_path)
    distribution = HarnessDistribution(
        harness=DEEPSEEK_HARNESS,
        version=version,
        platform=platform_key,
        directory=directory,
        binary_path=binary_path,
        sha256=checksum,
        size_bytes=size,
        files=tuple(sorted(files)),
    )
    write_manifest(directory, {**distribution.manifest(), "created_at": time.time()})
    logger.info("Seeded DeepSeek Harness runtime at %s", directory)
    return distribution


def seed_deepseek_distribution(
    config: Any,
    source: Path,
    *,
    expected_sha256: str | None = None,
) -> HarnessDistribution:
    """Seed the DeepSeek Harness cache from a wheel or an extracted runtime.

    ``source`` may be the published ``deepseek_harness_runtime_bin`` wheel, the
    extracted runtime directory (as installed by the Python SDK) or the extracted
    executable (its ``-rg`` sidecar and ``<platform>/downloads.json`` are
    collected from the same directory when present). No network is used.
    """
    platform_key = detect_platform(config.platform)
    if platform_key not in DSH_RUNTIME_EXECUTABLES:
        raise DistributionError(
            "DeepSeek Harness runtime wheels exist only for "
            f"{', '.join(sorted(DSH_RUNTIME_EXECUTABLES))}; got {platform_key!r}"
        )
    directory = distribution_dir(
        config.expanded_cache_dir, config.name, config.version, platform_key
    )
    source = Path(source).expanduser()
    if source.is_dir():
        source = source / DSH_RUNTIME_EXECUTABLES[platform_key]
        if not source.is_file():
            raise DistributionError(
                f"DeepSeek Harness runtime directory does not contain "
                f"{source.name}: {source.parent}"
            )
    if not source.is_file():
        raise DistributionError(f"DeepSeek Harness payload not found: {source}")
    if expected_sha256 and sha256_file(source) != expected_sha256.strip().lower():
        raise DistributionError(
            f"Checksum mismatch for {source}: "
            f"{sha256_file(source)} != {expected_sha256}"
        )
    directory.mkdir(parents=True, exist_ok=True)
    with distribution_lock(directory):
        if source.suffix == ".whl":
            if (
                expected_sha256
                and sha256_file(source) != expected_sha256.strip().lower()
            ):
                raise DistributionError(
                    f"Checksum mismatch for {source}: "
                    f"{sha256_file(source)} != {expected_sha256}"
                )
            binary_path, files = _extract_deepseek_runtime(source, directory)
            if host_can_run(platform_key):
                verify_binary(binary_path)
            distribution = HarnessDistribution(
                harness=DEEPSEEK_HARNESS,
                version=str(config.version),
                platform=platform_key,
                directory=directory,
                binary_path=binary_path,
                sha256=sha256_file(binary_path),
                size_bytes=binary_path.stat().st_size,
                files=files,
            )
            write_manifest(
                directory,
                {**distribution.manifest(), "created_at": time.time()},
            )
            logger.info("Seeded DeepSeek Harness runtime from wheel at %s", directory)
            return distribution

        sidecar = source.with_name(f"{source.name}-rg")
        resources = source.parent / platform_key / "downloads.json"
        return _publish_deepseek_payload(
            directory,
            version=str(config.version),
            platform_key=platform_key,
            binary_source=source,
            sidecar_source=sidecar,
            resources_source=resources if resources.is_file() else None,
        )


def ensure_distribution(
    config: Any,
    *,
    fetch: FetchJson = fetch_json,
    download_file: Download = download,
) -> HarnessDistribution:
    """Return the cached portable distribution, building it when missing.

    Safe on a shared/distributed cache (e.g. NFS): the cached entry is checked
    lock-free first, then an advisory :func:`distribution_lock` serializes
    first-time builds across workers/hosts, and payloads/manifests are published
    with unique temp files + atomic renames. The directory layout is per version
    and platform, so nodes preparing different container platforms never
    collide.
    """
    harness = str(config.name)
    version = str(config.version)
    platform_key = detect_platform(config.platform)
    directory = distribution_dir(
        config.expanded_cache_dir, harness, version, platform_key
    )

    cached = load_cached_distribution(
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

    if harness not in (CLAUDE_CODE_HARNESS, DEEPSEEK_HARNESS):
        raise DistributionError(f"Unsupported harness distribution: {harness!r}")

    with distribution_lock(directory):
        # Another host/worker may have finished the build while we waited.
        cached = load_cached_distribution(
            directory, harness=harness, version=version, platform_key=platform_key
        )
        if cached is not None:
            logger.info(
                "Harness distribution %s@%s (%s) was prepared concurrently at %s",
                harness,
                version,
                platform_key,
                cached.binary_path,
            )
            return cached
        if harness == DEEPSEEK_HARNESS:
            return _ensure_deepseek_harness_distribution(
                directory,
                version=version,
                platform_key=platform_key,
                download_file=download_file,
            )
        return _build_claude_code_distribution(
            directory,
            harness=harness,
            version=version,
            platform_key=platform_key,
            fetch=fetch,
            download_file=download_file,
        )


def host_can_run(platform_key: str) -> bool:
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
