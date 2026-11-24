"""Tests for the portable harness distribution cache."""

from __future__ import annotations

import hashlib
import io
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.utils.harnesses import distribution
from benchmarks.utils.harnesses.distribution import (
    MIN_BINARY_BYTES,
    DistributionError,
    HarnessDistribution,
    detect_platform,
    distribution_dir,
    download,
    ensure_distribution,
    release_binary_url,
    release_manifest_url,
    verify_binary,
    write_manifest,
)


BLOB = b"c" * (MIN_BINARY_BYTES + 16)
CHECKSUM = hashlib.sha256(BLOB).hexdigest()


def make_config(tmp_path: Path, platform: str = "linux-arm64") -> SimpleNamespace:
    return SimpleNamespace(
        name="claude-code",
        version="2.1.331",
        platform=platform,
        expanded_cache_dir=str(tmp_path / "cache"),
    )


def release_manifest(checksum: str = CHECKSUM, size: int = len(BLOB)) -> dict:
    return {
        "version": "2.1.331",
        "platforms": {
            "linux-arm64": {
                "binary": "claude",
                "checksum": checksum,
                "size": size,
            }
        },
    }


def test_detect_platform_override() -> None:
    assert detect_platform("linux-x64-musl") == "linux-x64-musl"


def test_detect_platform_from_host(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(distribution.platform_module, "system", lambda: "Linux")
    monkeypatch.setattr(distribution.platform_module, "machine", lambda: "x86_64")
    assert detect_platform() == "linux-x64"


def test_detect_platform_unsupported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(distribution.platform_module, "system", lambda: "Darwin")
    monkeypatch.setattr(distribution.platform_module, "machine", lambda: "arm64")
    with pytest.raises(DistributionError, match="--harness-platform"):
        detect_platform()


def test_release_urls() -> None:
    assert release_binary_url("2.1.331", "linux-x64") == (
        "https://downloads.claude.ai/claude-code-releases/2.1.331/linux-x64/claude"
    )
    assert release_manifest_url("2.1.331").endswith("/2.1.331/manifest.json")


def test_ensure_distribution_downloads_caches_and_reuses(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    requested: list[str] = []

    def fake_fetch(url: str) -> dict:
        requested.append(url)
        assert url == release_manifest_url("2.1.331")
        return release_manifest()

    def fake_download(url: str, destination: Path) -> None:
        requested.append(url)
        assert url == release_binary_url("2.1.331", "linux-arm64")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(BLOB)

    result = ensure_distribution(config, fetch=fake_fetch, download_file=fake_download)
    assert result.binary_path.read_bytes() == BLOB
    assert result.sha256 == CHECKSUM
    assert result.size_bytes == len(BLOB)
    assert result.platform == "linux-arm64"
    manifest = json.loads((result.directory / "manifest.json").read_text())
    assert manifest["sha256"] == CHECKSUM

    # A cached distribution must not touch the network again.
    def failing_fetch(url: str) -> dict:
        raise AssertionError(f"unexpected network call to {url}")

    cached = ensure_distribution(
        config,
        fetch=failing_fetch,
        download_file=lambda url, destination: None,
    )
    assert cached.binary_path == result.binary_path
    expected_dir = distribution_dir(
        config.expanded_cache_dir, "claude-code", "2.1.331", "linux-arm64"
    )
    assert result.directory == expected_dir


def test_ensure_distribution_rejects_bad_checksum(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    monkeypatch_manifest = release_manifest(checksum="0" * 64)

    def fake_download(url: str, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(BLOB)

    with pytest.raises(DistributionError, match="checksum"):
        ensure_distribution(
            config,
            fetch=lambda url: monkeypatch_manifest,
            download_file=fake_download,
        )


def test_ensure_distribution_rebuilds_on_truncated_cache(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    directory = distribution_dir(
        config.expanded_cache_dir, "claude-code", "2.1.331", "linux-arm64"
    )
    directory.mkdir(parents=True)
    (directory / "claude").write_bytes(b"short")
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "harness": "claude-code",
                "version": "2.1.331",
                "platform": "linux-arm64",
                "binary": "claude",
                "sha256": CHECKSUM,
                "size_bytes": len(BLOB),
            }
        )
    )
    calls: list[str] = []

    def fake_fetch(url: str) -> dict:
        calls.append(url)
        return release_manifest()

    def fake_download(url: str, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(BLOB)

    result = ensure_distribution(config, fetch=fake_fetch, download_file=fake_download)
    assert calls, "a truncated distribution must be re-downloaded"
    assert result.binary_path.stat().st_size == len(BLOB)


def test_ensure_distribution_unsupported_harness(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    config.name = "unknown-harness"
    with pytest.raises(DistributionError, match="Unsupported harness"):
        ensure_distribution(
            config, fetch=lambda url: {}, download_file=lambda u, d: None
        )


def test_verify_binary_reports_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binary = tmp_path / "claude"
    binary.write_text("#!/bin/sh\nexit 3\n")
    binary.chmod(0o755)
    with pytest.raises(DistributionError, match="version check"):
        verify_binary(binary)


def test_missing_platform_entry_is_reported(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    with pytest.raises(DistributionError, match="no binary for platform"):
        ensure_distribution(
            config,
            fetch=lambda url: {"platforms": {"darwin-arm64": {}}},
            download_file=lambda url, destination: None,
        )


def test_download_publishes_atomically_with_unique_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeResponse(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

    requested: list[str] = []

    def fake_urlopen(url, timeout=None):
        requested.append(url)
        return FakeResponse(BLOB)

    monkeypatch.setattr(distribution.urllib.request, "urlopen", fake_urlopen)
    destination = tmp_path / "cache" / "claude"
    url = release_binary_url("2.1.331", "linux-arm64")
    download(url, destination)
    assert requested == [url]
    assert destination.read_bytes() == BLOB
    leftovers = [
        path.name for path in destination.parent.iterdir() if path != destination
    ]
    assert leftovers == []


def test_write_manifest_is_atomic(tmp_path: Path) -> None:
    directory = tmp_path / "cache"
    write_manifest(directory, {"version": "2.1.331"})
    assert json.loads((directory / "manifest.json").read_text()) == {
        "version": "2.1.331"
    }
    leftovers = [
        path.name for path in directory.iterdir() if path.name != "manifest.json"
    ]
    assert leftovers == []


def test_ensure_distribution_shared_cache_downloads_once(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    calls = {"fetch": 0, "download": 0}
    guard = threading.Lock()

    def fake_fetch(url: str) -> dict:
        with guard:
            calls["fetch"] += 1
        return release_manifest()

    def fake_download(url: str, destination: Path) -> None:
        with guard:
            calls["download"] += 1
        time.sleep(0.3)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(BLOB)

    results: list[HarnessDistribution] = []
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            results.append(
                ensure_distribution(
                    config, fetch=fake_fetch, download_file=fake_download
                )
            )
        except BaseException as exc:  # noqa: BLE001 - reported after join
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)

    assert errors == []
    assert calls["download"] == 1
    assert len(results) == 3
    paths = {getattr(result, "binary_path") for result in results}
    assert paths == {results[0].binary_path}


def test_ensure_distribution_works_without_flock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = make_config(tmp_path)
    monkeypatch.setattr(distribution, "fcntl", None)

    def fake_download(url: str, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(BLOB)

    result = ensure_distribution(
        config, fetch=lambda url: release_manifest(), download_file=fake_download
    )
    assert result.binary_path.read_bytes() == BLOB


def test_ensure_distribution_works_when_flock_is_unsupported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = make_config(tmp_path)

    class UnsupportedFlock:
        LOCK_EX = 2
        LOCK_NB = 4
        LOCK_UN = 8

        @staticmethod
        def flock(fd, operation):  # noqa: ANN001 - mimics fcntl.flock
            raise OSError(37, "No locks available")

    monkeypatch.setattr(distribution, "fcntl", UnsupportedFlock)

    def fake_download(url: str, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(BLOB)

    result = ensure_distribution(
        config, fetch=lambda url: release_manifest(), download_file=fake_download
    )
    assert result.binary_path.read_bytes() == BLOB
