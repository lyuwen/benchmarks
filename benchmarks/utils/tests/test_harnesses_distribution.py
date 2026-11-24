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


# ---------------------------------------------------------------------------
# DeepSeek Harness: directory payload downloaded as a runtime wheel
# ---------------------------------------------------------------------------

DSH_EXE = "deepseek-harness-sdk-runtime-linux-x64"
DSH_BLOB = b"d" * (MIN_BINARY_BYTES + 4096)


def dsh_wheel_bytes() -> bytes:
    import zipfile

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(f"deepseek_harness_runtime/runtime/{DSH_EXE}", DSH_BLOB)
        archive.writestr(f"deepseek_harness_runtime/runtime/{DSH_EXE}-rg", b"rg-binary")
        archive.writestr(
            "deepseek_harness_runtime/runtime/linux-x64/downloads.json", b"{}"
        )
        archive.writestr("deepseek_harness_runtime/__init__.py", b"")
    return buffer.getvalue()


def make_dsh_config(tmp_path: Path, platform: str = "linux-x64") -> SimpleNamespace:
    return SimpleNamespace(
        name="deepseek-harness",
        version=distribution.DSH_RUNTIME_VERSION,
        platform=platform,
        expanded_cache_dir=str(tmp_path / "cache"),
    )


def pin_fake_wheel(monkeypatch: pytest.MonkeyPatch, blob: bytes) -> None:
    monkeypatch.setitem(
        distribution.DSH_RUNTIME_WHEELS,
        "linux-x64",
        {
            "filename": "fake-runtime-linux-x64.whl",
            "url": "https://example.invalid/fake.whl",
            "sha256": hashlib.sha256(blob).hexdigest(),
            "size": len(blob),
        },
    )
    monkeypatch.setattr(distribution, "host_can_run", lambda platform: False)


def test_ensure_deepseek_distribution_extracts_wheel(tmp_path, monkeypatch) -> None:
    blob = dsh_wheel_bytes()
    pin_fake_wheel(monkeypatch, blob)
    config = make_dsh_config(tmp_path)
    downloads: list[str] = []

    def fake_download(url: str, destination: Path) -> None:
        downloads.append(url)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(blob)

    result = ensure_distribution(config, download_file=fake_download)
    assert result.harness == "deepseek-harness"
    assert result.binary_path.name == DSH_EXE
    assert result.binary_path.read_bytes() == DSH_BLOB
    assert set(result.files) == {
        DSH_EXE,
        f"{DSH_EXE}-rg",
        "linux-x64/downloads.json",
    }
    manifest = json.loads((result.directory / "manifest.json").read_text())
    assert manifest["files"] == list(result.files)
    assert not (result.directory / "fake-runtime-linux-x64.whl").exists()

    def failing_download(url: str, destination: Path) -> None:
        raise AssertionError(f"cache hit should not download {url}")

    cached = ensure_distribution(config, download_file=failing_download)
    assert cached.binary_path == result.binary_path
    assert len(downloads) == 1


def test_ensure_deepseek_distribution_rebuilds_missing_payload(
    tmp_path, monkeypatch
) -> None:
    blob = dsh_wheel_bytes()
    pin_fake_wheel(monkeypatch, blob)
    config = make_dsh_config(tmp_path)
    calls: list[str] = []

    def fake_download(url: str, destination: Path) -> None:
        calls.append(url)
        destination.write_bytes(blob)

    result = ensure_distribution(config, download_file=fake_download)
    (result.directory / "linux-x64" / "downloads.json").unlink()
    rebuilt = ensure_distribution(config, download_file=fake_download)
    assert rebuilt.binary_path == result.binary_path
    assert len(calls) == 2


def test_ensure_deepseek_distribution_rejects_unpinned_version_and_musl(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(distribution, "host_can_run", lambda platform: False)
    config = make_dsh_config(tmp_path)
    config.version = "9.9.9"
    with pytest.raises(DistributionError, match="No pinned DeepSeek Harness"):
        ensure_distribution(config, download_file=lambda url, destination: None)

    musl = make_dsh_config(tmp_path, platform="linux-x64-musl")
    with pytest.raises(DistributionError, match="no runtime wheel"):
        ensure_distribution(musl, download_file=lambda url, destination: None)


def test_ensure_deepseek_distribution_checks_wheel_checksum(
    tmp_path, monkeypatch
) -> None:
    blob = dsh_wheel_bytes()
    pin_fake_wheel(monkeypatch, blob)
    monkeypatch.setitem(
        distribution.DSH_RUNTIME_WHEELS,
        "linux-x64",
        {
            "filename": "fake.whl",
            "url": "https://example.invalid/fake.whl",
            "sha256": "0" * 64,
            "size": len(blob),
        },
    )

    def fake_download(url: str, destination: Path) -> None:
        destination.write_bytes(blob)

    with pytest.raises(DistributionError, match="checksum"):
        ensure_distribution(make_dsh_config(tmp_path), download_file=fake_download)


def test_seed_deepseek_distribution_from_wheel(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(distribution, "host_can_run", lambda platform: False)
    wheel = tmp_path / "runtime.whl"
    wheel.write_bytes(dsh_wheel_bytes())
    config = make_dsh_config(tmp_path)
    result = distribution.seed_deepseek_distribution(config, wheel)
    assert result.binary_path.name == DSH_EXE
    assert "linux-x64/downloads.json" in result.files
    assert (
        distribution.load_cached_distribution(
            result.directory,
            harness="deepseek-harness",
            version=config.version,
            platform_key="linux-x64",
        )
        is not None
    )


def test_seed_deepseek_distribution_from_executable_collects_payload(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(distribution, "host_can_run", lambda platform: False)
    source = tmp_path / "runtime" / DSH_EXE
    source.parent.mkdir()
    source.write_bytes(DSH_BLOB)
    (source.parent / f"{DSH_EXE}-rg").write_bytes(b"rg")
    (source.parent / "linux-x64").mkdir()
    (source.parent / "linux-x64" / "downloads.json").write_text("{}")
    config = make_dsh_config(tmp_path)
    result = distribution.seed_deepseek_distribution(config, source)
    assert set(result.files) == {
        DSH_EXE,
        f"{DSH_EXE}-rg",
        "linux-x64/downloads.json",
    }
    assert (result.directory / "linux-x64" / "downloads.json").read_text() == "{}"


def test_build_distribution_bind_volumes_for_directory_payload(tmp_path) -> None:
    directory = tmp_path / "cache"
    directory.mkdir()
    binary = directory / DSH_EXE
    binary.write_bytes(b"x")
    payload = HarnessDistribution(
        harness="deepseek-harness",
        version="0.2.2rc2",
        platform="linux-x64",
        directory=directory,
        binary_path=binary,
        sha256="x",
        size_bytes=1,
        files=(DSH_EXE,),
    )
    mount_config = SimpleNamespace(dist_mode="mount", install_dir="/opt/harness/")
    assert distribution.build_distribution_bind_volumes(mount_config, payload) == [
        f"{directory}:/opt/harness:ro"
    ]
    copy_config = SimpleNamespace(dist_mode="copy", install_dir="/opt/harness")
    assert distribution.build_distribution_bind_volumes(copy_config, payload) == []
