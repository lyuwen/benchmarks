"""Tests for the ``harness-prepare`` cache warm-up/seed command."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.utils.harnesses import prepare
from benchmarks.utils.harnesses.config import HarnessConfig
from benchmarks.utils.harnesses.distribution import (
    MIN_BINARY_BYTES,
    DistributionError,
    release_manifest_url,
)


BLOB = b"b" * (MIN_BINARY_BYTES + 32)


def make_config(tmp_path: Path, platform: str = "linux-arm64") -> HarnessConfig:
    return HarnessConfig(
        version="2.1.331",
        platform=platform,
        cache_dir=str(tmp_path / "cache"),
    )


def write_binary(tmp_path: Path, name: str = "claude", blob: bytes = BLOB) -> Path:
    path = tmp_path / name
    path.write_bytes(blob)
    return path


def test_seed_distribution_writes_manifest(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    binary = write_binary(tmp_path)
    distribution = prepare.seed_distribution(config, binary)
    assert distribution.binary_path.read_bytes() == BLOB
    assert distribution.binary_path.name == "claude"
    assert distribution.platform == "linux-arm64"
    manifest = json.loads((distribution.directory / "manifest.json").read_text())
    assert manifest["harness"] == "claude-code"
    assert manifest["version"] == "2.1.331"
    assert manifest["size_bytes"] == len(BLOB)
    # A seeded cache is usable without touching the network.
    cached = prepare.check_distribution(config)
    assert cached is not None
    assert cached.binary_path == distribution.binary_path


def test_seed_renames_arbitrary_source_file(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    binary = write_binary(tmp_path, name="claude-2.1.331-download")
    distribution = prepare.seed_distribution(config, binary)
    assert distribution.binary_path.name == "claude"
    assert (distribution.directory / "claude").is_file()


def test_seed_checksum_mismatch(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    binary = write_binary(tmp_path)
    with pytest.raises(DistributionError, match="Checksum mismatch"):
        prepare.seed_distribution(config, binary, expected_sha256="0" * 64)


def test_seed_rejects_small_file_with_tarball_hint(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    binary = write_binary(tmp_path, name="claude.tgz", blob=b"tiny")
    with pytest.raises(DistributionError, match="npm tarball"):
        prepare.seed_distribution(config, binary)


def test_seed_missing_file(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    with pytest.raises(DistributionError, match="not found"):
        prepare.seed_distribution(config, tmp_path / "missing")


def test_check_distribution_when_empty(tmp_path: Path) -> None:
    assert prepare.check_distribution(make_config(tmp_path)) is None


def test_main_seed_mode(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    binary = write_binary(tmp_path)
    cache = tmp_path / "cache"
    exit_code = prepare.main(
        [
            "--harness-platform",
            "linux-arm64",
            "--harness-cache-dir",
            str(cache),
            "--binary",
            str(binary),
        ]
    )
    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Harness distribution ready" in output
    assert "linux-arm64" in output
    assert (cache / "claude-code" / "2.1.331" / "linux-arm64" / "claude").is_file()


def test_main_check_mode(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cache = tmp_path / "cache"
    exit_code = prepare.main(
        [
            "--harness-platform",
            "linux-arm64",
            "--harness-cache-dir",
            str(cache),
            "--check",
        ]
    )
    assert exit_code == 0
    assert "Not cached" in capsys.readouterr().out

    prepare.main(
        [
            "--harness-platform",
            "linux-arm64",
            "--harness-cache-dir",
            str(cache),
            "--binary",
            str(write_binary(tmp_path)),
        ]
    )
    capsys.readouterr()
    exit_code = prepare.main(
        [
            "--harness-platform",
            "linux-arm64",
            "--harness-cache-dir",
            str(cache),
            "--check",
        ]
    )
    assert exit_code == 0
    assert "Harness distribution ready" in capsys.readouterr().out


def test_main_reports_failure_with_offline_hint(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = prepare.main(
        [
            "--harness-platform",
            "linux-arm64",
            "--harness-cache-dir",
            str(tmp_path / "cache"),
            "--binary",
            str(tmp_path / "missing"),
        ]
    )
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "harness-prepare failed" in captured.err
    assert "Release manifest" in captured.err
    assert release_manifest_url("2.1.331") in captured.err


def test_cache_dir_env_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shared = tmp_path / "shared-cache"
    monkeypatch.setenv("HARNESS_DIST_CACHE_DIR", str(shared))
    assert HarnessConfig().cache_dir == str(shared)

    exit_code = prepare.main(
        [
            "--harness-platform",
            "linux-arm64",
            "--binary",
            str(write_binary(tmp_path)),
        ]
    )
    assert exit_code == 0
    assert (shared / "claude-code" / "2.1.331" / "linux-arm64" / "claude").is_file()


def test_cache_dir_flag_overrides_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HARNESS_DIST_CACHE_DIR", str(tmp_path / "from-env"))
    explicit = tmp_path / "from-flag"
    exit_code = prepare.main(
        [
            "--harness-platform",
            "linux-arm64",
            "--harness-cache-dir",
            str(explicit),
            "--binary",
            str(write_binary(tmp_path)),
        ]
    )
    assert exit_code == 0
    assert (explicit / "claude-code" / "2.1.331" / "linux-arm64" / "claude").is_file()
    assert not (tmp_path / "from-env").exists()
