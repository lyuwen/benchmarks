"""Pre-download (or seed) the portable harness distributions.

The evaluation entry points build the distribution lazily on their first run.
This command lets you warm the cache ahead of time — on a login node, in a
container image build, or on a machine with network access — and seed it from an
already-downloaded binary for air-gapped / offline evaluation hosts.

Usage::

    # Download the pinned Claude Code release into the default cache
    uv run harness-prepare

    # Same, but into a cache shared by all evaluation workers
    uv run harness-prepare --harness-cache-dir /shared/harness-dist

    # Offline: seed from a binary downloaded elsewhere (checksum optional but
    # recommended; the download mode prints it)
    uv run harness-prepare --binary /path/to/claude \\
        --sha256 4c00839cf46beb2b38ec310105f2d538f7f9321fe46c636299b74bcab94626b1

    # Show what is already cached without touching the network
    uv run harness-prepare --check
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

from benchmarks.utils.harnesses.config import (
    DEFAULT_CACHE_DIR,
    DEFAULT_CLAUDE_CODE_VERSION,
    HARNESS_PLATFORMS,
    SUPPORTED_HARNESSES,
    HarnessConfig,
)
from benchmarks.utils.harnesses.distribution import (
    BINARY_NAME,
    MANIFEST_FILENAME,
    MIN_BINARY_BYTES,
    DistributionError,
    HarnessDistribution,
    detect_platform,
    distribution_dir,
    ensure_distribution,
    host_can_run,
    load_cached_distribution,
    release_manifest_url,
    sha256_file,
    verify_binary,
)


logger = logging.getLogger(__name__)


def _config_from_args(args: argparse.Namespace) -> HarnessConfig:
    return HarnessConfig(
        name=args.harness,
        version=args.harness_version,
        platform=args.harness_platform,
        cache_dir=args.harness_cache_dir,
    )


def check_distribution(config: HarnessConfig) -> HarnessDistribution | None:
    """Return a cached, ready-to-use distribution without any network access."""
    platform_key = detect_platform(config.platform)
    directory = distribution_dir(
        config.expanded_cache_dir, config.name, config.version, platform_key
    )
    return load_cached_distribution(
        directory,
        harness=config.name,
        version=config.version,
        platform_key=platform_key,
    )


def seed_distribution(
    config: HarnessConfig,
    binary: str | os.PathLike[str],
    *,
    expected_sha256: str | None = None,
) -> HarnessDistribution:
    """Copy an already-downloaded harness binary into the cache.

    This is the offline path: no network call is made. When the binary matches
    the host platform it is smoke-checked with ``--version`` before being
    accepted.
    """
    source = Path(binary).expanduser()
    if not source.is_file():
        raise DistributionError(f"Harness binary not found: {source}")
    size = source.stat().st_size
    if size < MIN_BINARY_BYTES:
        hint = ""
        if source.suffix in {".tgz", ".tar", ".gz", ".xz"}:
            hint = (
                " This looks like an npm tarball; unpack it and pass the "
                f"extracted {BINARY_NAME!r} binary instead."
            )
        raise DistributionError(
            f"Harness binary {source} is unexpectedly small ({size} bytes).{hint}"
        )
    digest = sha256_file(source)
    if expected_sha256 and digest != expected_sha256.strip().lower():
        raise DistributionError(
            f"Checksum mismatch for {source}: {digest} != {expected_sha256}"
        )

    platform_key = detect_platform(config.platform)
    if host_can_run(platform_key):
        verify_binary(source)

    directory = distribution_dir(
        config.expanded_cache_dir, config.name, config.version, platform_key
    )
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / source.name
    if destination.resolve() != source.resolve():
        shutil.copy2(source, destination)
    destination.chmod(0o755)
    if destination.name != BINARY_NAME:
        # Keep the cache layout stable regardless of the source file name.
        canonical = directory / BINARY_NAME
        os.replace(destination, canonical)
        destination = canonical

    distribution = HarnessDistribution(
        harness=config.name,
        version=config.version,
        platform=platform_key,
        directory=directory,
        binary_path=destination,
        sha256=digest,
        size_bytes=size,
    )
    (directory / MANIFEST_FILENAME).write_text(
        json.dumps({**distribution.manifest(), "created_at": time.time()}, indent=2)
        + "\n"
    )
    logger.info("Seeded harness distribution at %s", directory)
    return distribution


def _print_distribution(distribution: HarnessDistribution) -> None:
    print("Harness distribution ready")
    print(
        f"  harness: {distribution.harness} {distribution.version} "
        f"({distribution.platform})"
    )
    print(
        f"  binary:  {distribution.binary_path} "
        f"({distribution.size_bytes / (1024 * 1024):.1f} MiB, "
        f"sha256 {distribution.sha256})"
    )
    print(f"  cache:   {distribution.directory}")
    print()
    print("Evaluation runs use this cache by default. For a shared/custom cache:")
    print(f"  --harness-cache-dir {distribution.directory.parents[2]}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="harness-prepare",
        description=(
            "Download (or seed) the portable blackbox-harness distributions "
            "used by run_infer_harnesses.py entry points."
        ),
    )
    parser.add_argument(
        "--harness",
        type=str,
        default="claude-code",
        choices=list(SUPPORTED_HARNESSES),
        help="Harness whose distribution to prepare (default: claude-code)",
    )
    parser.add_argument(
        "--harness-version",
        type=str,
        default=DEFAULT_CLAUDE_CODE_VERSION,
        help=f"Pinned harness release (default: {DEFAULT_CLAUDE_CODE_VERSION})",
    )
    parser.add_argument(
        "--harness-platform",
        type=str,
        default=None,
        choices=list(HARNESS_PLATFORMS),
        help="Container platform to prepare (default: auto-detected from host)",
    )
    parser.add_argument(
        "--harness-cache-dir",
        type=str,
        default=DEFAULT_CACHE_DIR,
        help=f"Host cache directory (default: {DEFAULT_CACHE_DIR})",
    )
    parser.add_argument(
        "--binary",
        type=str,
        default=None,
        help=(
            "Seed the cache from an already-downloaded harness binary instead "
            "of downloading it (offline mode)"
        ),
    )
    parser.add_argument(
        "--sha256",
        type=str,
        default=None,
        help="Expected sha256 of --binary; the seed fails when it does not match",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only report whether the distribution is already cached",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (default INFO)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = _config_from_args(args)
    try:
        platform_key = detect_platform(config.platform)
        if args.check:
            cached = check_distribution(config)
            if cached is None:
                print(
                    f"Not cached: {config.name} {config.version} "
                    f"({platform_key})\nRun `harness-prepare` to download it."
                )
                return 0
            _print_distribution(cached)
            return 0
        if args.binary:
            distribution = seed_distribution(
                config, args.binary, expected_sha256=args.sha256
            )
        else:
            distribution = ensure_distribution(config)
        _print_distribution(distribution)
        return 0
    except DistributionError as exc:
        if args.check:
            print(f"Not cached: {exc}")
            return 0
        print(f"harness-prepare failed: {exc}", file=sys.stderr)
        print(
            "If this host has no network access, download the binary elsewhere "
            f"and seed it:\n  harness-prepare --binary /path/to/{BINARY_NAME} "
            "--sha256 <checksum>\n"
            f"Release manifest: {release_manifest_url(config.version)}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
