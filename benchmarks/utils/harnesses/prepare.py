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
import logging
import os
import shutil
import sys
import time
import uuid
from pathlib import Path

from benchmarks.utils.harnesses.config import (
    DEFAULT_HARNESS_VERSIONS,
    HARNESS_CACHE_ENV,
    HARNESS_PLATFORMS,
    HARNESS_PLATFORMS_BY_HARNESS,
    SUPPORTED_HARNESSES,
    HarnessConfig,
    default_cache_dir,
)
from benchmarks.utils.harnesses.distribution import (
    BINARY_NAME,
    DEEPSEEK_HARNESS,
    DSH_RUNTIME_VERSION,
    DSH_RUNTIME_WHEELS,
    MIN_BINARY_BYTES,
    DistributionError,
    HarnessDistribution,
    detect_platform,
    distribution_dir,
    distribution_lock,
    ensure_distribution,
    host_can_run,
    load_cached_distribution,
    release_manifest_url,
    seed_deepseek_distribution,
    sha256_file,
    verify_binary,
    write_manifest,
)


logger = logging.getLogger(__name__)


def _config_from_args(args: argparse.Namespace) -> HarnessConfig:
    return HarnessConfig(
        name=args.harness,
        version=args.harness_version or DEFAULT_HARNESS_VERSIONS[args.harness],
        platform=args.harness_platform,
        cache_dir=args.harness_cache_dir or default_cache_dir(),
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
    """Copy an already-downloaded single-binary harness payload into the cache.

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
    with distribution_lock(directory):
        # Keep the cache layout stable regardless of the source file name, and
        # publish the binary atomically so concurrent readers never see a
        # half-copied file on a shared (NFS) cache.
        destination = directory / BINARY_NAME
        temporary = directory / (f".{BINARY_NAME}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
        try:
            if source.resolve() != destination.resolve():
                shutil.copy2(source, temporary)
                temporary.chmod(0o755)
                os.replace(temporary, destination)
            else:
                destination.chmod(0o755)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

        distribution = HarnessDistribution(
            harness=config.name,
            version=config.version,
            platform=platform_key,
            directory=directory,
            binary_path=destination,
            sha256=digest,
            size_bytes=size,
        )
        write_manifest(
            directory,
            {**distribution.manifest(), "created_at": time.time()},
        )
    logger.info("Seeded harness distribution at %s", directory)
    return distribution


def _print_distribution(distribution: HarnessDistribution) -> None:
    print("Harness distribution ready")
    print(
        f"  harness: {distribution.harness} {distribution.version} "
        f"({distribution.platform})"
    )
    if len(distribution.payload_files()) > 1:
        print(f"  payload: {len(distribution.payload_files())} file(s)")
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
        default=None,
        help=(
            "Pinned harness release (default: the selected harness's pinned version)"
        ),
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
        default=None,
        help=(
            "Host cache directory. Defaults to $"
            f"{HARNESS_CACHE_ENV} or {default_cache_dir()}"
        ),
    )
    parser.add_argument(
        "--binary",
        type=str,
        default=None,
        help=(
            "Seed the cache from an already-downloaded harness binary instead "
            "of downloading it (offline mode). For deepseek-harness this is the "
            "extracted runtime executable; its <exe>-rg sidecar and "
            "<platform>/downloads.json are collected automatically."
        ),
    )
    parser.add_argument(
        "--wheel",
        type=str,
        default=None,
        help=(
            "Seed deepseek-harness from an already-downloaded "
            "deepseek_harness_runtime_bin wheel instead of downloading it"
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
    platform_key: str | None = None
    try:
        platform_key = detect_platform(config.platform)
        allowed = HARNESS_PLATFORMS_BY_HARNESS.get(config.name, HARNESS_PLATFORMS)
        if platform_key not in allowed:
            raise DistributionError(
                f"{config.name} has no portable distribution for {platform_key}; "
                f"supported: {', '.join(allowed)}"
            )
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
        if args.wheel:
            if config.name != DEEPSEEK_HARNESS:
                parser.error("--wheel is only supported for --harness deepseek-harness")
            distribution = seed_deepseek_distribution(
                config, Path(args.wheel), expected_sha256=args.sha256
            )
        elif args.binary:
            if config.name == DEEPSEEK_HARNESS:
                distribution = seed_deepseek_distribution(
                    config, Path(args.binary), expected_sha256=args.sha256
                )
            else:
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
        if config.name == DEEPSEEK_HARNESS:
            spec = DSH_RUNTIME_WHEELS.get(platform_key) if platform_key else None
            source = spec["url"] if spec else f"DeepSeek Harness {DSH_RUNTIME_VERSION}"
            print(
                "If this host has no network access, download the runtime wheel "
                "elsewhere and seed it:\n  harness-prepare --harness "
                f"deepseek-harness --wheel /path/to/runtime.whl --sha256 "
                f"<checksum>\nWheel: {source}",
                file=sys.stderr,
            )
        else:
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
