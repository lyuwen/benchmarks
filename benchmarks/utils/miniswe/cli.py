"""Shared CLI helpers for the mini-swe-agent benchmark entry points.

Every ``run_infer_miniswe.py`` needs the same mini-specific flags, the same
critic/condenser restrictions and the same progress-file conventions. Keeping
them here means the four entry points (swebench, scaleswe, z021data,
swerebenchv2) cannot drift apart.
"""

from __future__ import annotations

import os
from argparse import ArgumentParser, Namespace
from typing import Literal


MINISWE_SUPPORTED_CRITICS: tuple[str, ...] = ("pass", "empty_patch_critic")
"""Critics that only need the patch (mini produces no OpenHands events)."""

MiniProgressMode = Literal["console", "file", "none"]
_PROGRESS_CHOICES = ("console", "file", "none")


def add_mini_common_args(parser: ArgumentParser) -> None:
    """Add the mini-swe-agent flags shared by all mini entry points."""
    parser.add_argument(
        "--mini-progress",
        choices=list(_PROGRESS_CHOICES),
        default="console",
        help=(
            "Live rollout progress: 'console' (default) writes a tail-able "
            "per-instance progress file and mirrors one-line events to the "
            "console, 'file' writes the file only, 'none' disables it"
        ),
    )
    parser.add_argument(
        "--mini-cost-limit",
        type=float,
        default=0.0,
        help="Stop a rollout after this cost in USD (default: 0, no limit)",
    )
    parser.add_argument(
        "--mini-command-timeout",
        type=int,
        default=60,
        help="Per-command timeout in seconds (default: 60)",
    )
    parser.add_argument(
        "--mini-bash-env-file",
        type=str,
        default=None,
        help=(
            "Optional rc file exported as BASH_ENV for non-login bash. Off by "
            "default: agent-server images already put the task environment on "
            "PATH and /root/.bashrc is unreadable to the non-root agent user. "
            "Example: --mini-bash-env-file /home/openhands/.bashrc"
        ),
    )


def validate_mini_args(parser: ArgumentParser, args: Namespace) -> None:
    """Reject OpenHands-only options that mini rollouts cannot support."""
    if args.critic not in MINISWE_SUPPORTED_CRITICS:
        parser.error(
            f"--critic {args.critic!r} relies on OpenHands SDK events, which mini "
            f"rollouts do not produce. Supported critics: "
            f"{', '.join(MINISWE_SUPPORTED_CRITICS)}."
        )
    if args.condenser != "none":
        parser.error(
            "--condenser is an OpenHands-only feature and is not supported by "
            "mini-swe-agent runs."
        )


def resolve_bash_env_file(raw: str | None) -> str | None:
    """Map the CLI spelling 'none'/'' to no ``BASH_ENV`` file."""
    if isinstance(raw, str) and raw.lower() in ("", "none"):
        return None
    return raw


def mini_progress_path(mode: str, eval_output_dir: str, instance_id: str) -> str | None:
    """Path of the dedicated progress log for one instance, if enabled."""
    if mode == "none":
        return None
    return os.path.join(eval_output_dir, "logs", f"instance_{instance_id}.mini.log")


def mini_progress_console(mode: str) -> bool:
    """Whether progress lines are also mirrored to the real console."""
    return mode == "console"
