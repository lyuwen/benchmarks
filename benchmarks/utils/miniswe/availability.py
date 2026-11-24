"""Availability guard for the optional mini-swe-agent dependency.

``mini-swe-agent`` is only needed by the ``run_infer_miniswe`` entry points, so
every other benchmark keeps working without it. Importing the package eagerly
would break those entry points *and* would write to the user's config directory
at import time, so all mini imports go through :func:`require_miniswe`.
"""

import os
from typing import Any

from openhands.sdk import get_logger


logger = get_logger(__name__)

MINISWE_VERSION = "2.4.6"
"""mini-swe-agent release this adapter is written against (pinned in pyproject)."""

INSTALL_HINT = (
    "mini-swe-agent is an optional benchmark-worker dependency. Install it with "
    f"`uv sync --extra miniswe` (pins mini-swe-agent=={MINISWE_VERSION}) or "
    f"`uv pip install mini-swe-agent=={MINISWE_VERSION}`."
)


def require_miniswe() -> Any:
    """Import :mod:`minisweagent`, raising one actionable error when missing.

    Returns:
        The imported ``minisweagent`` module.
    """
    # Avoid importing mini's startup banner into benchmark stdout; child-process
    # output is redirected to per-instance log files, where the banner is noise.
    os.environ.setdefault("MSWEA_SILENT_STARTUP", "1")
    try:
        import minisweagent
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise RuntimeError(INSTALL_HINT) from exc

    version = getattr(minisweagent, "__version__", "unknown")
    if version != MINISWE_VERSION:
        logger.warning(
            "mini-swe-agent %s is installed but this adapter is verified against "
            "%s; submission/transcript formats may differ. Install with `%s`.",
            version,
            MINISWE_VERSION,
            INSTALL_HINT,
        )
    return minisweagent
