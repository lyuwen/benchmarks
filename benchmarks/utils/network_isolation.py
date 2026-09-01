# benchmarks/utils/network_isolation.py
"""Container network isolation for inference runs.

When OH_DISABLE_NETWORK is set, new outbound connections from the container
are blocked via host-side nsenter + iptables. Established connections (e.g.,
the agent server responding to the host's workspace API) and loopback traffic
are preserved, so the agent server keeps working while the agent itself cannot
reach the internet.

Usage in a run_infer.py prepare_workspace hook (after all setup is done):

    from benchmarks.utils.network_isolation import maybe_disable_network
    ...
    maybe_disable_network(workspace)
    return workspace
"""

import os
import subprocess

from openhands.sdk.logger import get_logger


logger = get_logger(__name__)


def disable_container_network(container_id: str) -> bool:
    """Block new outbound connections from a Docker container.

    Runs iptables inside the container via ``docker exec --privileged``, which
    grants NET_ADMIN to the exec'd process without making the whole container
    privileged.  Three OUTPUT rules are inserted in order:

      1. ACCEPT loopback (lo) — local services like httpbin stay reachable
         within the container.
      2. ACCEPT ESTABLISHED/RELATED — the agent server can still respond to
         host API calls that were already open before the cutoff.
      3. DROP everything else — new outbound connections are silently dropped.

    Requires Docker socket access on the host (already needed to start
    containers).

    Args:
        container_id: Docker container ID or name.

    Returns:
        True if all rules were applied; False on any error.
    """
    rules: list[list[str]] = [
        # loopback — local services (e.g. httpbin) must remain reachable
        ["-A", "OUTPUT", "-o", "lo", "-j", "ACCEPT"],
        # established — agent server response packets back to the host
        ["-A", "OUTPUT", "-m", "conntrack", "--ctstate", "ESTABLISHED,RELATED", "-j", "ACCEPT"],
        # drop all new outbound connections
        ["-A", "OUTPUT", "-j", "DROP"],
    ]

    for rule in rules:
        result = subprocess.run(
            ["docker", "exec", "--privileged", container_id, "iptables"] + rule,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            logger.warning(
                "network isolation: iptables rule %r failed for container %s: %s",
                rule,
                container_id,
                result.stderr.strip(),
            )
            return False

    logger.info(
        "network isolation: outbound connections disabled for container %s",
        container_id,
    )
    return True


def maybe_disable_network(workspace) -> None:
    """Disable outbound network in the workspace container if OH_DISABLE_NETWORK is set.

    A no-op when OH_DISABLE_NETWORK is unset or empty.

    For docker/flex workspaces (DockerWorkspace and its subclasses), applies
    iptables rules via nsenter from the host.  For remote API workspaces there
    is no container to nsenter into, so isolation is skipped with a log notice.

    Non-fatal: logs a warning and continues on failure rather than aborting
    the run, since degrading to an unblocked run is preferable to losing the
    instance entirely.
    """
    if not os.getenv("OH_DISABLE_NETWORK"):
        return

    # Import lazily to avoid a hard dependency at module load time; workspace
    # types are only available when openhands-workspace is installed.
    from openhands.workspace import DockerWorkspace  # noqa: PLC0415

    if not isinstance(workspace, DockerWorkspace):
        logger.info(
            "OH_DISABLE_NETWORK is set but workspace is not docker/flex-based "
            "— skipping network isolation"
        )
        return

    container_id: str | None = workspace._container_id  # type: ignore[attr-defined]
    if not container_id:
        logger.warning(
            "network isolation: OH_DISABLE_NETWORK is set but workspace has no "
            "container ID — skipping"
        )
        return

    if not disable_container_network(container_id):
        logger.warning(
            "network isolation: failed to apply rules for container %s "
            "— continuing without isolation",
            container_id,
        )
