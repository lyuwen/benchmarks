"""Central resolution of the workspace egress policy.

This is the authority for validating OH_NETWORK_MODE and persisting the
resolved policy. It runs during Evaluation setup, before metadata.json is
written, and rejects combinations the SDK layer cannot enforce.
"""

import os
from collections.abc import Mapping

from openhands.sdk import get_logger
from openhands.workspace.docker.network_policy import (
    ENV_VAR,
    WorkspaceNetworkPolicy,
    parse_network_mode,
)


logger = get_logger(__name__)


def resolve_network_policy(
    workspace_type: str,
    explicit: WorkspaceNetworkPolicy | None = None,
    env: Mapping[str, str] | None = None,
) -> WorkspaceNetworkPolicy:
    """Resolve the effective egress policy, or raise.

    Raises:
        ValueError: on an invalid mode, a conflict between an explicit policy
            and the environment, or a non-public mode requested for a remote
            workspace (which has no local container to isolate).
    """
    source = os.environ if env is None else env
    env_mode = parse_network_mode(source.get(ENV_VAR))

    if explicit is not None:
        if source.get(ENV_VAR, "").strip() and explicit.mode != env_mode:
            raise ValueError(
                f"Network policy conflict: explicit mode {explicit.mode!r} does "
                f"not match {ENV_VAR}={env_mode!r}. Set them to the same value "
                "or supply only one."
            )
        policy = explicit
    else:
        policy = WorkspaceNetworkPolicy(mode=env_mode)

    if workspace_type == "remote" and policy.requires_sidecar:
        raise ValueError(
            f"workspace_type='remote' cannot enforce network mode "
            f"{policy.mode!r}: there is no local container to isolate. Use a "
            "docker or flex workspace, or set the mode to 'public'."
        )

    logger.info(
        "Resolved workspace network policy: mode=%s endpoints=%d",
        policy.mode,
        len(policy.resolved_endpoints()),
    )
    return policy
