"""Blackbox agent harness inference with backend request capture.

The harness (Claude Code first) runs *inside* the benchmark task container as a
black box. Its model traffic is routed through the `rollout-manager
<https://github.com/lyuwen/rollout-manager>`_ proxy, launched once per instance
next to the Docker daemon, which captures every request/response pair and
exports the session's trajectory series into the evaluation output directory.

Public entry points live in ``benchmarks/<task>/run_infer_harnesses.py``; this
package holds the task-independent plumbing:

- :mod:`benchmarks.utils.harnesses.config` -- CLI/config surface
- :mod:`benchmarks.utils.harnesses.distribution` -- portable harness bundles
- :mod:`benchmarks.utils.harnesses.workspace` -- task container via the docker CLI
- :mod:`benchmarks.utils.harnesses.rollout` -- capture proxy lifecycle + exports
- :mod:`benchmarks.utils.harnesses.claude_code` -- the Claude Code harness
- :mod:`benchmarks.utils.harnesses.runner` -- per-instance pipeline
"""

from benchmarks.utils.harnesses.config import (
    DEFAULT_CLAUDE_CODE_VERSION,
    DEFAULT_COMPACTION_THRESHOLD,
    DEFAULT_CONTEXT_WINDOW,
    SUPPORTED_HARNESSES,
    HarnessConfig,
    add_harness_args,
    harness_config_from_args,
    validate_harness_args,
)
from benchmarks.utils.harnesses.distribution import (
    DistributionError,
    HarnessDistribution,
    ensure_distribution,
)
from benchmarks.utils.harnesses.rollout import (
    RolloutManager,
    RolloutManagerConfig,
    RolloutManagerError,
    build_metrics,
    load_rollout_export,
    rollout_manager_available,
    upstream_credentials,
)
from benchmarks.utils.harnesses.runner import (
    HarnessRunRequest,
    HarnessRunResult,
    publish_canonical_histories,
    run_harness_instance,
)
from benchmarks.utils.harnesses.workspace import (
    CommandResult,
    DockerContainerWorkspace,
    WorkspaceError,
)


__all__ = [
    "DEFAULT_CLAUDE_CODE_VERSION",
    "DEFAULT_COMPACTION_THRESHOLD",
    "DEFAULT_CONTEXT_WINDOW",
    "SUPPORTED_HARNESSES",
    "CommandResult",
    "DistributionError",
    "DockerContainerWorkspace",
    "HarnessConfig",
    "HarnessDistribution",
    "HarnessRunRequest",
    "HarnessRunResult",
    "RolloutManager",
    "RolloutManagerConfig",
    "RolloutManagerError",
    "WorkspaceError",
    "add_harness_args",
    "build_metrics",
    "ensure_distribution",
    "harness_config_from_args",
    "load_rollout_export",
    "publish_canonical_histories",
    "rollout_manager_available",
    "run_harness_instance",
    "upstream_credentials",
    "validate_harness_args",
]
