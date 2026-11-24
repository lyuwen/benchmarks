"""Shared mini-swe-agent adapter for benchmark ``Evaluation`` classes.

The benchmarks harness keeps instance loading, workspace provisioning, retries,
resume, accounting and aggregation. This package adds the mini-swe-agent agent
loop on top of the workspace the benchmark already provisioned:

- :mod:`.model` translates an OpenHands ``LLM`` config into mini's LiteLLM model.
- :mod:`.environment` implements mini's ``Environment`` over the workspace.
- :mod:`.rollout` runs ``DefaultAgent`` for one attempt and returns a normalized
  result, writing the redacted native trace.
- :mod:`.history`` maps mini's transcript onto the ``*.history.json`` contract.
- :mod:`.submission` validates the submitted patch against the task baseline.

``mini-swe-agent`` is optional: it is imported lazily so every other benchmark
entry point keeps working without the extra installed.
"""

from benchmarks.utils.miniswe.availability import (
    INSTALL_HINT,
    MINISWE_VERSION,
    require_miniswe,
)
from benchmarks.utils.miniswe.cli import (
    MINISWE_SUPPORTED_CRITICS,
    add_mini_common_args,
    mini_progress_console,
    mini_progress_path,
    resolve_bash_env_file,
    validate_mini_args,
)
from benchmarks.utils.miniswe.environment import (
    DEFAULT_ENV,
    MAX_SINGLE_OUTPUT_CHARS,
    SUBMISSION_MARKER,
    SubmissionTruncatedError,
    WorkspaceCommandError,
    WorkspaceEnvironment,
    WorkspaceEnvironmentConfig,
)
from benchmarks.utils.miniswe.history import (
    build_history_dump,
    build_trajectories,
    history_filename,
    normalize_messages,
    publish_canonical_histories,
    redact_secrets,
    write_history,
)
from benchmarks.utils.miniswe.model import (
    ROLLOUT_ID_HEADERS,
    MiniModelSpec,
    build_mini_model,
    litellm_model_kwargs,
)
from benchmarks.utils.miniswe.progress import (
    LoggingEnvironment,
    LoggingModel,
    MiniProgress,
)
from benchmarks.utils.miniswe.rollout import (
    MiniRolloutResult,
    MiniRolloutSettings,
    run_mini_rollout,
)
from benchmarks.utils.miniswe.submission import (
    check_patch_applies,
    get_head_commit,
    head_matches_commit,
)


__all__ = [
    "DEFAULT_ENV",
    "INSTALL_HINT",
    "MAX_SINGLE_OUTPUT_CHARS",
    "MINISWE_SUPPORTED_CRITICS",
    "MINISWE_VERSION",
    "ROLLOUT_ID_HEADERS",
    "SUBMISSION_MARKER",
    "MiniModelSpec",
    "LoggingEnvironment",
    "LoggingModel",
    "MiniProgress",
    "MiniRolloutResult",
    "MiniRolloutSettings",
    "SubmissionTruncatedError",
    "WorkspaceCommandError",
    "WorkspaceEnvironment",
    "WorkspaceEnvironmentConfig",
    "add_mini_common_args",
    "build_history_dump",
    "build_mini_model",
    "build_trajectories",
    "check_patch_applies",
    "get_head_commit",
    "head_matches_commit",
    "history_filename",
    "litellm_model_kwargs",
    "mini_progress_console",
    "mini_progress_path",
    "normalize_messages",
    "publish_canonical_histories",
    "redact_secrets",
    "require_miniswe",
    "resolve_bash_env_file",
    "run_mini_rollout",
    "validate_mini_args",
    "write_history",
]
