"""Configuration for blackbox agent harnesses.

A "harness" here is an external coding agent CLI (Claude Code or DeepSeek
Harness) that runs *inside* the task container as a black box. Its model traffic is routed through
the ``rollout-manager`` proxy, which captures every request/response pair and
exports the session's trajectory series next to the evaluation outputs.

The configuration is a pydantic model so it can be stored in
``EvalMetadata.details`` and therefore inherited by the worker processes that
run each instance.
"""

from __future__ import annotations

import json
import os
from argparse import ArgumentParser
from typing import Any, Literal

from pydantic import BaseModel, Field


DEFAULT_HARNESS = "claude-code"

# Pinned native package version for the first supported harness. Bump together
# with the tests/docs when validating a new Claude Code release.
DEFAULT_CLAUDE_CODE_VERSION = "2.1.331"

# Pinned published DeepSeek Harness runtime carrier (PyPI wheel tag). The
# repository checkout can be ahead of the last published runtime; evaluation
# consumes the published one for reproducibility.
DEFAULT_DEEPSEEK_HARNESS_VERSION = "0.2.2rc2"

DEFAULT_HARNESS_VERSIONS = {
    "claude-code": DEFAULT_CLAUDE_CODE_VERSION,
    "deepseek-harness": DEFAULT_DEEPSEEK_HARNESS_VERSION,
}

# Claude Code assumes a 1M context window for model IDs it does not recognize.
# Benchmark backends are usually much smaller, so we tell the harness the real
# window (and where to compact) instead of letting it run to ~967K tokens.
DEFAULT_CONTEXT_WINDOW = 200_000

# Compact at this percentage of the auto-compact window. Lower than Claude
# Code's own threshold so long benchmark rollouts compact (and are captured)
# instead of dying on a context overflow.
DEFAULT_COMPACTION_THRESHOLD = 80

DEFAULT_CACHE_DIR = "~/.cache/openhands-benchmarks/harness-dist"
HARNESS_CACHE_ENV = "HARNESS_DIST_CACHE_DIR"
DEFAULT_INSTALL_DIR = "/opt/harness"

# Long tool calls (test suites, builds) are common in SWE tasks; the Claude Code
# default is two minutes, which kills legitimate commands.
DEFAULT_BASH_TIMEOUT_MS = 600_000
DEFAULT_API_TIMEOUT_MS = 1_200_000

# Claude Code closes aux requests (session titles, safety classifiers) through
# the same proxy. They are kept in the export but masked out of training.
DEFAULT_EXCLUDE_REQUEST_CLASSES = "auxiliary"

# Supported container platforms per harness: Claude Code publishes glibc and musl
# native binaries; the DeepSeek Harness runtime wheel is manylinux (glibc) only.
HARNESS_PLATFORMS_BY_HARNESS = {
    "claude-code": ("linux-x64", "linux-arm64", "linux-x64-musl", "linux-arm64-musl"),
    "deepseek-harness": ("linux-x64", "linux-arm64"),
}
HARNESS_PLATFORMS = tuple(
    dict.fromkeys(
        platform
        for platforms in HARNESS_PLATFORMS_BY_HARNESS.values()
        for platform in platforms
    )
)

SUPPORTED_HARNESSES = tuple(DEFAULT_HARNESS_VERSIONS)

# Permission-mode vocabulary accepted by each adapter. Claude Code passes its
# own mode names through; DeepSeek Harness maps the shared vocabulary onto dsh
# permission presets (danger-full-access / workspace-write / read-only).
HARNESS_PERMISSION_MODES = {
    "claude-code": ("auto", "bypassPermissions", "acceptEdits", "default", "plan"),
    "deepseek-harness": (
        "danger-full-access",
        "workspace-write",
        "read-only",
        "bypassPermissions",
        "acceptEdits",
        "default",
        "plan",
    ),
}

# Structural request fields the rollout manager refuses to override: the
# benchmark supplies sampling parameters only, not the conversation/tools.
REQUEST_PARAM_FORBIDDEN_KEYS = frozenset(
    {"messages", "stream", "stream_options", "tools"}
)


def default_cache_dir() -> str:
    """Cache directory used when no CLI flag is given (env-overridable)."""
    return os.getenv(HARNESS_CACHE_ENV) or DEFAULT_CACHE_DIR


class HarnessConfig(BaseModel):
    """Everything the harness runner needs, independent of the benchmark task."""

    name: str = Field(
        default=DEFAULT_HARNESS,
        description="Harness implementation to run (see SUPPORTED_HARNESSES)",
    )
    version: str = Field(
        default=DEFAULT_CLAUDE_CODE_VERSION,
        description=(
            "Pinned harness release (Claude Code npm version or the published "
            "DeepSeek Harness runtime wheel version)"
        ),
    )
    platform: str | None = Field(
        default=None,
        description=(
            "Container platform of the portable distribution; auto-detected "
            "from the host when unset"
        ),
    )
    cache_dir: str = Field(
        default_factory=default_cache_dir,
        description=(
            "Host cache directory for portable harness distributions "
            f"(default: ${HARNESS_CACHE_ENV} or {DEFAULT_CACHE_DIR})"
        ),
    )
    install_dir: str = Field(
        default=DEFAULT_INSTALL_DIR,
        description="Directory inside the task container the harness is installed to",
    )
    dist_mode: Literal["copy", "mount"] = Field(
        default="copy",
        description=(
            "How the portable distribution reaches the container: 'copy' uploads "
            "the binary with docker cp; 'mount' bind-mounts the cached directory "
            "(faster on large local runs)"
        ),
    )

    # ---- model / context configuration ------------------------------------
    context_window: int = Field(
        default=DEFAULT_CONTEXT_WINDOW,
        ge=0,
        description=(
            "Real context window of the served model. Claude Code is told this "
            "window so its compaction logic matches the backend; 0 leaves "
            "Claude Code's own assumption untouched"
        ),
    )
    compaction_threshold: int = Field(
        default=DEFAULT_COMPACTION_THRESHOLD,
        ge=0,
        le=100,
        description=(
            "Percentage of the auto-compact window at which the harness "
            "compacts; 0 leaves Claude Code's default threshold untouched"
        ),
    )
    max_output_tokens: int | None = Field(
        default=None,
        ge=1,
        description="Optional cap on output tokens per model call",
    )
    max_turns: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Optional cap on agentic turns; falls back to the benchmark's "
            "--max-iterations when unset"
        ),
    )
    bash_timeout_ms: int = Field(
        default=DEFAULT_BASH_TIMEOUT_MS,
        ge=1_000,
        description="Default and maximum timeout of harness shell commands",
    )
    api_timeout_ms: int = Field(
        default=DEFAULT_API_TIMEOUT_MS,
        ge=1_000,
        description="Per-request timeout the harness applies to model calls",
    )
    bare: bool = Field(
        default=False,
        description=(
            "Run Claude Code with --bare (skip hooks/skills/MCP/CLAUDE.md "
            "discovery) for faster, more deterministic sessions"
        ),
    )
    skip_permissions: bool = Field(
        default=True,
        description="Run the harness in bypass-permissions mode",
    )
    permission_mode: (
        Literal[
            "auto",
            "bypassPermissions",
            "acceptEdits",
            "default",
            "plan",
            "danger-full-access",
            "workspace-write",
            "read-only",
        ]
        | None
    ) = Field(
        default=None,
        description=(
            "Optional Claude Code permission mode. 'auto' routes every action "
            "through Claude Code's auto-mode classifier (captured as auxiliary "
            "requests); unset keeps the bypass-permissions default"
        ),
    )
    auto_mode_model: str | None = Field(
        default=None,
        description=(
            "Optional explicit model ID for auto-mode classifier requests "
            "(CLAUDE_CODE_AUTO_MODE_MODEL). Unset means Claude Code uses the "
            "session model (the main agent model); the rollout manager forces "
            "the served backend model on every request either way"
        ),
    )
    extra_env: dict[str, str] = Field(
        default_factory=dict,
        description="Extra environment variables injected into the harness process",
    )
    extra_args: list[str] = Field(
        default_factory=list,
        description="Extra command-line arguments appended to the harness command",
    )
    custom_headers: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Extra HTTP headers sent by the harness to the proxy/upstream "
            "(e.g. provider beta flags); forwarded as ANTHROPIC_CUSTOM_HEADERS"
        ),
    )
    request_params: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Fixed Chat-Completions parameters merged (overriding) into every "
            "upstream model request by the rollout manager, e.g. temperature, "
            "top_p, top_k, reasoning_effort, seed, max_tokens. Populated from "
            "the LLM config and --request-param/--request-params overrides"
        ),
    )
    use_llm_request_params: bool = Field(
        default=True,
        description=(
            "Forward sampling parameters from the LLM config to the rollout "
            "manager (disable with --no-llm-request-params)"
        ),
    )

    # ---- rollout manager ---------------------------------------------------
    upstream_base_url: str | None = Field(
        default=None,
        description=(
            "OpenAI-compatible upstream URL for the rollout manager; defaults "
            "to the LLM config base_url with /v1 appended"
        ),
    )
    upstream_api_key: str | None = Field(
        default=None,
        description="Bearer token for the upstream; defaults to the LLM config key",
    )
    upstream_model: str | None = Field(
        default=None,
        description=(
            "Model name requested from the upstream; defaults to the LLM config "
            "model with client-side LiteLLM prefixes stripped"
        ),
    )
    harness_model: str | None = Field(
        default=None,
        description=(
            "Model identifier the harness believes it is talking to; defaults "
            "to the resolved upstream model"
        ),
    )
    rollout_request_timeout_sec: float = Field(
        default=900.0,
        gt=0,
        description="Upstream read timeout of the rollout manager",
    )
    rollout_bind_host: str | None = Field(
        default=None,
        description=(
            "Host interface the rollout manager binds to; defaults to the "
            "Docker bridge gateway so only the task container can reach it"
        ),
    )
    rollout_start_timeout_sec: float = Field(
        default=90.0,
        gt=0,
        description="How long to wait for the rollout manager to become healthy",
    )
    rollout_finish_timeout_sec: float = Field(
        default=120.0,
        gt=0,
        description="How long the rollout manager may take to finalize its export",
    )
    rollout_log_upstream: Literal["off", "summary", "full"] = Field(
        default="off",
        description=(
            "Log the requests the rollout manager sends to the model backend: "
            "'summary' (shape only) or 'full' (redacted headers + body, "
            "truncated to rollout_log_upstream_max_chars)"
        ),
    )
    rollout_log_upstream_max_chars: int = Field(
        default=20000,
        ge=0,
        description=(
            "Maximum request-body characters logged in 'full' mode (0 = unlimited)"
        ),
    )
    write_streamlined_history: bool = Field(
        default=True,
        description=(
            "Write a streamlined <id>.attemptN.history.streamlined.json (merged "
            "fragments, original tool arguments, stop sequences re-attached) "
            "next to the raw history at task finalization"
        ),
    )
    exclude_request_classes: str = Field(
        default=DEFAULT_EXCLUDE_REQUEST_CLASSES,
        description=(
            "Comma-separated request classes kept in the export but excluded "
            "from training (e.g. auxiliary)"
        ),
    )

    # ---- images ------------------------------------------------------------
    skip_image_pull: bool = Field(
        default=False,
        description="Do not pull a missing task image; fail instead",
    )

    @property
    def harness_output_subdir(self) -> str:
        return os.path.join("harnesses", self.name)

    @property
    def expanded_cache_dir(self) -> str:
        return os.path.expanduser(self.cache_dir)

    def for_metadata(self) -> dict[str, Any]:
        """Serializable view without upstream credentials."""
        data = self.model_dump(mode="json")
        data.pop("upstream_api_key", None)
        return data

    def resolve_max_turns(self, max_iterations: int) -> int:
        if self.max_turns is not None:
            return self.max_turns
        return max_iterations


def add_harness_args(parser: ArgumentParser) -> None:
    """Register the shared ``--harness*`` / ``--rollout*`` CLI flags."""
    group = parser.add_argument_group("harness")
    group.add_argument(
        "--harness",
        type=str,
        default=DEFAULT_HARNESS,
        choices=list(SUPPORTED_HARNESSES),
        help="Blackbox agent harness to run inside the task container",
    )
    group.add_argument(
        "--harness-version",
        type=str,
        default=None,
        help=(
            "Pinned harness release (default: the selected harness's pinned "
            "version, e.g. Claude Code 2.1.331 or DeepSeek Harness 0.2.2rc2)"
        ),
    )
    group.add_argument(
        "--harness-platform",
        type=str,
        default=None,
        choices=list(HARNESS_PLATFORMS),
        help="Container platform of the portable distribution (default: auto)",
    )
    group.add_argument(
        "--harness-cache-dir",
        type=str,
        default=None,
        help=(
            "Host cache directory for portable harness distributions "
            f"(default: ${HARNESS_CACHE_ENV} or {DEFAULT_CACHE_DIR})"
        ),
    )
    group.add_argument(
        "--harness-install-dir",
        type=str,
        default=DEFAULT_INSTALL_DIR,
        help="Directory inside the task container the harness is installed to",
    )
    group.add_argument(
        "--harness-dist-mode",
        type=str,
        default="copy",
        choices=["copy", "mount"],
        help=(
            "How the portable harness distribution reaches the container: "
            "'copy' (docker cp) or 'mount' (bind mount, faster locally)"
        ),
    )
    group.add_argument(
        "--harness-context-window",
        type=int,
        default=DEFAULT_CONTEXT_WINDOW,
        help=(
            "Context window of the served model in tokens, reported to the "
            "harness so compaction matches the backend (0: harness default)"
        ),
    )
    group.add_argument(
        "--harness-compaction-threshold",
        type=int,
        default=DEFAULT_COMPACTION_THRESHOLD,
        help=(
            "Percentage (1-100) of the auto-compact window at which the "
            "harness compacts (0: harness default)"
        ),
    )
    group.add_argument(
        "--harness-max-output-tokens",
        type=int,
        default=None,
        help="Optional cap on output tokens per model call",
    )
    group.add_argument(
        "--harness-max-turns",
        type=int,
        default=None,
        help="Optional cap on agentic turns (default: --max-iterations)",
    )
    group.add_argument(
        "--harness-bash-timeout-ms",
        type=int,
        default=DEFAULT_BASH_TIMEOUT_MS,
        help="Default and maximum timeout of harness shell commands",
    )
    group.add_argument(
        "--harness-api-timeout-ms",
        type=int,
        default=DEFAULT_API_TIMEOUT_MS,
        help="Per-request timeout the harness applies to model calls",
    )
    group.add_argument(
        "--harness-bare",
        action="store_true",
        help="Run the harness in bare mode (skip hooks/skills/MCP discovery)",
    )
    group.add_argument(
        "--harness-no-skip-permissions",
        action="store_true",
        help="Do not run the harness in bypass-permissions mode",
    )
    group.add_argument(
        "--harness-permission-mode",
        type=str,
        default=None,
        choices=list(
            dict.fromkeys(
                mode for modes in HARNESS_PERMISSION_MODES.values() for mode in modes
            )
        ),
        help=(
            "Harness permission mode. Claude Code: 'auto' runs the auto-mode "
            "classifier (captured as auxiliary); default: bypass permissions. "
            "DeepSeek Harness: danger-full-access (default), workspace-write or "
            "read-only"
        ),
    )
    group.add_argument(
        "--harness-auto-mode-model",
        type=str,
        default=None,
        help=(
            "Optional model ID for Claude Code's auto-mode classifier requests; "
            "default: the main agent model (the proxy forces the served backend "
            "model on every request anyway)"
        ),
    )
    group.add_argument(
        "--harness-env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra environment variable for the harness process (repeatable)",
    )
    group.add_argument(
        "--harness-arg",
        action="append",
        default=[],
        metavar="ARG",
        help="Extra argument appended to the harness command (repeatable)",
    )
    group.add_argument(
        "--harness-header",
        action="append",
        default=[],
        metavar="NAME: VALUE",
        help=(
            "Extra HTTP header sent by the harness to the proxy/upstream, "
            "repeatable (e.g. 'anthropic-beta: interleaved-thinking-2025-05-14')"
        ),
    )
    group.add_argument(
        "--request-param",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Upstream request parameter to force, parsed as JSON when possible "
            "(e.g. temperature=0.7, reasoning_effort=high); repeatable and "
            "overrides values taken from the LLM config"
        ),
    )
    group.add_argument(
        "--request-params",
        type=str,
        default=None,
        metavar="JSON",
        help=(
            "JSON object of upstream request parameters, e.g. "
            '{"temperature": 0.7, "top_p": 0.95} (overrides LLM config)'
        ),
    )
    group.add_argument(
        "--no-llm-request-params",
        action="store_true",
        help=(
            "Do not forward temperature/top_p/top_k/reasoning_effort/seed/"
            "max_tokens/provider extra_body from the LLM config"
        ),
    )
    group.add_argument(
        "--upstream-base-url",
        type=str,
        default=None,
        help=(
            "OpenAI-compatible upstream URL for the rollout manager "
            "(default: LLM config base_url + /v1)"
        ),
    )
    group.add_argument(
        "--upstream-api-key",
        type=str,
        default=None,
        help="Bearer token for the upstream (default: LLM config api_key)",
    )
    group.add_argument(
        "--upstream-model",
        type=str,
        default=None,
        help="Model name requested from the upstream",
    )
    group.add_argument(
        "--harness-model",
        type=str,
        default=None,
        help="Model identifier the harness is told it is talking to",
    )
    group.add_argument(
        "--rollout-request-timeout",
        type=float,
        default=900.0,
        help="Upstream read timeout of the rollout manager in seconds",
    )
    group.add_argument(
        "--rollout-bind-host",
        type=str,
        default=None,
        help=(
            "Host interface the rollout manager binds to "
            "(default: Docker bridge gateway)"
        ),
    )
    group.add_argument(
        "--rollout-start-timeout",
        type=float,
        default=90.0,
        help="Seconds to wait for the rollout manager to become healthy",
    )
    group.add_argument(
        "--rollout-finish-timeout",
        type=float,
        default=120.0,
        help="Seconds the rollout manager may take to finalize its export",
    )
    group.add_argument(
        "--harness-include-auxiliary",
        action="store_true",
        help=(
            "Train auxiliary requests (auto-mode classifier/security-monitor "
            "calls, session titles) instead of masking them out: equivalent to "
            "--exclude-request-classes ''"
        ),
    )
    group.add_argument(
        "--no-streamline-history",
        dest="write_streamlined_history",
        action="store_false",
        help=(
            "Do not write the streamlined history companion file at task "
            "finalization (the raw history is still written)"
        ),
    )
    group.add_argument(
        "--rollout-log-upstream",
        choices=["off", "summary", "full"],
        default="off",
        help=(
            "Log the requests sent to the backend by the rollout manager: "
            "'summary' logs shape/headers names, 'full' logs redacted headers "
            "and the request body (in the per-instance manager log)"
        ),
    )
    group.add_argument(
        "--rollout-log-upstream-max-chars",
        type=int,
        default=20000,
        help=(
            "Maximum body characters logged by --rollout-log-upstream full "
            "(0 = unlimited, default 20000)"
        ),
    )
    group.add_argument(
        "--exclude-request-classes",
        type=str,
        default=DEFAULT_EXCLUDE_REQUEST_CLASSES,
        help=(
            "Comma-separated harness request classes masked out of training "
            "but kept in the rollout export"
        ),
    )
    group.add_argument(
        "--skip-image-pull",
        action="store_true",
        help="Do not pull a missing task image; fail instead",
    )


def validate_request_params(params: dict[str, Any], *, origin: str) -> dict[str, Any]:
    """Reject request-param overlays that would corrupt the capture."""
    forbidden = set(params) & REQUEST_PARAM_FORBIDDEN_KEYS
    if forbidden:
        raise ValueError(
            f"{origin} may not override structural fields: "
            + ", ".join(sorted(forbidden))
        )
    return params


def parse_request_params_json(raw: str | None) -> dict[str, Any]:
    """Parse ``--request-params`` as a JSON object."""
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"--request-params is not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("--request-params must be a JSON object")
    return validate_request_params(parsed, origin="--request-params")


def parse_request_param_values(values: list[str]) -> dict[str, Any]:
    """Parse repeated ``--request-param KEY=VALUE`` flags (JSON value or str)."""
    params: dict[str, Any] = {}
    for value in values:
        key, separator, raw = value.partition("=")
        if not separator or not key.strip():
            raise ValueError(f"--request-param expects KEY=VALUE, got {value!r}")
        try:
            parsed: Any = json.loads(raw)
        except json.JSONDecodeError:
            parsed = raw
        params[key.strip()] = parsed
    return validate_request_params(params, origin="--request-param")


def merge_request_params(
    base: dict[str, Any], overrides: dict[str, Any]
) -> dict[str, Any]:
    """Merge request-parameter dicts (later values win), validating the result."""
    if not overrides:
        return dict(base)
    return validate_request_params({**base, **overrides}, origin="request parameters")


def parse_extra_env(values: list[str]) -> dict[str, str]:
    """Parse repeated ``KEY=VALUE`` flags into a dict."""
    env: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--harness-env expects KEY=VALUE, got {value!r}")
        key, _, raw = value.partition("=")
        if not key:
            raise ValueError(f"--harness-env expects KEY=VALUE, got {value!r}")
        env[key] = raw
    return env


def parse_headers(values: list[str]) -> dict[str, str]:
    """Parse repeated ``Name: Value`` header flags into a dict."""
    headers: dict[str, str] = {}
    for value in values:
        name, separator, raw = value.partition(":")
        if not separator or not name.strip():
            raise ValueError(f"--harness-header expects 'Name: Value', got {value!r}")
        headers[name.strip()] = raw.strip()
    return headers


def harness_config_from_args(args: Any) -> HarnessConfig:
    """Build a :class:`HarnessConfig` from parsed CLI arguments."""
    return HarnessConfig(
        name=args.harness,
        version=args.harness_version or DEFAULT_HARNESS_VERSIONS[args.harness],
        platform=args.harness_platform,
        cache_dir=args.harness_cache_dir or default_cache_dir(),
        install_dir=args.harness_install_dir,
        dist_mode=args.harness_dist_mode,
        context_window=args.harness_context_window,
        compaction_threshold=args.harness_compaction_threshold,
        max_output_tokens=args.harness_max_output_tokens,
        max_turns=args.harness_max_turns,
        bash_timeout_ms=args.harness_bash_timeout_ms,
        api_timeout_ms=args.harness_api_timeout_ms,
        bare=args.harness_bare,
        skip_permissions=not args.harness_no_skip_permissions,
        permission_mode=args.harness_permission_mode,
        auto_mode_model=args.harness_auto_mode_model,
        extra_env=parse_extra_env(args.harness_env),
        extra_args=list(args.harness_arg),
        custom_headers=parse_headers(args.harness_header),
        request_params=merge_request_params(
            parse_request_params_json(args.request_params),
            parse_request_param_values(args.request_param),
        ),
        use_llm_request_params=not args.no_llm_request_params,
        upstream_base_url=args.upstream_base_url,
        upstream_api_key=args.upstream_api_key,
        upstream_model=args.upstream_model,
        harness_model=args.harness_model,
        rollout_request_timeout_sec=args.rollout_request_timeout,
        rollout_bind_host=args.rollout_bind_host,
        rollout_start_timeout_sec=args.rollout_start_timeout,
        rollout_finish_timeout_sec=args.rollout_finish_timeout,
        exclude_request_classes=(
            "" if args.harness_include_auxiliary else args.exclude_request_classes
        ),
        write_streamlined_history=args.write_streamlined_history,
        rollout_log_upstream=args.rollout_log_upstream,
        rollout_log_upstream_max_chars=args.rollout_log_upstream_max_chars,
        skip_image_pull=args.skip_image_pull,
    )


def validate_harness_args(parser: ArgumentParser, args: Any) -> None:
    """Reject configurations the harness pipeline cannot honor."""
    if args.workspace != "docker":
        parser.error(
            "--workspace must be 'docker' for harness runs: the harness executes "
            "inside the task container and its rollout manager proxy runs next "
            "to the docker daemon"
        )
    if args.max_attempts < 1:
        parser.error(f"--max-attempts must be >= 1, got {args.max_attempts}")
    if not 0 <= args.harness_compaction_threshold <= 100:
        parser.error("--harness-compaction-threshold must be between 0 and 100")
    if args.harness_context_window < 0:
        parser.error("--harness-context-window must be >= 0")
    if args.rollout_log_upstream_max_chars < 0:
        parser.error("--rollout-log-upstream-max-chars must be >= 0")
    allowed_platforms = HARNESS_PLATFORMS_BY_HARNESS.get(
        args.harness, HARNESS_PLATFORMS
    )
    if args.harness_platform and args.harness_platform not in allowed_platforms:
        parser.error(
            f"--harness-platform {args.harness_platform!r} is not available for "
            f"{args.harness}; supported: {', '.join(allowed_platforms)}"
        )
    allowed_modes = HARNESS_PERMISSION_MODES.get(args.harness, ())
    if (
        args.harness_permission_mode
        and args.harness_permission_mode not in allowed_modes
    ):
        parser.error(
            f"--harness-permission-mode {args.harness_permission_mode!r} is not "
            f"supported by {args.harness}; supported: {', '.join(allowed_modes)}"
        )
    if args.harness == "deepseek-harness":
        if args.harness_bare:
            parser.error("--harness-bare is Claude Code-specific")
        if args.harness_auto_mode_model:
            parser.error("--harness-auto-mode-model is Claude Code-specific")
        if args.harness_max_turns is not None:
            parser.error(
                "--harness-max-turns is not supported by deepseek-harness; "
                "bound the run with --conversation-timeout instead"
            )
    parse_extra_env(args.harness_env)
    parse_headers(args.harness_header)
    try:
        merge_request_params(
            parse_request_params_json(args.request_params),
            parse_request_param_values(args.request_param),
        )
    except ValueError as exc:
        parser.error(str(exc))
    if args.harness_include_auxiliary and args.exclude_request_classes not in (
        "",
        DEFAULT_EXCLUDE_REQUEST_CLASSES,
    ):
        parser.error(
            "--harness-include-auxiliary conflicts with an explicit "
            "--exclude-request-classes; pass --exclude-request-classes '' instead"
        )
