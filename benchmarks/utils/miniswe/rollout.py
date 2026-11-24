"""Run one mini-swe-agent rollout against a benchmark-provisioned workspace.

The benchmark ``Evaluation`` keeps ownership of instance selection, workspace
provisioning, retries, resume, accounting and cleanup. This module only drives
mini's ``DefaultAgent`` for a single rollout:

- the task instruction is rendered by the task's own prompt (passed in as
  ``instruction`` and used as mini's ``{{task}}`` template variable);
- the rollout deadline from ``--conversation-timeout`` bounds both model requests
  and workspace commands;
- the raw mini transcript is written as a diagnostic trace, with credentials
  redacted, before the workspace is released.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

from pydantic import BaseModel, Field

from benchmarks.utils.miniswe.availability import require_miniswe
from benchmarks.utils.miniswe.environment import (
    DEFAULT_ENV,
    WorkspaceEnvironment,
)
from benchmarks.utils.miniswe.history import redact_secrets
from benchmarks.utils.miniswe.model import MiniModelSpec, build_mini_model
from benchmarks.utils.miniswe.progress import (
    LoggingEnvironment,
    LoggingModel,
    MiniProgress,
)
from benchmarks.utils.miniswe.templates import SYSTEM_TEMPLATE
from openhands.sdk import LLM, get_logger
from openhands.sdk.llm import Metrics
from openhands.sdk.llm.utils.metrics import TokenUsage
from openhands.sdk.workspace import RemoteWorkspace


logger = get_logger(__name__)

TERMINAL_EXCEPTION = "Exception"
TERMINAL_UNFINISHED = "Unfinished"


class MiniRolloutSettings(BaseModel):
    """Per-rollout execution settings (task-specific parts come from the task)."""

    cwd: str = Field(description="Task working directory every action starts in")
    step_limit: int = Field(
        default=0, description="Maximum model steps (0 = unlimited)"
    )
    cost_limit: float = Field(
        default=0.0, description="Stop after this cost in USD (0 = unlimited)"
    )
    timeout_seconds: int | None = Field(
        default=None, description="Wall-clock rollout deadline in seconds"
    )
    command_timeout: int = Field(
        default=60, description="Per-command timeout in seconds"
    )
    interpreter: list[str] = Field(
        default_factory=lambda: ["bash", "-lc"],
        description="Shell interpreter prefix used inside the task container",
    )
    env: dict[str, str] = Field(
        default_factory=lambda: dict(DEFAULT_ENV),
        description="Environment variables prepended to every action",
    )
    bash_env_file: str | None = Field(
        default=None,
        description=(
            "Exported as BASH_ENV so a non-login bash sources the image's rc file "
            "(used by SWE-bench images to activate the testbed conda env)"
        ),
    )
    trace_path: str | None = Field(
        default=None, description="Path of the redacted native mini trace"
    )
    progress_path: str | None = Field(
        default=None,
        description=(
            "Dedicated human-readable progress log (tail -f). Progress lines are "
            "also mirrored to the per-instance framework log"
        ),
    )
    progress_console: bool = Field(
        default=False,
        description="Also print progress lines to the real console (sys.__stdout__)",
    )
    instance_id: str | None = Field(
        default=None, description="Instance label for progress lines"
    )
    attempt: int | None = Field(
        default=None, description="Attempt label for progress lines"
    )


class MiniRolloutResult(BaseModel):
    """Terminal state and normalized transcript of one mini rollout."""

    status: str = Field(
        description=(
            "mini exit status: Submitted, LimitsExceeded, TimeExceeded, "
            "RepeatedFormatError, Unfinished or Exception"
        )
    )
    submission: str = Field(default="", description="Submitted patch payload")
    messages: list[dict[str, Any]] = Field(default_factory=list)
    tools: list[dict[str, Any]] = Field(default_factory=list)
    model: MiniModelSpec
    metrics: Metrics | None = None
    n_calls: int = 0
    cost: float = 0.0
    duration_seconds: float = 0.0
    trace_path: str | None = None


def run_mini_rollout(
    *,
    llm: LLM,
    workspace: RemoteWorkspace,
    instruction: str,
    settings: MiniRolloutSettings,
    rollout_key: str | None = None,
    model: Any = None,
    model_spec: MiniModelSpec | None = None,
) -> MiniRolloutResult:
    """Run mini's ``DefaultAgent`` once and return its normalized result.

    Args:
        llm: Model config for this rollout (read at rollout time so the
            attempt-level temperature adjustment applies).
        workspace: Provisioned benchmark workspace; borrowed, never released here.
        instruction: Fully rendered task instruction (mini's ``task`` variable).
        settings: Execution limits and shell configuration.
        rollout_key: Fresh router-affinity key, reused for this rollout's requests.
        model: Optional pre-built mini model (used by tests); when provided the
            model factory is bypassed.
        model_spec: Redacted description matching ``model`` when injected.

    Raises:
        Exception: Uncaught model/agent failures are re-raised after the trace is
            saved so ``Evaluation`` can apply its retry policy.
    """
    minisweagent = require_miniswe()
    from minisweagent.agents.default import DefaultAgent
    from minisweagent.models.utils.actions_toolcall import BASH_TOOL

    # Evaluation already retries transport failures; keep mini's internal
    # tenacity retries out of the way unless the operator configured otherwise.
    os.environ.setdefault("MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT", "1")

    tools = [BASH_TOOL]
    started = time.monotonic()
    deadline = (
        started + float(settings.timeout_seconds) if settings.timeout_seconds else None
    )
    env = WorkspaceEnvironment(
        workspace,
        cwd=settings.cwd,
        timeout=settings.command_timeout,
        interpreter=settings.interpreter,
        env=settings.env,
        bash_env_file=settings.bash_env_file,
        deadline=deadline,
    )

    if model is None:
        model, spec = build_mini_model(
            llm=llm,
            rollout_key=rollout_key,
            timeout=settings.timeout_seconds,
        )
    else:
        spec = model_spec or MiniModelSpec(
            model_name=str(
                getattr(getattr(model, "config", None), "model_name", "injected")
            )
        )

    progress = MiniProgress(
        path=settings.progress_path,
        console=settings.progress_console,
        label=_progress_label(settings),
    )
    env = LoggingEnvironment(env, progress)
    model = LoggingModel(model, progress)

    agent = DefaultAgent(
        model=model,
        env=env,
        system_template=SYSTEM_TEMPLATE,
        instance_template="{{task}}",
        step_limit=settings.step_limit,
        cost_limit=settings.cost_limit,
        wall_time_limit_seconds=settings.timeout_seconds or 0,
        output_path=None,
    )

    progress.rollout_start(
        model=spec.model_name,
        cwd=settings.cwd,
        step_limit=settings.step_limit,
        command_timeout=settings.command_timeout,
        timeout_seconds=settings.timeout_seconds,
    )
    try:
        exit_info = agent.run(task=instruction)
    except Exception as exc:
        duration = time.monotonic() - started
        progress.rollout_error(exc, duration_seconds=duration)
        _write_trace(
            path=settings.trace_path,
            agent=agent,
            env=env,
            minisweagent=minisweagent,
            spec=spec,
            status=TERMINAL_EXCEPTION,
            submission="",
            error=f"{type(exc).__name__}: {exc}",
            duration_seconds=duration,
        )
        raise
    else:
        duration = time.monotonic() - started
        status = _exit_status(exit_info)
        submission = str(exit_info.get("submission") or "")
        messages = list(agent.messages)
        result = MiniRolloutResult(
            status=status,
            submission=submission,
            messages=messages,
            tools=tools,
            model=spec,
            metrics=build_metrics(
                model_name=spec.model_name,
                messages=messages,
                cost=agent.cost,
            ),
            n_calls=agent.n_calls,
            cost=agent.cost,
            duration_seconds=duration,
            trace_path=settings.trace_path,
        )
        _write_trace(
            path=settings.trace_path,
            agent=agent,
            env=env,
            minisweagent=minisweagent,
            spec=spec,
            status=status,
            submission=submission,
            error=None,
            duration_seconds=duration,
        )
        progress.rollout_end(
            status=status,
            n_calls=agent.n_calls,
            cost=agent.cost,
            duration_seconds=duration,
            submission_chars=len(submission),
        )
        return result
    finally:
        progress.close()


def _progress_label(settings: MiniRolloutSettings) -> str:
    parts: list[str] = []
    if settings.instance_id:
        parts.append(f"instance={settings.instance_id}")
    if settings.attempt is not None:
        parts.append(f"attempt={settings.attempt}")
    return " ".join(parts)


def build_metrics(
    *,
    model_name: str,
    messages: list[dict[str, Any]],
    cost: float,
) -> Metrics | None:
    """Summarize token usage/cost from the responses captured on mini messages."""
    usages: list[TokenUsage] = []
    for message in messages:
        extra = message.get("extra")
        if not isinstance(extra, dict):
            continue
        response = extra.get("response")
        if not isinstance(response, dict):
            continue
        usage = response.get("usage")
        if not isinstance(usage, dict):
            continue
        prompt_details = usage.get("prompt_tokens_details")
        completion_details = usage.get("completion_tokens_details")
        prompt_tokens = _as_int(usage.get("prompt_tokens"))
        completion_tokens = _as_int(usage.get("completion_tokens"))
        usages.append(
            TokenUsage(
                model=model_name,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cache_read_tokens=_as_int(
                    prompt_details.get("cached_tokens")
                    if isinstance(prompt_details, dict)
                    else 0
                ),
                reasoning_tokens=_as_int(
                    completion_details.get("reasoning_tokens")
                    if isinstance(completion_details, dict)
                    else 0
                ),
                per_turn_token=prompt_tokens + completion_tokens,
            )
        )
    if not usages:
        return None

    accumulated = usages[0]
    for usage in usages[1:]:
        accumulated = accumulated + usage
    return Metrics(
        model_name=model_name,
        accumulated_cost=max(0.0, float(cost)),
        token_usages=usages,
        accumulated_token_usage=accumulated,
    )


def _exit_status(exit_info: dict[str, Any]) -> str:
    status = exit_info.get("exit_status")
    if isinstance(status, str) and status:
        return status
    return TERMINAL_UNFINISHED


def _as_int(value: Any) -> int:
    if isinstance(value, (int, float)):
        return max(0, int(value))
    return 0


def _write_trace(
    *,
    path: str | None,
    agent: Any,
    env: Any,
    minisweagent: Any,
    spec: MiniModelSpec,
    status: str,
    submission: str,
    error: str | None,
    duration_seconds: float,
) -> None:
    """Persist mini's native transcript with credentials redacted."""
    if not path:
        return
    payload = {
        "info": {
            "model_stats": {
                "instance_cost": agent.cost,
                "api_calls": agent.n_calls,
            },
            "config": {
                "agent": agent.config.model_dump(mode="json"),
                "agent_type": (
                    f"{agent.__class__.__module__}.{agent.__class__.__name__}"
                ),
            },
            "mini_version": getattr(minisweagent, "__version__", "unknown"),
            "exit_status": status,
            "submission_chars": len(submission),
            "duration_seconds": duration_seconds,
            "error": error,
        },
        "messages": agent.messages,
        "trajectory_format": "mini-swe-agent-1.1",
        "model": {
            "model_name": spec.model_name,
            "model_kwargs": redact_secrets(spec.model_kwargs),
        },
        "environment": env.serialize(),
    }
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    tmp_path = f"{path}.tmp.{os.getpid()}"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(tmp_path, path)
