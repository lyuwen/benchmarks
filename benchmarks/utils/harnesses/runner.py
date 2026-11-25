"""Per-instance pipeline shared by every blackbox harness run.

The task-specific ``run_infer_harnesses.py`` entry points only decide the image,
the repository path and the prompt. This module does the rest for one instance:

1. install the portable harness distribution into the task container;
2. reset the repository to the dataset base commit and record pre-existing
   untracked files (image build artifacts must never leak into the patch);
3. start one ``rollout-manager`` proxy session on the host;
4. run the harness headless inside the container, with its model endpoint
   pointing at the proxy;
5. finalize the capture, export the trajectory series, and collect the patch;
6. optionally judge the patch and write the per-attempt history file.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shlex
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from benchmarks.utils.execution_judge import ExecutionBasedJudge
from benchmarks.utils.harnesses.base import create_harness
from benchmarks.utils.harnesses.config import HarnessConfig
from benchmarks.utils.harnesses.distribution import ensure_distribution
from benchmarks.utils.harnesses.rollout import (
    RolloutManager,
    RolloutManagerConfig,
    RolloutManagerError,
    attach_stop_sequences_to_history,
    build_metrics,
    collect_tool_sets,
    export_summary,
    load_rollout_export,
    main_trajectory,
    resolve_trajectory_tools,
    upstream_credentials,
)
from benchmarks.utils.harnesses.streamline import (
    STREAMLINED_SUFFIX,
    streamline_history,
)
from benchmarks.utils.harnesses.workspace import DockerContainerWorkspace
from benchmarks.utils.models import EvalInstance
from benchmarks.utils.patch_utils import remove_files_from_patch
from openhands.sdk.llm import LLM, Metrics


logger = logging.getLogger(__name__)

GIT_AUTHOR_NAME = "OpenHands Evaluation"
GIT_AUTHOR_EMAIL = "evaluation@openhands.dev"


@dataclass
class HarnessRunRequest:
    """Task-specific inputs for one harness run."""

    instance: EvalInstance
    attempt: int
    repo_path: str
    base_commit: str
    instruction: str
    output_dir: str
    max_iterations: int
    conversation_timeout: float | None = None
    extra_env: dict[str, str] = field(default_factory=dict)


@dataclass
class HarnessRunResult:
    """Everything the evaluation layer needs from one harness attempt."""

    git_patch: str
    instruction: str
    history: dict[str, Any]
    history_path: str
    evaluation: bool | None
    metrics: Metrics | None
    rollout_export_path: str | None = None
    streamlined_history_path: str | None = None
    warnings: list[str] = field(default_factory=list)


def run_harness_instance(
    request: HarnessRunRequest,
    *,
    workspace: DockerContainerWorkspace,
    config: HarnessConfig,
    llm: LLM,
    judge: ExecutionBasedJudge | None = None,
) -> HarnessRunResult:
    """Run one harness attempt and return the patch plus history payload."""
    instance = request.instance
    instance_id = instance.id
    output_dir = request.output_dir
    rollout_dir = os.path.join(output_dir, "rollout")
    logs_dir = os.path.join(output_dir, "logs")
    os.makedirs(rollout_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    warnings: list[str] = []
    distribution = ensure_distribution(config)
    harness = create_harness(config.name)
    installed = harness.install(workspace, distribution, config)

    checkout_commit, baseline_untracked = prepare_repository(
        workspace, request.repo_path, request.base_commit
    )

    upstream_url, upstream_api_key, upstream_model, harness_model = (
        upstream_credentials(llm, config)
    )
    harness_model = config.harness_model or harness_model
    logger.info(
        "Harness %s upstream=%s model=%s harness_model=%s repo=%s base=%s",
        config.name,
        upstream_url,
        upstream_model,
        harness_model,
        request.repo_path,
        checkout_commit,
    )

    max_turns = config.resolve_max_turns(request.max_iterations)
    harness_log = os.path.join(
        logs_dir, f"{instance_id}.attempt{request.attempt}.harness.jsonl"
    )
    rollout_export = os.path.join(
        rollout_dir, f"{instance_id}.attempt{request.attempt}.json"
    )
    rollout_log = os.path.join(
        logs_dir, f"{instance_id}.attempt{request.attempt}.rollout-manager.log"
    )

    manager = _start_rollout_manager(
        workspace=workspace,
        config=config,
        instance_id=instance_id,
        attempt=request.attempt,
        output_path=rollout_export,
        log_path=rollout_log,
        upstream_url=upstream_url,
        upstream_api_key=upstream_api_key,
        upstream_model=upstream_model,
        harness_name=config.name,
        harness_version=config.version,
    )

    run_record: dict[str, Any] = {}
    try:
        harness_run = harness.run(
            workspace,
            config,
            installed,
            proxy_url=f"http://{workspace.host_alias}:{manager.port}",
            harness_model=harness_model,
            instruction=request.instruction,
            max_turns=max_turns,
            repo_path=request.repo_path,
            log_path=harness_log,
            timeout_sec=request.conversation_timeout,
            extra_env=request.extra_env,
        )
        run_record = {
            "exit_code": harness_run.exit_code,
            "timed_out": harness_run.timed_out,
            "duration_seconds": round(harness_run.outcome.duration_seconds, 3),
            "log_path": harness_log,
            "result": harness_run.result_event,
        }
        if harness_run.timed_out:
            warnings.append(f"harness timed out after {request.conversation_timeout}s")
        if harness_run.exit_code != 0:
            warnings.append(f"harness exited with code {harness_run.exit_code}")
    finally:
        try:
            manager.finish(config.rollout_finish_timeout_sec)
        except RolloutManagerError as exc:
            warnings.append(f"rollout-manager finish failed: {exc}")
            logger.warning("rollout-manager finish failed for %s: %s", instance_id, exc)
        finally:
            manager.stop()

    export = _load_export(rollout_export, warnings)
    main_trajectory_payload = main_trajectory(export) if export else None
    metrics = build_metrics(export, model_name=llm.model) if export else None
    stats = export_summary(export) if export else {}
    if export is not None and int(stats.get("requests") or 0) == 0:
        warnings.append("rollout export contains no captured requests")
        logger.warning("No rollout requests captured for %s", instance_id)

    git_patch = collect_patch(
        workspace,
        request.repo_path,
        request.base_commit,
        baseline_untracked=baseline_untracked,
    )

    evaluation = _judge_instance(judge, instance_id, git_patch, instance.data)

    history = build_history_payload(
        request=request,
        config=config,
        installed_version=installed.version,
        model=getattr(llm, "model", None),
        upstream_model=upstream_model,
        harness_model=harness_model,
        rollout_export_path=rollout_export,
        export=export,
        main_trajectory_payload=main_trajectory_payload,
        stats=stats,
        run_record=run_record,
        git_patch=git_patch,
        evaluation=evaluation,
        baseline_untracked=baseline_untracked,
        warnings=warnings,
    )
    history_path = history_file_path(output_dir, instance_id, request.attempt)
    streamlined_path = streamlined_history_file_path(
        output_dir, instance_id, request.attempt
    )
    streamlined: dict[str, Any] | None = None
    if config.write_streamlined_history:
        try:
            streamlined = streamline_history(history)
        except Exception as exc:  # noqa: BLE001 - never fail a scored attempt
            warnings.append(f"history streamlining failed: {exc}")
            logger.warning(
                "History streamlining failed for %s: %s",
                instance_id,
                exc,
                exc_info=True,
            )
    try:
        restored_stops = attach_stop_sequences_to_history(history)
        if restored_stops:
            history["restored_stop_sequences"] = restored_stops
    except Exception as exc:  # noqa: BLE001 - best-effort normalization
        warnings.append(f"stop-sequence restoration failed: {exc}")
        logger.warning(
            "Stop-sequence restoration failed for %s: %s",
            instance_id,
            exc,
            exc_info=True,
        )
    if streamlined is not None:
        history["streamlined_history"] = os.path.relpath(streamlined_path, output_dir)
    write_json_atomic(history_path, history)
    if streamlined is not None:
        write_json_atomic(streamlined_path, streamlined)
    logger.info(
        "Harness run for %s finished: patch=%d bytes, captures=%s, trajectories=%s",
        instance_id,
        len(git_patch),
        stats.get("requests"),
        stats.get("trajectories"),
    )
    return HarnessRunResult(
        git_patch=git_patch,
        instruction=request.instruction,
        history=history,
        history_path=history_path,
        streamlined_history_path=streamlined_path if streamlined is not None else None,
        evaluation=evaluation,
        metrics=metrics,
        rollout_export_path=rollout_export if export is not None else None,
        warnings=warnings,
    )


def env_from_setup_commands(commands: list[str] | None) -> dict[str, str]:
    """Convert ``EvalMetadata.env_setup_commands`` exports into harness env vars.

    Task metadata carries shell snippets such as ``export PIP_CACHE_DIR=...``
    and mirror configuration. A blackbox harness runs its own shell, so those
    values are passed through the harness process environment instead of being
    executed inside a long-lived agent-server shell.
    """
    env: dict[str, str] = {}
    pattern = re.compile(r"^\s*export\s+([A-Za-z_][A-Za-z0-9_]*)=(.*)$")
    for command in commands or []:
        for line in str(command).splitlines():
            match = pattern.match(line)
            if not match:
                continue
            key, raw = match.group(1), match.group(2).strip()
            try:
                parts = shlex.split(raw, posix=True)
            except ValueError:
                env[key] = raw
                continue
            env[key] = parts[0] if parts else ""
    return env


# ---------------------------------------------------------------------------
# rollout manager
# ---------------------------------------------------------------------------


def rollout_bind_host(
    workspace: DockerContainerWorkspace, config: HarnessConfig
) -> str:
    """Interface the proxy binds to, reachable from the task container."""
    if config.rollout_bind_host:
        return config.rollout_bind_host
    if workspace.gateway_ip:
        return workspace.gateway_ip
    # No inspectable bridge: fall back to all interfaces so the container can
    # still reach it (the capture proxy has no auth and must stay local-only).
    logger.warning(
        "Cannot determine the Docker bridge gateway; binding the rollout "
        "manager to all host interfaces"
    )
    return "0.0.0.0"


def _start_rollout_manager(
    *,
    workspace: DockerContainerWorkspace,
    config: HarnessConfig,
    instance_id: str,
    attempt: int,
    output_path: str,
    log_path: str,
    upstream_url: str,
    upstream_api_key: str | None,
    upstream_model: str,
    harness_name: str,
    harness_version: str,
) -> RolloutManager:
    bind_host = rollout_bind_host(workspace, config)
    metadata = {
        "instance_id": instance_id,
        "attempt": attempt,
        "harness": harness_name,
        "harness_version": harness_version,
        "upstream_model": upstream_model,
    }
    manager = RolloutManager(
        RolloutManagerConfig(
            upstream_base_url=upstream_url,
            session_id=f"{instance_id}-attempt{attempt}",
            output_path=output_path,
            log_path=log_path,
            upstream_api_key=upstream_api_key,
            model=upstream_model,
            metadata=metadata,
            exclude_request_classes=config.exclude_request_classes,
            request_params=config.request_params,
            log_upstream_requests=config.rollout_log_upstream,
            log_upstream_max_chars=config.rollout_log_upstream_max_chars,
            request_timeout_sec=config.rollout_request_timeout_sec,
            finish_wait_sec=config.rollout_finish_timeout_sec,
            start_timeout_sec=config.rollout_start_timeout_sec,
            host=bind_host,
        )
    )
    manager.start()
    return manager


def _load_export(path: str, warnings: list[str]) -> dict[str, Any] | None:
    try:
        return load_rollout_export(path)
    except RolloutManagerError as exc:
        message = f"rollout export unavailable: {exc}"
        warnings.append(message)
        logger.warning(message)
        return None


# ---------------------------------------------------------------------------
# repository/patch handling
# ---------------------------------------------------------------------------


def _git(workspace: DockerContainerWorkspace, repo_path: str, command: str) -> Any:
    return workspace.execute_command(
        f"cd {shlex.quote(repo_path)} && {command}", timeout=300
    )


def prepare_repository(
    workspace: DockerContainerWorkspace, repo_path: str, base_commit: str
) -> tuple[str, set[str]]:
    """Reset the checkout to ``base_commit`` and record pre-existing untracked files.

    Returns the resolved commit and the baseline untracked file set.
    """
    # Some task images are owned by another user/root; make git accept the repo.
    workspace.execute_command(
        "git config --global --add safe.directory '*'", timeout=30
    )

    resolved = _git(
        workspace, repo_path, f"git rev-parse {shlex.quote(base_commit)}^{{commit}}"
    )
    if not resolved.ok:
        raise RuntimeError(
            f"Base commit {base_commit} is not present in {repo_path}: "
            f"{(resolved.stderr or resolved.stdout).strip()[:1000]}"
        )
    commit = (resolved.stdout or "").strip()
    reset = _git(workspace, repo_path, f"git reset --hard {shlex.quote(commit)}")
    if not reset.ok:
        raise RuntimeError(
            f"Failed to reset {repo_path} to {commit}: "
            f"{(reset.stderr or reset.stdout).strip()[:1000]}"
        )
    untracked = _git(workspace, repo_path, "git ls-files --others --exclude-standard")
    if not untracked.ok:
        raise RuntimeError(
            f"Failed to list untracked files in {repo_path}: {untracked.stderr}"
        )
    baseline = {
        line.strip() for line in (untracked.stdout or "").splitlines() if line.strip()
    }
    if baseline:
        logger.info(
            "Ignoring %d pre-existing untracked file(s) when extracting the "
            "patch (image build artifacts)",
            len(baseline),
        )
    return commit, baseline


def collect_patch(
    workspace: DockerContainerWorkspace,
    repo_path: str,
    base_commit: str,
    *,
    baseline_untracked: set[str] | None = None,
) -> str:
    """Stage, commit and diff the working tree against ``base_commit``."""
    add = _git(workspace, repo_path, "git add -A")
    if not add.ok:
        raise RuntimeError(
            f"Failed to stage changes in {repo_path}: {add.stderr.strip()[:1000]}"
        )
    # The harness may have committed already; a failed commit then only means
    # "nothing to commit" and is expected.
    commit = _git(
        workspace,
        repo_path,
        "git -c user.name="
        f"{shlex.quote(GIT_AUTHOR_NAME)} -c user.email={shlex.quote(GIT_AUTHOR_EMAIL)} "
        "commit --no-verify --allow-empty-message -m 'harness patch'",
    )
    if not commit.ok:
        logger.debug("Commit after harness produced nothing: %s", commit.stderr.strip())
    # Hooks/aliases configured by the harness must not alter patch extraction.
    diff = _git(
        workspace,
        repo_path,
        f"git --no-pager diff --no-color --binary {shlex.quote(base_commit)} HEAD",
    )
    if not diff.ok:
        raise RuntimeError(
            f"Failed to extract the git patch: {(diff.stderr or diff.stdout)[:1000]}"
        )
    patch = diff.stdout or ""
    if baseline_untracked and patch:
        filtered = remove_files_from_patch(patch, baseline_untracked)
        if filtered != patch:
            logger.info(
                "Removed pre-existing untracked files from the patch so the "
                "judge's `git apply` does not fail"
            )
        patch = filtered
    return patch


# ---------------------------------------------------------------------------
# judge + history
# ---------------------------------------------------------------------------


def _judge_instance(
    judge: ExecutionBasedJudge | None,
    instance_id: str,
    git_patch: str,
    instance_data: dict[str, Any],
) -> bool | None:
    if judge is None:
        return None
    try:
        result = judge.judge(
            instance_id=instance_id,
            git_patch=git_patch,
            instance_data=instance_data,
        )
    except Exception as exc:  # noqa: BLE001 - a judge crash must not lose the patch
        logger.error("Judge failed for %s: %s", instance_id, exc)
        return None
    logger.info("Judge result for %s: %s", instance_id, result)
    return result


def build_history_payload(
    *,
    request: HarnessRunRequest,
    config: HarnessConfig,
    installed_version: str,
    model: str | None,
    upstream_model: str,
    harness_model: str,
    rollout_export_path: str,
    export: dict[str, Any] | None,
    main_trajectory_payload: dict[str, Any] | None,
    stats: dict[str, Any],
    run_record: dict[str, Any],
    git_patch: str,
    evaluation: bool | None,
    baseline_untracked: set[str],
    warnings: list[str],
) -> dict[str, Any]:
    """History JSON compatible with the other benchmark history dumps."""
    messages = (main_trajectory_payload or {}).get("messages") or []
    # ``tools`` is the OpenAI tool list actually used by the main trajectory
    # (directly usable by trainers); the deduplicated raw variants are kept
    # alongside as ``tool_sets`` for provenance.
    tools = (
        resolve_trajectory_tools(main_trajectory_payload, export)
        if export is not None and main_trajectory_payload is not None
        else []
    )
    tool_sets = collect_tool_sets(export) if export else []
    try:
        rollout_relative = os.path.relpath(rollout_export_path, request.output_dir)
    except ValueError:  # pragma: no cover - different drives on Windows
        rollout_relative = rollout_export_path
    payload: dict[str, Any] = {
        "instance_id": request.instance.id,
        "attempt": request.attempt,
        "harness": config.name,
        "harness_version": config.version,
        "harness_install_version": installed_version,
        "model": model,
        "upstream_model": upstream_model,
        "harness_model": harness_model,
        "temperature": None,
        "top_p": None,
        "instruction": request.instruction,
        "messages": messages,
        "tools": tools,
        "tool_sets": tool_sets,
        "trajectories": (export or {}).get("trajectories") or [],
        "request_log": (export or {}).get("request_log") or [],
        "stats": stats,
        "run": run_record,
        "rollout_export": rollout_relative,
        "repository": {
            "path": request.repo_path,
            "base_commit": request.base_commit,
            "ignored_untracked_files": sorted(baseline_untracked),
        },
        "test_result": {"git_patch": git_patch},
        "warnings": warnings,
    }
    if evaluation is not None:
        payload["evaluation"] = evaluation
    return payload


def history_file_path(output_dir: str, instance_id: str, attempt: int) -> str:
    return os.path.join(output_dir, f"{instance_id}.attempt{attempt}.history.json")


def streamlined_history_file_path(
    output_dir: str, instance_id: str, attempt: int
) -> str:
    return os.path.join(
        output_dir, f"{instance_id}.attempt{attempt}{STREAMLINED_SUFFIX}"
    )


def canonical_streamlined_history_path(output_dir: str, instance_id: str) -> str:
    return os.path.join(output_dir, f"{instance_id}{STREAMLINED_SUFFIX}")


def canonical_history_path(output_dir: str, instance_id: str) -> str:
    return os.path.join(output_dir, f"{instance_id}.history.json")


def write_json_atomic(path: str, payload: dict[str, Any]) -> None:
    """Write a JSON document atomically (tmp file in the same directory)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=str(Path(path).parent),
        prefix=f".{Path(path).name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary = handle.name
    try:
        with handle:
            # One JSON document per line (+ trailing newline) so
            # ``cat <output>/*.history.json`` is directly valid JSONL.
            json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def publish_canonical_histories(output_dir: str, output_path: str) -> None:
    """Copy each instance's winning-attempt history to ``<id>.history.json``."""
    selected: dict[str, int] = {}
    candidates = [output_path, output_path.replace(".jsonl", "_errors.jsonl")]
    for path in candidates:
        if not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                instance_id = row.get("instance_id")
                if instance_id:
                    selected[str(instance_id)] = int(row.get("attempt") or 1)
    for instance_id, attempt in selected.items():
        source = history_file_path(output_dir, instance_id, attempt)
        if not os.path.isfile(source):
            continue
        try:
            with open(source, encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:  # pragma: no cover - best effort
            logger.warning("Failed to publish history for %s: %s", instance_id, exc)
            continue
        try:
            restored_stops = attach_stop_sequences_to_history(payload)
            if restored_stops:
                payload["restored_stop_sequences"] = restored_stops
            streamlined = streamline_history(payload)
        except Exception as exc:  # noqa: BLE001 - publish raw history regardless
            logger.warning(
                "Failed to streamline canonical history for %s: %s", instance_id, exc
            )
            streamlined = None
        canonical_streamlined = canonical_streamlined_history_path(
            output_dir, instance_id
        )
        if streamlined is not None:
            # The canonical payload is loaded from an already-finalized attempt,
            # so stop sequences were attached before this pass; keep the count
            # visible in the streamlined summary.
            if not streamlined["streamlined"].get(
                "restored_stop_sequences"
            ) and payload.get("restored_stop_sequences"):
                streamlined["streamlined"]["restored_stop_sequences"] = payload[
                    "restored_stop_sequences"
                ]
            payload["streamlined_history"] = os.path.relpath(
                canonical_streamlined, output_dir
            )
        try:
            write_json_atomic(canonical_history_path(output_dir, instance_id), payload)
            if streamlined is not None:
                write_json_atomic(canonical_streamlined, streamlined)
        except OSError as exc:  # pragma: no cover - best effort
            logger.warning(
                "Failed to write canonical history for %s: %s", instance_id, exc
            )
