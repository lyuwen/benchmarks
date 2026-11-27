"""Harbor-format task inference through the customized OpenHands SDK harness.

The benchmark runner owns the trial lifecycle (workspace, agent, verifier,
artifacts, result records); Harbor supplies the task specification and verifier
contract.  This module is the ``harbor-infer`` entry point.

Example::

    uv run harbor-infer .llm_config/model.json \
        --tasks-path thirdparty/CalibForge/multi_solver \
        --num-workers 4 --n-limit 10 --max-attempts 1 --max-retries 0

Phase 1 supports single-container, single-step tasks with a shared verifier.
Everything else is rejected during task loading/preflight (see
``benchmarks/harbor/task.py``) instead of being silently relaxed.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf
from pydantic import Field, PrivateAttr

from benchmarks.harbor.judge import collect_artifacts, run_shared_verifier
from benchmarks.harbor.task import (
    HARBOR_REFERENCE_COMMIT,
    HARBOR_REFERENCE_VERSION,
    HarborTask,
    HarborTaskConfig,
    HarborTaskError,
    HarborTaskSet,
    check_resume_compatible,
    compute_run_settings_hash,
    load_tasks,
    resolve_env_templates,
    should_upload_environment_dir,
    validate_supported,
    write_task_manifest,
)
from benchmarks.harbor.workspace import (
    HarborDockerWorkspace,
    ensure_local_image,
    read_local_digest,
    resolve_effective_workdir,
    resolve_image_reference,
)
from benchmarks.utils.args_parser import get_parser
from benchmarks.utils.condenser import get_condenser
from benchmarks.utils.conversation import build_event_persistence_callback
from benchmarks.utils.critics import create_critic
from benchmarks.utils.evaluation import Evaluation, OnResult
from benchmarks.utils.evaluation_utils import (
    construct_eval_output_dir,
    get_default_on_result_writer,
)
from benchmarks.utils.fake_user_response import (
    run_conversation_with_fake_user_response,
    sync_remote_conversation_state,
)
from benchmarks.utils.mirror_config import get_mirror_env_commands
from benchmarks.utils.models import EvalInstance, EvalMetadata, EvalOutput
from benchmarks.utils.trajectory import (
    build_messages,
    build_trajectories,
    pruning_metadata,
    resolve_pruning_config,
)
from openhands.sdk import LLM, Agent, BaseConversation, Conversation, get_logger
from openhands.sdk.conversation.state import ConversationExecutionStatus
from openhands.sdk.event.llm_convertible.system import SystemPromptEvent
from openhands.sdk.tool.tool import ToolDefinition
from openhands.sdk.workspace import RemoteWorkspace
from openhands.tools.preset.default import get_default_tools


logger = get_logger(__name__)

AGENT_STATUS_COMPLETED = "completed"
AGENT_STATUS_TIMEOUT = "timeout"
AGENT_STATUS_ERROR = "error"

DEFAULT_AGENT_PLUGIN_IMAGE = os.getenv("AGENT_PLUGIN_IMAGE", "openhands/agent-plugin")
DEFAULT_WORKSPACE_MEMORY_LIMIT = os.getenv("HARBOR_WORKSPACE_MEMORY_LIMIT", "14g")

_TRIAL_JSON = "trial.json"
_HISTORY_JSON = "history.json"
_AGENT_STOP_TIMEOUT_SEC = 60.0


# ---------------------------------------------------------------------------
# Image resolution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResolvedTaskImage:
    """The image a trial will run, plus its provenance."""

    reference: str
    digest: str | None
    built: bool


def _sanitize_built_image_name(task_id: str, fingerprint: str) -> str:
    sanitized = "".join(
        char if char.isalnum() or char in "._-" else "-" for char in task_id.lower()
    ).strip("-.")
    suffix = fingerprint[:12]
    return f"harbor-task/{sanitized[:100] or 'task'}:{suffix}"


def resolve_task_image(
    task: HarborTask,
    docker_image_prefix: str | None,
    *,
    trial_dir: str | Path,
) -> ResolvedTaskImage:
    """Resolve the prebuilt image or build the task's Dockerfile."""
    docker_image = task.config.environment.docker_image
    if docker_image:
        reference = resolve_image_reference(docker_image, docker_image_prefix)
        # Pull first: the effective working directory falls back to the image
        # WORKDIR, which must be readable before the container starts.
        ensure_local_image(
            reference, timeout_sec=float(task.config.environment.build_timeout_sec)
        )
        return ResolvedTaskImage(
            reference=reference,
            digest=read_local_digest(reference),
            built=False,
        )

    dockerfile = task.environment_dir / "Dockerfile"
    if not dockerfile.is_file():
        raise HarborTaskError(
            f"Task {task.id!r} has no docker_image and no environment/Dockerfile."
        )
    reference = _sanitize_built_image_name(task.id, task.fingerprint)
    build_timeout = max(60.0, float(task.config.environment.build_timeout_sec))
    logger.info("Building %s from %s", reference, dockerfile)
    try:
        completed = subprocess.run(
            [
                "docker",
                "build",
                "--platform",
                "linux/amd64",
                "-f",
                str(dockerfile),
                "-t",
                reference,
                str(task.environment_dir),
            ],
            capture_output=True,
            text=True,
            timeout=build_timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        _write_build_log(trial_dir, f"docker build timed out after {build_timeout}s\n")
        raise RuntimeError(
            f"Building task image for {task.id!r} timed out after {build_timeout}s"
        ) from exc
    build_log = completed.stdout + completed.stderr
    _write_build_log(trial_dir, build_log)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Building task image for {task.id!r} failed "
            f"(exit {completed.returncode}): {completed.stderr.strip()[-4000:]}"
        )
    return ResolvedTaskImage(
        reference=reference,
        digest=read_local_digest(reference),
        built=True,
    )


def _write_build_log(trial_dir: str | Path, text: str) -> None:
    try:
        path = Path(trial_dir) / "image-build.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    except OSError:  # pragma: no cover - diagnostics only
        logger.debug("Failed to write image build log", exc_info=True)


def image_workdir(image_reference: str) -> str | None:
    """Return the image's configured ``WORKDIR`` (may be empty)."""
    try:
        completed = subprocess.run(
            [
                "docker",
                "image",
                "inspect",
                "--format",
                "{{.Config.WorkingDir}}",
                image_reference,
            ],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    value = completed.stdout.strip()
    return value or None


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


class HarborEvaluation(Evaluation):
    """Harbor task evaluation using the customized OpenHands agent harness."""

    tasks_path: str
    docker_image_prefix: str | None = Field(
        default=None,
        description=(
            "Optional registry/namespace prefix prepended to bare docker_image "
            "references (e.g. a local mirror). Never hard-coded."
        ),
    )
    agent_plugin_image: str = Field(default=DEFAULT_AGENT_PLUGIN_IMAGE)
    verifier_enabled: bool = Field(default=True)
    primary_reward_key: str = Field(default="reward")
    workspace_memory_limit: str | None = Field(default=DEFAULT_WORKSPACE_MEMORY_LIMIT)
    agent_timeout_override: float | None = Field(default=None)
    verifier_timeout_override: float | None = Field(default=None)
    detach_docker_logs: bool = Field(default=True)

    _task_set: HarborTaskSet | None = PrivateAttr(default=None)
    _tasks_by_id: dict[str, HarborTask] = PrivateAttr(default_factory=dict)
    _active_trial_dirs: dict[str, str] = PrivateAttr(default_factory=dict)
    _trial_settings: dict[str, dict[str, Any]] = PrivateAttr(default_factory=dict)

    # -- lifecycle ---------------------------------------------------------

    def model_post_init(self, __context: Any) -> None:
        """Create the output directory but defer metadata until tasks are loaded.

        The base class overwrites ``metadata.json`` immediately; resume
        validation needs the task fingerprints first, so the write happens in
        :meth:`run` after ``prepare_instances``.
        """
        os.makedirs(self.metadata.eval_output_dir, exist_ok=True)

    def run(self, *, on_result: OnResult | None = None) -> list[EvalOutput]:
        self.prepare_instances()
        self._write_run_metadata()
        return super().run(on_result=on_result)

    def prepare_instances(self) -> list[EvalInstance]:
        if self._task_set is None:
            logger.info("Loading Harbor tasks from %s", self.tasks_path)
            self._task_set = load_tasks(
                self.tasks_path,
                selected_instances_file=self.metadata.selected_instances_file,
                eval_limit=self.metadata.eval_limit,
            )
            self._tasks_by_id = {task.id: task for task in self._task_set.tasks}
            logger.info(
                "Loaded %d Harbor task(s) (dataset revision %s)",
                len(self._task_set.tasks),
                self._task_set.dataset_revision,
            )
        return [
            EvalInstance(id=task.id, data=task.to_instance_data())
            for task in self._task_set.tasks
        ]

    def _run_settings(self) -> dict[str, Any]:
        return {
            "reference_version": HARBOR_REFERENCE_VERSION,
            "reference_commit": HARBOR_REFERENCE_COMMIT,
            "workspace_backend": "docker",
            "verifier_enabled": self.verifier_enabled,
            "primary_reward_key": self.primary_reward_key,
            "agent_plugin_image": self.agent_plugin_image,
            "docker_image_prefix": self.docker_image_prefix or "",
            "workspace_memory_limit": self.workspace_memory_limit or "",
            "agent_timeout_override": self.agent_timeout_override,
            "verifier_timeout_override": self.verifier_timeout_override,
            "max_iterations": self.metadata.max_iterations,
            "conversation_timeout": self.metadata.conversation_timeout,
            "condenser": self.metadata.condenser,
            "condenser_config": self.metadata.condenser_config,
            "llm_model": self.metadata.llm.model,
        }

    def _write_run_metadata(self) -> None:
        if self._task_set is None:
            self.prepare_instances()
        assert self._task_set is not None
        output_dir = Path(self.metadata.eval_output_dir)
        run_settings = self._run_settings()
        run_settings_hash = compute_run_settings_hash(run_settings)
        check_resume_compatible(
            output_dir,
            manifest_hash=self._task_set.manifest_hash,
            run_settings_hash=run_settings_hash,
        )
        manifest_file = write_task_manifest(output_dir, self._task_set)
        self.metadata.details = {
            "harbor": {
                "reference_version": HARBOR_REFERENCE_VERSION,
                "reference_commit": HARBOR_REFERENCE_COMMIT,
                "tasks_path": str(self._task_set.tasks_path),
                "dataset_revision": self._task_set.dataset_revision,
                "task_count": len(self._task_set.tasks),
                "manifest_file": manifest_file.name,
                "manifest_hash": self._task_set.manifest_hash,
                "run_settings_hash": run_settings_hash,
                "run_settings": run_settings,
            }
        }
        metadata_file = output_dir / "metadata.json"
        metadata_file.write_text(
            self.metadata.model_dump_json(indent=2), encoding="utf-8"
        )
        logger.info("Saved Harbor run metadata to %s", metadata_file)

    # -- trial layout ------------------------------------------------------

    def _begin_trial(self, instance: EvalInstance) -> str:
        execution_id = (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            + "-"
            + uuid.uuid4().hex[:8]
        )
        trial_dir = (
            Path(self.metadata.eval_output_dir)
            / "trials"
            / instance.id
            / f"attempt-{self.current_attempt}"
            / execution_id
        )
        (trial_dir / "verifier").mkdir(parents=True, exist_ok=True)
        (trial_dir / "artifacts").mkdir(parents=True, exist_ok=True)
        trial_record = {
            "instance_id": instance.id,
            "task_id": instance.data.get("task_id"),
            "relative_path": instance.data.get("relative_path"),
            "attempt": self.current_attempt,
            "execution_id": execution_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "fingerprint": instance.data.get("fingerprint"),
        }
        (trial_dir / _TRIAL_JSON).write_text(
            json.dumps(trial_record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self._active_trial_dirs[instance.id] = str(trial_dir)
        return str(trial_dir)

    def _task_for_instance(self, instance: EvalInstance) -> HarborTask:
        task = self._tasks_by_id.get(instance.id)
        if task is None:
            task = task_from_instance_data(instance.data)
            self._tasks_by_id[instance.id] = task
        return task

    # -- hooks -------------------------------------------------------------

    def prepare_workspace(
        self,
        instance: EvalInstance,
        resource_factor: int = 1,
        forward_env: list[str] | None = None,
    ) -> RemoteWorkspace:
        task = self._task_for_instance(instance)
        trial_dir = self._begin_trial(instance)

        image = resolve_task_image(task, self.docker_image_prefix, trial_dir=trial_dir)
        workdir = resolve_effective_workdir(
            task.config.environment.workdir, image_workdir(image.reference)
        )
        task_env = resolve_env_templates(task.config.environment.env)
        settings = {
            "image": image.reference,
            "image_digest": image.digest,
            "image_built": image.built,
            "workdir": workdir,
            "cpus": task.config.environment.cpus,
            "memory_mb": task.config.environment.memory_mb,
            "workspace_memory_limit": self.workspace_memory_limit,
            "storage_mb": task.config.environment.storage_mb,
            "agent_plugin_image": self.agent_plugin_image,
            "env_keys": sorted(task_env),
            "network_mode": "public",
            "agent_timeout_sec": self._agent_timeout(task),
            "verifier_timeout_sec": self._verifier_timeout(task),
        }
        self._trial_settings[instance.id] = settings

        environment_upload = None
        if should_upload_environment_dir(
            task.environment_dir, docker_image=task.config.environment.docker_image
        ):
            environment_upload = str(task.environment_dir)
            settings["environment_dir_uploaded_to"] = workdir
        workspace = HarborDockerWorkspace(
            base_image=image.reference,
            working_dir=workdir,
            agent_plugin_image=self.agent_plugin_image,
            task_env=task_env,
            cpus=task.config.environment.cpus,
            memory_mb=task.config.environment.memory_mb,
            memory_limit=self.workspace_memory_limit,
            healthcheck=task.config.environment.healthcheck,
            upload_environment_dir=environment_upload,
            forward_env=list(forward_env or []),
            detach_logs=self.detach_docker_logs,
        )
        settings["image_digest"] = workspace.image_digest or image.digest
        try:
            for command in self.metadata.env_setup_commands or []:
                result = workspace.execute_command(command, cwd=workdir)
                if result.exit_code != 0:
                    raise RuntimeError(
                        f"Environment setup command failed ({result.exit_code}): "
                        f"{command!r}: {result.stderr.strip()[:1000]}"
                    )
        except Exception:
            workspace.cleanup()
            raise
        logger.info(
            "Workspace ready for %s (image=%s workdir=%s)",
            instance.id,
            image.reference,
            workdir,
        )
        return workspace

    def _agent_timeout(self, task: HarborTask) -> float | None:
        if self.agent_timeout_override is not None:
            return self.agent_timeout_override
        return task.config.agent.timeout_sec

    def _verifier_timeout(self, task: HarborTask) -> float:
        if self.verifier_timeout_override is not None:
            return self.verifier_timeout_override
        return task.config.verifier.timeout_sec

    def evaluate_instance(
        self, instance: EvalInstance, workspace: RemoteWorkspace
    ) -> EvalOutput:
        task = self._task_for_instance(instance)
        trial_dir = self._active_trial_dirs.get(instance.id) or self._begin_trial(
            instance
        )
        settings = dict(self._trial_settings.get(instance.id, {}))
        assert isinstance(workspace, HarborDockerWorkspace)

        instruction = task.instruction
        agent_started = time.monotonic()
        agent_status = AGENT_STATUS_COMPLETED
        agent_error: str | None = None
        conversation: BaseConversation | None = None
        try:
            agent = Agent(
                llm=self.metadata.llm,
                condenser=get_condenser(
                    self.metadata.condenser,
                    self.metadata.llm,
                    self.metadata.condenser_config,
                ),
                tools=get_default_tools(enable_browser=False),
                system_prompt_kwargs={"cli_mode": True},
            )
            persist_callback = build_event_persistence_callback(
                run_id=self.metadata.eval_output_dir,
                instance_id=instance.id,
                attempt=self.current_attempt,
            )
            conversation = Conversation(
                agent=agent,
                workspace=workspace,
                callbacks=[persist_callback],
                max_iteration_per_run=self.metadata.max_iterations,
            )
            conversation.send_message(instruction)

            agent_timeout = self._agent_timeout(task)
            deadline = time.monotonic() + agent_timeout if agent_timeout else None
            try:
                run_conversation_with_fake_user_response(
                    conversation,
                    timeout=self.metadata.conversation_timeout,
                    deadline=deadline,
                )
            except TimeoutError as exc:
                agent_status = AGENT_STATUS_TIMEOUT
                agent_error = str(exc) or "agent execution timed out"
                logger.warning(
                    "Agent timed out for %s after %ss; stopping execution before "
                    "grading",
                    instance.id,
                    agent_timeout,
                )
                _stop_agent_execution(conversation)
            except Exception as exc:
                agent_status = AGENT_STATUS_ERROR
                agent_error = f"{type(exc).__name__}: {exc}"
                logger.error("Agent run failed for %s: %s", instance.id, agent_error)
                _stop_agent_execution(conversation)
        except Exception:
            # A failure before the conversation can run means there is no usable
            # rollout; let the worker retry / record a top-level error.
            if conversation is not None:
                _close_conversation(conversation)
            raise

        agent_duration = time.monotonic() - agent_started
        events, messages, trajectories, tools_list, metrics = _export_trajectory(
            conversation
        )

        artifacts = collect_artifacts(
            workspace,
            task,
            trial_dir=trial_dir,
            run_dir=self.metadata.eval_output_dir,
        )
        verifier = run_shared_verifier(
            workspace,
            task,
            trial_dir=trial_dir,
            run_dir=self.metadata.eval_output_dir,
            primary_reward_key=self.primary_reward_key,
            timeout_override=self.verifier_timeout_override,
            enabled=self.verifier_enabled,
        )

        test_result = {
            "verifier": verifier,
            "agent": {
                "status": agent_status,
                "error": agent_error,
                "duration_sec": round(agent_duration, 3),
                "timeout_sec": self._agent_timeout(task),
            },
            "artifacts": artifacts,
            "trial": {
                "instance_id": instance.id,
                "attempt": self.current_attempt,
                "trial_dir": _relative_to_run_dir(
                    trial_dir, self.metadata.eval_output_dir
                ),
                "image": settings.get("image"),
                "image_digest": settings.get("image_digest"),
                "image_built": settings.get("image_built"),
                "workdir": settings.get("workdir"),
                "effective_settings": settings,
            },
            "task": {
                "task_id": task.id,
                "relative_path": task.relative_path,
                "schema_version": task.config.schema_version,
                "fingerprint": task.fingerprint,
                "verifier_enabled": self.verifier_enabled,
                "primary_reward_key": self.primary_reward_key,
            },
        }

        history_dump = {
            "instance_id": instance.id,
            "messages": messages,
            "model": self.metadata.llm.model,
            "tools": tools_list,
            "temperature": self.metadata.llm.temperature,
            "top_p": self.metadata.llm.top_p,
            "test_result": test_result,
            "trajectories": trajectories,
        }
        pruning = pruning_metadata(resolve_pruning_config(events))
        if pruning is not None:
            history_dump["pruning"] = pruning
        _write_history(trial_dir, history_dump)

        _close_conversation(conversation)
        logger.info(
            "Trial %s finished: agent=%s verifier=%s reward=%s",
            instance.id,
            agent_status,
            verifier.get("status"),
            verifier.get("reward"),
        )
        return EvalOutput(
            instance_id=instance.id,
            attempt=self.current_attempt,
            test_result=test_result,
            instruction=instruction,
            error=None,
            history=events,
            metrics=metrics,
            instance=instance.data,
        )

    def _capture_conversation_archive(
        self, workspace: RemoteWorkspace, instance: EvalInstance
    ) -> None:
        """Archive the agent-server conversation store into the trial directory.

        The base worker writes ``conversations/<id>.tar.gz`` which would
        overwrite earlier attempts; Harbor trial directories are per execution.
        """
        trial_dir = self._active_trial_dirs.get(instance.id)
        if not trial_dir:
            logger.debug("No trial directory for %s; skipping archive", instance.id)
            return
        conversations_dir = getattr(
            workspace, "conversations_dir", "/tmp/harbor/conversations"
        )
        remote_tar = "/tmp/harbor-conversations.tar.gz"
        relative = shlex.quote(str(conversations_dir).lstrip("/"))
        try:
            built = workspace.execute_command(
                f"if [ -d {shlex.quote(str(conversations_dir))} ]; then "
                f"tar -czf {remote_tar} -C / {relative}; else exit 3; fi",
                timeout=300,
            )
            if built.exit_code == 3:
                logger.debug(
                    "No conversation archive for %s (directory not found)",
                    instance.id,
                )
                return
            if built.exit_code != 0:
                logger.warning(
                    "Failed to build conversation archive for %s: %s",
                    instance.id,
                    built.stderr.strip()[:500],
                )
                return
            destination = Path(trial_dir) / "conversations.tar.gz"
            result = workspace.file_download(remote_tar, destination)
            if not result.success:
                logger.warning(
                    "Failed to download conversation archive for %s: %s",
                    instance.id,
                    result.error,
                )
        except Exception as exc:  # pragma: no cover - best-effort archive
            logger.warning(
                "Failed to capture conversation archive for %s: %s",
                instance.id,
                exc,
            )


# ---------------------------------------------------------------------------
# Helpers shared with tests
# ---------------------------------------------------------------------------


def task_from_instance_data(data: dict[str, Any]) -> HarborTask:
    """Rebuild a :class:`HarborTask` from normalized ``EvalInstance.data``."""
    config = HarborTaskConfig.model_validate(data["config"])
    validate_supported(config, str(data.get("relative_path", data.get("task_id", "?"))))
    return HarborTask(
        id=str(data["task_id"]),
        relative_path=str(data["relative_path"]),
        path=Path(str(data["task_dir"])),
        dataset_root=Path(str(data["dataset_root"])),
        config=config,
        instruction=str(data.get("instruction", "")),
        fingerprint=str(data.get("fingerprint", "")),
        test_script=str(data.get("test_script", "test.sh")),
    )


def _relative_to_run_dir(path: str | Path, run_dir: str) -> str:
    try:
        return Path(path).resolve().relative_to(Path(run_dir).resolve()).as_posix()
    except ValueError:
        return str(path)


def _stop_agent_execution(conversation: BaseConversation) -> None:
    """Ask the agent server to pause and wait for a non-running state."""
    try:
        conversation.pause()
    except Exception as exc:
        logger.warning("Failed to pause conversation: %s", exc)
        return
    deadline = time.monotonic() + _AGENT_STOP_TIMEOUT_SEC
    while time.monotonic() < deadline:
        try:
            status = conversation.state.execution_status
        except Exception:
            return
        if status.is_terminal() or status in (
            ConversationExecutionStatus.PAUSED,
            ConversationExecutionStatus.IDLE,
        ):
            return
        time.sleep(0.5)
    logger.warning(
        "Agent execution did not stop within %ss after pause",
        _AGENT_STOP_TIMEOUT_SEC,
    )


def _close_conversation(conversation: BaseConversation | None) -> None:
    if conversation is None:
        return
    try:
        conversation.close()
    except Exception as exc:  # pragma: no cover - best effort cleanup
        logger.debug("Failed to close conversation: %s", exc)


def _export_trajectory(
    conversation: BaseConversation | None,
) -> tuple[list[Any], list[dict[str, Any]], list[Any], list[Any], Any]:
    """Export events, model-view messages/trajectories, tools, and metrics."""
    events: list[Any] = []
    messages: list[dict[str, Any]] = []
    trajectories: list[Any] = []
    tools_list: list[Any] = []
    metrics = None
    if conversation is None:
        return events, messages, trajectories, tools_list, metrics
    try:
        sync_remote_conversation_state(conversation)
    except Exception:  # pragma: no cover - best effort resync
        logger.debug("Remote conversation resync failed", exc_info=True)
    try:
        events = list(conversation.state.events)
    except Exception:  # pragma: no cover - best effort export
        logger.warning("Failed to read conversation events")
        return events, messages, trajectories, tools_list, metrics
    try:
        pruning_config = resolve_pruning_config(events)
        messages = build_messages(events, pruning_config)
        trajectories = build_trajectories(events, pruning_config)
    except Exception:
        logger.warning("Failed to build model-view trajectory", exc_info=True)
        messages = []
        trajectories = []

    for event in events:
        if isinstance(event, SystemPromptEvent):
            for tool in event.tools:
                if isinstance(tool, ToolDefinition):
                    tools_list.append(tool.to_openai_tool())
    try:
        metrics = conversation.conversation_stats.get_combined_metrics()
    except Exception:  # pragma: no cover - remote stats may be unavailable
        logger.debug("Failed to read conversation metrics", exc_info=True)
    return events, messages, trajectories, tools_list, metrics


def _write_history(trial_dir: str | Path, dump_data: dict[str, Any]) -> None:
    path = Path(trial_dir) / _HISTORY_JSON
    try:
        path.write_text(
            json.dumps(dump_data, indent=2, default=str) + "\n", encoding="utf-8"
        )
        logger.info("Dumped conversation history to %s", path)
    except OSError:
        logger.warning("Failed to write history for %s", trial_dir, exc_info=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser():
    parser = get_parser()
    parser.set_defaults(
        dataset=None,
        workspace="docker",
        max_attempts=1,
        max_retries=0,
    )
    parser.add_argument(
        "--tasks-path",
        type=str,
        required=True,
        help=(
            "Path to an individual Harbor task directory or a dataset root "
            "(nested subsets are discovered recursively)."
        ),
    )
    parser.add_argument(
        "--docker-image-prefix",
        type=str,
        default=None,
        help=(
            "Registry/namespace prefix prepended to a bare docker_image "
            "reference (e.g. a local mirror). Ignored when the reference "
            "already contains a registry host."
        ),
    )
    parser.add_argument(
        "--agent-plugin-image",
        type=str,
        default=DEFAULT_AGENT_PLUGIN_IMAGE,
        help=(
            "Image providing the /agent-server volume injected into task images "
            "(default: %(default)s, or AGENT_PLUGIN_IMAGE)."
        ),
    )
    parser.add_argument(
        "--skip-verifier",
        action="store_true",
        help=(
            "Run inference only and mark trials as unscored (explicit "
            "inference-only mode)."
        ),
    )
    parser.add_argument(
        "--primary-reward-key",
        type=str,
        default="reward",
        help=(
            "Reward key used as the primary score (default: 'reward'). A "
            "dataset with a different primary metric must pass it explicitly."
        ),
    )
    parser.add_argument(
        "--workspace-memory-limit",
        type=str,
        default=DEFAULT_WORKSPACE_MEMORY_LIMIT,
        help=(
            "Fallback Docker memory limit for task images without memory_mb "
            "(default: %(default)s, or HARBOR_WORKSPACE_MEMORY_LIMIT)."
        ),
    )
    parser.add_argument(
        "--agent-timeout",
        type=float,
        default=None,
        help="Override [agent].timeout_sec for every task (seconds).",
    )
    parser.add_argument(
        "--verifier-timeout",
        type=float,
        default=None,
        help="Override [verifier].timeout_sec for every task (seconds).",
    )
    parser.add_argument(
        "--no-detach-docker-logs",
        action="store_true",
        help="Do not stream task container logs into the instance log.",
    )
    return parser


def _preflight_docker() -> None:
    try:
        completed = subprocess.run(
            ["docker", "version"], capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SystemExit(f"Docker is not available: {exc}") from exc
    if completed.returncode != 0:
        raise SystemExit(
            "Docker daemon is not available; harbor-infer requires local Docker:\n"
            f"{completed.stderr.strip()}"
        )


def _validate_cli(args: Any) -> None:
    if args.workspace != "docker":
        raise SystemExit(
            f"Unsupported --workspace {args.workspace!r}: Harbor Phase 1 supports "
            "only the local Docker backend (--workspace docker)."
        )
    if args.max_attempts != 1:
        raise SystemExit(
            "Harbor Phase 1 supports only --max-attempts 1: verifier-driven "
            "retries need Harbor-aware critic policies that are not implemented."
        )
    if args.critic != "pass":
        raise SystemExit(
            f"Unsupported --critic {args.critic!r}: Harbor Phase 1 requires the "
            "'pass' critic (rollout retries are separate from verifier grading)."
        )
    if args.max_retries < 0:
        raise SystemExit("--max-retries must be >= 0")
    if not os.path.isdir(args.tasks_path):
        raise SystemExit(f"--tasks-path is not a directory: {args.tasks_path}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    _validate_cli(args)
    _preflight_docker()

    llm_config_path = args.llm_config_path
    if not os.path.isfile(llm_config_path):
        raise ValueError(f"LLM config file {llm_config_path} does not exist")
    llm_config = json.dumps(
        OmegaConf.to_container(OmegaConf.load(llm_config_path), resolve=True)
    )
    llm = LLM.model_validate_json(llm_config)
    logger.info("Using LLM config: %s", llm.model_dump_json(indent=2))

    tasks_path = Path(args.tasks_path).resolve()
    dataset_label = args.dataset or tasks_path.name
    dataset_description = dataset_label.replace("/", "__")
    structured_output_dir = construct_eval_output_dir(
        base_dir=args.output_dir,
        dataset_name=dataset_description,
        model_name=llm.model,
        max_iterations=args.max_iterations,
        eval_note=args.note,
    )

    critic = create_critic(args)
    logger.info("Using critic: %s", type(critic).__name__)

    metadata = EvalMetadata(
        llm=llm,
        dataset=dataset_label,
        dataset_split=args.split,
        max_iterations=args.max_iterations,
        eval_output_dir=structured_output_dir,
        details=None,
        prompt_path=None,
        eval_limit=args.n_limit,
        env_setup_commands=get_mirror_env_commands()
        + ["export PIP_CACHE_DIR=~/.cache/pip"],
        max_attempts=args.max_attempts,
        critic=critic,
        selected_instances_file=args.select,
        max_retries=args.max_retries,
        workspace_type=args.workspace,
        conversation_timeout=args.conversation_timeout,
        condenser=args.condenser,
        condenser_config=args.condenser_config,
    )

    evaluator = HarborEvaluation(
        metadata=metadata,
        num_workers=args.num_workers,
        tasks_path=str(tasks_path),
        docker_image_prefix=args.docker_image_prefix,
        agent_plugin_image=args.agent_plugin_image,
        verifier_enabled=not args.skip_verifier,
        primary_reward_key=args.primary_reward_key,
        workspace_memory_limit=args.workspace_memory_limit,
        agent_timeout_override=args.agent_timeout,
        verifier_timeout_override=args.verifier_timeout,
        detach_docker_logs=not args.no_detach_docker_logs,
    )
    evaluator.run(on_result=get_default_on_result_writer(evaluator.output_path))

    logger.info("Harbor evaluation completed!")


if __name__ == "__main__":
    main()
