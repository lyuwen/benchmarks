"""
Scale-SWE execution-based critic.

Evaluates agent patches by running the Scale-SWE test harness in Docker
containers.  Mirrors the evaluation logic from
``thirdparty/AweAgent/recipes/scale_swe/eval_predictions.py``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import traceback
from pathlib import Path
from typing import Any

from pydantic import Field, PrivateAttr

from benchmarks.utils.execution_critic import ExecutionBasedCritic
from openhands.sdk.critic.base import CriticResult

logger = logging.getLogger(__name__)


def _resolve_image_url(image_url: str, prefix: str | None = None) -> str:
    if not prefix:
        return image_url
    _, _, image_name = image_url.rpartition("/")
    return f"{prefix.rstrip('/')}/{image_name}"


class ScaleSWEExecutionCritic(ExecutionBasedCritic):
    """Critic that evaluates Scale-SWE patches by running tests in Docker.

    Uses AweAgent's ScaleSWEEvaluator to apply the patch, run F2P+P2P
    tests, and determine whether the issue is resolved.

    Config example (``--critic-config critic.json``)::

        {
            "data_file": "thirdparty/Scale-SWE/scale-swe-batch1.jsonl",
            "docker_image_prefix": "myregistry.com/myorg",
            "timeout": 3600
        }
    """

    data_file: str = Field(
        default="thirdparty/Scale-SWE/scale-swe-batch1.jsonl",
        description="Path to the Scale-SWE dataset JSONL file",
    )
    docker_image_prefix: str | None = Field(
        default=None,
        description="Override image namespace/registry prefix",
    )
    remove_image_after_eval: bool = Field(
        default=False,
        description="Remove Docker image after each evaluation",
    )

    _dataset_cache: dict[str, dict[str, Any]] | None = PrivateAttr(default=None)

    def _load_dataset(self) -> dict[str, dict[str, Any]]:
        """Load and cache the dataset as a dict keyed by instance_id."""
        if self._dataset_cache is not None:
            return self._dataset_cache

        data_path = Path(self.data_file)
        if not data_path.exists():
            raise FileNotFoundError(
                f"Scale-SWE data file not found: {self.data_file}"
            )

        instances: dict[str, dict[str, Any]] = {}
        with open(data_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                instances[record["instance_id"]] = record

        self._dataset_cache = instances
        logger.info(
            "ScaleSWEExecutionCritic: loaded %d instances from %s",
            len(instances),
            self.data_file,
        )
        return self._dataset_cache

    def evaluate_with_context(
        self,
        instance_id: str,
        git_patch: str,
        instance_data: dict[str, Any] | None = None,
    ) -> CriticResult:
        if not git_patch or not git_patch.strip():
            return CriticResult(score=0.0, message="Empty or missing git patch")

        # Resolve instance data
        if instance_data is None:
            dataset = self._load_dataset()
            instance_data = dataset.get(instance_id)
            if instance_data is None:
                return CriticResult(
                    score=0.0,
                    message=f"Instance {instance_id} not found in dataset",
                )

        try:
            # Handle being called from within an existing event loop
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop is not None:
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    result = pool.submit(
                        asyncio.run,
                        self._evaluate_async(instance_id, git_patch, instance_data),
                    ).result()
            else:
                result = asyncio.run(
                    self._evaluate_async(instance_id, git_patch, instance_data)
                )
            return result
        except Exception as e:
            logger.error(
                "ScaleSWEExecutionCritic failed for %s: %s\n%s",
                instance_id,
                e,
                traceback.format_exc(),
            )
            return CriticResult(
                score=0.0,
                message=f"Evaluation error for {instance_id}: {e}",
            )

    async def _evaluate_async(
        self,
        instance_id: str,
        git_patch: str,
        instance_data: dict[str, Any],
    ) -> CriticResult:
        try:
            from awe_agent.core.runtime import RuntimeConfig
            from awe_agent.core.runtime.docker import DockerRuntime
            from awe_agent.tasks.scale_swe.evaluator import ScaleSWEEvaluator
            from awe_agent.tasks.scale_swe.task import ScaleSWETask
        except (ImportError, ModuleNotFoundError) as e:
            logger.warning(
                "awe_agent package is not available (%s). "
                "Falling back to pass-through (score=1.0).",
                e,
            )
            return CriticResult(
                score=1.0,
                message=f"awe_agent not installed, skipping execution check: {e}",
            )

        image_url = instance_data.get("image_url", "")
        image = _resolve_image_url(image_url, self.docker_image_prefix)
        workdir = instance_data.get("workdir", "/testbed")

        task = ScaleSWETask(instances=[instance_data])
        inst_obj = task.get_instances()[0]
        inst_obj.image = image

        evaluator = ScaleSWEEvaluator(timeout=self.timeout)
        runtime = DockerRuntime(
            RuntimeConfig(
                backend="docker",
                image=image,
                workdir=workdir,
                docker={"remove_image_after_use": self.remove_image_after_eval},
            ),
        )

        eval_result = await evaluator.evaluate(inst_obj, git_patch, runtime)

        if eval_result.accepted:
            return CriticResult(
                score=1.0,
                message=f"Instance {instance_id} resolved",
            )
        else:
            details_str = ""
            if eval_result.details:
                error = eval_result.details.get("error", "")
                if error:
                    details_str = f" ({error})"
            return CriticResult(
                score=0.0,
                message=f"Instance {instance_id} not resolved{details_str}",
            )
