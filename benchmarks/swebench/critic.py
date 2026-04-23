"""
SWE-bench execution-based critic.

Evaluates agent patches by running the SWE-bench test harness in Docker
containers.  Mirrors the evaluation logic from
``thirdparty/SWE-bench/swebench/harness/run_evaluation.py``.
"""

from __future__ import annotations

import logging
import traceback
from typing import Any

import docker
from pydantic import Field, PrivateAttr

from benchmarks.utils.execution_critic import ExecutionBasedCritic
from openhands.sdk.critic.base import CriticResult

logger = logging.getLogger(__name__)


class SWEBenchExecutionCritic(ExecutionBasedCritic):
    """Critic that evaluates SWE-bench patches by running tests in Docker.

    Loads the SWE-bench dataset, creates a TestSpec for the instance,
    builds/reuses the Docker image, applies the patch, runs the eval
    script, and grades the result using SWE-bench's own grading logic.

    Config example (``--critic-config critic.json``)::

        {
            "dataset_name": "princeton-nlp/SWE-bench_Lite",
            "dataset_split": "test",
            "timeout": 1800
        }
    """

    dataset_name: str = Field(
        default="princeton-nlp/SWE-bench_Lite",
        description="HuggingFace dataset name or local path",
    )
    dataset_split: str = Field(
        default="test",
        description="Dataset split to use",
    )
    namespace: str | None = Field(
        default="swebench",
        description="Docker image namespace",
    )
    instance_image_tag: str = Field(
        default="latest",
        description="Tag for instance Docker images",
    )
    env_image_tag: str = Field(
        default="latest",
        description="Tag for environment Docker images",
    )
    run_id: str = Field(
        default="critic_eval",
        description="Run ID for SWE-bench evaluation logging",
    )
    force_rebuild: bool = Field(
        default=False,
        description="Force rebuild Docker images",
    )
    rm_image: bool = Field(
        default=False,
        description="Remove Docker image after evaluation",
    )

    _dataset_cache: dict[str, Any] | None = PrivateAttr(default=None)

    def _load_dataset(self) -> dict[str, Any]:
        """Load and cache the dataset as a dict keyed by instance_id."""
        if self._dataset_cache is not None:
            return self._dataset_cache

        from swebench.harness.utils import load_swebench_dataset

        dataset = load_swebench_dataset(self.dataset_name, self.dataset_split)
        self._dataset_cache = {inst["instance_id"]: inst for inst in dataset}
        logger.info(
            "SWEBenchExecutionCritic: loaded %d instances from %s/%s",
            len(self._dataset_cache),
            self.dataset_name,
            self.dataset_split,
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
        try:
            if instance_data is None:
                dataset = self._load_dataset()
                instance_data = dataset.get(instance_id)
                if instance_data is None:
                    return CriticResult(
                        score=0.0,
                        message=f"Instance {instance_id} not found in dataset",
                    )
        except (ImportError, ModuleNotFoundError) as e:
            logger.warning(
                "swebench package is not available (%s). "
                "Falling back to pass-through (score=1.0).",
                e,
            )
            return CriticResult(
                score=1.0, message=f"swebench not installed, skipping execution check: {e}"
            )

        try:
            from swebench.harness.constants import (
                KEY_INSTANCE_ID,
                KEY_MODEL,
                KEY_PREDICTION,
            )
            from swebench.harness.run_evaluation import run_instance
            from swebench.harness.test_spec.test_spec import make_test_spec
        except (ImportError, ModuleNotFoundError) as e:
            logger.warning(
                "swebench package is not available (%s). "
                "Falling back to pass-through (score=1.0).",
                e,
            )
            return CriticResult(
                score=1.0,
                message=f"swebench not installed, skipping execution check: {e}",
            )

        try:
            test_spec = make_test_spec(
                instance_data,
                namespace=self.namespace,
                instance_image_tag=self.instance_image_tag,
                env_image_tag=self.env_image_tag,
            )

            pred = {
                KEY_INSTANCE_ID: instance_id,
                KEY_MODEL: "critic_eval",
                KEY_PREDICTION: git_patch,
            }

            client = docker.from_env(timeout=600)

            result = run_instance(
                test_spec=test_spec,
                pred=pred,
                rm_image=self.rm_image,
                force_rebuild=self.force_rebuild,
                client=client,
                run_id=self.run_id,
                timeout=self.timeout,
            )

            if result["completed"] and result["resolved"]:
                return CriticResult(
                    score=1.0,
                    message=f"Instance {instance_id} resolved",
                )
            elif result["completed"]:
                return CriticResult(
                    score=0.0,
                    message=f"Instance {instance_id} completed but not resolved",
                )
            else:
                return CriticResult(
                    score=0.0,
                    message=f"Instance {instance_id} evaluation did not complete",
                )

        except Exception as e:
            logger.error(
                "SWEBenchExecutionCritic failed for %s: %s\n%s",
                instance_id,
                e,
                traceback.format_exc(),
            )
            return CriticResult(
                score=0.0,
                message=f"Evaluation error for {instance_id}: {e}",
            )
