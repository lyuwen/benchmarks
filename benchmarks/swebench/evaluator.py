"""
SWE-bench execution-based evaluator.

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

from benchmarks.utils.execution_evaluator import ExecutionBasedEvaluator

logger = logging.getLogger(__name__)


class SWEBenchEvaluator(ExecutionBasedEvaluator):
    """Evaluator that runs the SWE-bench test harness in Docker.

    Loads the SWE-bench dataset, creates a TestSpec for the instance,
    builds/reuses the Docker image, applies the patch, runs the eval
    script, and grades the result using SWE-bench's own grading logic.

    Config example (``--evaluator-config evaluator.json``)::

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
        default="evaluator",
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
        if self._dataset_cache is not None:
            return self._dataset_cache

        from swebench.harness.utils import load_swebench_dataset

        dataset = load_swebench_dataset(self.dataset_name, self.dataset_split)
        self._dataset_cache = {inst["instance_id"]: inst for inst in dataset}
        logger.info(
            "SWEBenchEvaluator: loaded %d instances from %s/%s",
            len(self._dataset_cache),
            self.dataset_name,
            self.dataset_split,
        )
        return self._dataset_cache

    def evaluate(
        self,
        instance_id: str,
        git_patch: str,
        instance_data: dict[str, Any],
    ) -> bool:
        if not git_patch or not git_patch.strip():
            logger.warning("Empty or missing git patch for %s", instance_id)
            return False

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
                "swebench package is not available (%s). Skipping evaluation.",
                e,
            )
            return False

        swebench_data = instance_data
        if "FAIL_TO_PASS" not in swebench_data:
            dataset = self._load_dataset()
            swebench_data = dataset.get(instance_id)
            if swebench_data is None:
                logger.error("Instance %s not found in dataset", instance_id)
                return False

        try:
            test_spec = make_test_spec(
                swebench_data,
                namespace=self.namespace,
                instance_image_tag=self.instance_image_tag,
                env_image_tag=self.env_image_tag,
            )

            pred = {
                KEY_INSTANCE_ID: instance_id,
                KEY_MODEL: "evaluator",
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

            resolved = result.get("completed", False) and result.get("resolved", False)
            logger.info(
                "SWEBenchEvaluator %s: completed=%s resolved=%s",
                instance_id,
                result.get("completed"),
                result.get("resolved"),
            )
            return resolved

        except Exception as e:
            logger.error(
                "SWEBenchEvaluator failed for %s: %s\n%s",
                instance_id,
                e,
                traceback.format_exc(),
            )
            return False
