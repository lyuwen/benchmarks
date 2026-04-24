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
from pydantic import Field

from benchmarks.utils.execution_evaluator import ExecutionBasedEvaluator

logger = logging.getLogger(__name__)


class SWEBenchEvaluator(ExecutionBasedEvaluator):
    """Evaluator that runs the SWE-bench test harness in Docker.

    Creates a TestSpec from the instance data already provided by the
    inference harness, builds/reuses the Docker image, applies the patch,
    runs the eval script, and grades the result using SWE-bench's grading.

    Config example (``--evaluator-config evaluator.json``)::

        {
            "timeout": 1800,
            "force_rebuild": false
        }
    """

    force_rebuild: bool = Field(
        default=False,
        description="Force rebuild Docker images",
    )
    rm_image: bool = Field(
        default=False,
        description="Remove Docker image after evaluation",
    )

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

        try:
            test_spec = make_test_spec(instance_data)

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
                run_id="evaluator",
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
