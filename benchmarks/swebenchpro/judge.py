from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any

from benchmarks.swebenchpro import constants
from benchmarks.swebenchpro._evaluator import evaluate_instance
from benchmarks.utils.execution_judge import ExecutionBasedJudge, register_judge
from openhands.sdk import get_logger
from pydantic import Field

logger = get_logger(__name__)


@register_judge("swebenchpro")
class SWEBenchProJudge(ExecutionBasedJudge):
    harness_dir: Path = Field(
        default_factory=lambda: constants.HARNESS_SUBMODULE_PATH,
        description="Reserved harness checkout path override for future evaluator support",
    )
    rm_image: bool = Field(
        default=False,
        description="Request Docker image cleanup after evaluation when evaluator supports it",
    )
    block_network: bool = Field(
        default=False,
        description="Disable network access during Docker-based judge evaluation",
    )
    docker_platform: str | None = Field(
        default=None,
        description="Optional Docker platform passed through to the evaluator",
    )
    mirror: str | None = Field(
        default=None,
        description="Package manager mirror configuration for faster installations",
    )

    def judge(
        self,
        instance_id: str,
        git_patch: str,
        instance_data: dict[str, Any],
    ) -> bool | None:
        if not git_patch or not git_patch.strip():
            logger.warning("Empty or missing git patch for %s", instance_id)
            return False

        try:
            if self.harness_dir != constants.HARNESS_SUBMODULE_PATH:
                logger.warning(
                    "SWEBenchProJudge %s configured harness_dir=%s, but the current "
                    "evaluator ignores harness overrides and uses %s",
                    instance_id,
                    self.harness_dir,
                    constants.HARNESS_SUBMODULE_PATH,
                )
            if self.rm_image:
                logger.warning(
                    "SWEBenchProJudge %s configured rm_image=%s, but the current "
                    "evaluator does not support image cleanup overrides yet",
                    instance_id,
                    self.rm_image,
                )

            result = evaluate_instance(
                instance_data,
                git_patch,
                timeout=self.timeout,
                block_network=self.block_network,
                docker_platform=self.docker_platform,
                remove_image=self.rm_image,
                mirror=self.mirror,
            )
            logger.info(
                "SWEBenchProJudge %s: resolved=%s exit_code=%s error=%s",
                instance_id,
                result.get("resolved"),
                result.get("exit_code"),
                result.get("error"),
            )
            return bool(result["resolved"])
        except Exception as exc:
            logger.error(
                "SWEBenchProJudge failed for %s: %s\n%s",
                instance_id,
                exc,
                traceback.format_exc(),
            )
            return None
