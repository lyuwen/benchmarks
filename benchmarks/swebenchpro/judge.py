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
    harness_dir: Path = Field(default_factory=lambda: constants.HARNESS_SUBMODULE_PATH)
    rm_image: bool = Field(default=False)
    block_network: bool = Field(default=False)
    docker_platform: str | None = Field(default=None)

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
            result = evaluate_instance(
                instance_data,
                git_patch,
                self.harness_dir,
                timeout=self.timeout,
                block_network=self.block_network,
                docker_platform=self.docker_platform,
                rm_image=self.rm_image,
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
