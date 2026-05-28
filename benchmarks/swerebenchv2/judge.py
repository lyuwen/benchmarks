"""
SWE-rebench-V2 execution-based judge.

Evaluates agent patches by applying them alongside the ground-truth test
patch in the instance's Docker image, running the test command, and parsing
results with the instance-specific log parser from
``thirdparty/SWE-rebench-V2/lib/agent/log_parsers.py``.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any

from pydantic import Field

from benchmarks.utils.execution_judge import ExecutionBasedJudge, register_judge

logger = logging.getLogger(__name__)

_V2_ROOT = Path(__file__).resolve().parents[2] / "thirdparty" / "SWE-rebench-V2"
_LIB_DIR = _V2_ROOT / "lib"

_TIMING_NORMALIZE_RES = [
    re.compile(r"\s*\[\s*\d+(?:\.\d+)?\s*(?:ms|s)\s*\]\s*$", re.IGNORECASE),
    re.compile(r"\s+in\s+\d+(?:\.\d+)?\s+(?:msec|sec)\b", re.IGNORECASE),
    re.compile(r"\s*\(\s*\d+(?:\.\d+)?\s*(?:ms|s)\s*\)\s*$", re.IGNORECASE),
]


def _normalize_test_name(name: str) -> str:
    for pattern in _TIMING_NORMALIZE_RES:
        name = pattern.sub("", name)
    return name.strip()


def _get_log_parsers():
    """Lazy-import V2 log parsers to avoid import-time side effects."""
    if str(_V2_ROOT) not in sys.path:
        sys.path.insert(0, str(_V2_ROOT))
    if str(_LIB_DIR) not in sys.path:
        sys.path.insert(0, str(_LIB_DIR))
    from agent import log_parsers
    return log_parsers


def _get_parser(parser_name: str):
    log_parsers = _get_log_parsers()
    parser = log_parsers.NAME_TO_PARSER.get(parser_name)
    if parser is None:
        parser = getattr(log_parsers, parser_name, None)
    if parser is None:
        raise ValueError(f"Unknown log parser: {parser_name}")
    return parser


@register_judge("swerebenchv2")
class SWERebenchV2Judge(ExecutionBasedJudge):
    """Judge that evaluates patches using V2's Docker images and log parsers.

    Applies the model patch and ground-truth test patch inside the instance
    Docker image, runs the configured test command, and checks FAIL_TO_PASS /
    PASS_TO_PASS expectations.
    """

    rm_image: bool = Field(
        default=False,
        description="Remove Docker image after evaluation",
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
            return self._run_eval(instance_id, git_patch, instance_data)
        except Exception as e:
            logger.error(
                "SWERebenchV2Judge failed for %s: %s\n%s",
                instance_id,
                e,
                traceback.format_exc(),
            )
            return None

    def _run_eval(
        self,
        instance_id: str,
        git_patch: str,
        instance_data: dict[str, Any],
    ) -> bool | None:
        repo = instance_data.get("repo", "")
        if not repo or "/" not in repo:
            logger.error("Instance %s missing repo field", instance_id)
            return None

        install_config = instance_data.get("install_config", {})
        if isinstance(install_config, str):
            install_config = json.loads(install_config)

        test_cmd = install_config.get("test_cmd", [])
        if isinstance(test_cmd, str):
            test_cmd = [test_cmd]
        test_cmds = [c for c in test_cmd if isinstance(c, str) and c.strip()]
        if not test_cmds:
            logger.error("Instance %s has no test_cmd", instance_id)
            return None

        parser_name = install_config.get("log_parser")
        if not parser_name:
            logger.error("Instance %s has no log_parser", instance_id)
            return None

        parser = _get_parser(parser_name)

        test_patch = instance_data.get("test_patch", "")
        if not test_patch:
            logger.error("Instance %s has no test_patch", instance_id)
            return None

        image = instance_data.get("image_name", "")
        if not image:
            logger.error("Instance %s has no image_name", instance_id)
            return None

        workdir = f"/{repo.split('/')[1]}"

        with tempfile.TemporaryDirectory(prefix="judge_patches_") as tmp:
            patch_dir = Path(tmp)
            (patch_dir / "patch.diff").write_text(git_patch, encoding="utf-8")
            (patch_dir / "test_patch.diff").write_text(test_patch, encoding="utf-8")

            cmd_lines = [
                "set -e",
                "git reset --hard HEAD",
                "git apply -v --3way --recount --ignore-space-change --whitespace=nowarn /patches/patch.diff",
                "git apply -v --3way --recount --ignore-space-change --whitespace=nowarn /patches/test_patch.diff",
            ]
            cmd_lines.extend(test_cmds)
            script = "\n".join(cmd_lines)

            docker_cmd = [
                "docker", "run", "--rm",
                "--network", "host",
                "-e", "_JAVA_OPTIONS=-Djava.net.preferIPv6Addresses=false",
                "-v", f"{patch_dir}:/patches:ro",
                "-w", workdir,
                image,
                "/bin/bash", "-c", script,
            ]

            result = subprocess.run(
                docker_cmd,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            output = (result.stdout or "") + (result.stderr or "")

        if self.rm_image:
            subprocess.run(
                ["docker", "rmi", "-f", image],
                check=False,
                capture_output=True,
            )

        parsed = parser(output)
        parsed = {_normalize_test_name(k): v for k, v in parsed.items()}
        passed = sorted(k for k, v in parsed.items() if v == "PASSED")

        expected_passed = sorted(
            _normalize_test_name(n)
            for n in instance_data.get("PASS_TO_PASS", [])
            + instance_data.get("FAIL_TO_PASS", [])
        )

        resolved = passed == expected_passed

        fail_to_pass = {_normalize_test_name(n) for n in instance_data.get("FAIL_TO_PASS", [])}
        f2p_passed = sorted(set(passed) & fail_to_pass)

        logger.info(
            "SWERebenchV2Judge %s: resolved=%s, F2P passed %d/%d, exit_code=%d",
            instance_id,
            resolved,
            len(f2p_passed),
            len(fail_to_pass),
            result.returncode,
        )
        return resolved
