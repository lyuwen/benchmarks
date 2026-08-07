"""
SWE-rebench-V2 execution-based judge.

Evaluates agent patches by applying them alongside the ground-truth test
patch in the instance's Docker image, running the test command, and parsing
results with the instance-specific log parser from the vendored
``benchmarks/swerebenchv2/lib/agent/log_parsers.py``.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import sys
import tempfile
import traceback
import uuid
from pathlib import Path
from typing import Any

from pydantic import Field
from unidiff import PatchSet

from benchmarks.utils.execution_judge import ExecutionBasedJudge, register_judge

logger = logging.getLogger(__name__)

# Vendored copy of the SWE-rebench-V2 log parsers, kept alongside this task so
# evaluation does not depend on the (large) thirdparty checkout being present.
_LIB_PARENT = Path(__file__).resolve().parent

_TIMING_NORMALIZE_RES = [
    re.compile(r"\s*\[\s*\d+(?:\.\d+)?\s*(?:ms|s)\s*\]\s*$", re.IGNORECASE),
    re.compile(r"\s+in\s+\d+(?:\.\d+)?\s+(?:msec|sec)\b", re.IGNORECASE),
    re.compile(r"\s*\(\s*\d+(?:\.\d+)?\s*(?:ms|s)\s*\)\s*$", re.IGNORECASE),
]


def _normalize_test_name(name: str) -> str:
    for pattern in _TIMING_NORMALIZE_RES:
        name = pattern.sub("", name)
    return name.strip()


def _normalize_path(path: str) -> str | None:
    if path == "/dev/null":
        return None
    return path.removeprefix("a/").removeprefix("b/")


def _get_patch_files(patch: str) -> set[str]:
    files: set[str] = set()
    for f in PatchSet(patch):
        for path in (f.source_file, f.target_file):
            if path := _normalize_path(path):
                files.add(path)
    return files


def _strip_test_files(fix_patch: str, test_patch: str) -> tuple[str, list[str]]:
    """Drop from the fix patch any file also touched by the test patch.

    Keeps the candidate solution from smuggling edits to the test files that
    the ground-truth test patch owns. Returns the filtered fix patch plus the
    sorted list of stripped file paths.
    """
    test_files = _get_patch_files(test_patch)

    kept: list[str] = []
    stripped: list[str] = []

    for f in PatchSet(fix_patch):
        touched = {
            path
            for path in (
                _normalize_path(f.source_file),
                _normalize_path(f.target_file),
            )
            if path
        }
        overlap = touched & test_files
        if overlap:
            stripped.extend(overlap)
        else:
            kept.append(str(f))

    return "".join(kept), sorted(set(stripped))


def _resolve_image_name(image_name: str, prefix: str | None = None) -> str:
    """Override an image's registry/namespace prefix, keeping name and tag.

    Mirrors ``get_official_docker_image`` so judge evaluation targets the
    same image the inference workspace was built from.
    """
    if not prefix:
        return image_name
    _, _, name = image_name.rpartition("/")
    return f"{prefix.rstrip('/')}/{name}"


def _get_log_parsers():
    """Lazy-import vendored V2 log parsers to avoid import-time side effects."""
    if str(_LIB_PARENT) not in sys.path:
        sys.path.insert(0, str(_LIB_PARENT))
    from lib.agent import log_parsers
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
    force_rebuild: bool = Field(
        default=False,
        description="Force rebuild Docker images (unused, for CLI compat)",
    )
    docker_image_prefix: str | None = Field(
        default=None,
        description="Override image namespace/registry prefix",
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
        image = _resolve_image_name(image, self.docker_image_prefix)

        workdir = f"/{repo.split('/')[-1]}"

        clean_patch, stripped_files = _strip_test_files(
            fix_patch=git_patch,
            test_patch=test_patch,
        )
        if stripped_files:
            logger.info(
                "Instance %s: stripped test-file edits from candidate patch: %s",
                instance_id,
                stripped_files,
            )

        # Unique container name so we can force-kill it on timeout: since
        # subprocess.run's timeout only kills the docker *client*, the --rm
        # container would otherwise keep running (and holding the image).
        container_name = f"judge_{instance_id}_{uuid.uuid4().hex[:8]}"
        timed_out = False

        with tempfile.TemporaryDirectory(prefix="judge_patches_") as tmp:
            patch_dir = Path(tmp)
            (patch_dir / "patch.diff").write_text(clean_patch, encoding="utf-8")
            (patch_dir / "test_patch.diff").write_text(test_patch, encoding="utf-8")

            cmd_lines = [
                "set -e",
                "git reset --hard HEAD",
                "git apply -v --3way --recount --ignore-space-change --whitespace=nowarn /patches/patch.diff",
                "git apply -v --3way --recount --ignore-space-change --whitespace=nowarn /patches/test_patch.diff",
                "set +e",
            ]
            cmd_lines.extend(test_cmds)
            script = "\n".join(cmd_lines)

            docker_cmd = [
                "docker", "run", "--rm",
                "--name", container_name,
                "--network", "host",
                "-e", "_JAVA_OPTIONS=-Djava.net.preferIPv6Addresses=false",
                "-v", f"{patch_dir}:/patches:ro",
                "-w", workdir,
                image,
                "/bin/bash", "-c", script,
            ]

            try:
                result = subprocess.run(
                    docker_cmd,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                )
                output = (result.stdout or "") + (result.stderr or "")
                exit_code = result.returncode
            except subprocess.TimeoutExpired as exc:
                timed_out = True
                # Capture whatever partial output was produced before the kill.
                # exc.stdout/stderr may be bytes even in text mode.
                def _as_text(chunk: object) -> str:
                    if chunk is None:
                        return ""
                    if isinstance(chunk, bytes):
                        return chunk.decode("utf-8", errors="replace")
                    return str(chunk)

                output = _as_text(exc.stdout) + _as_text(exc.stderr)
                exit_code = -1
                logger.warning(
                    "SWERebenchV2Judge %s: test run timed out after %ss; killing container %s",
                    instance_id,
                    self.timeout,
                    container_name,
                )
                # Force-kill the still-running container (--rm cleans it up).
                subprocess.run(
                    ["docker", "kill", container_name],
                    check=False,
                    capture_output=True,
                )

        if self.rm_image:
            subprocess.run(
                ["docker", "rmi", "-f", image],
                check=False,
                capture_output=True,
            )

        if timed_out:
            return False

        parsed = parser(output)
        parsed = {_normalize_test_name(k): v for k, v in parsed.items()}
        passed = sorted(k for k, v in parsed.items() if v == "PASSED")

        pass_to_pass = instance_data.get("PASS_TO_PASS", [])
        if isinstance(pass_to_pass, str):
            pass_to_pass = json.loads(pass_to_pass)
        pass_to_pass = list(pass_to_pass)
        fail_to_pass_list = instance_data.get("FAIL_TO_PASS", [])
        if isinstance(fail_to_pass_list, str):
            fail_to_pass_list = json.loads(fail_to_pass_list)
        fail_to_pass_list = list(fail_to_pass_list)

        expected_passed = sorted(
            _normalize_test_name(n)
            for n in pass_to_pass + fail_to_pass_list
        )

        resolved = passed == expected_passed

        fail_to_pass = {_normalize_test_name(n) for n in fail_to_pass_list}
        f2p_passed = sorted(set(passed) & fail_to_pass)

        logger.info(
            "SWERebenchV2Judge %s: resolved=%s, F2P passed %d/%d, exit_code=%d",
            instance_id,
            resolved,
            len(f2p_passed),
            len(fail_to_pass),
            exit_code,
        )
        return resolved
