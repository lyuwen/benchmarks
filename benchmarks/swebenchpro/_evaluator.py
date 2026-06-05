#!/usr/bin/env python3
"""Shared execution evaluator for SWE-bench Pro patches."""

import ast
import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from benchmarks.swebenchpro import constants
from benchmarks.swebenchpro.build_images import get_official_docker_image


def _save_logs(log_dir: str, instance_id: str, workspace_dir: Path, result: dict[str, Any]) -> None:
    """Save evaluation logs for an instance."""
    instance_log_dir = Path(log_dir) / instance_id
    instance_log_dir.mkdir(parents=True, exist_ok=True)

    # Save all workspace files
    for file_name in ["stdout.log", "stderr.log", "output.json", "patch.diff", "entryscript.sh", "run_script.sh", "parser.py"]:
        src = workspace_dir / file_name
        if src.exists():
            shutil.copy2(src, instance_log_dir / file_name)

    # Save result JSON
    (instance_log_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


_BINARY_PATCH_MARKERS = (
    "GIT binary patch",
    "Binary files ",
)
_ENV_LINE_RE = re.compile(r"^\s*ENV\s+(.+?)\s*$")
_TIMING_SUFFIX_RE = re.compile(
    r"\s*(?:\[\s*\d+(?:\.\d+)?\s*(?:ms|s)\s*\]|\(\s*\d+(?:\.\d+)?\s*(?:ms|s)\s*\)|in\s+\d+(?:\.\d+)?\s+(?:msec|sec))\s*$",
    re.IGNORECASE,
)


def _parse_literal_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item).strip()]
    if not isinstance(value, str):
        raise TypeError(f"Expected string or list-like value, got {type(value).__name__}")

    stripped = value.strip()
    if not stripped:
        return []

    parsed = ast.literal_eval(stripped)
    if not isinstance(parsed, (list, tuple, set)):
        raise ValueError(f"Expected a literal list, got {type(parsed).__name__}")
    return [str(item) for item in parsed if str(item).strip()]


def _validate_harness_dir(harness_dir: str | Path) -> Path:
    harness_path = Path(harness_dir)
    if not harness_path.is_dir() or not any(harness_path.iterdir()):
        raise FileNotFoundError(
            "Expected SWE-bench Pro harness checkout at "
            f"{constants.HARNESS_SUBMODULE_PATH}. "
            "Run: git submodule update --init benchmarks/swebenchpro/SWE-bench_Pro-os"
        )
    return harness_path


def _load_instance_assets(harness_dir: str | Path, instance_id: str) -> dict[str, str]:
    harness_path = _validate_harness_dir(harness_dir)
    instance_dir = harness_path / "run_scripts" / instance_id
    base_docker_dir = harness_path / "dockerfiles" / "base_dockerfile" / instance_id
    instance_docker_dir = harness_path / "dockerfiles" / "instance_dockerfile" / instance_id

    asset_paths = {
        "run_script": instance_dir / "run_script.sh",
        "parser": instance_dir / "parser.py",
        "base_dockerfile": base_docker_dir / "Dockerfile",
        "instance_dockerfile": instance_docker_dir / "Dockerfile",
    }

    missing = [str(path) for path in asset_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing SWE-bench Pro harness assets for {instance_id}: {', '.join(missing)}"
        )

    return {
        name: path.read_text(encoding="utf-8")
        for name, path in asset_paths.items()
    }


def _strip_binary_hunks(git_patch: str) -> str:
    if not git_patch.strip():
        return ""

    lines = git_patch.splitlines(keepends=True)
    if not any(line.startswith("diff --git ") for line in lines):
        if any(marker in git_patch for marker in _BINARY_PATCH_MARKERS):
            return ""
        return git_patch.strip()

    sections: list[list[str]] = []
    current: list[str] = []
    for line in lines:
        if line.startswith("diff --git ") and current:
            sections.append(current)
            current = [line]
        else:
            current.append(line)
    if current:
        sections.append(current)

    kept_sections: list[str] = []
    for section in sections:
        section_text = "".join(section)
        if any(marker in section_text for marker in _BINARY_PATCH_MARKERS):
            continue
        kept_sections.append(section_text)
    return "".join(kept_sections).strip()


def _extract_env_exports(dockerfile_text: str) -> list[str]:
    exports: list[str] = []
    for raw_line in dockerfile_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _ENV_LINE_RE.match(line)
        if not match:
            continue

        payload = match.group(1).strip()
        tokens = shlex.split(payload)
        if not tokens:
            continue

        if all("=" in token for token in tokens):
            for token in tokens:
                key, value = token.split("=", 1)
                exports.append(f"export {key}={shlex.quote(value)}")
            continue

        if len(tokens) >= 2:
            key = tokens[0]
            value = " ".join(tokens[1:])
            exports.append(f"export {key}={shlex.quote(value)}")
    return exports


def _normalize_test_name(name: str) -> str:
    normalized = str(name).strip()
    if not normalized:
        return ""

    while True:
        updated = _TIMING_SUFFIX_RE.sub("", normalized)
        if updated == normalized:
            break
        normalized = updated.strip()

    parts = [part.strip() for part in normalized.split("::") if part.strip()]
    if not parts:
        return ""

    path_part = parts[0].replace("\\", "/")
    tail_parts = parts[1:]

    if path_part.endswith(".py"):
        path_part = re.sub(r"/+", "/", path_part).strip("/")
    elif "." in path_part:
        dotted_parts = [part for part in path_part.split(".") if part]
        if len(dotted_parts) >= 2:
            path_part = "/".join(dotted_parts[:-1]) + ".py"
            tail_parts = [dotted_parts[-1], *tail_parts]
        else:
            path_part = path_part.replace(".", "/")
        path_part = re.sub(r"/+", "/", path_part).strip("/")

    return "::".join([path_part, *tail_parts]) if tail_parts else path_part


def _create_entryscript(
    spec: dict[str, Any],
    base_dockerfile: str,
    instance_dockerfile: str,
) -> str:
    env_exports = _extract_env_exports(base_dockerfile) + _extract_env_exports(instance_dockerfile)
    export_block = "\n".join(env_exports)
    export_block = f"{export_block}\n" if export_block else ""

    selected_tests = " ".join(
        shlex.quote(test)
        for test in _parse_literal_list(spec.get("selected_test_files_to_run", []))
    )
    selected_tests_suffix = f" {selected_tests}" if selected_tests else ""

    before_repo_set_cmd = str(spec.get("before_repo_set_cmd", "") or "").strip()
    before_repo_block = f"{before_repo_set_cmd}\n" if before_repo_set_cmd else ""

    base_commit = str(spec["base_commit"]).strip()

    return f"""#!/bin/bash
set -euo pipefail
{export_block}cd /app
git reset --hard {shlex.quote(base_commit)}
git checkout {shlex.quote(base_commit)}
# Try to apply patch with various fallback strategies
if git apply --reject -v /workspace/patch.diff; then
    echo "Patch applied successfully"
elif git apply --reject --3way -v /workspace/patch.diff 2>/dev/null; then
    echo "Patch applied with 3-way merge"
elif git apply --reject --ignore-whitespace -v /workspace/patch.diff 2>/dev/null; then
    echo "Patch applied ignoring whitespace differences"
else
    echo "Patch applied with failures, continuing anyway. Check .rej files for failed hunks."
fi
{before_repo_block}set +e
bash /workspace/run_script.sh{selected_tests_suffix} > /workspace/stdout.log 2> /workspace/stderr.log
run_script_exit_code=$?
set -e
python /workspace/parser.py /workspace/stdout.log /workspace/stderr.log /workspace/output.json
exit "$run_script_exit_code"
"""


def _run_in_container(
    image: str,
    workspace_dir: str | Path,
    timeout: int = 3600,
    block_network: bool = False,
    docker_platform: str | None = None,
    remove_image: bool = False,
) -> dict[str, Any]:
    docker_cmd = ["docker", "run", "--rm"]
    if docker_platform:
        docker_cmd.extend(["--platform", docker_platform])
    if block_network:
        docker_cmd.extend(["--network", "none"])
    docker_cmd.extend(
        [
            "-v",
            f"{Path(workspace_dir)}:/workspace",
            "--entrypoint",
            "/bin/bash",
            image,
            "/workspace/entryscript.sh",
        ]
    )

    try:
        completed = subprocess.run(
            docker_cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "exit_code": completed.returncode,
            "stdout": completed.stdout or "",
            "stderr": completed.stderr or "",
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "exit_code": 124,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
        }


def _score_result(
    spec: dict[str, Any],
    output_data: dict[str, Any],
    exit_code: int,
    git_patch: str,
) -> dict[str, Any]:
    passed_tests = _parse_literal_list(output_data.get("passed_tests", []))
    if not passed_tests:
        for key in ("test_status_map", "tests_status", "tests"):
            statuses = output_data.get(key)
            if isinstance(statuses, dict):
                passed_tests = [
                    str(name)
                    for name, status in statuses.items()
                    if str(status).strip().upper() in {"PASSED", "PASS"}
                ]
                if passed_tests:
                    break

    fail_to_pass_expected = {
        _normalize_test_name(name)
        for name in _parse_literal_list(spec.get("fail_to_pass", []))
        if _normalize_test_name(name)
    }
    pass_to_pass_expected = {
        _normalize_test_name(name)
        for name in _parse_literal_list(spec.get("pass_to_pass", []))
        if _normalize_test_name(name)
    }
    passed_set = {
        _normalize_test_name(name)
        for name in passed_tests
        if _normalize_test_name(name)
    }

    from_fail_to_pass = sorted(passed_set & fail_to_pass_expected)
    failed_from_pass_to_pass = sorted(pass_to_pass_expected - passed_set)
    resolved = (
        fail_to_pass_expected.issubset(passed_set)
        and not failed_from_pass_to_pass
        and exit_code == 0
    )

    return {
        "instance_id": str(spec.get("instance_id") or spec.get("id") or "").strip(),
        "resolved": resolved,
        "exit_code": exit_code,
        "from_fail_to_pass": from_fail_to_pass,
        "fail_to_pass_total": len(fail_to_pass_expected),
        "failed_from_pass_to_pass": failed_from_pass_to_pass,
        "pass_to_pass_total": len(pass_to_pass_expected),
        "passed_tests": sorted(passed_set),
        "test_result": {
            "git_patch": git_patch,
            "output": output_data,
        },
        "error": "",
    }


def evaluate_instance(
    instance: Any,
    git_patch: str,
    timeout: int = 3600,
    block_network: bool = False,
    docker_platform: str | None = None,
    docker_image_prefix: str | None = None,
    log_dir: str | None = None,
    remove_image: bool = False,
) -> dict[str, Any]:
    spec = getattr(instance, "data", instance)
    if not isinstance(spec, dict):
        raise TypeError("instance must be a dict or expose a dict-like .data attribute")

    cleaned_patch = _strip_binary_hunks(git_patch)
    instance_id = str(spec.get("instance_id") or spec.get("id") or "").strip()
    if not cleaned_patch:
        return {
            "instance_id": instance_id,
            "resolved": False,
            "exit_code": 0,
            "error": "empty git patch",
            "test_result": {"git_patch": ""},
        }

    harness_dir = _validate_harness_dir(constants.HARNESS_SUBMODULE_PATH)
    assets = _load_instance_assets(harness_dir, instance_id)
    image = get_official_docker_image(str(spec["dockerhub_tag"]), docker_image_prefix)

    with tempfile.TemporaryDirectory(prefix="swebenchpro_eval_") as tmp_dir:
        workspace_dir = Path(tmp_dir)
        (workspace_dir / "patch.diff").write_text(cleaned_patch if cleaned_patch.endswith("\n") else cleaned_patch + "\n", encoding="utf-8")
        (workspace_dir / "run_script.sh").write_text(assets["run_script"], encoding="utf-8")
        (workspace_dir / "parser.py").write_text(assets["parser"], encoding="utf-8")
        (workspace_dir / "entryscript.sh").write_text(
            _create_entryscript(spec, assets["base_dockerfile"], assets["instance_dockerfile"]),
            encoding="utf-8",
        )
        (workspace_dir / "run_script.sh").chmod(0o755)
        (workspace_dir / "entryscript.sh").chmod(0o755)

        run_result = _run_in_container(
            image=image,
            workspace_dir=workspace_dir,
            timeout=timeout,
            block_network=block_network,
            docker_platform=docker_platform,
        )

        output_path = workspace_dir / "output.json"
        if not output_path.is_file():
            result = {
                "instance_id": instance_id,
                "resolved": False,
                "exit_code": run_result["exit_code"],
                "error": f"no output.json (exit_code={run_result['exit_code']})",
                "test_result": {"git_patch": cleaned_patch},
                "docker_stdout": run_result.get("stdout", ""),
                "docker_stderr": run_result.get("stderr", ""),
            }
            if log_dir:
                _save_logs(log_dir, instance_id, workspace_dir, result)
            return result

        output_data = json.loads(output_path.read_text(encoding="utf-8"))
        result = _score_result(
            spec=spec,
            output_data=output_data,
            exit_code=run_result["exit_code"],
            git_patch=cleaned_patch,
        )

        if log_dir:
            _save_logs(log_dir, instance_id, workspace_dir, result)

        return result
