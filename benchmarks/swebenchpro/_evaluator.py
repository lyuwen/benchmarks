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
from benchmarks.swebenchpro.mirror_config import format_mirror_env_exports


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


def _strip_binary_hunks(patch: str) -> str:
    """Remove binary diff sections from a git patch."""
    if not patch:
        return patch

    sections = re.split(r'(?=^diff --git )', patch, flags=re.MULTILINE)

    kept: list[str] = []
    for section in sections:
        if not section.strip():
            continue
        if re.search(r'^Binary files .* differ$', section, re.MULTILINE):
            continue
        if re.search(r'^GIT binary patch$', section, re.MULTILINE):
            continue
        kept.append(section)

    return "".join(kept)


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
    """No normalization - return name as-is to match official evaluator."""
    return str(name)

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
    mirror: str | None = None,
) -> str:
    # Add package manager mirror configurations if specified
    from benchmarks.swebenchpro.mirror_config import format_mirror_env_exports
    mirror_env_block = format_mirror_env_exports(mirror)

    env_exports = _extract_env_exports(base_dockerfile) + _extract_env_exports(instance_dockerfile)
    export_block = "\n".join(env_exports)
    export_block = f"{export_block}\n" if export_block else ""

    selected_tests_list = _parse_literal_list(spec.get("selected_test_files_to_run", []))
    selected_tests = ",".join(selected_tests_list) if selected_tests_list else ""

    before_repo_set_cmd = str(spec.get("before_repo_set_cmd", "") or "").strip()
    if before_repo_set_cmd:
        before_repo_set_cmd = before_repo_set_cmd.split("\n")[-1].strip()
    before_repo_block = f"{before_repo_set_cmd}\n" if before_repo_set_cmd else ""

    base_commit = str(spec["base_commit"]).strip()

    return f"""#!/bin/bash
{mirror_env_block}{export_block}
# apply patch
cd /app
git reset --hard {shlex.quote(base_commit)}
git checkout {shlex.quote(base_commit)}
# Try multiple patch application strategies for robustness
if git apply -v /workspace/patch.diff 2>/dev/null; then
    echo "Patch applied successfully"
elif git reset --hard {shlex.quote(base_commit)} && git apply --reject -v /workspace/patch.diff 2>/dev/null; then
    echo "Patch applied with --reject"
elif git reset --hard {shlex.quote(base_commit)} && git apply --3way -v /workspace/patch.diff 2>/dev/null; then
    echo "Patch applied with 3-way merge"
elif git reset --hard {shlex.quote(base_commit)} && git apply --ignore-whitespace -v /workspace/patch.diff 2>/dev/null; then
    echo "Patch applied ignoring whitespace"
else
    git reset --hard {shlex.quote(base_commit)}
    echo "Warning: Patch application had issues, continuing anyway"
    git apply --reject --ignore-whitespace -v /workspace/patch.diff 2>&1 || true
fi
{before_repo_block}# run test and save stdout and stderr to separate files
bash /workspace/run_script.sh {selected_tests} > /workspace/stdout.log 2> /workspace/stderr.log
# run parsing script
python /workspace/parser.py /workspace/stdout.log /workspace/stderr.log /workspace/output.json
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
    finally:
        # Remove Docker image if requested
        if remove_image:
            try:
                subprocess.run(
                    ["docker", "rmi", "-f", image],
                    check=False, capture_output=True, timeout=60
                )
            except Exception:
                pass


def _score_result(
    spec: dict[str, Any],
    output_data: dict[str, Any],
    exit_code: int,
    git_patch: str,
) -> dict[str, Any]:
    """Score result matching official evaluator logic exactly.

    Official (swe_bench_pro_eval.py:560-563):
        passed_tests = {x["name"] for x in output["tests"] if x["status"] == "PASSED"}
        f2p = set(eval(raw_sample["fail_to_pass"]))
        p2p = set(eval(raw_sample["pass_to_pass"]))
        result = (f2p | p2p) <= passed_tests
    """
    instance_id = str(spec.get("instance_id") or spec.get("id") or "").strip()
    tests = output_data.get("tests") or []

    # NO normalization - match official exactly
    passed_tests = {x["name"] for x in tests if x.get("status") == "PASSED"}
    f2p = set(_parse_literal_list(spec.get("fail_to_pass", [])))
    p2p = set(_parse_literal_list(spec.get("pass_to_pass", [])))

    # Match official scoring
    result = (f2p | p2p) <= passed_tests

    return {
        "instance_id": instance_id,
        "resolved": result,
        "test_result": {"git_patch": git_patch},
        "tests": tests,
        "exit_code": exit_code,
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
    mirror: str | None = None,
) -> dict[str, Any]:
    spec = getattr(instance, "data", instance)
    if not isinstance(spec, dict):
        raise TypeError("instance must be a dict or expose a dict-like .data attribute")

    instance_id = str(spec.get("instance_id") or spec.get("id") or "").strip()
    cleaned_patch = _strip_binary_hunks(git_patch)

    if cleaned_patch != git_patch:
        print(f"Stripped binary diff hunks from patch for {instance_id}")

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
        (workspace_dir / "patch.diff").write_text(cleaned_patch + "\n", encoding="utf-8")
        (workspace_dir / "run_script.sh").write_text(assets["run_script"], encoding="utf-8")
        (workspace_dir / "parser.py").write_text(assets["parser"], encoding="utf-8")
        (workspace_dir / "entryscript.sh").write_text(
            _create_entryscript(spec, assets["base_dockerfile"], assets["instance_dockerfile"], mirror),
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
            remove_image=remove_image,
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
