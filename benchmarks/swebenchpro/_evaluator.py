#!/usr/bin/env python3
"""Shared execution evaluator for SWE-bench Pro patches."""

import ast
import json
import re
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from benchmarks.swebenchpro import constants
from benchmarks.swebenchpro.build_images import get_official_docker_image

_BINARY_PATCH_MARKERS = (
    "GIT binary patch",
    "Binary files ",
)
_EXPORT_RE = re.compile(r"^\s*export\s+([A-Za-z_][A-Za-z0-9_]*)=(.*)$")


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


def _validate_harness_dir() -> Path:
    harness_dir = constants.HARNESS_SUBMODULE_PATH
    if not harness_dir.is_dir() or not any(harness_dir.iterdir()):
        raise FileNotFoundError(
            "Expected SWE-bench Pro harness checkout at "
            f"{constants.HARNESS_SUBMODULE_PATH}. "
            "Run: git submodule update --init benchmarks/swebenchpro/SWE-bench_Pro-os"
        )
    return harness_dir


def _load_text_file(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def _load_instance_assets(instance: Any) -> dict[str, Any]:
    row = getattr(instance, "data", instance)
    if not isinstance(row, dict):
        raise TypeError("instance must be a dict or expose a dict-like .data attribute")

    dockerhub_tag = str(row["dockerhub_tag"]).strip()
    base_commit = str(row["base_commit"]).strip()
    before_repo_set_cmd = str(row.get("before_repo_set_cmd", "") or "")
    selected_test_files_to_run = _parse_literal_list(
        row.get("selected_test_files_to_run", [])
    )
    fail_to_pass = _parse_literal_list(row.get("fail_to_pass", []))
    pass_to_pass = _parse_literal_list(row.get("pass_to_pass", []))

    repo = str(row.get("repo", "")).strip()
    repo_dirname = repo.rsplit("/", 1)[-1] if repo else "repo"
    instance_id = str(row.get("instance_id") or row.get("id") or repo_dirname).strip()

    return {
        "instance_id": instance_id,
        "repo": repo,
        "repo_dirname": repo_dirname,
        "dockerhub_tag": dockerhub_tag,
        "docker_image": get_official_docker_image(dockerhub_tag),
        "base_commit": base_commit,
        "before_repo_set_cmd": before_repo_set_cmd,
        "selected_test_files_to_run": selected_test_files_to_run,
        "fail_to_pass": fail_to_pass,
        "pass_to_pass": pass_to_pass,
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


def _extract_env_exports(before_repo_set_cmd: str) -> list[str]:
    exports: list[str] = []
    for line in before_repo_set_cmd.splitlines():
        match = _EXPORT_RE.match(line)
        if match:
            exports.append(f"{match.group(1)}={match.group(2).strip()}")
    return exports


def _create_entryscript(assets: dict[str, Any]) -> str:
    repo_dir = "/workspace/repo"
    selected_tests = assets["selected_test_files_to_run"]
    test_args = " ".join(shlex.quote(test) for test in selected_tests)
    before_repo_set_cmd = assets["before_repo_set_cmd"].strip()
    before_repo_block = f"\n{before_repo_set_cmd}\n" if before_repo_set_cmd else "\n"

    test_command = "python3 -m pytest --color=no --junitxml=/eval/junit.xml"
    if test_args:
        test_command = f"{test_command} {test_args}"

    script = f"""#!/bin/bash
set -euo pipefail
rm -rf {repo_dir}
mkdir -p {repo_dir}
cp -r /testbed/. {repo_dir}
cd {repo_dir}
git reset --hard
git checkout {shlex.quote(assets['base_commit'])}
git apply --whitespace=nowarn /eval/model.patch{before_repo_block}set +e
{test_command} >/eval/test.stdout 2>/eval/test.stderr
test_exit=$?
set -e
python3 - <<'PY'
import json
import os
import xml.etree.ElementTree as ET

passed = []
failed = []
skipped = []
xml_path = '/eval/junit.xml'
if os.path.exists(xml_path):
    root = ET.parse(xml_path).getroot()
    for testcase in root.iter('testcase'):
        classname = testcase.attrib.get('classname', '').strip()
        name = testcase.attrib.get('name', '').strip()
        test_name = '::'.join(part for part in (classname, name) if part)
        if testcase.find('failure') is not None or testcase.find('error') is not None:
            failed.append(test_name)
        elif testcase.find('skipped') is not None:
            skipped.append(test_name)
        else:
            passed.append(test_name)
payload = {{
    'passed_tests': passed,
    'failed_tests': failed,
    'skipped_tests': skipped,
    'exit_code': test_exit,
    'selected_test_files_to_run': {assets['selected_test_files_to_run']!r},
}}
with open('/eval/output.json', 'w', encoding='utf-8') as fh:
    json.dump(payload, fh)
PY
exit "$test_exit"
"""
    return script


def _run_in_container(
    assets: dict[str, Any],
    git_patch: str,
    timeout: int = 1800,
) -> dict[str, Any]:
    harness_dir = _validate_harness_dir()
    with tempfile.TemporaryDirectory(prefix="swebenchpro_eval_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        patch_path = tmp_path / "model.patch"
        script_path = tmp_path / "entry.sh"
        output_path = tmp_path / "output.json"

        patch_path.write_text(git_patch, encoding="utf-8")
        script_path.write_text(_create_entryscript(assets), encoding="utf-8")
        script_path.chmod(0o755)

        docker_cmd = [
            "docker",
            "run",
            "--rm",
            "--network",
            "host",
            "-v",
            f"{tmp_path}:/eval",
            "-v",
            f"{harness_dir}:/harness:ro",
            assets["docker_image"],
            "/bin/bash",
            "/eval/entry.sh",
        ]
        for export in _extract_env_exports(assets["before_repo_set_cmd"]):
            docker_cmd[1:1] = ["-e", export]

        try:
            completed = subprocess.run(
                docker_cmd,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            output_text = output_path.read_text(encoding="utf-8") if output_path.exists() else None
            return {
                "exit_code": completed.returncode,
                "stdout": completed.stdout or "",
                "stderr": completed.stderr or "",
                "output_text": output_text,
            }
        except subprocess.TimeoutExpired as exc:
            output_text = output_path.read_text(encoding="utf-8") if output_path.exists() else None
            return {
                "exit_code": 124,
                "stdout": exc.stdout or "",
                "stderr": exc.stderr or "",
                "output_text": output_text,
            }


def _score_result(
    assets: dict[str, Any],
    output_data: dict[str, Any],
    exit_code: int,
    git_patch: str,
) -> dict[str, Any]:
    passed_tests = _parse_literal_list(output_data.get("passed_tests", []))
    failed_tests = _parse_literal_list(output_data.get("failed_tests", []))
    skipped_tests = _parse_literal_list(output_data.get("skipped_tests", []))

    fail_to_pass_expected = set(assets["fail_to_pass"])
    pass_to_pass_expected = set(assets["pass_to_pass"])
    passed_set = set(passed_tests)

    from_fail_to_pass = sorted(passed_set & fail_to_pass_expected)
    failed_from_pass_to_pass = sorted(pass_to_pass_expected - passed_set)
    resolved = (
        fail_to_pass_expected.issubset(passed_set)
        and not failed_from_pass_to_pass
        and exit_code == 0
    )

    return {
        "instance_id": assets["instance_id"],
        "resolved": resolved,
        "exit_code": exit_code,
        "from_fail_to_pass": from_fail_to_pass,
        "fail_to_pass_total": len(fail_to_pass_expected),
        "failed_from_pass_to_pass": failed_from_pass_to_pass,
        "pass_to_pass_total": len(pass_to_pass_expected),
        "passed_tests": passed_tests,
        "failed_tests": failed_tests,
        "skipped_tests": skipped_tests,
        "test_result": {
            "git_patch": git_patch,
            "output": output_data,
        },
        "error": "",
    }


def evaluate_instance(
    instance: Any,
    git_patch: str,
    timeout: int = 1800,
) -> dict[str, Any]:
    assets = _load_instance_assets(instance)
    cleaned_patch = _strip_binary_hunks(git_patch)
    if not cleaned_patch:
        return {
            "instance_id": assets["instance_id"],
            "resolved": False,
            "exit_code": 0,
            "error": "empty git patch",
            "test_result": {"git_patch": ""},
        }

    run_result = _run_in_container(assets=assets, git_patch=cleaned_patch, timeout=timeout)
    if not run_result["output_text"]:
        return {
            "instance_id": assets["instance_id"],
            "resolved": False,
            "exit_code": run_result["exit_code"],
            "error": f"no output.json (exit_code={run_result['exit_code']})",
            "test_result": {"git_patch": cleaned_patch},
        }

    output_data = json.loads(run_result["output_text"])
    return _score_result(
        assets=assets,
        output_data=output_data,
        exit_code=run_result["exit_code"],
        git_patch=cleaned_patch,
    )
