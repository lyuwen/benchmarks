#!/usr/bin/env python3
"""
Evaluate model-generated patches against SWE-rebench-V2 ground truth.

Uses the SWE-rebench-V2 evaluation machinery (log parsers, Docker-based test
execution) to check whether patches produced by the inference harness actually
fix the target issues.

Example:
  uv run benchmarks/swerebenchv2/eval_infer.py \
    --predictions output/output.jsonl \
    --dataset thirdparty/SWE-rebench-V2/SWE-rebench-V2-py.jsonl \
    --report-json eval_report.json
"""

import argparse
import json
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from benchmarks.utils.dataset import get_dataset
from openhands.sdk import get_logger


logger = get_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
V2_ROOT = REPO_ROOT / "thirdparty" / "SWE-rebench-V2"
LIB_DIR = V2_ROOT / "lib"

sys.path.insert(0, str(V2_ROOT))
sys.path.insert(0, str(LIB_DIR))

from agent import log_parsers  # noqa: E402

_TIMING_NORMALIZE_RES = [
    re.compile(r"\s*\[\s*\d+(?:\.\d+)?\s*(?:ms|s)\s*\]\s*$", re.IGNORECASE),
    re.compile(r"\s+in\s+\d+(?:\.\d+)?\s+(?:msec|sec)\b", re.IGNORECASE),
    re.compile(r"\s*\(\s*\d+(?:\.\d+)?\s*(?:ms|s)\s*\)\s*$", re.IGNORECASE),
]


def _normalize_test_name(name: str) -> str:
    for pattern in _TIMING_NORMALIZE_RES:
        name = pattern.sub("", name)
    return name.strip()


def _get_parser(parser_name: str):
    parser = log_parsers.NAME_TO_PARSER.get(parser_name)
    if parser is None:
        parser = getattr(log_parsers, parser_name, None)
    if parser is None:
        raise ValueError(f"Unknown log parser: {parser_name}")
    return parser


def _normalize_command_list(value: object, field_name: str, instance_id: str) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        raise ValueError(f"Task {instance_id} missing {field_name}.")
    commands = [item for item in value if isinstance(item, str) and item.strip()]
    if not commands:
        raise ValueError(f"Task {instance_id} missing {field_name}.")
    return commands


def run_in_container(
    image: str,
    workdir: str,
    patch_dir: Path,
    patch_name: str,
    test_patch_name: str,
    test_cmds: list[str],
) -> tuple[int, str]:
    cmd_lines = [
        "set -e",
        "git reset --hard HEAD",
        f"git apply -v --3way --recount --ignore-space-change --whitespace=nowarn /patches/{patch_name}",
        f"git apply -v --3way --recount --ignore-space-change --whitespace=nowarn /patches/{test_patch_name}",
    ]
    cmd_lines.extend(test_cmds)
    script = "\n".join(cmd_lines)

    docker_cmd = [
        "docker",
        "run",
        "--rm",
        "--network", "host",
        "-e", "_JAVA_OPTIONS=-Djava.net.preferIPv6Addresses=false",
        "-v", f"{patch_dir}:/patches:ro",
        "-w", workdir,
        image,
        "/bin/bash",
        "-c",
        script,
    ]

    result = subprocess.run(docker_cmd, check=False, capture_output=True, text=True)
    output = (result.stdout or "") + (result.stderr or "")
    return result.returncode, output


def evaluate_instance(
    spec: dict,
    model_patch: str,
) -> dict:
    instance_id = spec.get("instance_id", "")
    repo = spec.get("repo", "")

    install_config = spec.get("install_config", {})
    if isinstance(install_config, str):
        install_config = json.loads(install_config)

    test_cmds = _normalize_command_list(
        install_config.get("test_cmd", []),
        "install_config.test_cmd",
        instance_id,
    )

    parser_name = install_config.get("log_parser")
    if not parser_name:
        raise ValueError(f"Task {instance_id} missing install_config.log_parser.")
    parser = _get_parser(parser_name)

    test_patch = spec.get("test_patch", "")
    if not model_patch or not test_patch:
        return {
            "instance_id": instance_id,
            "resolved": False,
            "error": "missing patch or test_patch",
        }

    image = spec.get("image_name", "")
    if not image:
        raise ValueError(f"Task {instance_id} missing image_name.")

    workdir = f"/{repo.split('/')[1]}"

    with tempfile.TemporaryDirectory(prefix="eval_patches_") as tmp:
        patch_dir = Path(tmp)
        (patch_dir / "patch.diff").write_text(model_patch, encoding="utf-8")
        (patch_dir / "test_patch.diff").write_text(test_patch, encoding="utf-8")
        exit_code, output = run_in_container(
            image=image,
            workdir=workdir,
            patch_dir=patch_dir,
            patch_name="patch.diff",
            test_patch_name="test_patch.diff",
            test_cmds=test_cmds,
        )

    parsed = parser(output)
    parsed = {_normalize_test_name(k): v for k, v in parsed.items()}
    passed = sorted(k for k, v in parsed.items() if v == "PASSED")

    expected_passed = sorted(
        _normalize_test_name(n)
        for n in spec.get("PASS_TO_PASS", []) + spec.get("FAIL_TO_PASS", [])
    )

    fail_to_pass_expected = {_normalize_test_name(n) for n in spec.get("FAIL_TO_PASS", [])}
    pass_to_pass_expected = {_normalize_test_name(n) for n in spec.get("PASS_TO_PASS", [])}
    passed_set = set(passed)

    from_fail_to_pass = sorted(passed_set & fail_to_pass_expected)
    failed_from_pass_to_pass = sorted(pass_to_pass_expected - passed_set)

    resolved = passed == expected_passed

    return {
        "instance_id": instance_id,
        "resolved": resolved,
        "exit_code": exit_code,
        "from_fail_to_pass": from_fail_to_pass,
        "fail_to_pass_total": len(fail_to_pass_expected),
        "failed_from_pass_to_pass": failed_from_pass_to_pass,
        "pass_to_pass_total": len(pass_to_pass_expected),
        "error": "",
    }


def load_predictions(path: Path) -> dict[str, str]:
    """Load model predictions from output.jsonl, extracting git_patch per instance."""
    predictions: dict[str, str] = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            instance_id = record.get("instance_id", "")
            git_patch = ""
            test_result = record.get("test_result", {})
            if isinstance(test_result, dict):
                git_patch = test_result.get("git_patch", "")
            predictions[instance_id] = git_patch
    return predictions


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate SWE-rebench-V2 model predictions."
    )
    parser.add_argument(
        "--predictions",
        required=True,
        help="Path to output.jsonl from inference run.",
    )
    parser.add_argument(
        "--dataset",
        default="thirdparty/SWE-rebench-V2/SWE-rebench-V2-py.jsonl",
        help="Path to dataset (JSONL or HuggingFace dataset name).",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Number of parallel evaluation workers.",
    )
    parser.add_argument(
        "--report-json",
        default="eval_report.json",
        help="Path to write JSON evaluation report.",
    )
    args = parser.parse_args()

    predictions = load_predictions(Path(args.predictions))
    logger.info("Loaded %d predictions", len(predictions))

    df = get_dataset(
        dataset_name=args.dataset,
        split=args.split,
    )
    specs_by_id = {}
    for _, row in df.iterrows():
        iid = str(row["instance_id"])
        if iid in predictions:
            specs_by_id[iid] = row.to_dict()

    logger.info(
        "Matched %d predictions to dataset instances (out of %d total in dataset)",
        len(specs_by_id),
        len(df),
    )

    results = []
    resolved_count = 0
    error_count = 0

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_id = {}
        for iid, spec in specs_by_id.items():
            future = executor.submit(evaluate_instance, spec, predictions[iid])
            future_to_id[future] = iid

        for future in as_completed(future_to_id):
            iid = future_to_id[future]
            try:
                result = future.result()
            except Exception as exc:
                logger.error("Evaluation failed for %s: %s", iid, exc)
                result = {"instance_id": iid, "resolved": False, "error": str(exc)}
                error_count += 1

            results.append(result)
            if result.get("resolved"):
                resolved_count += 1

            logger.info(
                "[%d/%d] %s: resolved=%s",
                len(results),
                len(specs_by_id),
                iid,
                result.get("resolved"),
            )

    results.sort(key=lambda r: r["instance_id"])

    report = {
        "total": len(results),
        "resolved": resolved_count,
        "errors": error_count,
        "resolve_rate": resolved_count / len(results) if results else 0,
        "items": results,
    }

    report_path = Path(args.report_json)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(
        "Evaluation complete: %d/%d resolved (%.1f%%). Report: %s",
        resolved_count,
        len(results),
        report["resolve_rate"] * 100,
        report_path,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
