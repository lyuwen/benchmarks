#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
import traceback
import types
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _install_openhands_shims() -> None:
    openhands_module = types.ModuleType("openhands")
    sdk_module = types.ModuleType("openhands.sdk")
    tools_module = types.ModuleType("openhands.tools")
    file_editor_module = types.ModuleType("openhands.tools.file_editor")
    task_tracker_module = types.ModuleType("openhands.tools.task_tracker")
    terminal_module = types.ModuleType("openhands.tools.terminal")

    def _shim_get_logger(name: str):
        return logging.getLogger(name)

    sdk_module.get_logger = _shim_get_logger
    file_editor_module.FileEditorTool = type("FileEditorTool", (), {})
    task_tracker_module.TaskTrackerTool = type("TaskTrackerTool", (), {})
    terminal_module.TerminalTool = type("TerminalTool", (), {})

    openhands_module.sdk = sdk_module
    openhands_module.tools = tools_module
    tools_module.file_editor = file_editor_module
    tools_module.task_tracker = task_tracker_module
    tools_module.terminal = terminal_module

    sys.modules.setdefault("openhands", openhands_module)
    sys.modules.setdefault("openhands.sdk", sdk_module)
    sys.modules.setdefault("openhands.tools", tools_module)
    sys.modules.setdefault("openhands.tools.file_editor", file_editor_module)
    sys.modules.setdefault("openhands.tools.task_tracker", task_tracker_module)
    sys.modules.setdefault("openhands.tools.terminal", terminal_module)


try:
    from openhands.sdk import get_logger
except ModuleNotFoundError:
    _install_openhands_shims()
    from openhands.sdk import get_logger


def _runtime_harness_dir_default() -> str:
    return str(Path(__file__).resolve().parent / "SWE-bench_Pro-os")


def _import_constants():
    from benchmarks.swebenchpro import constants

    return constants


def _import_evaluate_instance():
    from benchmarks.swebenchpro._evaluator import evaluate_instance

    return evaluate_instance


def _import_get_dataset():
    from benchmarks.utils.dataset import get_dataset

    return get_dataset

logger = get_logger(__name__)


def load_predictions(path: Path) -> dict[str, str]:
    predictions: dict[str, str] = {}

    with path.open("r", encoding="utf-8") as handle:
        for line_num, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line:
                continue

            record = json.loads(line)
            instance_id = str(record.get("instance_id") or "").strip()
            if not instance_id:
                logger.warning(
                    "Skipping predictions line %s with missing instance_id", line_num
                )
                continue

            test_result = record.get("test_result") or {}
            if not isinstance(test_result, dict):
                logger.warning(
                    "Skipping predictions line %s for %s with invalid test_result",
                    line_num,
                    instance_id,
                )
                continue

            git_patch = test_result.get("git_patch")
            if git_patch is None:
                logger.warning(
                    "Skipping predictions line %s for %s with missing git_patch",
                    line_num,
                    instance_id,
                )
                continue

            predictions[instance_id] = str(git_patch)

    return predictions


def _evaluate_row(
    row: dict[str, Any],
    git_patch: str,
    timeout: int,
    block_network: bool,
    docker_platform: str | None,
) -> dict[str, Any]:
    instance_id = str(row.get("instance_id") or row.get("id") or "").strip()
    evaluate_instance = _import_evaluate_instance()
    try:
        result = evaluate_instance(
            row,
            git_patch,
            timeout=timeout,
            block_network=block_network,
            docker_platform=docker_platform,
        )
        if "instance_id" not in result or not result.get("instance_id"):
            result["instance_id"] = instance_id
        return result
    except Exception as exc:
        logger.error("Evaluation failed for %s: %s", instance_id, exc)
        return {
            "instance_id": instance_id,
            "resolved": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "test_result": {"git_patch": git_patch},
        }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the SWE-bench Pro offline evaluator on output.jsonl predictions"
    )
    parser.add_argument("--predictions", required=True, help="Path to output.jsonl")
    parser.add_argument("--dataset", default="ScaleAI/SWE-bench_Pro")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--report-json", default="eval_report.json")
    parser.add_argument(
        "--harness-dir",
        default=_runtime_harness_dir_default(),
        help="Reserved for forward compatibility; current evaluator uses the built-in harness path",
    )
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--block-network", action="store_true")
    parser.add_argument("--docker-platform", default=None)
    parser.add_argument(
        "--rm-image",
        action="store_true",
        help="Reserved for forward compatibility; current evaluator does not remove images",
    )
    args = parser.parse_args()

    predictions_path = Path(args.predictions)
    report_path = Path(args.report_json)
    constants = _import_constants()

    if args.harness_dir != str(constants.HARNESS_SUBMODULE_PATH):
        logger.warning(
            "Ignoring --harness-dir=%s because the current evaluator always uses %s",
            args.harness_dir,
            constants.HARNESS_SUBMODULE_PATH,
        )
    if args.rm_image:
        logger.warning(
            "Ignoring --rm-image because the current evaluator does not support image cleanup overrides"
        )

    predictions = load_predictions(predictions_path)
    logger.info("Loaded %s predictions from %s", len(predictions), predictions_path)

    get_dataset = _import_get_dataset()
    dataset = get_dataset(dataset_name=args.dataset, split=args.split)
    dataset_rows = dataset.to_dict(orient="records")
    matched_rows = [
        row
        for row in dataset_rows
        if str(row.get("instance_id") or row.get("id") or "").strip() in predictions
    ]
    logger.info(
        "Loaded %s dataset rows from %s/%s; evaluating %s matched instances",
        len(dataset_rows),
        args.dataset,
        args.split,
        len(matched_rows),
    )

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_instance = {
            executor.submit(
                _evaluate_row,
                row,
                predictions[str(row.get("instance_id") or row.get("id") or "").strip()],
                args.timeout,
                args.block_network,
                args.docker_platform,
            ): str(row.get("instance_id") or row.get("id") or "").strip()
            for row in matched_rows
        }
        for future in as_completed(future_to_instance):
            instance_id = future_to_instance[future]
            result = future.result()
            results.append(result)
            logger.info(
                "Finished %s: resolved=%s error=%s",
                instance_id,
                result.get("resolved"),
                result.get("error"),
            )

    results.sort(key=lambda item: str(item.get("instance_id") or ""))
    resolved_count = sum(1 for item in results if item.get("resolved") is True)
    error_count = sum(1 for item in results if item.get("error"))
    report = {
        "total": len(results),
        "resolved": resolved_count,
        "errors": error_count,
        "resolve_rate": (resolved_count / len(results)) if results else 0.0,
        "items": results,
    }

    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    logger.info(
        "Wrote SWE-bench Pro evaluation report to %s (total=%s resolved=%s errors=%s resolve_rate=%.4f)",
        report_path,
        report["total"],
        report["resolved"],
        report["errors"],
        report["resolve_rate"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
