#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_constants_module():
    constants_path = Path(__file__).resolve().parent / "constants.py"
    spec = importlib.util.spec_from_file_location("benchmarks.swebenchpro.constants", constants_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load constants module from {constants_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


constants = _load_constants_module()

try:
    from openhands.sdk import get_logger
except ModuleNotFoundError:
    get_logger = logging.getLogger

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
    docker_image_prefix: str | None,
    log_dir: str | None,
) -> dict[str, Any]:
    from benchmarks.swebenchpro._evaluator import evaluate_instance

    instance_id = str(row.get("instance_id") or row.get("id") or "").strip()
    try:
        result = evaluate_instance(
            row,
            git_patch,
            timeout=timeout,
            block_network=block_network,
            docker_platform=docker_platform,
            docker_image_prefix=docker_image_prefix,
            log_dir=log_dir,
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
        default=str(constants.HARNESS_SUBMODULE_PATH),
        help="Reserved for forward compatibility; current evaluator uses the built-in harness path",
    )
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--block-network", action="store_true")
    parser.add_argument("--docker-platform", default=None)
    parser.add_argument(
        "--docker-image-prefix",
        type=str,
        default=None,
        help="Override Docker image repository prefix (e.g., 'myregistry.com/myorg/sweap-images')",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Directory to save per-instance evaluation logs (stdout, stderr, output.json, etc.)",
    )
    parser.add_argument(
        "--rm-image",
        action="store_true",
        help="Reserved for forward compatibility; current evaluator does not remove images",
    )
    args = parser.parse_args()

    from benchmarks.utils.dataset import get_dataset

    predictions_path = Path(args.predictions)
    report_path = Path(args.report_json)

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

    # Create log directory if specified
    if args.log_dir:
        Path(args.log_dir).mkdir(parents=True, exist_ok=True)
        logger.info("Per-instance logs will be saved to %s", args.log_dir)

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
                args.docker_image_prefix,
                args.log_dir,
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
