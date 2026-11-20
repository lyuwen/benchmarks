"""Re-score recorded SWE-rebench-V2 runs with the current execution judge.

This module powers the ``scripts/swerebenchv2_rescore/`` SLURM workflow: it
splits a ``histories.jsonl`` file into one work item per instance, runs the
judge for a single item (so each work item can be scheduled as its own SLURM
array task), and merges the fresh verdicts back into a new histories file.

The judge verdict replaces the recorded ``evaluation`` field; every other field
of the history row is preserved byte-for-byte at the JSON level.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)

# Fields the execution judge reads from a dataset row.
TASK_KEYS = (
    "repo",
    "image_name",
    "install_config",
    "test_patch",
    "PASS_TO_PASS",
    "FAIL_TO_PASS",
)


def _instance_path(workdir: Path, index: int) -> Path:
    return workdir / "instances" / f"{index}.json"


def _task_path(workdir: Path, index: int) -> Path:
    return workdir / "tasks" / f"{index}.json"


def _result_path(workdir: Path, index: int) -> Path:
    return workdir / "results" / f"{index}.json"


def prepare_workdir(
    histories_path: str | Path,
    dataset_path: str | Path,
    workdir: str | Path,
) -> int:
    """Split histories + dataset rows into per-index work items.

    Writes, under ``workdir``:

    - ``instances/<index>.json``: ``instance_id``, ``old_evaluation``, ``git_patch``
    - ``tasks/<index>.json``: the dataset fields the judge needs
    - ``manifest.jsonl``: one line per index (for reporting/debugging)
    - ``count``: number of work items

    Indices follow the line order of ``histories.jsonl``. Returns the count.
    """
    workdir = Path(workdir)
    instances_dir = workdir / "instances"
    tasks_dir = workdir / "tasks"
    results_dir = workdir / "results"
    for directory in (instances_dir, tasks_dir, results_dir):
        directory.mkdir(parents=True, exist_ok=True)

    index_by_id: dict[str, int] = {}
    manifest_path = workdir / "manifest.jsonl"
    count = 0
    with (
        open(histories_path, encoding="utf-8") as histories,
        open(manifest_path, "w", encoding="utf-8") as manifest,
    ):
        for index, line in enumerate(histories):
            record = json.loads(line)
            instance_id = str(record.get("instance_id") or "")
            if not instance_id:
                raise ValueError(f"History row {index} has no instance_id")
            if instance_id in index_by_id:
                raise ValueError(f"Duplicate instance_id in histories: {instance_id}")
            instance = {
                "index": index,
                "instance_id": instance_id,
                "old_evaluation": record.get("evaluation"),
                "git_patch": (record.get("test_result") or {}).get("git_patch") or "",
            }
            _instance_path(workdir, index).write_text(
                json.dumps(instance), encoding="utf-8"
            )
            manifest.write(
                json.dumps(
                    {
                        "index": index,
                        "instance_id": instance_id,
                        "old_evaluation": instance["old_evaluation"],
                    }
                )
                + "\n"
            )
            index_by_id[instance_id] = index
            count = index + 1
            if count % 200 == 0:
                logger.info("read %d histories", count)

    missing = set(index_by_id)
    with open(dataset_path, encoding="utf-8") as dataset:
        for line in dataset:
            row = json.loads(line)
            instance_id = str(row.get("instance_id") or "")
            index = index_by_id.get(instance_id)
            if index is None:
                continue
            task = {key: row.get(key) for key in TASK_KEYS}
            _task_path(workdir, index).write_text(json.dumps(task), encoding="utf-8")
            missing.discard(instance_id)

    if missing:
        raise ValueError(
            f"{len(missing)} instance(s) from histories are missing in the dataset, "
            f"e.g. {sorted(missing)[:5]}"
        )

    (workdir / "count").write_text(str(count), encoding="utf-8")
    logger.info("prepared %d work items in %s", count, workdir)
    return count


def run_one(
    workdir: str | Path,
    index: int,
    *,
    docker_image_prefix: str | None = None,
    timeout: int = 1800,
    rm_image: bool = False,
) -> dict[str, Any]:
    """Run the judge for one prepared work item and write ``results/<index>.json``.

    ``evaluation`` is ``None`` when the judge could not produce a verdict; the
    caller is expected to retry or report those indices.
    """
    workdir = Path(workdir)
    instance = json.loads(_instance_path(workdir, index).read_text(encoding="utf-8"))
    task = json.loads(_task_path(workdir, index).read_text(encoding="utf-8"))

    # Imported lazily so listing/preparing work does not need the judge deps.
    from benchmarks.swerebenchv2.judge import SWERebenchV2Judge

    judge = SWERebenchV2Judge(
        timeout=timeout,
        rm_image=rm_image,
        docker_image_prefix=docker_image_prefix,
    )
    instance_id = str(instance["instance_id"])
    evaluation: bool | None
    error: str | None = None
    try:
        evaluation = judge.judge(instance_id, instance["git_patch"], task)
        if evaluation is None:
            error = "judge returned no verdict (see judge logs for details)"
    except Exception as exc:  # noqa: BLE001 - recorded and retried by the caller
        evaluation = None
        error = f"{type(exc).__name__}: {exc}"

    result: dict[str, Any] = {
        "index": index,
        "instance_id": instance_id,
        "old_evaluation": instance.get("old_evaluation"),
        "evaluation": evaluation,
        "error": error,
    }
    _result_path(workdir, index).write_text(
        json.dumps(result, indent=1), encoding="utf-8"
    )
    return result


def merge_histories(
    workdir: str | Path,
    histories_path: str | Path,
    output_path: str | Path,
    report_path: str | Path | None = None,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Write a copy of ``histories_path`` with re-scored ``evaluation`` values.

    Rows with a fresh boolean verdict get ``evaluation`` replaced; rows whose
    judge run errored or was never executed keep their original value. A JSON
    report summarises old/new transitions.
    """
    workdir = Path(workdir)
    counts: dict[str, int] = {
        "total": 0,
        "updated": 0,
        "recovered": 0,
        "regressed": 0,
        "unchanged": 0,
        "missing_result": 0,
        "errors": 0,
    }
    changes: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    missing_indices: list[int] = []
    error_indices: list[int] = []

    output = None
    if not dry_run:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output = open(output_path, "w", encoding="utf-8")

    try:
        with open(histories_path, encoding="utf-8") as histories:
            for index, line in enumerate(histories):
                record = json.loads(line)
                counts["total"] += 1
                result_file = _result_path(workdir, index)
                if result_file.is_file():
                    result = json.loads(result_file.read_text(encoding="utf-8"))
                else:
                    result = None

                new_value = None if result is None else result.get("evaluation")
                if new_value is None:
                    if result is not None:
                        counts["errors"] += 1
                        error_indices.append(index)
                        errors.append(
                            {
                                "index": index,
                                "instance_id": record.get("instance_id"),
                                "error": result.get("error"),
                            }
                        )
                    else:
                        counts["missing_result"] += 1
                        missing_indices.append(index)
                else:
                    old_value = bool(record.get("evaluation"))
                    new_bool = bool(new_value)
                    if new_bool != old_value:
                        counts["updated"] += 1
                        counts[("recovered" if new_bool else "regressed")] += 1
                        changes.append(
                            {
                                "index": index,
                                "instance_id": record.get("instance_id"),
                                "old_evaluation": old_value,
                                "new_evaluation": new_bool,
                            }
                        )
                    else:
                        counts["unchanged"] += 1
                    record["evaluation"] = new_bool

                if output is not None:
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
    finally:
        if output is not None:
            output.close()

    report: dict[str, Any] = {
        "workdir": str(workdir),
        "histories": str(histories_path),
        "output": str(output_path),
        "dry_run": dry_run,
        "counts": counts,
        "changes": changes,
        "errors": errors,
        "missing_indices": missing_indices,
        "error_indices": error_indices,
    }
    if report_path is not None:
        Path(report_path).write_text(
            json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8"
        )
    logger.info("merge summary: %s", counts)
    return report
