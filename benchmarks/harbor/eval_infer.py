"""Summarize a completed ``harbor-infer`` run without rerunning inference.

``harbor-eval`` reads the persisted attempt records (``output.critic_attempt_*.jsonl``,
falling back to ``output.jsonl``/``output_errors.jsonl``) and the selected-task
manifest, then reports:

* selected task count, recorded/attempt counts, and scored coverage;
* counts by agent status (completed/timeout/error) and verifier status
  (scored/verifier_error/verifier_timeout/unscored);
* infrastructure-error counts from the attempt/error records without
  double-counting the callback copies;
* the primary-score mean over scored tasks, with its explicit denominator.

Phase 2 adds replay grading; this command never re-executes a verifier.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


SUMMARY_FILENAME = "summary.json"
_MANIFEST_FILENAME = "harbor_tasks.json"
_MAIN_OUTPUT = "output.jsonl"
_ERROR_OUTPUT = "output_errors.jsonl"
_ATTEMPT_GLOB = "output.critic_attempt_*.jsonl"

VERIFIER_STATUSES = (
    "scored",
    "verifier_error",
    "verifier_timeout",
    "unscored",
    "missing",
)
AGENT_STATUSES = ("completed", "timeout", "error", "missing")


@dataclass
class InstanceSummary:
    instance_id: str
    attempts: int = 0
    attempt_numbers: list[int] = field(default_factory=list)
    agent_status: str = "missing"
    verifier_status: str = "missing"
    primary_reward: float | int | None = None
    verifier_reward: float | int | None = None
    top_level_error: str | None = None
    verifier_error: str | None = None
    trial_dir: str | None = None
    task_relative_path: str | None = None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


def _attempt_number(path: Path) -> int:
    stem = path.stem  # output.critic_attempt_2
    try:
        return int(stem.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return 1


def load_attempt_records(run_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """Load per-attempt records, keyed by ``instance_id``.

    Attempt files are the canonical record store (they include error rows).
    Only when no attempt file exists do we fall back to the aggregated
    ``output.jsonl`` plus ``output_errors.jsonl``; the latter is a callback copy
    of rows already present in the attempt files and is never counted twice.
    """
    records: dict[str, list[dict[str, Any]]] = {}
    attempt_files = sorted(run_dir.glob(_ATTEMPT_GLOB), key=_attempt_number)
    if attempt_files:
        for path in attempt_files:
            attempt = _attempt_number(path)
            for record in _read_jsonl(path):
                instance_id = record.get("instance_id")
                if not isinstance(instance_id, str):
                    continue
                record = dict(record)
                record.setdefault("_attempt", attempt)
                records.setdefault(instance_id, []).append(record)
        return records

    for path in (run_dir / _MAIN_OUTPUT, run_dir / _ERROR_OUTPUT):
        for record in _read_jsonl(path):
            instance_id = record.get("instance_id")
            if not isinstance(instance_id, str):
                continue
            record = dict(record)
            record.setdefault("_attempt", int(record.get("attempt") or 1))
            records.setdefault(instance_id, []).append(record)
    return records


def _verifier_payload(record: dict[str, Any]) -> dict[str, Any]:
    test_result = record.get("test_result")
    if not isinstance(test_result, dict):
        return {}
    verifier = test_result.get("verifier")
    return verifier if isinstance(verifier, dict) else {}


def _agent_payload(record: dict[str, Any]) -> dict[str, Any]:
    test_result = record.get("test_result")
    if not isinstance(test_result, dict):
        return {}
    agent = test_result.get("agent")
    return agent if isinstance(agent, dict) else {}


def _trial_payload(record: dict[str, Any]) -> dict[str, Any]:
    test_result = record.get("test_result")
    if not isinstance(test_result, dict):
        return {}
    trial = test_result.get("trial")
    return trial if isinstance(trial, dict) else {}


def _task_payload(record: dict[str, Any]) -> dict[str, Any]:
    test_result = record.get("test_result")
    if not isinstance(test_result, dict):
        return {}
    task = test_result.get("task")
    return task if isinstance(task, dict) else {}


def _is_scored_with_primary(record: dict[str, Any], primary_key: str) -> bool:
    verifier = _verifier_payload(record)
    if verifier.get("status") != "scored":
        return False
    rewards = verifier.get("rewards")
    if not isinstance(rewards, dict):
        return False
    value = rewards.get(primary_key)
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _record_rank(record: dict[str, Any], primary_key: str) -> int:
    verifier = _verifier_payload(record)
    status = verifier.get("status")
    if status == "scored" and _is_scored_with_primary(record, primary_key):
        return 3
    if status == "scored":
        return 2
    if record.get("error"):
        return 0
    return 1


def select_representative(
    records: list[dict[str, Any]], primary_key: str
) -> dict[str, Any] | None:
    """Pick the best record for an instance (primary score, then latest attempt)."""
    if not records:
        return None
    return max(
        records,
        key=lambda record: (
            _record_rank(record, primary_key),
            int(record.get("_attempt") or 1),
        ),
    )


def summarize_instance(
    instance_id: str, records: list[dict[str, Any]], primary_key: str
) -> InstanceSummary:
    summary = InstanceSummary(instance_id=instance_id)
    summary.attempts = len(records)
    summary.attempt_numbers = sorted(
        {int(record.get("_attempt") or 1) for record in records}
    )
    representative = select_representative(records, primary_key)
    if representative is None:
        return summary
    verifier = _verifier_payload(representative)
    agent = _agent_payload(representative)
    trial = _trial_payload(representative)
    task = _task_payload(representative)
    summary.verifier_status = str(verifier.get("status") or "missing")
    summary.verifier_reward = verifier.get("reward")  # type: ignore[assignment]
    rewards = verifier.get("rewards")
    if isinstance(rewards, dict):
        value = rewards.get(primary_key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            summary.primary_reward = value
    summary.agent_status = str(agent.get("status") or "missing")
    summary.top_level_error = representative.get("error")
    summary.verifier_error = verifier.get("error")
    summary.trial_dir = trial.get("trial_dir")
    summary.task_relative_path = task.get("relative_path")
    return summary


def _load_manifest(run_dir: Path) -> dict[str, Any]:
    path = run_dir / _MANIFEST_FILENAME
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _mean(values: list[float | int]) -> float | None:
    if not values:
        return None
    return sum(float(value) for value in values) / len(values)


def summarize_run(
    run_dir: str | Path,
    *,
    primary_key: str = "reward",
    errors_as_zero: bool = False,
    success_threshold: float | None = None,
) -> dict[str, Any]:
    """Build the full summary payload for a completed Harbor run."""
    run_path = Path(run_dir)
    if not run_path.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {run_path}")
    manifest = _load_manifest(run_path)
    records = load_attempt_records(run_path)

    selected_ids: list[str] = []
    selected_paths: dict[str, str] = {}
    tasks = manifest.get("tasks")
    if isinstance(tasks, list):
        for entry in tasks:
            if not isinstance(entry, dict):
                continue
            task_id = entry.get("id")
            if isinstance(task_id, str):
                selected_ids.append(task_id)
                relative = entry.get("relative_path")
                if isinstance(relative, str):
                    selected_paths[task_id] = relative
    for task_id in records:
        if task_id not in selected_ids:
            selected_ids.append(task_id)
    selected_ids.sort()

    summaries = [
        summarize_instance(task_id, records.get(task_id, []), primary_key)
        for task_id in selected_ids
    ]
    for summary in summaries:
        if summary.task_relative_path is None:
            summary.task_relative_path = selected_paths.get(summary.instance_id)

    recorded = [summary for summary in summaries if summary.attempts > 0]
    missing = [summary for summary in summaries if summary.attempts == 0]

    agent_counts = {status: 0 for status in AGENT_STATUSES}
    verifier_counts = {status: 0 for status in VERIFIER_STATUSES}
    for summary in summaries:
        agent_counts[summary.agent_status] = (
            agent_counts.get(summary.agent_status, 0) + 1
        )
        verifier_counts[summary.verifier_status] = (
            verifier_counts.get(summary.verifier_status, 0) + 1
        )

    scored = [summary for summary in summaries if summary.verifier_status == "scored"]
    scored_primary = [
        summary for summary in scored if summary.primary_reward is not None
    ]
    primary_values = [
        summary.primary_reward
        for summary in scored_primary
        if summary.primary_reward is not None
    ]
    success_values: list[InstanceSummary] = []
    if success_threshold is not None:
        for summary in scored_primary:
            value = summary.primary_reward
            if value is not None and value >= success_threshold:
                success_values.append(summary)

    selected_total = len(summaries)
    infrastructure_errors = [
        summary
        for summary in summaries
        if summary.top_level_error
        or summary.verifier_status in ("verifier_error", "verifier_timeout")
    ]

    scores: dict[str, Any] = {
        "primary_key": primary_key,
        "scored_mean": _mean(primary_values),
        "scored_denominator": len(primary_values),
        "scored_coverage": round(len(scored_primary) / selected_total, 6)
        if selected_total
        else None,
        "scored_count": len(scored),
        "scored_without_primary_count": len(scored) - len(scored_primary),
    }
    if errors_as_zero:
        zero_filled = list(primary_values) + [0.0] * (
            selected_total - len(primary_values)
        )
        scores["errors_as_zero_mean"] = _mean(zero_filled)
        scores["errors_as_zero_denominator"] = selected_total
    if success_threshold is not None:
        scores["success_threshold"] = success_threshold
        scores["success_count"] = len(success_values)
        scores["success_rate"] = (
            round(len(success_values) / selected_total, 6) if selected_total else None
        )

    payload: dict[str, Any] = {
        "run_dir": str(run_path.resolve()),
        "reference_version": manifest.get("reference_version"),
        "dataset_revision": manifest.get("dataset_revision"),
        "manifest_hash": manifest.get("manifest_hash"),
        "task_count": selected_total,
        "recorded_count": len(recorded),
        "missing_count": len(missing),
        "attempt_records": sum(summary.attempts for summary in summaries),
        "multiple_attempt_instances": sum(
            1 for summary in summaries if summary.attempts > 1
        ),
        "agent_status_counts": agent_counts,
        "verifier_status_counts": verifier_counts,
        "infrastructure_error_count": len(infrastructure_errors),
        "scores": scores,
        "instances": [
            {
                "instance_id": summary.instance_id,
                "relative_path": summary.task_relative_path,
                "attempts": summary.attempts,
                "agent_status": summary.agent_status,
                "verifier_status": summary.verifier_status,
                "primary_reward": summary.primary_reward,
                "verifier_reward": summary.verifier_reward,
                "top_level_error": summary.top_level_error,
                "verifier_error": summary.verifier_error,
                "trial_dir": summary.trial_dir,
            }
            for summary in summaries
        ],
    }
    payload["notes"] = _build_notes(payload)
    return payload


def _build_notes(payload: dict[str, Any]) -> list[str]:
    notes: list[str] = []
    scores = payload["scores"]
    notes.append(
        f"Scored coverage: {payload['scores']['scored_count']}/"
        f"{payload['task_count']} tasks"
    )
    if payload["missing_count"]:
        notes.append(
            f"{payload['missing_count']} selected task(s) have no persisted "
            "record (interrupted or filtered run)."
        )
    if payload["infrastructure_error_count"]:
        notes.append(
            f"{payload['infrastructure_error_count']} task(s) have "
            "infrastructure/verifier errors; they are excluded from the "
            "scored-only mean."
        )
    if scores.get("scored_without_primary_count"):
        notes.append(
            f"{scores['scored_without_primary_count']} scored task(s) lack the "
            f"primary key {scores['primary_key']!r}; provide "
            "--primary-reward-key for that dataset profile."
        )
    if "errors_as_zero_mean" in scores:
        notes.append(
            "errors_as_zero_mean counts unscored/errored tasks as 0 over the "
            "selected denominator; report it alongside the scored-only mean."
        )
    return notes


def format_report(payload: dict[str, Any]) -> str:
    scores = payload["scores"]
    lines = [
        f"Harbor run: {payload['run_dir']}",
        f"Reference: Harbor v{payload.get('reference_version') or '?'} "
        f"(dataset revision {payload.get('dataset_revision') or 'unknown'})",
        f"Tasks selected: {payload['task_count']}  "
        f"records: {payload['recorded_count']}  "
        f"attempts: {payload['attempt_records']}  "
        f"missing: {payload['missing_count']}",
        "",
        "Agent status:    "
        + "  ".join(f"{k}={v}" for k, v in payload["agent_status_counts"].items()),
        "Verifier status: "
        + "  ".join(f"{k}={v}" for k, v in payload["verifier_status_counts"].items()),
        "",
    ]
    if scores["scored_mean"] is None:
        lines.append("Primary score: no scored tasks with a primary reward")
    else:
        lines.append(
            f"Primary score mean ({scores['primary_key']}): "
            f"{scores['scored_mean']:.4f} over {scores['scored_denominator']} "
            "scored task(s)"
        )
        lines.append(f"Scored coverage: {scores['scored_coverage']:.1%}")
    if "errors_as_zero_mean" in scores:
        errors_zero = scores["errors_as_zero_mean"]
        rendered = "n/a" if errors_zero is None else f"{errors_zero:.4f}"
        lines.append(
            f"Errors-as-zero mean: {rendered} over "
            f"{scores['errors_as_zero_denominator']} selected task(s)"
        )
    if "success_rate" in scores:
        success_rate = scores["success_rate"]
        rendered_rate = "n/a" if success_rate is None else f"{success_rate:.1%}"
        lines.append(
            f"Success rate (>= {scores['success_threshold']}): {rendered_rate} "
            f"({scores['success_count']}/{payload['task_count']})"
        )
    lines.append("")
    lines.extend(f"- {note}" for note in payload["notes"])
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize persisted harbor-infer results (no re-execution)."
    )
    parser.add_argument(
        "run_dir",
        type=str,
        help="Evaluation output directory produced by harbor-infer.",
    )
    parser.add_argument(
        "--primary-reward-key",
        type=str,
        default="reward",
        help="Reward key used as the primary score (default: 'reward').",
    )
    parser.add_argument(
        "--errors-as-zero",
        action="store_true",
        help="Also report a mean that counts unscored/errored tasks as zero.",
    )
    parser.add_argument(
        "--success-threshold",
        type=float,
        default=None,
        help="Also report a success rate for primary reward >= threshold.",
    )
    parser.add_argument(
        "--json-out",
        type=str,
        default=None,
        help=(
            "Write the JSON summary here (default: <run_dir>/summary.json; "
            "use '-' to print it)."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = summarize_run(
            args.run_dir,
            primary_key=args.primary_reward_key,
            errors_as_zero=args.errors_as_zero,
            success_threshold=args.success_threshold,
        )
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(format_report(payload))
    if args.json_out == "-":
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        out_path = (
            Path(args.json_out)
            if args.json_out
            else Path(args.run_dir) / SUMMARY_FILENAME
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"\nSummary written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
