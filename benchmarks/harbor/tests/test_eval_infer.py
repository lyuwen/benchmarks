"""Aggregation tests for ``harbor-eval`` (no inference or Docker)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.harbor.eval_infer import (
    format_report,
    load_attempt_records,
    main,
    summarize_run,
)
from benchmarks.harbor.tests.conftest import (
    errored_record,
    scored_record,
    verifier_error_record,
    write_jsonl,
)


def test_summary_counts_and_scored_mean(run_dir: Path) -> None:
    write_jsonl(
        run_dir / "output.critic_attempt_1.jsonl",
        [
            scored_record("task-a", reward=1.0),
            verifier_error_record("task-b"),
        ],
    )
    # Callback copies must not be double counted.
    write_jsonl(
        run_dir / "output.jsonl",
        [scored_record("task-a", reward=1.0)],
    )
    write_jsonl(
        run_dir / "output_errors.jsonl",
        [verifier_error_record("task-b")],
    )

    payload = summarize_run(run_dir)

    assert payload["task_count"] == 3
    assert payload["recorded_count"] == 2
    assert payload["missing_count"] == 1
    assert payload["attempt_records"] == 2
    assert payload["scores"]["scored_count"] == 1
    assert payload["scores"]["scored_mean"] == 1.0
    assert payload["scores"]["scored_denominator"] == 1
    assert payload["scores"]["scored_coverage"] == pytest.approx(1 / 3)
    assert payload["verifier_status_counts"]["scored"] == 1
    assert payload["verifier_status_counts"]["verifier_error"] == 1
    assert payload["verifier_status_counts"]["missing"] == 1
    assert payload["agent_status_counts"]["completed"] == 2
    assert payload["infrastructure_error_count"] == 1


def test_summary_chooses_best_attempt(run_dir: Path) -> None:
    write_jsonl(
        run_dir / "output.critic_attempt_1.jsonl",
        [scored_record("task-a", attempt=1, reward=0.0)],
    )
    write_jsonl(
        run_dir / "output.critic_attempt_2.jsonl",
        [verifier_error_record("task-a", attempt=2)],
    )

    payload = summarize_run(run_dir)
    instance = next(
        item for item in payload["instances"] if item["instance_id"] == "task-a"
    )
    assert instance["attempts"] == 2
    assert instance["verifier_status"] == "scored"
    assert instance["primary_reward"] == 0.0
    assert payload["multiple_attempt_instances"] == 1
    # A valid zero still counts as scored.
    assert payload["scores"]["scored_count"] == 1
    assert payload["scores"]["scored_mean"] == 0.0


def test_summary_errors_as_zero_and_threshold(run_dir: Path) -> None:
    write_jsonl(
        run_dir / "output.critic_attempt_1.jsonl",
        [
            scored_record("task-a", reward=1.0),
            verifier_error_record("task-b"),
            errored_record("task-c"),
        ],
    )

    payload = summarize_run(run_dir, errors_as_zero=True, success_threshold=0.5)

    assert payload["scores"]["errors_as_zero_mean"] == pytest.approx(1 / 3)
    assert payload["scores"]["errors_as_zero_denominator"] == 3
    assert payload["scores"]["success_rate"] == pytest.approx(1 / 3)
    assert payload["infrastructure_error_count"] == 2
    # Only the top-level-errored record has no agent status; the verifier_error
    # record still completed its agent phase.
    assert payload["agent_status_counts"]["missing"] == 1


def test_summary_named_primary_key(run_dir: Path) -> None:
    record = {
        "instance_id": "task-a",
        "attempt": 1,
        "test_result": {
            "verifier": {
                "status": "scored",
                "reward": None,
                "rewards": {"f2p": 1, "p2p": 0.5},
            },
            "agent": {"status": "completed"},
            "task": {"relative_path": "task-a"},
        },
        "error": None,
    }
    write_jsonl(run_dir / "output.critic_attempt_1.jsonl", [record])

    payload = summarize_run(run_dir, primary_key="f2p")
    assert payload["scores"]["primary_key"] == "f2p"
    assert payload["scores"]["scored_mean"] == 1.0

    default_payload = summarize_run(run_dir)
    assert default_payload["scores"]["scored_count"] == 1
    assert default_payload["scores"]["scored_mean"] is None
    assert default_payload["scores"]["scored_without_primary_count"] == 1
    assert any("primary key" in note for note in default_payload["notes"])


def test_fallback_to_output_files_when_no_attempt_records(run_dir: Path) -> None:
    write_jsonl(run_dir / "output.jsonl", [scored_record("task-a", reward=2.0)])
    write_jsonl(run_dir / "output_errors.jsonl", [errored_record("task-b")])

    records = load_attempt_records(run_dir)
    assert set(records) == {"task-a", "task-b"}
    payload = summarize_run(run_dir)
    assert payload["recorded_count"] == 2
    assert payload["scores"]["scored_mean"] == 2.0
    # task-b (errored record, no agent status) and task-c (no record).
    assert payload["agent_status_counts"]["missing"] == 2


def test_summary_is_deterministic(run_dir: Path) -> None:
    write_jsonl(
        run_dir / "output.critic_attempt_1.jsonl",
        [scored_record("task-c", reward=1.0), scored_record("task-a", reward=0.5)],
    )
    first = summarize_run(run_dir)
    second = summarize_run(run_dir)
    assert first == second
    assert [item["instance_id"] for item in first["instances"]] == [
        "task-a",
        "task-b",
        "task-c",
    ]


def test_format_report_mentions_denominator(run_dir: Path) -> None:
    write_jsonl(
        run_dir / "output.critic_attempt_1.jsonl",
        [scored_record("task-a", reward=0.5)],
    )
    report = format_report(summarize_run(run_dir))
    assert "Primary score mean (reward): 0.5000 over 1" in report
    assert "Scored coverage" in report
    assert "Verifier status" in report


def test_main_writes_summary(run_dir: Path) -> None:
    write_jsonl(
        run_dir / "output.critic_attempt_1.jsonl",
        [scored_record("task-a", reward=1.0)],
    )
    exit_code = main([str(run_dir)])
    assert exit_code == 0
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["task_count"] == 3
    assert summary["scores"]["scored_mean"] == 1.0


def test_main_missing_run_dir(tmp_path: Path) -> None:
    assert main([str(tmp_path / "nope")]) == 2
