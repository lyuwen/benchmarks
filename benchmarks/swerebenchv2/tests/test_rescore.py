"""Tests for the SWE-rebench-V2 re-scoring workflow (no Docker)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.swerebenchv2.rescore import (
    TASK_KEYS,
    merge_histories,
    prepare_workdir,
    run_one,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def _histories() -> list[dict]:
    return [
        {
            "instance_id": "owner__one-1",
            "evaluation": False,
            "test_result": {"git_patch": "PATCH_ONE"},
            "messages": [{"role": "user", "content": "hello"}],
        },
        {
            "instance_id": "owner__two-2",
            "evaluation": True,
            "test_result": {"git_patch": "PATCH_TWO"},
            "messages": [],
        },
        {
            "instance_id": "owner__three-3",
            "evaluation": False,
            "test_result": {"git_patch": "PATCH_THREE"},
            "messages": [],
        },
    ]


def _dataset() -> list[dict]:
    rows = []
    for instance_id, repo in (
        ("owner__one-1", "owner/one"),
        ("owner__two-2", "owner/two"),
        ("owner__three-3", "owner/three"),
    ):
        row = {
            "instance_id": instance_id,
            "repo": repo,
            "image_name": f"registry/{repo.replace('/', '-')}:tag",
            "install_config": {
                "test_cmd": "go test -v ./...",
                "log_parser": "parse_log_gotest",
            },
            "test_patch": f"TEST_PATCH_{instance_id}",
            "PASS_TO_PASS": [f"Pass_{instance_id}"],
            "FAIL_TO_PASS": [f"Fail_{instance_id}"],
            "unused_dataset_field": "must not be copied to tasks/",
        }
        rows.append(row)
    return rows


def _prepared(tmp_path: Path) -> tuple[Path, Path, Path]:
    histories = tmp_path / "histories.jsonl"
    dataset = tmp_path / "tasks.jsonl"
    workdir = tmp_path / "work"
    _write_jsonl(histories, _histories())
    _write_jsonl(dataset, _dataset())
    prepare_workdir(histories, dataset, workdir)
    return histories, dataset, workdir


def test_prepare_writes_one_item_per_history_line(tmp_path: Path) -> None:
    histories, _, workdir = _prepared(tmp_path)

    assert (workdir / "count").read_text() == "3"
    manifest = [
        json.loads(line)
        for line in (workdir / "manifest.jsonl").read_text().splitlines()
    ]
    assert [item["instance_id"] for item in manifest] == [
        "owner__one-1",
        "owner__two-2",
        "owner__three-3",
    ]
    instance = json.loads((workdir / "instances" / "1.json").read_text())
    assert instance == {
        "index": 1,
        "instance_id": "owner__two-2",
        "old_evaluation": True,
        "git_patch": "PATCH_TWO",
    }
    task = json.loads((workdir / "tasks" / "1.json").read_text())
    assert set(task) == set(TASK_KEYS)
    assert task["repo"] == "owner/two"
    assert len(histories.read_text().splitlines()) == 3


def test_prepare_fails_when_dataset_is_missing_an_instance(tmp_path: Path) -> None:
    histories = tmp_path / "histories.jsonl"
    dataset = tmp_path / "tasks.jsonl"
    _write_jsonl(histories, _histories())
    _write_jsonl(dataset, _dataset()[:2])

    with pytest.raises(ValueError, match="missing in the dataset"):
        prepare_workdir(histories, dataset, tmp_path / "work")


def test_run_one_writes_the_judge_verdict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, workdir = _prepared(tmp_path)
    monkeypatch.setattr(
        "benchmarks.swerebenchv2.judge.SWERebenchV2Judge.judge",
        lambda self, instance_id, git_patch, instance_data: True,
    )

    result = run_one(workdir, 1)

    assert result["instance_id"] == "owner__two-2"
    assert result["old_evaluation"] is True
    assert result["evaluation"] is True
    assert result["error"] is None
    stored = json.loads((workdir / "results" / "1.json").read_text())
    assert stored["evaluation"] is True


def test_run_one_records_a_missing_verdict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, workdir = _prepared(tmp_path)
    monkeypatch.setattr(
        "benchmarks.swerebenchv2.judge.SWERebenchV2Judge.judge",
        lambda self, instance_id, git_patch, instance_data: None,
    )

    result = run_one(workdir, 0)

    assert result["evaluation"] is None
    assert result["error"] is not None


def test_merge_updates_only_the_evaluation_field(tmp_path: Path) -> None:
    histories, _, workdir = _prepared(tmp_path)
    results_dir = workdir / "results"
    (results_dir / "0.json").write_text(
        json.dumps({"index": 0, "evaluation": True, "error": None})
    )
    (results_dir / "1.json").write_text(
        json.dumps({"index": 1, "evaluation": None, "error": "judge returned None"})
    )

    output = tmp_path / "histories.rescored.jsonl"
    report = merge_histories(workdir, histories, output, tmp_path / "report.json")

    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["evaluation"] for row in rows] == [True, True, False]
    # Non-evaluation fields are preserved.
    assert rows[0]["messages"] == [{"role": "user", "content": "hello"}]
    assert rows[1]["test_result"] == {"git_patch": "PATCH_TWO"}

    counts = report["counts"]
    assert counts["total"] == 3
    assert counts["updated"] == 1
    assert counts["recovered"] == 1
    assert counts["regressed"] == 0
    assert counts["unchanged"] == 0
    assert counts["missing_result"] == 1
    assert counts["errors"] == 1
    assert report["missing_indices"] == [2]
    assert report["error_indices"] == [1]

    saved_report = json.loads((tmp_path / "report.json").read_text())
    assert saved_report["counts"] == counts


def test_merge_dry_run_writes_no_output(tmp_path: Path) -> None:
    histories, _, workdir = _prepared(tmp_path)
    (workdir / "results" / "0.json").write_text(
        json.dumps({"index": 0, "evaluation": True, "error": None})
    )

    output = tmp_path / "histories.rescored.jsonl"
    report = merge_histories(workdir, histories, output, dry_run=True)

    assert not output.exists()
    assert report["dry_run"] is True
    assert report["counts"]["recovered"] == 1


def test_merge_counts_unchanged_rows(tmp_path: Path) -> None:
    histories, _, workdir = _prepared(tmp_path)
    results_dir = workdir / "results"
    (results_dir / "0.json").write_text(
        json.dumps({"index": 0, "evaluation": False, "error": None})
    )
    (results_dir / "2.json").write_text(
        json.dumps({"index": 2, "evaluation": False, "error": None})
    )

    report = merge_histories(workdir, histories, tmp_path / "out.jsonl", dry_run=True)

    assert report["counts"]["updated"] == 0
    assert report["counts"]["unchanged"] == 2
