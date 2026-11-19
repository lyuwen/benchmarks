"""Tests for loading local JSONL datasets (benchmarks.utils.dataset)."""

import json
from pathlib import Path
from typing import Any

import pytest

from benchmarks.utils.dataset import get_dataset


# Hugging Face datasets' JSON builder reads a local JSONL file in 10 MB chunks
# and infers the Arrow schema from the first chunk; every later chunk is cast
# to that schema. A nested field that is null throughout the first chunk but
# holds a string later therefore used to abort loading with
# `TypeError: Couldn't cast array of type string to null`.
HF_JSON_CHUNK_BYTES = 10 << 20
_PADDING = 300_000


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _row(index: int, helm_version: str | None) -> dict[str, Any]:
    return {
        "instance_id": f"inst-{index:03d}",
        "problem_statement": "some problem",
        "install_config": {"docker_specs": {"helm_version": helm_version}},
        # Pad each row so the file spans more than one 10 MB JSON chunk.
        "padding": "x" * _PADDING,
    }


def test_local_jsonl_with_heterogeneous_nested_types(tmp_path: Path) -> None:
    """Null-then-string nested fields load and are preserved as serialized."""
    path = tmp_path / "data.jsonl"
    _write_jsonl(
        path,
        [
            _row(0, None),
            _row(1, "3.14.0"),
            _row(2, None),
        ],
    )

    df = get_dataset(dataset_name=str(path), split="train")

    assert len(df) == 3
    specs = [row["install_config"]["docker_specs"] for _, row in df.iterrows()]
    assert [s["helm_version"] for s in specs] == [None, "3.14.0", None]
    assert list(df["instance_id"]) == ["inst-000", "inst-001", "inst-002"]


def test_local_jsonl_field_null_only_in_first_chunk(tmp_path: Path) -> None:
    """A nested field null in the first HF chunk and a string later must load.

    This reproduces the SWE-rebench-V2 failure mode: ``helm_version`` is null
    on every row of the leading >10 MB and only populated on the last row.
    """
    path = tmp_path / "data.jsonl"
    rows = [_row(i, None) for i in range(40)]
    rows[-1] = _row(39, "3.14.0")
    _write_jsonl(path, rows)

    assert path.stat().st_size > HF_JSON_CHUNK_BYTES

    df = get_dataset(dataset_name=str(path), split="train")

    assert len(df) == 40
    specs = [row["install_config"]["docker_specs"] for _, row in df.iterrows()]
    assert specs[-1]["helm_version"] == "3.14.0"
    assert all(s["helm_version"] is None for s in specs[:-1])


def test_local_jsonl_select_filter(tmp_path: Path) -> None:
    path = tmp_path / "data.jsonl"
    _write_jsonl(path, [_row(0, None), _row(1, "3.14.0"), _row(2, None)])
    select = tmp_path / "select.txt"
    select.write_text("inst-001\n", encoding="utf-8")

    df = get_dataset(
        dataset_name=str(path), split="train", selected_instances_file=str(select)
    )

    assert list(df["instance_id"]) == ["inst-001"]


def test_local_jsonl_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "data.jsonl"
    path.write_text("\n\n", encoding="utf-8")

    with pytest.raises(ValueError, match="empty"):
        get_dataset(dataset_name=str(path), split="train")
