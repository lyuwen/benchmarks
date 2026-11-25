"""Tests for the harness-history flattening tool (no network, no Docker)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.utils.harnesses import flatten


TOOL = {
    "type": "function",
    "function": {
        "name": "Bash",
        "description": "run a command",
        "parameters": {"type": "object", "properties": {}},
    },
}

SYSTEM = {"role": "system", "content": "system prompt"}
USER = {"role": "user", "content": "fix the bug"}
ASSISTANT_1_VERBATIM = {
    "role": "assistant",
    "content": "",
    "tool_calls": [
        {
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "Bash",
                "arguments": '{"command": "cd /testbed && ls"}',
            },
        }
    ],
}
ASSISTANT_1_REPLAYED = {
    "role": "assistant",
    "content": "",
    "tool_calls": [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "Bash", "arguments": '{"command": "ls"}'},
        }
    ],
}
TOOL_RESULT = {"role": "tool", "tool_call_id": "call_1", "content": "ok"}
ASSISTANT_2 = {"role": "assistant", "content": "done"}


def make_turn(turn_index: int, message_index: int, trained: bool, tool_set: int = 0):
    return {
        "turn_index": turn_index,
        "message_index": message_index,
        "trained": trained,
        "tool_set": tool_set,
        "request_class": "main",
        "request": {"messages": [SYSTEM, USER]},
        "response": ASSISTANT_2,
    }


def make_history(*, excluded: bool = False, extra_tools: bool = False) -> dict:
    fragment_one = {
        "trajectory_id": "s:1",
        "trajectory_index": 0,
        "kind": "conversation",
        "parent_trajectory_id": None,
        "branch_message_index": None,
        "branch_turn_index": None,
        "first_turn_index": 1,
        "leaf_turn_index": 1,
        "contains_compaction": False,
        "exclude_from_training": False,
        "path_turn_indices": [1],
        "messages": [SYSTEM, USER, ASSISTANT_1_REPLAYED],
        "loss_mask": [0, 0, 1],
        "reasoning_content": [None, None, "thinking"],
        "tool_sets": [[TOOL]],
        "turns": [make_turn(1, 2, True)],
    }
    tool_sets = [
        [TOOL],
        [{"type": "function", "function": {"name": "Read", "parameters": {}}}],
    ]
    turns = [
        make_turn(1, 2, False),
        make_turn(2, 4, True, tool_set=1 if extra_tools else 0),
    ]
    fragment_two = {
        "trajectory_id": "s:2",
        "trajectory_index": 1,
        "kind": "conversation",
        "parent_trajectory_id": "s:1",
        "branch_message_index": 3,
        "branch_turn_index": 1,
        "first_turn_index": 1,
        "leaf_turn_index": 2,
        "contains_compaction": False,
        "exclude_from_training": excluded,
        "path_turn_indices": [1, 2],
        "messages": [SYSTEM, USER, ASSISTANT_1_REPLAYED, TOOL_RESULT, ASSISTANT_2],
        "loss_mask": [0, 0, 0, 0, 1],
        "reasoning_content": [None, None, None, None, None],
        "tool_sets": tool_sets,
        "turns": turns,
    }
    return {
        "instance_id": "owner__repo-1",
        "attempt": 1,
        "harness": "claude-code",
        "harness_version": "2.1.331",
        "model": "openai/test-model",
        "upstream_model": "test-model",
        "harness_model": "test-model",
        "instruction": "fix the bug",
        "test_result": {"git_patch": "diff --git a/x b/x\n"},
        "evaluation": True,
        "store": {"instance_id": "owner__repo-1"},
        "rollout_export": "rollout/owner__repo-1.attempt1.json",
        "stats": {"requests": 2},
        "tools": tool_sets,
        "trajectories": [fragment_one, fragment_two],
        "request_log": [make_turn(1, 2, True), make_turn(2, 4, True)],
    }


def test_iter_json_documents_handles_jsonl_and_pretty_cat(tmp_path: Path) -> None:
    jsonl = tmp_path / "rows.jsonl"
    jsonl.write_text(json.dumps({"a": 1}) + "\n" + json.dumps({"a": 2}) + "\n")
    assert [doc["a"] for doc in flatten.iter_json_documents(jsonl)] == [1, 2]

    pretty = tmp_path / "concat.json"
    pretty.write_text(
        json.dumps({"a": 1}, indent=2) + "\n" + json.dumps({"a": 2}, indent=2) + "\n"
    )
    assert [doc["a"] for doc in flatten.iter_json_documents(pretty)] == [1, 2]


def test_iter_json_documents_rejects_truncated(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text('{"a": 1}{"b": ')
    with pytest.raises(ValueError, match="invalid JSON"):
        list(flatten.iter_json_documents(path))


def test_flatten_trajectories_mode_has_messages_and_tools() -> None:
    records = list(flatten.iter_records(iter([make_history()])))
    assert len(records) == 2
    first = records[0]
    assert first["source"] == "trajectory"
    assert first["messages"][:2] == [SYSTEM, USER]
    assert first["messages"][2] == {
        **ASSISTANT_1_REPLAYED,
        "reasoning_content": "thinking",
    }
    assert first["tools"] == [TOOL]
    assert first["loss_mask"] == [0, 0, 1]
    assert first["reasoning_content"] == [None, None, "thinking"]
    assert first["trajectory_id"] == "s:1"
    assert first["test_result"]["git_patch"].startswith("diff --git")
    assert first["evaluation"] is True
    assert flatten.validate_record(first) == []


def test_excluded_trajectories_are_skipped_by_default() -> None:
    history = make_history(excluded=True)
    default = list(flatten.iter_records(iter([history])))
    assert [record["trajectory_id"] for record in default] == ["s:1"]
    included = list(flatten.iter_records(iter([history]), include_excluded=True))
    assert [record["trajectory_id"] for record in included] == ["s:1", "s:2"]


def test_projects_total_turn_set_into_single_stitched_record() -> None:
    records = list(flatten.iter_records(iter([make_history()]), mode="stitch"))
    assert len(records) == 1
    record = records[0]
    assert record["stitched"] is True
    assert record["fragmented"] is True
    assert record["num_fragments"] == 2
    assert record["messages"] == [
        SYSTEM,
        USER,
        ASSISTANT_1_REPLAYED,
        TOOL_RESULT,
        ASSISTANT_2,
    ]
    assert record["loss_mask"] == [0, 0, 1, 0, 1]
    assert record["tools"] == [TOOL]
    assert record["turn_indices"] == [1, 2]
    assert flatten.validate_record(record) == []


def test_train_excluded_recomputes_loss_mask() -> None:
    history = make_history(excluded=True)
    default = list(flatten.iter_records(iter([history])))
    assert [record["trajectory_id"] for record in default] == ["s:1"]

    included = list(flatten.iter_records(iter([history]), train_excluded=True))
    assert [record["trajectory_id"] for record in included] == ["s:1", "s:2"]
    masked = included[1]
    assert masked["exclude_from_training"] is True
    assert masked["loss_mask_recomputed"] is True
    # s:2 messages: system, user, assistant(tool call), tool, assistant
    assert masked["loss_mask"] == [0, 0, 1, 0, 1]
    assert flatten.validate_record(masked) == []


def test_stitch_verbatim_restores_original_tool_arguments() -> None:
    history = make_history()
    # The captured response (turn 1) kept the original cd-prefixed command.
    history["trajectories"][0]["turns"][0]["response"] = ASSISTANT_1_VERBATIM
    plain = flatten.stitch_row(history)
    assert plain is not None
    assert (
        "cd /testbed"
        not in plain["messages"][2]["tool_calls"][0]["function"]["arguments"]
    )
    verbatim = flatten.stitch_row(history, verbatim=True)
    assert verbatim is not None
    assert (
        verbatim["messages"][2]["tool_calls"][0]["function"]["arguments"]
        == '{"command": "cd /testbed && ls"}'
    )


def test_resolve_tools_uses_last_turn_tool_set() -> None:
    history = make_history(extra_tools=True)
    trajectory = history["trajectories"][1]
    tools = flatten.resolve_tools(trajectory, history)
    assert tools[0]["function"]["name"] == "Read"


def test_resolve_tools_falls_back_to_top_level_tools() -> None:
    history = make_history()
    trajectory = {
        "trajectory_id": "s:x",
        "messages": [SYSTEM],
        "turns": [],
        "tool_sets": None,
    }
    assert flatten.resolve_tools(trajectory, history) == [TOOL]
    # New-format history dumps already resolve the root tools field.
    assert flatten.resolve_tools({}, {"tools": [TOOL]}) == [TOOL]
    assert flatten.resolve_tools({"tool_sets": []}, {"tools": []}) == []


def test_resolve_tools_respects_explicitly_empty_tool_sets() -> None:
    history = make_history()
    tool_less = {
        "trajectory_id": "s:aux",
        "messages": [SYSTEM, USER, ASSISTANT_2],
        "tool_sets": [],
        "turns": [
            {
                "turn_index": 3,
                "message_index": 2,
                "trained": False,
                "tool_set": None,
            }
        ],
    }
    assert flatten.resolve_tools(tool_less, history) == []

    record = flatten.flatten_trajectory(history, tool_less)
    assert record["tools"] == []
    assert flatten.validate_record(record) == []


def test_stitch_ignores_excluded_trajectories() -> None:
    history = make_history()
    # Give the auxiliary (excluded) fragment the highest leaf turn index so a
    # naive "longest conversation" pick would choose it.
    auxiliary = {
        "trajectory_id": "s:aux",
        "trajectory_index": 2,
        "kind": "conversation",
        "exclude_from_training": True,
        "leaf_turn_index": 99,
        "messages": [SYSTEM, USER, ASSISTANT_2],
        "loss_mask": [0, 0, 0],
        "tool_sets": [],
        "turns": [{"turn_index": 99, "message_index": 2, "trained": False}],
    }
    history["trajectories"].append(auxiliary)
    record = flatten.stitch_row(history)
    assert record is not None
    main = [t for t in history["trajectories"] if not t.get("exclude_from_training")][
        -1
    ]
    assert record["messages"] == main["messages"]
    assert record["tools"] == [TOOL]


def test_validate_record_reports_problems() -> None:
    record = {
        "instance_id": "i",
        "trajectory_id": "s:1",
        "messages": [
            SYSTEM,
            {"role": "tool", "tool_call_id": "missing", "content": "x"},
            {"role": "assistant", "tool_calls": [{"id": "c", "function": {}}]},
        ],
        "tools": "nope",
        "loss_mask": [0, 1],
    }
    problems = flatten.validate_record(record)
    assert any("tools is not a list" in problem for problem in problems)
    assert any("has no preceding call" in problem for problem in problems)
    assert any("loss_mask is not aligned" in problem for problem in problems)


def test_validate_accepts_fragment_with_trailing_tool_call() -> None:
    records = list(flatten.iter_records(iter([make_history()])))
    # s:1 ends in an assistant tool call whose result lives in the next fragment.
    assert flatten.validate_record(records[0]) == []


def test_main_writes_jsonl_and_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    history = make_history()
    source = tmp_path / "histories.jsonl"
    source.write_text(json.dumps(history) + "\n" + json.dumps(history) + "\n")
    output = tmp_path / "flat.jsonl"
    exit_code = flatten.main([str(source), "-o", str(output)])
    assert exit_code == 0
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == 4
    assert all("messages" in row and "tools" in row for row in rows)
    stderr = capsys.readouterr().err
    assert "histories=2" in stderr
    assert "emitted=4" in stderr
    assert "warnings=0" in stderr


def test_main_stitch_mode_and_strict(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "histories.jsonl"
    source.write_text(json.dumps(make_history()) + "\n")
    output = tmp_path / "flat.jsonl"
    exit_code = flatten.main(
        [str(source), "-o", str(output), "--mode", "stitch", "--strict"]
    )
    assert exit_code == 0
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["stitched"] is True

    broken = make_history()
    broken["trajectories"] = []
    broken["tools"] = []
    source.write_text(json.dumps(broken) + "\n")
    assert flatten.main([str(source), "-o", str(output), "--strict"]) == 1


TOOL_B = {
    "type": "function",
    "function": {
        "name": "Read",
        "description": "read a file",
        "parameters": {"type": "object", "properties": {}},
    },
}


def make_multi_tool_history() -> dict:
    history = make_history()
    call_1 = {
        "id": "call_a",
        "type": "function",
        "function": {"name": "Bash", "arguments": '{"command": "ls"}'},
    }
    call_2 = {
        "id": "call_b",
        "type": "function",
        "function": {"name": "Bash", "arguments": '{"command": "pwd"}'},
    }
    messages = [
        SYSTEM,
        USER,
        {"role": "assistant", "content": "", "tool_calls": [call_1]},
        {"role": "tool", "tool_call_id": "call_a", "content": "a"},
        {"role": "assistant", "content": "", "tool_calls": [call_2]},
        {"role": "tool", "tool_call_id": "call_b", "content": "b"},
        ASSISTANT_2,
    ]
    trajectory = {
        "trajectory_id": "s:multi",
        "trajectory_index": 0,
        "kind": "conversation",
        "exclude_from_training": False,
        "leaf_turn_index": 3,
        "path_turn_indices": [1, 2, 3],
        "messages": messages,
        "loss_mask": [0, 0, 1, 0, 1, 0, 1],
        "reasoning_content": [None] * len(messages),
        "tool_sets": [[TOOL], [TOOL, TOOL_B]],
        "turns": [
            {
                "turn_index": 1,
                "message_index": 2,
                "trained": True,
                "tool_set": 0,
                "request_class": "main",
            },
            {
                "turn_index": 2,
                "message_index": 4,
                "trained": True,
                "tool_set": 0,
                "request_class": "main",
            },
            {
                "turn_index": 3,
                "message_index": 6,
                "trained": True,
                "tool_set": 1,
                "request_class": "main",
            },
        ],
    }
    history["trajectories"] = [trajectory]
    return history


def test_split_tool_sets_emits_segment_per_tool_phase() -> None:
    history = make_multi_tool_history()
    records = list(flatten.iter_records(iter([history]), split_tool_sets=True))
    assert len(records) == 2

    first, second = records
    assert first["source"] == "trajectory-split"
    assert first["tools"] == [TOOL]
    assert first["tool_set_index"] == 0
    assert first["split_index"] == 0
    assert first["split_count"] == 2
    assert first["turn_indices"] == [1, 2]
    assert first["trained_turns"] == [1, 2]
    # Context up to the last turn of the phase, only its turns are targets.
    assert first["messages"] == history["trajectories"][0]["messages"][:5]
    assert first["loss_mask"] == [0, 0, 1, 0, 1]

    assert second["tools"] == [TOOL, TOOL_B]
    assert second["tool_set_index"] == 1
    assert second["split_index"] == 1
    assert second["turn_indices"] == [3]
    assert second["messages"] == history["trajectories"][0]["messages"]
    assert second["loss_mask"] == [0, 0, 0, 0, 0, 0, 1]
    for record in records:
        assert flatten.validate_record(record) == []


def test_split_tool_sets_is_noop_for_single_phase() -> None:
    history = make_history()
    split = list(flatten.iter_records(iter([history]), split_tool_sets=True))
    plain = list(flatten.iter_records(iter([history])))
    assert [record["tools"] for record in split] == [
        record["tools"] for record in plain
    ]
    assert all(record.get("source") == "trajectory" for record in split)


def test_split_tool_sets_respects_excluded_and_train_excluded() -> None:
    history = make_history(excluded=True)
    masked = list(
        flatten.iter_records(
            iter([history]), split_tool_sets=True, include_excluded=True
        )
    )
    assert masked[1]["loss_mask"] == [0, 0, 0, 0, 0]

    recomputed = list(
        flatten.iter_records(iter([history]), split_tool_sets=True, train_excluded=True)
    )
    # Single-phase trajectory: all of its assistant messages become targets.
    assert recomputed[1]["loss_mask"] == [0, 0, 1, 0, 1]
    assert recomputed[1]["loss_mask_recomputed"] is True


def test_split_tool_sets_requires_trajectories_mode(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "histories.jsonl"
    source.write_text(json.dumps(make_history()) + "\n")
    with pytest.raises(SystemExit):
        flatten.main([str(source), "--mode", "stitch", "--split-tool-sets"])


def test_main_split_tool_sets_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "histories.jsonl"
    source.write_text(json.dumps(make_multi_tool_history()) + "\n")
    output = tmp_path / "flat.jsonl"
    assert flatten.main([str(source), "-o", str(output), "--split-tool-sets"]) == 0
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == 2
    assert [len(row["tools"]) for row in rows] == [1, 2]
    assert [row["split_index"] for row in rows] == [0, 1]
    assert "emitted=2" in capsys.readouterr().err


def make_aux_stop_history(
    *,
    content: str = "<block>no",
    stop: object = ["</block>"],
    finish_reason: str = "stop",
) -> dict:
    history = make_history()
    turn = {
        "turn_index": 3,
        "message_index": 2,
        "trained": False,
        "tool_set": None,
        "request_class": "auxiliary",
        "finish_reason": finish_reason,
        "request": {
            "model": "m",
            "max_tokens": 256,
            "messages": [],
            "stop": stop,
        },
    }
    trajectory = {
        "trajectory_id": "s:aux",
        "trajectory_index": 1,
        "kind": "conversation",
        "exclude_from_training": True,
        "leaf_turn_index": 3,
        "messages": [SYSTEM, USER, {"role": "assistant", "content": content}],
        "loss_mask": [0, 0, 0],
        "tool_sets": [],
        "turns": [turn],
    }
    history["trajectories"] = [trajectory]
    return history


def test_auxiliary_stop_sequence_is_reattached() -> None:
    history = make_aux_stop_history()
    records = list(flatten.iter_records(iter([history]), include_excluded=True))
    record = records[0]
    assert record["messages"][2]["content"] == "<block>no</block>"
    assert record["restored_stop_sequences"] == [
        {"message_index": 2, "turn_index": 3, "stop_sequence": "</block>"}
    ]
    assert flatten.validate_record(record) == []


def test_stop_sequence_restoration_is_conservative() -> None:
    # Already terminated: nothing to do.
    record = list(
        flatten.iter_records(
            iter([make_aux_stop_history(content="<block>no</block>")]),
            include_excluded=True,
        )
    )[0]
    assert record["messages"][2]["content"] == "<block>no</block>"
    assert "restored_stop_sequences" not in record

    # Truncated by max_tokens: the model never emitted the stop sequence.
    record = list(
        flatten.iter_records(
            iter([make_aux_stop_history(finish_reason="length")]),
            include_excluded=True,
        )
    )[0]
    assert record["messages"][2]["content"] == "<block>no"
    assert "restored_stop_sequences" not in record

    # Multiple stop sequences are ambiguous (the API does not say which one hit).
    record = list(
        flatten.iter_records(
            iter([make_aux_stop_history(stop=["</block>", "\n\n"])]),
            include_excluded=True,
        )
    )[0]
    assert record["messages"][2]["content"] == "<block>no"
    assert "restored_stop_sequences" not in record

    # A string stop is a valid single stop sequence.
    record = list(
        flatten.iter_records(
            iter([make_aux_stop_history(stop="</verdict>")]),
            include_excluded=True,
        )
    )[0]
    assert record["messages"][2]["content"] == "<block>no</verdict>"


def test_restore_stop_sequences_can_be_disabled() -> None:
    history = make_aux_stop_history()
    record = list(
        flatten.iter_records(
            iter([history]),
            include_excluded=True,
            restore_stop_sequences=False,
        )
    )[0]
    assert record["messages"][2]["content"] == "<block>no"
    assert "restored_stop_sequences" not in record


def test_stop_sequence_restoration_skips_tool_calls() -> None:
    history = make_aux_stop_history()
    history["trajectories"][0]["messages"][2] = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_x",
                "type": "function",
                "function": {"name": "Bash", "arguments": "{}"},
            }
        ],
    }
    record = list(flatten.iter_records(iter([history]), include_excluded=True))[0]
    assert record["messages"][2]["content"] == ""
    assert "restored_stop_sequences" not in record


def test_main_cli_no_restore_stop_sequences(tmp_path: Path) -> None:
    source = tmp_path / "histories.jsonl"
    source.write_text(json.dumps(make_aux_stop_history()) + "\n")
    output = tmp_path / "flat.jsonl"
    assert (
        flatten.main(
            [
                str(source),
                "-o",
                str(output),
                "--include-excluded",
                "--no-restore-stop-sequences",
            ]
        )
        == 0
    )
    row = json.loads(output.read_text())
    assert row["messages"][2]["content"] == "<block>no"
