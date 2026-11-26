"""Tests for the fragmented-history streamlining tool."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from benchmarks.utils.harnesses import streamline


BASH = {
    "type": "function",
    "function": {
        "name": "Bash",
        "description": "run a command",
        "parameters": {"type": "object", "properties": {}},
    },
}

SYSTEM = {"role": "system", "content": "You are a coding agent."}
USER = {"role": "user", "content": "Fix the failing test."}
AUX_SYSTEM = {"role": "system", "content": "You are a security monitor."}
AUX_USER = {"role": "user", "content": "<transcript>..."}


def call(call_id: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def assistant(
    *, calls: list[dict[str, Any]] | None = None, content: Any = None
) -> dict:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if calls:
        message["tool_calls"] = calls
    return message


def tool_result(call_id: str, content: str) -> dict:
    return {"role": "tool", "tool_call_id": call_id, "content": content}


def turn(
    turn_index: int,
    message_index: int,
    *,
    trained: bool,
    response: dict | None = None,
    request_class: str = "main",
    agent_id: str | None = None,
    agent_type: str | None = None,
    tool_set: int | None = 0,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "turn_index": turn_index,
        "message_index": message_index,
        "trained": trained,
        "request_class": request_class,
        "agent_id": agent_id,
        "agent_type": agent_type,
        "tool_set": tool_set,
    }
    if response is not None:
        entry["response"] = response
    if request_class == "auxiliary":
        entry["finish_reason"] = "stop"
        entry["request"] = {"stop": ["</block>"], "messages": []}
    return entry


def trajectory(
    trajectory_id: str,
    messages: list[dict],
    turns: list[dict],
    *,
    kind: str = "conversation",
    excluded: bool = False,
    parent: str | None = None,
    leaf: int | None = None,
    trajectory_index: int = 0,
    tool_sets: list[list[dict]] | None = None,
    agent_id: str | None = None,
) -> dict[str, Any]:
    loss_mask = [0] * len(messages)
    for entry in turns:
        if entry.get("trained"):
            loss_mask[entry["message_index"]] = 1
    return {
        "trajectory_id": trajectory_id,
        "trajectory_index": trajectory_index,
        "kind": kind,
        "exclude_from_training": excluded,
        "parent_trajectory_id": parent,
        "leaf_turn_index": leaf,
        "path_turn_indices": sorted(entry["turn_index"] for entry in turns),
        "messages": messages,
        "loss_mask": loss_mask,
        "tool_sets": tool_sets if tool_sets is not None else [[BASH]],
        "turns": turns,
        "agent_id": agent_id,
    }


def make_history() -> dict[str, Any]:
    original_1 = assistant(
        calls=[call("call_1", "Bash", {"command": "cd /testbed && ls"})]
    )
    replayed_1 = assistant(calls=[call("call_1", "Bash", {"command": "ls"})])
    original_2 = assistant(
        calls=[call("call_2", "Bash", {"command": "cd /testbed && pwd"})]
    )
    result_1 = tool_result("call_1", "files")

    fragment_a = trajectory(
        "run:1",
        [SYSTEM, USER, original_1],
        [turn(1, 2, trained=True, response=original_1)],
        leaf=1,
        trajectory_index=0,
    )
    fragment_b = trajectory(
        "run:2",
        [SYSTEM, USER, replayed_1, result_1, original_2],
        [
            turn(1, 2, trained=False, response=replayed_1),
            turn(2, 4, trained=True, response=original_2),
        ],
        parent="run:1",
        leaf=2,
        trajectory_index=1,
    )

    # Auxiliary classifier request: kept verbatim (not merged, not modified).
    aux_assistant = assistant(content="<block>no")
    auxiliary = trajectory(
        "run:aux",
        [AUX_SYSTEM, AUX_USER, aux_assistant],
        [
            turn(
                3,
                2,
                trained=True,
                response=aux_assistant,
                request_class="auxiliary",
            )
        ],
        leaf=3,
        trajectory_index=2,
        tool_sets=[],
    )

    # Standalone compaction/summarization: kept verbatim.
    compaction = trajectory(
        "run:comp",
        [SYSTEM, USER, assistant(content="Summary of work so far.")],
        [
            turn(
                4,
                2,
                trained=True,
                response=assistant(content="Summary of work so far."),
            )
        ],
        kind="compaction",
        leaf=4,
        trajectory_index=3,
    )

    # Sub-agent fragments split the same way by normalized tool arguments.
    sub_original_1 = assistant(
        calls=[call("sub_call_1", "Bash", {"command": "cd /repo && ls"})]
    )
    sub_replayed_1 = assistant(calls=[call("sub_call_1", "Bash", {"command": "ls"})])
    sub_original_2 = assistant(
        calls=[call("sub_call_2", "Bash", {"command": "cd /repo && cat a.txt"})]
    )
    sub_a = trajectory(
        "run:sub1",
        [AUX_SYSTEM, USER, sub_original_1],
        [
            turn(
                5,
                2,
                trained=True,
                response=sub_original_1,
                agent_id="sub-1",
                agent_type="subagent",
            )
        ],
        leaf=5,
        trajectory_index=4,
    )
    sub_b = trajectory(
        "run:sub2",
        [
            AUX_SYSTEM,
            USER,
            sub_replayed_1,
            tool_result("sub_call_1", "a"),
            sub_original_2,
        ],
        [
            turn(
                5,
                2,
                trained=False,
                response=sub_replayed_1,
                agent_id="sub-1",
                agent_type="subagent",
            ),
            turn(
                6,
                4,
                trained=True,
                response=sub_original_2,
                agent_id="sub-1",
                agent_type="subagent",
            ),
        ],
        parent="run:sub1",
        leaf=6,
        trajectory_index=5,
    )

    return {
        "instance_id": "owner__repo-1",
        "attempt": 1,
        "harness": "claude-code",
        "messages": [SYSTEM, USER, replayed_1, result_1, original_2],
        "tools": [BASH],
        "tool_sets": [[BASH]],
        "trajectories": [fragment_a, fragment_b, auxiliary, compaction, sub_a, sub_b],
        "stats": {"requests": 6, "trajectories": 6},
        "request_log": [{"turn_index": 1}, {"turn_index": 2}],
        "test_result": {"git_patch": "diff --git a/x b/x"},
        "run": {"exit_code": 0},
    }


def by_id(history: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        trajectory["trajectory_id"]: trajectory
        for trajectory in history["trajectories"]
    }


def test_streamline_merges_fragments_and_restores_original_arguments() -> None:
    result = streamline.streamline_history(make_history())
    trajectories = by_id(result)

    # run:1 was absorbed into run:2, run:sub1 into run:sub2; aux + compaction kept.
    assert set(trajectories) == {"run:2", "run:aux", "run:comp", "run:sub2"}
    assert len(result["trajectories"]) == 4

    main = trajectories["run:2"]
    assert main["streamlined"]["absorbed_trajectory_ids"] == ["run:1"]
    assert main["streamlined"]["restored_tool_call_messages"] == 1
    assert main["parent_trajectory_id"] is None
    # Replayed "<tool> ls" is replaced by the model's original "cd /testbed && ls".
    arguments = main["messages"][2]["tool_calls"][0]["function"]["arguments"]
    assert "cd /testbed && ls" in arguments
    # Union of the fragments' trained turns: both assistant responses are targets.
    assert main["loss_mask"] == [0, 0, 1, 0, 1]
    assert main["trained_turns"] == [1, 2]
    assert main["leaf_turn_index"] == 2
    assert streamline.resolve_trajectory_tools(main, result) == [BASH]

    sub = trajectories["run:sub2"]
    assert sub["streamlined"]["absorbed_trajectory_ids"] == ["run:sub1"]
    assert (
        "cd /repo && ls" in sub["messages"][2]["tool_calls"][0]["function"]["arguments"]
    )
    assert sub["loss_mask"] == [0, 0, 1, 0, 1]


def test_streamline_reattaches_stop_sequences_but_does_not_merge_aux() -> None:
    history = make_history()
    before = by_id(history)
    result = streamline.streamline_history(history)
    after = by_id(result)

    # Auxiliary request: not merged, but its stripped stop sequence is re-attached.
    auxiliary = after["run:aux"]
    assert auxiliary["messages"][2]["content"] == "<block>no</block>"
    assert auxiliary["restored_stop_sequences"] == [
        {"message_index": 2, "turn_index": 3, "stop_sequence": "</block>"}
    ]
    assert "streamlined" not in auxiliary

    # Compaction trajectory: no stop sequence configured; only the folded
    # per-message loss_mask is added to its assistant message.
    compaction_before = before["run:comp"]
    compaction_after = after["run:comp"]
    assert [
        {**message, "loss_mask": 1} if message.get("role") == "assistant" else message
        for message in compaction_before["messages"]
    ] == compaction_after["messages"]
    assert compaction_after["turns"] == compaction_before["turns"]

    assert result["streamlined"]["auxiliary_kept"] == 1
    assert result["streamlined"]["compaction_kept"] == 1
    assert result["streamlined"]["restored_stop_sequences"] == 1

    # Opt-out keeps the API-verbatim assistant text.
    verbatim = streamline.streamline_history(history, restore_stop_sequences=False)
    assert by_id(verbatim)["run:aux"]["messages"][2]["content"] == "<block>no"


def test_streamline_root_messages_from_latest_main_trajectory() -> None:
    history = make_history()
    # Latest traffic overall is auxiliary; it must not become the root messages.
    history["trajectories"][2]["leaf_turn_index"] = 99
    result = streamline.streamline_history(history)
    main = by_id(result)["run:2"]
    assert result["messages"] == main["messages"]
    assert result["streamlined"]["root_messages_from_trajectory_id"] == "run:2"
    assert result["tools"] == [BASH]


def test_streamline_keeps_divergent_branches() -> None:
    original_1 = assistant(calls=[call("call_1", "Bash", {"command": "cd /x && ls"})])
    replayed_1 = assistant(calls=[call("call_1", "Bash", {"command": "ls"})])
    tool_1 = tool_result("call_1", "ok")
    original_2 = assistant(calls=[call("call_2", "Bash", {"command": "cd /x && pwd"})])
    other_2 = assistant(
        calls=[call("call_3", "Bash", {"command": "cd /x && git diff"})]
    )

    base = trajectory(
        "run:1",
        [SYSTEM, USER, original_1],
        [turn(1, 2, trained=True, response=original_1)],
        leaf=1,
    )
    branch_a = trajectory(
        "run:2",
        [SYSTEM, USER, replayed_1, tool_1, original_2],
        [
            turn(1, 2, trained=False, response=replayed_1),
            turn(2, 4, trained=True, response=original_2),
        ],
        leaf=2,
        trajectory_index=1,
    )
    branch_b = trajectory(
        "run:3",
        [SYSTEM, USER, replayed_1, tool_1, other_2],
        [
            turn(1, 2, trained=False, response=replayed_1),
            turn(2, 4, trained=True, response=other_2),
        ],
        leaf=3,
        trajectory_index=2,
    )
    history = {
        "instance_id": "owner__repo-1",
        "messages": [],
        "tools": [BASH],
        "trajectories": [base, branch_a, branch_b],
    }

    result = streamline.streamline_history(history)
    trajectories = by_id(result)
    # The base fragment is absorbed into one branch; both divergent leaves stay.
    assert "run:1" not in trajectories
    assert {"run:2", "run:3"} <= set(trajectories)
    assert result["streamlined"]["fragments_before"] == 3
    assert result["streamlined"]["trajectories_after"] == 2
    # The branch leaf that was not merged still gets the model's original args.
    branch = trajectories["run:2"]
    assert (
        "cd /x && ls" in branch["messages"][2]["tool_calls"][0]["function"]["arguments"]
    )
    assert branch["restored_tool_call_messages"] == 1


def test_streamline_history_file_and_attempt_skip(tmp_path: Path) -> None:
    source = tmp_path / "owner__repo-1.history.json"
    source.write_text(json.dumps(make_history()), encoding="utf-8")
    attempt = tmp_path / "owner__repo-1.attempt1.history.json"
    attempt.write_text(json.dumps(make_history()), encoding="utf-8")

    summary = streamline.streamline_history_file(source)
    assert summary is not None
    target = tmp_path / "owner__repo-1.history.streamlined.json"
    assert target.is_file()
    text = target.read_text(encoding="utf-8")
    assert text.endswith("\n") and text.count("\n") == 1
    data = json.loads(text)
    assert data["streamlined"]["fragments_before"] == 6
    assert data["streamlined"]["trajectories_after"] == 4

    # Idempotent: a second run skips the existing output.
    assert streamline.streamline_history_file(source) is None
    assert streamline.streamline_history_file(source, overwrite=True) is not None


def test_streamline_main_processes_only_final_histories(
    tmp_path: Path,
) -> None:
    source = tmp_path / "claude-code"
    source.mkdir()
    (source / "owner__repo-1.history.json").write_text(
        json.dumps(make_history()), encoding="utf-8"
    )
    (source / "owner__repo-1.attempt1.history.json").write_text(
        json.dumps(make_history()), encoding="utf-8"
    )

    assert streamline.main(["--dir", str(source)]) == 0
    assert (source / "owner__repo-1.history.streamlined.json").is_file()
    assert not (source / "owner__repo-1.attempt1.history.streamlined.json").exists()


def test_streamline_list_discovers_claude_code_dirs(tmp_path: Path) -> None:
    directory = tmp_path / "0" / "dataset" / "harnesses" / "claude-code"
    directory.mkdir(parents=True)
    (directory / "owner__repo-1.history.json").write_text(
        json.dumps(make_history()), encoding="utf-8"
    )
    assert streamline.main(["--root", str(tmp_path), "--list"]) == 0


def test_streamline_dry_run_writes_nothing(tmp_path: Path) -> None:
    source = tmp_path / "claude-code"
    source.mkdir()
    (source / "owner__repo-1.history.json").write_text(
        json.dumps(make_history()), encoding="utf-8"
    )
    assert streamline.main(["--dir", str(source), "--dry-run"]) == 0
    assert not (source / "owner__repo-1.history.streamlined.json").exists()


def test_argument_normalization_matches_replay_rewrites_only() -> None:
    bash_original = call("c1", "Bash", {"command": "cd /testbed && ls"})
    bash_replayed = call("c1", "Bash", {"command": "ls"})
    assert streamline._arguments_match(bash_original, bash_replayed)
    # A real command change is not just a stripped cwd.
    assert not streamline._arguments_match(
        call("c1", "Bash", {"command": "cd /testbed && pwd"}),
        bash_replayed,
    )

    edit_original = call(
        "c2",
        "Edit",
        {"file_path": "/x", "old_string": "a", "new_string": "b  \n"},
    )
    edit_replayed = call(
        "c2",
        "Edit",
        {
            "replace_all": False,
            "file_path": "/x",
            "old_string": "a",
            "new_string": "b",
        },
    )
    assert streamline._arguments_match(edit_original, edit_replayed)
    assert not streamline._arguments_match(
        call("c2", "Edit", {"file_path": "/x", "old_string": "a", "new_string": "c"}),
        edit_replayed,
    )


def test_prefix_matching_uses_restored_original_arguments() -> None:
    """A replay rewrite outside the small whitelist must not keep a branch."""
    original = call("call_read", "Read", {"file_path": "/x"})
    replayed = call("call_read", "Read", {"file_path": "/x", "limit": 2000})
    result = tool_result("call_read", "contents")
    later = call("call_2", "Bash", {"command": "cd /x && pwd"})

    fragment_a = trajectory(
        "run:1",
        [SYSTEM, USER, assistant(calls=[original])],
        [turn(1, 2, trained=True, response=assistant(calls=[original]))],
        leaf=1,
        trajectory_index=0,
    )
    fragment_b = trajectory(
        "run:2",
        [SYSTEM, USER, assistant(calls=[replayed]), result, assistant(calls=[later])],
        [
            turn(1, 2, trained=False, response=assistant(calls=[replayed])),
            turn(2, 4, trained=True, response=assistant(calls=[later])),
        ],
        leaf=2,
        trajectory_index=1,
    )
    # Raw replay copies differ beyond the known normalizations...
    assert not streamline._messages_match(
        fragment_b["messages"][2], fragment_a["messages"][2]
    )

    history = {
        "instance_id": "owner__repo-1",
        "messages": [],
        "tools": [BASH],
        "trajectories": [fragment_a, fragment_b],
    }
    streamlined = streamline.streamline_history(history)
    # ... but restoring the captured original arguments makes them a prefix.
    assert len(streamlined["trajectories"]) == 1
    merged = streamlined["trajectories"][0]
    assert merged["streamlined"]["absorbed_trajectory_ids"] == ["run:1"]
    assert merged["loss_mask"] == [0, 0, 1, 0, 1]
    assert json.loads(
        merged["messages"][2]["tool_calls"][0]["function"]["arguments"]
    ) == {"file_path": "/x"}


def test_merge_fills_reasoning_missing_from_base_context() -> None:
    original = assistant(calls=[call("call_1", "Bash", {"command": "cd /x && ls"})])
    replayed = assistant(calls=[call("call_1", "Bash", {"command": "ls"})])
    later = assistant(content="done")
    fragment_a = trajectory(
        "run:1",
        [SYSTEM, USER, original],
        [turn(1, 2, trained=True, response=original)],
        leaf=1,
        trajectory_index=0,
    )
    fragment_a["reasoning_content"] = [None, None, "absorbed thought"]
    fragment_b = trajectory(
        "run:2",
        [SYSTEM, USER, replayed, tool_result("call_1", "ok"), later],
        [
            turn(1, 2, trained=False, response=replayed),
            turn(2, 4, trained=True, response=later),
        ],
        leaf=2,
        trajectory_index=1,
    )
    fragment_b["reasoning_content"] = [None, None, None, None, "own thought"]
    history = {
        "instance_id": "owner__repo-1",
        "messages": [],
        "tools": [BASH],
        "trajectories": [fragment_a, fragment_b],
    }
    merged = streamline.streamline_history(history)["trajectories"][0]
    assert merged["streamlined"]["absorbed_trajectory_ids"] == ["run:1"]
    assert merged["reasoning_content"] == [
        None,
        None,
        "absorbed thought",
        None,
        "own thought",
    ]
    assert merged["messages"][2]["reasoning_content"] == "absorbed thought"
    assert merged["messages"][4]["reasoning_content"] == "own thought"
    # reasoning_content sits right after role, before content/tool_calls.
    assert list(merged["messages"][2].keys()) == [
        "role",
        "reasoning_content",
        "content",
        "tool_calls",
        "loss_mask",
    ]
    assert list(merged["messages"][4].keys()) == [
        "role",
        "reasoning_content",
        "content",
        "loss_mask",
    ]
    assert merged["messages"][2]["loss_mask"] == 1
    assert merged["messages"][4]["loss_mask"] == 1
    assert (
        streamline.streamline_history(history)["streamlined"][
            "loss_mask_embedded_messages"
        ]
        == 2
    )
    assert (
        streamline.streamline_history(history)["streamlined"][
            "reasoning_embedded_messages"
        ]
        == 2
    )


def test_streamline_regenerates_stale_companion(tmp_path: Path) -> None:
    source = tmp_path / "owner__repo-1.history.json"
    source.write_text(json.dumps(make_history()), encoding="utf-8")
    stale = tmp_path / "owner__repo-1.history.streamlined.json"
    # An older output: pretty-printed and without the new summary marker.
    # Older outputs may have the reasoning marker but not the loss-mask one.
    stale.write_text(
        json.dumps(
            {
                "streamlined": {
                    "fragments_before": 99,
                    "reasoning_embedded_messages": 1,
                }
            }
        ),
        encoding="utf-8",
    )

    summary = streamline.streamline_history_file(source)
    assert summary is not None
    refreshed = json.loads(stale.read_text(encoding="utf-8"))
    assert refreshed["streamlined"]["trajectories_after"] == 4
    assert "reasoning_embedded_messages" in refreshed["streamlined"]
    assert "loss_mask_embedded_messages" in refreshed["streamlined"]
    # A current companion is skipped on the next run.
    assert streamline.streamline_history_file(source) is None


def test_streamline_progress_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "claude-code"
    source.mkdir()
    (source / "owner__repo-1.history.json").write_text(
        json.dumps(make_history()), encoding="utf-8"
    )
    assert (
        streamline.main(
            [
                "--dir",
                str(source),
                "--progress",
                "--progress-interval",
                "0.01",
            ]
        )
        == 0
    )
    assert "[streamline]" in capsys.readouterr().err
