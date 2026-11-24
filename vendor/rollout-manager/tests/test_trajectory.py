"""Unit tests for the single-session trajectory manager."""

from __future__ import annotations

from rollout_manager.trajectory import (
    TrajectoryManager,
    TurnRecord,
    common_prefix_len,
)

STRICT_MESSAGE_FIELDS = {"role", "content", "name", "tool_calls", "tool_call_id"}


def user(text: str) -> dict:
    return {"role": "user", "content": text}


def assistant(text: str = "", *, tool_calls=None, reasoning: str | None = None) -> dict:
    message: dict = {"role": "assistant", "content": text}
    if reasoning:
        message["reasoning_content"] = reasoning
    if tool_calls:
        message["tool_calls"] = [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
            for call_id, name, arguments in tool_calls
        ]
    return message


def tool(text: str, call_id: str = "call_1") -> dict:
    return {"role": "tool", "tool_call_id": call_id, "content": text}


def turn(text: str = "", *, tools=None, **kwargs) -> TurnRecord:
    return TurnRecord(response_message=assistant(text), tools=tools, **kwargs)


def trajectory_messages(trajectory: dict) -> list[tuple[str, str]]:
    return [(message["role"], message["content"]) for message in trajectory["messages"]]


def assert_strict_messages(trajectory: dict) -> None:
    """Exported messages must be valid OpenAI Chat-Completions shapes, and
    every tool result must correlate with a preceding tool call in the same
    trajectory."""
    messages = trajectory["messages"]
    assert len(trajectory["loss_mask"]) == len(messages)
    assert len(trajectory["reasoning_content"]) == len(messages)
    call_ids: set[str] = set()
    for message in messages:
        assert set(message) <= STRICT_MESSAGE_FIELDS, message
        assert "loss_mask" not in message and "reasoning_content" not in message
        assert message["role"] in {"system", "developer", "user", "assistant", "tool"}
        assert message["content"] is None or isinstance(message["content"], (str, list))
        for call in message.get("tool_calls") or []:
            assert isinstance(call["id"], str) and call["id"]
            assert isinstance(call["function"]["arguments"], str)
            call_ids.add(call["id"])
        if message["role"] == "tool":
            assert isinstance(message["tool_call_id"], str)
            assert message["tool_call_id"] in call_ids, f"orphan tool result: {message['tool_call_id']}"


def manager() -> TrajectoryManager:
    return TrajectoryManager("s")


def test_common_prefix_len():
    base = [user("hi"), assistant("hello")]
    assert common_prefix_len(base, base) == 2
    assert common_prefix_len(base, [*base, user("next")]) == 2
    assert common_prefix_len(base, [user("hi"), assistant("different")]) == 1
    assert common_prefix_len([], base) == 0


def test_prefix_extension_assembles_one_trajectory():
    trajectory_manager = manager()
    trajectory_manager.record_turn(prompt_messages=[user("hi")], turn=turn("A1"))
    trajectory_manager.record_turn(prompt_messages=[user("hi"), assistant("A1"), user("next")], turn=turn("A2"))

    trajectories = trajectory_manager.get_trajectories()
    assert len(trajectories) == 1
    trajectory = trajectories[0]
    assert trajectory["trajectory_id"] == "s:2"
    assert trajectory["trajectory_index"] == 0
    assert trajectory["first_turn_index"] == 1
    assert trajectory["kind"] == "conversation"
    assert trajectory["contains_compaction"] is False
    assert trajectory["exclude_from_training"] is False
    assert trajectory["parent_trajectory_id"] is None
    assert trajectory_messages(trajectory) == [
        ("user", "hi"),
        ("assistant", "A1"),
        ("user", "next"),
        ("assistant", "A2"),
    ]
    assert trajectory["loss_mask"] == [0, 1, 0, 1]
    assert trajectory["reasoning_content"] == [None, None, None, None]
    assert [turn["message_index"] for turn in trajectory["turns"]] == [1, 3]
    assert [turn["turn_index"] for turn in trajectory["turns"]] == [1, 2]
    assert trajectory["path_turn_indices"] == [1, 2]
    assert all(turn["trained"] for turn in trajectory["turns"])
    assert trajectory_manager.turn_count == 2
    assert_strict_messages(trajectory)


def test_duplicate_tool_call_ids_do_not_cross_mount():
    """Two identical calls with different ids must not swap tool results."""
    call_a = assistant("", tool_calls=[("call_a", "bash", '{"cmd": "ls"}')])
    call_b = assistant("", tool_calls=[("call_b", "bash", '{"cmd": "ls"}')])
    trajectory_manager = manager()
    trajectory_manager.record_turn(prompt_messages=[user("run")], turn=TurnRecord(response_message=call_a))
    # The continuation belongs to call_b; prefix matching must not mount it
    # under the identical-looking call_a node.
    trajectory_manager.record_turn(
        prompt_messages=[user("run"), call_b, tool("42", "call_b"), user("next")],
        turn=turn("done"),
    )

    trajectories = trajectory_manager.get_trajectories()
    assert len(trajectories) == 2
    call_a_leaf = next(trajectory for trajectory in trajectories if len(trajectory["messages"]) == 2)
    call_b_branch = next(trajectory for trajectory in trajectories if len(trajectory["messages"]) == 5)

    assert call_a_leaf["messages"][1]["tool_calls"][0]["id"] == "call_a"
    assert [message["role"] for message in call_a_leaf["messages"]] == ["user", "assistant"]

    assert call_b_branch["messages"][1]["tool_calls"][0]["id"] == "call_b"
    assert call_b_branch["messages"][2]["tool_call_id"] == "call_b"
    assert call_b_branch["messages"][2]["content"] == "42"
    for trajectory in trajectories:
        assert_strict_messages(trajectory)


def test_branching_exports_one_trajectory_per_leaf_and_claims_shared_turns_once():
    trajectory_manager = manager()
    trajectory_manager.record_turn(prompt_messages=[user("hi")], turn=turn("A1"))
    trajectory_manager.record_turn(prompt_messages=[user("hi"), assistant("A1"), user("x")], turn=turn("B1"))
    trajectory_manager.record_turn(
        prompt_messages=[user("hi"), assistant("A1"), user("x"), assistant("B1"), user("y")],
        turn=turn("C1"),
    )
    trajectory_manager.record_turn(prompt_messages=[user("hi"), assistant("A1"), user("z")], turn=turn("B2"))

    trajectories = trajectory_manager.get_trajectories()
    assert len(trajectories) == 2
    deep, sibling = trajectories
    assert trajectory_messages(deep) == [
        ("user", "hi"),
        ("assistant", "A1"),
        ("user", "x"),
        ("assistant", "B1"),
        ("user", "y"),
        ("assistant", "C1"),
    ]
    assert [turn["trained"] for turn in deep["turns"]] == [True, True, True]
    assert deep["loss_mask"] == [0, 1, 0, 1, 0, 1]
    assert deep["trajectory_index"] == 0

    assert trajectory_messages(sibling) == [
        ("user", "hi"),
        ("assistant", "A1"),
        ("user", "z"),
        ("assistant", "B2"),
    ]
    assert [(turn["turn_index"], turn["trained"]) for turn in sibling["turns"]] == [(1, False), (4, True)]
    assert sibling["loss_mask"] == [0, 0, 0, 1]
    assert sibling["path_turn_indices"] == [1, 4]
    assert sibling["trajectory_index"] == 1
    assert sibling["parent_trajectory_id"] == deep["trajectory_id"]
    assert sibling["branch_message_index"] == 2
    assert sibling["branch_turn_index"] == 1
    assert_strict_messages(sibling)


def test_explicit_compaction_splits_into_marked_trajectory():
    tools = [{"type": "function", "function": {"name": "bash", "parameters": {}}}]
    trajectory_manager = manager()
    trajectory_manager.record_turn(prompt_messages=[user("task")], turn=turn("work", tools=tools))
    trajectory_manager.record_turn(
        prompt_messages=[user("task"), assistant("work"), user("summarize the session")],
        turn=turn(
            "summary text",
            tools=None,
            compaction=True,
            standalone=True,
            client={"request_class": "compaction", "compaction_trigger": "auto"},
        ),
    )
    trajectory_manager.record_turn(
        prompt_messages=[user("summary text"), user("keep going")],
        turn=turn("post-compaction work", tools=tools, client={"context_compacted": "auto"}),
    )

    trajectories = trajectory_manager.get_trajectories()
    assert len(trajectories) == 3
    before, compaction, after = trajectories

    assert before["kind"] == "conversation"
    assert before["loss_mask"] == [0, 1]

    assert compaction["kind"] == "compaction"
    assert compaction["contains_compaction"] is True
    assert [(m["role"], m["content"]) for m in compaction["messages"]] == [
        ("user", "task"),
        ("assistant", "work"),
        ("user", "summarize the session"),
        ("assistant", "summary text"),
    ]
    # Only the final summary message is trainable.
    assert compaction["loss_mask"] == [0, 0, 0, 1]
    assert [(turn["message_index"], turn["turn_index"], turn["trained"]) for turn in compaction["turns"]] == [
        (3, 2, True)
    ]
    assert compaction["turns"][0]["request_class"] == "compaction"
    assert compaction["turns"][0]["compaction_trigger"] == "auto"
    assert compaction["parent_trajectory_id"] == before["trajectory_id"]
    assert compaction["branch_message_index"] == 2
    assert compaction["branch_turn_index"] == 1

    assert after["kind"] == "conversation"
    assert after["loss_mask"] == [0, 0, 1]
    assert after["turns"][0]["context_compacted"] == "auto"
    assert_strict_messages(compaction)


def test_compaction_with_tools_is_labelled_but_not_split():
    tools = [{"type": "function", "function": {"name": "bash", "parameters": {}}}]
    trajectory_manager = manager()
    trajectory_manager.record_turn(prompt_messages=[user("task")], turn=turn("work", tools=tools))
    trajectory_manager.record_turn(
        prompt_messages=[user("task"), assistant("work"), user("summarize")],
        turn=turn(
            "summary",
            tools=tools,
            compaction=True,
            standalone=False,
            client={"request_class": "compaction", "compaction_trigger": "manual"},
        ),
    )

    trajectories = trajectory_manager.get_trajectories()
    assert len(trajectories) == 1
    trajectory = trajectories[0]
    assert trajectory["kind"] == "conversation"
    assert trajectory["contains_compaction"] is True
    assert trajectory["loss_mask"] == [0, 1, 0, 1]
    assert [turn["compaction"] for turn in trajectory["turns"]] == [False, True]


def test_rewritten_assistant_fork_keeps_generated_leaf():
    trajectory_manager = manager()
    trajectory_manager.record_turn(prompt_messages=[user("hi")], turn=turn("A1"))
    trajectory_manager.record_turn(
        prompt_messages=[user("hi"), assistant("A1 rewritten"), user("next")],
        turn=turn("A2"),
    )

    trajectories = trajectory_manager.get_trajectories()
    assert len(trajectories) == 2
    original, rewritten_branch = trajectories
    assert trajectory_messages(original) == [("user", "hi"), ("assistant", "A1")]
    assert original["loss_mask"] == [0, 1]
    assert trajectory_messages(rewritten_branch) == [
        ("user", "hi"),
        ("assistant", "A1 rewritten"),
        ("user", "next"),
        ("assistant", "A2"),
    ]
    # A1 was claimed by the original leaf, so it is context here.
    assert rewritten_branch["loss_mask"] == [0, 0, 0, 1]
    assert rewritten_branch["parent_trajectory_id"] is None


def test_assistant_prefill_is_routing_only_context():
    trajectory_manager = manager()
    trajectory_manager.record_turn(prompt_messages=[user("write a poem"), assistant("Roses are")], turn=turn(" red"))
    trajectories = trajectory_manager.get_trajectories()
    assert len(trajectories) == 1
    assert [turn["message_index"] for turn in trajectories[0]["turns"]] == [2]
    assert trajectories[0]["loss_mask"] == [0, 0, 1]
    assert trajectory_messages(trajectories[0])[1] == ("assistant", "Roses are")


def test_reasoning_is_separate_and_aligned():
    trajectory_manager = manager()
    trajectory_manager.record_turn(
        prompt_messages=[user("hi")],
        turn=TurnRecord(response_message=assistant("A", reasoning="because")),
    )
    trajectory = trajectory_manager.get_trajectories()[0]
    assert trajectory["messages"][-1] == {"role": "assistant", "content": "A"}
    assert trajectory["reasoning_content"] == [None, "because"]


def test_auxiliary_turn_can_be_excluded_from_training():
    trajectory_manager = manager()
    trajectory_manager.record_turn(
        prompt_messages=[user("classify")],
        turn=turn("no", client={"request_class": "auxiliary"}, exclude_from_training=True),
    )
    trajectory = trajectory_manager.get_trajectories()[0]
    assert trajectory["exclude_from_training"] is True
    assert trajectory["loss_mask"] == [0, 0]
    assert trajectory["turns"][0]["excluded"] is True
    assert trajectory["turns"][0]["request_class"] == "auxiliary"


def test_request_and_response_are_retained():
    trajectory_manager = manager()
    request = {"model": "m", "temperature": 0.2, "max_tokens": 64, "message_count": 1}
    response = assistant("A")
    trajectory_manager.record_turn(
        prompt_messages=[user("hi")],
        turn=TurnRecord(response_message=response, request=request, response=response),
    )
    exported_turn = trajectory_manager.get_trajectories()[0]["turns"][0]
    assert exported_turn["request"] == request
    assert exported_turn["response"] == response


def test_ordered_request_log_is_the_timeline():
    trajectory_manager = manager()
    trajectory_manager.record_turn(prompt_messages=[user("one")], turn=turn("A1", client={"request_class": "main"}))
    trajectory_manager.record_turn(
        prompt_messages=[user("one"), assistant("A1"), user("two")],
        turn=turn("A2", client={"request_class": "subagent", "agent_id": "agent-1"}),
    )
    trajectories = trajectory_manager.get_trajectories()
    log = trajectory_manager.get_request_log(trajectories)
    assert [entry["turn_index"] for entry in log] == [1, 2]
    assert [entry["request_class"] for entry in log] == ["main", "subagent"]
    assert [entry["trajectory_id"] for entry in log] == [trajectories[0]["trajectory_id"]] * 2
    assert log[1]["agent_id"] == "agent-1"


def test_deep_history_does_not_recurse():
    trajectory_manager = manager()
    prompt = [user(f"message-{index}") for index in range(1500)]
    trajectory_manager.record_turn(prompt_messages=prompt, turn=turn("done"))
    trajectories = trajectory_manager.get_trajectories()
    assert len(trajectories) == 1
    assert len(trajectories[0]["messages"]) == len(prompt) + 1


def test_get_trajectories_is_repeatable_and_clear_resets():
    trajectory_manager = manager()
    trajectory_manager.record_turn(prompt_messages=[user("hi")], turn=turn("A1"))
    first = trajectory_manager.get_trajectories()
    second = trajectory_manager.get_trajectories()
    assert first == second
    assert trajectory_manager.turn_count == 1
    trajectory_manager.clear()
    assert trajectory_manager.get_trajectories() == []
    assert trajectory_manager.turn_count == 0
    assert trajectory_manager.get_request_log([]) == []


def test_tool_sets_are_interned_per_trajectory_and_keep_strict():
    tools_a = [
        {
            "type": "function",
            "function": {"name": "read", "parameters": {}, "strict": True},
        }
    ]
    tools_b = [{"type": "function", "function": {"name": "write", "parameters": {}}}]
    trajectory_manager = manager()
    trajectory_manager.record_turn(prompt_messages=[user("a")], turn=turn("A", tools=tools_a))
    trajectory_manager.record_turn(
        prompt_messages=[user("a"), assistant("A"), user("b")], turn=turn("B", tools=tools_a)
    )
    trajectory_manager.record_turn(
        prompt_messages=[user("a"), assistant("A"), user("b"), assistant("B"), user("c")],
        turn=turn("C", tools=tools_b),
    )
    trajectory = trajectory_manager.get_trajectories()[0]
    assert trajectory["tool_sets"] == [tools_a, tools_b]
    assert trajectory["tool_sets"][0][0]["function"]["strict"] is True
    assert [turn["tool_set"] for turn in trajectory["turns"]] == [0, 0, 1]


def test_tool_call_ids_and_argument_strings_survive_export():
    tool_calls = [("call_1", "bash", '{"cmd": "ls"}')]
    trajectory_manager = manager()
    trajectory_manager.record_turn(
        prompt_messages=[user("run")],
        turn=TurnRecord(response_message=assistant("", tool_calls=tool_calls)),
    )
    trajectory = trajectory_manager.get_trajectories()[0]
    exported = trajectory["messages"][1]
    assert exported["tool_calls"] == [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "bash", "arguments": '{"cmd": "ls"}'},
        }
    ]
    assert trajectory["loss_mask"] == [0, 1]
