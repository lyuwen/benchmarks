"""Trajectory capture for one agent session.

An agent resends its whole conversation on every model turn, so request N+1's
message list is, in the common case, request N's message list plus the assistant
reply plus an observation. That "one request is a prefix of the next" relation is
what lets the capture service avoid storing duplicated histories: each request is
routed into the session's message tree and only the new tail is mounted.

Branches fall out of the same rule. A context compaction starts a new history
rooted at a summary, a sub-agent starts from a different system prompt, and a
client may rewrite history -- each one diverges from the held prefix and becomes
a new branch. Each routing leaf is exported as its own trajectory, so the turns
from before a compaction and the turns after it are both preserved.

Compaction is never inferred from request shape. It is only recognised from the
harness's explicit gateway hint headers (Claude Code's
``x-claude-code-request-class: compaction`` / ``x-claude-code-compaction``); a
tools-less request is just a tools-less request. Request purpose and lineage
metadata are recorded per turn and in the ordered request log.

Exported ``messages`` are strict OpenAI Chat-Completions messages: tool-call
ids, ``tool_call_id`` and JSON-encoded ``function.arguments`` strings are
preserved. The training mask and provider-specific reasoning are separate,
message-aligned arrays rather than message fields.

One :class:`TrajectoryManager` owns exactly one session: the service is launched
per agent run and torn down when that run finishes.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from collections.abc import Iterator
from typing import Any

from .convert import flatten_content, message_identity

logger = logging.getLogger(__name__)

__all__ = ["MessageNode", "TrajectoryManager", "TurnRecord", "common_prefix_len"]


# ===========================================================================
# TurnRecord
# ===========================================================================


@dataclasses.dataclass(frozen=True)
class TurnRecord:
    """One model turn as observed by the capture service.

    ``response_message`` is the normalized assistant message stored in the tree;
    ``request`` is the effective upstream chat-completions request (its
    ``messages`` are trimmed by the capture layer and replaced with
    ``message_count``); ``response`` is the original assistant message returned
    by the upstream. ``client`` holds request-purpose/lineage metadata extracted
    from Claude Code gateway hint headers.
    """

    response_message: dict[str, Any]
    finish_reason: str = "stop"
    usage: dict[str, Any] = dataclasses.field(default_factory=dict)
    response_id: str | None = None
    model: str | None = None
    protocol: str = "chat"
    tools: list[dict[str, Any]] | None = None
    latency_ms: float = 0.0
    request: dict[str, Any] = dataclasses.field(default_factory=dict)
    response: dict[str, Any] | None = None
    client: dict[str, Any] = dataclasses.field(default_factory=dict)
    # Explicitly signalled compaction request (see x-claude-code-* headers).
    compaction: bool = False
    # Recorded as its own branch instead of extending the current trajectory.
    standalone: bool = False
    # Excluded from the training mask (e.g. auxiliary classifier/title calls).
    exclude_from_training: bool = False
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)


# ===========================================================================
# tree
# ===========================================================================


class MessageNode:
    """One node in the session's routing tree, carrying one canonical chat message.

    ``turn`` is set only on assistant messages the model generated for this
    capture; every other node (system/user/tool, replayed assistant context) is
    routing-only.
    """

    __slots__ = ("role", "message", "parent", "children", "turn", "turn_index")

    def __init__(
        self,
        *,
        role: str | None = None,
        message: dict[str, Any] | None = None,
        parent: MessageNode | None = None,
    ) -> None:
        self.role = role
        self.message = message
        self.parent: MessageNode | None = parent
        self.children: list[MessageNode] = []
        self.turn: TurnRecord | None = None
        self.turn_index: int | None = None

    @property
    def is_root(self) -> bool:
        return self.parent is None

    def add_child(self, child: MessageNode) -> MessageNode:
        child.parent = self
        self.children.append(child)
        return child

    def path_from_root(self) -> list[MessageNode]:
        """Nodes from the first non-root ancestor down to self, in order."""
        chain: list[MessageNode] = []
        node: MessageNode | None = self
        while node is not None and not node.is_root:
            chain.append(node)
            node = node.parent
        chain.reverse()
        return chain

    def leaves(self) -> Iterator[MessageNode]:
        """Iterative depth-first leaves traversal (histories can be very deep)."""
        stack: list[MessageNode] = [self]
        while stack:
            node = stack.pop()
            if not node.children:
                yield node
                continue
            stack.extend(reversed(node.children))


def common_prefix_len(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> int:
    """Length of the shared message prefix of two conversations."""
    matched = 0
    for left_message, right_message in zip(left, right, strict=False):
        if message_identity(left_message) != message_identity(right_message):
            return matched
        matched += 1
    return matched


def _find_mount_point(root: MessageNode, messages: list[dict[str, Any]]) -> tuple[MessageNode, int]:
    """Walk down the tree matching each message by identity, returning where to mount the rest."""
    node = root
    depth = 0
    while depth < len(messages):
        target = message_identity(messages[depth])
        next_child = None
        for child in node.children:
            if child.message is not None and message_identity(child.message) == target:
                next_child = child
                break
        if next_child is None:
            break
        node = next_child
        depth += 1
    return node, depth


# ===========================================================================
# TrajectoryManager (one session)
# ===========================================================================


class TrajectoryManager:
    """Builds one session's prefix tree and linearizes its leaves into trajectories.

    ``record_turn`` mounts a request's prompt messages under the deepest matching
    node and attaches the assistant reply as a generated leaf. ``get_trajectories``
    is non-destructive: the capture service snapshots the session after every
    turn, so it can be called repeatedly.

    Traversal is depth-first in insertion order. When two leaves share generated
    turns (a prefix that both continuations extend), the first leaf to reach a
    turn exports it with ``trained=True``; later leaves export the same response
    as context, so a consumer trains each response once. Array order is tree
    order, not the session timeline; the ordered request log is the timeline.
    """

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self._root = MessageNode()
        self._turn_count = 0
        # Ordered capture log, independent of tree/leaf ordering.
        self._requests: list[dict[str, Any]] = []

    # -------------------- public ------------------------------------------

    @property
    def turn_count(self) -> int:
        return self._turn_count

    def record_turn(self, *, prompt_messages: list[dict[str, Any]], turn: TurnRecord) -> None:
        if not prompt_messages and not turn.response_message:
            logger.warning("record_turn: empty request and response; skipping")
            return
        if turn.standalone:
            # Explicit compaction/side request: keep its full prompt as fresh
            # routing-only context so the branch it continues stays a leaf and
            # is exported on its own. Only this response is trained there.
            node = self._root
            for message in prompt_messages:
                node = node.add_child(MessageNode(role=message.get("role"), message=message))
        else:
            node, depth = _find_mount_point(self._root, prompt_messages)
            for message in prompt_messages[depth:]:
                node = node.add_child(MessageNode(role=message.get("role"), message=message))

        assistant = MessageNode(role="assistant", message=turn.response_message)
        assistant.turn = turn
        assistant.turn_index = self._turn_count + 1
        self._turn_count = assistant.turn_index
        node.add_child(assistant)

        self._requests.append(
            {
                "turn_index": assistant.turn_index,
                "protocol": turn.protocol,
                "finish_reason": turn.finish_reason,
                "model": turn.model,
                "response_id": turn.response_id,
                "message_count": int(turn.request.get("message_count") or 0),
                "timestamp": turn.metadata.get("timestamp"),
                "compaction": turn.compaction,
                "standalone": turn.standalone,
                "excluded": turn.exclude_from_training,
                "request_class": turn.client.get("request_class"),
                "compaction_trigger": turn.client.get("compaction_trigger"),
                "context_compacted": turn.client.get("context_compacted"),
                "agent_id": turn.client.get("agent_id"),
                "agent_type": turn.client.get("agent_type"),
                "parent_agent_id": turn.client.get("parent_agent_id"),
                "prompt_id": turn.client.get("prompt_id"),
                "client_session_id": turn.client.get("client_session_id"),
            }
        )

    def get_trajectories(self) -> list[dict[str, Any]]:
        """Linearize every routing leaf into a strict-format trajectory record."""
        claimed: set[int] = set()
        trajectories: list[dict[str, Any]] = []

        for leaf in self._root.leaves():
            if leaf.is_root:
                continue
            chain = leaf.path_from_root()
            if not any(node.turn is not None for node in chain):
                continue

            tool_sets: list[list[dict[str, Any]]] = []
            tool_index: dict[str, int] = {}
            turns: list[dict[str, Any]] = []
            path_turn_indices: list[int] = []
            for message_index, node in enumerate(chain):
                turn = node.turn
                if turn is None:
                    continue
                trained = id(node) not in claimed and not turn.exclude_from_training
                claimed.add(id(node))
                path_turn_indices.append(int(node.turn_index or 0))
                client = turn.client
                turns.append(
                    {
                        "message_index": message_index,
                        "turn_index": node.turn_index,
                        "trained": trained,
                        "excluded": bool(turn.exclude_from_training),
                        "compaction": turn.compaction,
                        "standalone": turn.standalone,
                        "request_class": client.get("request_class"),
                        "compaction_trigger": client.get("compaction_trigger"),
                        "context_compacted": client.get("context_compacted"),
                        "agent_id": client.get("agent_id"),
                        "agent_type": client.get("agent_type"),
                        "parent_agent_id": client.get("parent_agent_id"),
                        "prompt_id": client.get("prompt_id"),
                        "client_session_id": client.get("client_session_id"),
                        "finish_reason": turn.finish_reason,
                        "response_id": turn.response_id,
                        "model": turn.model,
                        "protocol": turn.protocol,
                        "usage": dict(turn.usage),
                        "tool_set": _tool_set_index(tool_sets, tool_index, turn.tools),
                        "timestamp": turn.metadata.get("timestamp", 0.0),
                        "latency_ms": turn.latency_ms,
                        "request": dict(turn.request),
                        "response": dict(turn.response) if turn.response is not None else None,
                    }
                )
            if not turns:
                continue

            standalone = any(turn["standalone"] for turn in turns)
            contains_compaction = any(turn["compaction"] for turn in turns)
            if standalone:
                # A standalone compaction trajectory is context plus one summary:
                # only its final turn is trainable.
                for index, turn in enumerate(turns):
                    turn["trained"] = index == len(turns) - 1 and not turn["excluded"]

            loss_mask = [0] * len(chain)
            for turn in turns:
                if turn["trained"]:
                    loss_mask[turn["message_index"]] = 1
            messages = [_strict_message(node.message) for node in chain]
            reasoning_content = [node.message.get("reasoning_content") or None for node in chain]
            leaf_turn_index = path_turn_indices[-1]
            trajectories.append(
                {
                    "trajectory_id": f"{self.session_id}:{leaf_turn_index}",
                    "leaf_turn_index": leaf_turn_index,
                    "kind": "compaction" if standalone else "conversation",
                    "contains_compaction": contains_compaction,
                    "exclude_from_training": not any(loss_mask),
                    "path_turn_indices": path_turn_indices,
                    "messages": messages,
                    "loss_mask": loss_mask,
                    "reasoning_content": reasoning_content,
                    "tool_sets": tool_sets,
                    "turns": turns,
                }
            )

        _annotate_series(trajectories)
        return trajectories

    def get_request_log(self, trajectories: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Capture-ordered request log with the trajectory that owns each turn.

        This is the session timeline; the ``trajectories`` array is ordered by
        tree traversal and is not chronological.
        """
        owners: dict[int, str] = {}
        for trajectory in trajectories:
            for turn_index in trajectory["path_turn_indices"]:
                owners.setdefault(turn_index, trajectory["trajectory_id"])
        log: list[dict[str, Any]] = []
        for entry in self._requests:
            log.append({**entry, "trajectory_id": owners.get(entry["turn_index"])})
        return log

    def clear(self) -> None:
        self._root = MessageNode()
        self._turn_count = 0
        self._requests = []


# ===========================================================================
# helpers
# ===========================================================================


def _strict_message(message: dict[str, Any]) -> dict[str, Any]:
    """Render a stored message as a strict OpenAI Chat-Completions message.

    Keeps only schema fields; ``reasoning_content`` and training masks live in
    separate message-aligned arrays.
    """
    role = message.get("role")
    content = message.get("content")
    if content is not None and not isinstance(content, (str, list)):
        content = flatten_content(content)
    if role == "tool" and content is None:
        content = ""
    strict: dict[str, Any] = {"role": role, "content": content}
    if message.get("name"):
        strict["name"] = str(message["name"])
    if role == "tool":
        strict["tool_call_id"] = str(message.get("tool_call_id") or "")
        return strict
    if role != "assistant":
        return strict
    tool_calls: list[dict[str, Any]] = []
    for call in message.get("tool_calls") or []:
        function = call.get("function") if isinstance(call.get("function"), dict) else {}
        arguments = function.get("arguments")
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments, ensure_ascii=False) if arguments is not None else "{}"
        tool_calls.append(
            {
                "id": str(call.get("id") or ""),
                "type": "function",
                "function": {"name": str(function.get("name") or "tool"), "arguments": arguments},
            }
        )
    if tool_calls:
        strict["tool_calls"] = tool_calls
    return strict


def _annotate_series(trajectories: list[dict[str, Any]]) -> None:
    """Record tree order and branch provenance for the trajectory series."""
    for index, trajectory in enumerate(trajectories):
        trajectory["trajectory_index"] = index
        trajectory["first_turn_index"] = trajectory["path_turn_indices"][0]
        parent: dict[str, Any] | None = None
        shared_len = 0
        branch_turn: int | None = None
        for earlier in trajectories[:index]:
            shared = common_prefix_len(earlier["messages"], trajectory["messages"])
            if shared <= shared_len:
                continue
            candidate_turn = None
            for earlier_turn in earlier["turns"]:
                if earlier_turn["message_index"] < shared:
                    candidate_turn = earlier_turn["turn_index"]
            if candidate_turn is None:
                continue
            parent = earlier
            shared_len = shared
            branch_turn = candidate_turn
        trajectory["parent_trajectory_id"] = parent["trajectory_id"] if parent is not None else None
        trajectory["branch_message_index"] = shared_len if parent is not None else None
        trajectory["branch_turn_index"] = branch_turn


def _tool_set_index(
    tool_sets: list[list[dict[str, Any]]],
    tool_index: dict[str, int],
    tools: list[dict[str, Any]] | None,
) -> int | None:
    """Intern a turn's tool schema, returning its index in the trajectory's tool sets."""
    if not tools:
        return None
    key = json.dumps(tools, sort_keys=True, ensure_ascii=False)
    index = tool_index.get(key)
    if index is None:
        index = len(tool_sets)
        tool_index[key] = index
        tool_sets.append(tools)
    return index
