"""Flatten harness history dumps into training-ready trajectory records.

``run_infer_harnesses.py`` writes one gzipped
``attempts/<instance>.attemptN.history.json.gz`` per attempt. Typical downstream use is to concatenate those into a single file and
turn every captured trajectory into one record that carries OpenAI
Chat-Completions ``messages`` and ``tools``.

The input reader accepts both a real JSONL file (one JSON document per line) and
a concatenation of pretty-printed JSON documents (e.g. plain ``cat`` of the
history files); it streams with an incremental JSON decoder instead of assuming
one document per line.

Two output shapes are available:

``--mode trajectories`` (default)
    One record per captured trajectory fragment. Each record is a valid
    loss-masked sample: the shared conversation prefix is context
    (``loss_mask=0``) and the fragment's own turns are targets
    (``loss_mask=1``). This is the raw export shape, flattened.

``--mode stitch``
    One record per history row: the longest conversation leaf already contains
    the complete linear history, so it is emitted as a single trajectory with
    ``loss_mask=1`` on every assistant message. Use ``--stitch-verbatim`` to
    restore the model's original tool-call arguments where the client (Claude
    Code) rewrote them when replaying history.

Every emitted record has ``messages`` (strict OpenAI chat messages) and
``tools`` (one OpenAI function-tool list, resolved from the trajectory's
``tool_sets``/``turn.tool_set``); ``loss_mask`` and ``reasoning_content`` are
kept as the aligned arrays the capture uses.

One training-side normalization: when a request configured a single stop
sequence and the reply finished on it (e.g. Claude Code's auto-mode classifier
sends ``stop=["</block>"]`` and receives ``"<block>no"``), the API strips that
sequence from the returned content, so the flattener appends it back and records
the change in ``restored_stop_sequences``. Pass ``--no-restore-stop-sequences``
for verbatim API content.
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import os
import sys
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor, wait
from pathlib import Path
from queue import Empty
from typing import Any

import orjson
from tqdm import tqdm

from benchmarks.utils.harnesses.history_io import (
    open_binary_auto,
    open_text_auto,
)
from benchmarks.utils.harnesses.normalize import classifier_label, normalize_record
from benchmarks.utils.harnesses.progress import (
    add_progress_arguments,
    reporter_from_args,
)
from benchmarks.utils.harnesses.rollout import (
    embed_loss_mask_into_messages,
    embed_reasoning_into_messages,
    resolve_trajectory_tools,
    restore_stop_sequences as attach_stop_sequences,
)


logger = logging.getLogger(__name__)

READ_CHUNK_BYTES = 1 << 20
VALID_ROLES = {"system", "user", "assistant", "tool", "developer"}
MODES = ("trajectories", "stitch")


# ---------------------------------------------------------------------------
# input
# ---------------------------------------------------------------------------


def is_jsonl_file(path: str | Path) -> bool:
    """Whether the first non-empty line is a complete JSON document."""
    try:
        with open(path, "rb") as handle:
            for raw in handle:
                if raw.strip():
                    try:
                        orjson.loads(raw)
                    except orjson.JSONDecodeError:
                        return False
                    return True
    except OSError:
        return False
    return False


def iter_json_documents(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield JSON objects from JSONL (fast orjson path) or concatenated docs."""
    seen_first = False
    with open_binary_auto(path) as handle:
        for raw in handle:
            if not raw.strip():
                continue
            if not seen_first:
                seen_first = True
                try:
                    document = orjson.loads(raw)
                except orjson.JSONDecodeError:
                    # Pretty-printed/concatenated documents: use the scanner.
                    yield from _iter_concatenated_json_documents(path)
                    return
                if not isinstance(document, dict):
                    raise ValueError(
                        f"{path}: expected JSON objects, got {type(document)}"
                    )
                yield document
                continue
            try:
                document = orjson.loads(raw)
            except orjson.JSONDecodeError as exc:
                raise ValueError(f"{path}: invalid JSONL line: {exc}") from exc
            if not isinstance(document, dict):
                raise ValueError(f"{path}: expected JSON objects, got {type(document)}")
            yield document


def _iter_concatenated_json_documents(
    path: str | Path,
) -> Iterator[dict[str, Any]]:
    """Yield JSON objects from JSONL or concatenated pretty-printed documents."""
    decoder = json.JSONDecoder()
    with open_text_auto(path) as handle:
        buffer = ""
        position = 0
        eof = False
        while True:
            while position < len(buffer) and buffer[position].isspace():
                position += 1
            if position >= len(buffer):
                if eof:
                    return
                chunk = handle.read(READ_CHUNK_BYTES)
                buffer = buffer[position:] + chunk
                position = 0
                eof = chunk == ""
                continue
            try:
                document, end = decoder.raw_decode(buffer, position)
            except json.JSONDecodeError as exc:
                if eof:
                    raise ValueError(
                        f"{path}: invalid JSON at character {position}: {exc}"
                    ) from exc
                chunk = handle.read(READ_CHUNK_BYTES)
                buffer = buffer[position:] + chunk
                position = 0
                if chunk == "":
                    eof = True
                continue
            if not isinstance(document, dict):
                raise ValueError(f"{path}: expected JSON objects, got {type(document)}")
            yield document
            position = end


# ---------------------------------------------------------------------------
# field resolution
# ---------------------------------------------------------------------------


def resolve_tools(
    trajectory: dict[str, Any], history: dict[str, Any]
) -> list[dict[str, Any]]:
    """Resolve the OpenAI function-tool list used by one captured trajectory."""
    return resolve_trajectory_tools(trajectory, history)


def _aligned(value: Any, size: int) -> list[Any] | None:
    return value if isinstance(value, list) and len(value) == size else None


def _turn_indices(trajectory: dict[str, Any]) -> list[int]:
    indices = [
        int(turn["turn_index"])
        for turn in trajectory.get("turns") or []
        if isinstance(turn, dict) and isinstance(turn.get("turn_index"), int)
    ]
    return sorted(set(indices))


def _trained_turns(trajectory: dict[str, Any]) -> list[int]:
    return sorted(
        {
            int(turn["turn_index"])
            for turn in trajectory.get("turns") or []
            if isinstance(turn, dict)
            and turn.get("trained")
            and isinstance(turn.get("turn_index"), int)
        }
    )


# ---------------------------------------------------------------------------
# flattening
# ---------------------------------------------------------------------------


def _base_record(history: dict[str, Any]) -> dict[str, Any]:
    record: dict[str, Any] = {
        "instance_id": history.get("instance_id"),
        "attempt": history.get("attempt"),
        "harness": history.get("harness"),
        "harness_version": history.get("harness_version"),
        "model": history.get("model"),
        "upstream_model": history.get("upstream_model"),
        "harness_model": history.get("harness_model"),
        "instruction": history.get("instruction"),
        "test_result": history.get("test_result") or {},
        "rollout_export": history.get("rollout_export"),
        "stats": history.get("stats") or {},
    }
    if "evaluation" in history:
        record["evaluation"] = history.get("evaluation")
    repository = history.get("repository")
    if isinstance(repository, dict):
        record["repository"] = repository
    return record


def _loss_mask_from_messages(messages: list[dict[str, Any]]) -> list[int] | None:
    """Recover the aligned loss mask from folded per-message fields.

    Streamlined histories fold ``loss_mask`` into assistant messages and the
    compact ``streamlined-lite`` companion drops the trajectory-level array, so
    flattening that file recovers the mask here. Returns ``None`` when no
    message carries the field (raw captures keep the array instead).
    """
    if not any(
        isinstance(message, dict)
        and message.get("role") == "assistant"
        and "loss_mask" in message
        for message in messages
    ):
        return None
    return [
        1
        if isinstance(message, dict)
        and message.get("role") == "assistant"
        and message.get("loss_mask")
        else 0
        for message in messages
    ]


def _reasoning_from_messages(
    messages: list[dict[str, Any]],
) -> list[str | None] | None:
    """Recover the aligned reasoning array from folded per-message fields."""
    if not any(
        isinstance(message, dict) and message.get("reasoning_content")
        for message in messages
    ):
        return None
    return [
        message.get("reasoning_content") if isinstance(message, dict) else None
        for message in messages
    ]


def flatten_trajectory(
    history: dict[str, Any],
    trajectory: dict[str, Any],
    *,
    restore_stop_sequences: bool = True,
) -> dict[str, Any]:
    """Flatten one captured trajectory fragment into a training-ready record."""
    messages = [message for message in trajectory.get("messages") or [] if message]
    embed_reasoning_into_messages(messages, trajectory.get("reasoning_content"))
    restored = (
        attach_stop_sequences(trajectory, messages) if restore_stop_sequences else []
    )
    record = _base_record(history)
    record.update(
        {
            "source": "trajectory",
            "trajectory_id": trajectory.get("trajectory_id"),
            "trajectory_index": trajectory.get("trajectory_index"),
            "parent_trajectory_id": trajectory.get("parent_trajectory_id"),
            "branch_message_index": trajectory.get("branch_message_index"),
            "branch_turn_index": trajectory.get("branch_turn_index"),
            "kind": trajectory.get("kind", "conversation"),
            "contains_compaction": bool(trajectory.get("contains_compaction")),
            "exclude_from_training": bool(trajectory.get("exclude_from_training")),
            "path_turn_indices": trajectory.get("path_turn_indices") or [],
            "turn_indices": _turn_indices(trajectory),
            "trained_turns": _trained_turns(trajectory),
            "messages": messages,
            "tools": resolve_tools(trajectory, history),
        }
    )
    if restored:
        record["restored_stop_sequences"] = restored
    loss_mask = _aligned(trajectory.get("loss_mask"), len(messages))
    if loss_mask is None:
        loss_mask = _loss_mask_from_messages(messages)
    if loss_mask is not None:
        record["loss_mask"] = loss_mask
        embed_loss_mask_into_messages(messages, loss_mask)
    reasoning = _aligned(trajectory.get("reasoning_content"), len(messages))
    if reasoning is None:
        reasoning = _reasoning_from_messages(messages)
    if reasoning is not None:
        record["reasoning_content"] = reasoning
    return record


def split_trajectory_by_tool_sets(
    history: dict[str, Any],
    trajectory: dict[str, Any],
    *,
    excluded: bool = False,
    train_excluded: bool = False,
) -> list[dict[str, Any]]:
    """Split one trajectory into contiguous segments with a constant tool list.

    The capture keeps a trajectory intact when the harness changes its tool
    schema mid-session (tools are not part of message identity), storing the
    variants in ``tool_sets`` and a per-turn ``tool_set`` index. For training
    that is ambiguous (an OpenAI sample has exactly one ``tools`` field), so this
    emits one record per consecutive tool-set phase: the shared history before
    each phase is duplicated as context and only the turns produced under that
    tool list are training targets.
    """
    base = flatten_trajectory(history, trajectory)
    messages = base["messages"]
    full_loss = base.get("loss_mask")
    full_reasoning = base.get("reasoning_content")
    turns = [
        turn
        for turn in trajectory.get("turns") or []
        if isinstance(turn, dict) and isinstance(turn.get("message_index"), int)
    ]
    turns.sort(key=lambda turn: int(turn["message_index"]))
    tool_sets = trajectory.get("tool_sets")
    if not isinstance(tool_sets, list):
        tool_sets = []

    groups: list[list[dict[str, Any]]] = []
    for turn in turns:
        if groups and groups[-1][-1].get("tool_set") == turn.get("tool_set"):
            groups[-1].append(turn)
        else:
            groups.append([turn])

    if len(groups) <= 1:
        if excluded and not train_excluded:
            # Captured as excluded: honour the exclusion (no targets).
            base["loss_mask"] = [0] * len(messages)
        elif excluded and train_excluded and groups:
            # Only this trajectory's own turns are targets (context stays 0).
            base["loss_mask"] = [0] * len(messages)
            for turn in groups[0]:
                index = int(turn["message_index"])
                if (
                    0 <= index < len(messages)
                    and messages[index].get("role") == "assistant"
                ):
                    base["loss_mask"][index] = 1
            base["loss_mask_recomputed"] = True
            embed_loss_mask_into_messages(base["messages"], base["loss_mask"])
        return [base]

    records: list[dict[str, Any]] = []
    for split_index, group in enumerate(groups):
        end_index = min(
            max(int(turn["message_index"]) for turn in group) + 1, len(messages)
        )
        segment_messages = messages[:end_index]
        segment_loss = [0] * end_index
        for turn in group:
            index = int(turn["message_index"])
            if not 0 <= index < end_index:
                continue
            is_assistant = segment_messages[index].get("role") == "assistant"
            if excluded and train_excluded:
                if is_assistant:
                    segment_loss[index] = 1
            elif not excluded and turn.get("trained") and is_assistant:
                if full_loss is None or (index < len(full_loss) and full_loss[index]):
                    segment_loss[index] = 1
        tool_set_index = group[-1].get("tool_set")
        if (
            isinstance(tool_set_index, int)
            and 0 <= tool_set_index < len(tool_sets)
            and isinstance(tool_sets[tool_set_index], list)
        ):
            tools = tool_sets[tool_set_index]
        else:
            tools = []
        record = dict(base)
        record["messages"] = segment_messages
        record["tools"] = tools
        record["loss_mask"] = segment_loss
        embed_loss_mask_into_messages(segment_messages, segment_loss)
        if full_reasoning is not None:
            record["reasoning_content"] = full_reasoning[:end_index]
        record["turn_indices"] = [
            int(turn["turn_index"])
            for turn in group
            if isinstance(turn.get("turn_index"), int)
        ]
        record["trained_turns"] = [
            int(turn["turn_index"])
            for turn in group
            if turn.get("trained") and isinstance(turn.get("turn_index"), int)
        ]
        record["source"] = "trajectory-split"
        record["split_from_trajectory_id"] = trajectory.get("trajectory_id")
        record["split_index"] = split_index
        record["split_count"] = len(groups)
        record["tool_set_index"] = (
            tool_set_index if isinstance(tool_set_index, int) else None
        )
        if excluded and train_excluded:
            record["loss_mask_recomputed"] = True
        records.append(record)
    return records


def _longest_conversation_trajectory(
    history: dict[str, Any],
) -> dict[str, Any] | None:
    candidates = [
        trajectory
        for trajectory in history.get("trajectories") or []
        if isinstance(trajectory, dict)
        and trajectory.get("kind", "conversation") == "conversation"
        and not trajectory.get("exclude_from_training")
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda trajectory: (
            int(trajectory.get("leaf_turn_index") or 0),
            len(trajectory.get("messages") or []),
        ),
    )


def _captured_responses(history: dict[str, Any]) -> dict[tuple[str, ...], dict]:
    """Map tool-call-id sets -> original upstream assistant message."""
    responses: dict[tuple[str, ...], dict] = {}
    for trajectory in history.get("trajectories") or []:
        if not isinstance(trajectory, dict):
            continue
        for turn in trajectory.get("turns") or []:
            if not isinstance(turn, dict):
                continue
            response = turn.get("response")
            if not isinstance(response, dict):
                continue
            calls = response.get("tool_calls") or []
            key = tuple(sorted(str(call.get("id") or "") for call in calls if call))
            if key:
                responses.setdefault(key, response)
    return responses


def restore_verbatim_messages(
    messages: list[dict[str, Any]], history: dict[str, Any]
) -> list[dict[str, Any]]:
    """Replace replayed (client-normalized) assistant messages with the originals."""
    responses = _captured_responses(history)
    restored: list[dict[str, Any]] = []
    for message in messages:
        if message.get("role") != "assistant" or not message.get("tool_calls"):
            restored.append(message)
            continue
        key = tuple(
            sorted(
                str(call.get("id") or "") for call in message.get("tool_calls") or []
            )
        )
        original = responses.get(key)
        if original is None:
            restored.append(message)
            continue
        replacement: dict[str, Any] = {
            "role": "assistant",
            "content": original.get("content"),
        }
        if original.get("tool_calls"):
            replacement["tool_calls"] = original["tool_calls"]
        restored.append(replacement)
    return restored


def stitch_row(
    history: dict[str, Any],
    *,
    verbatim: bool = False,
    restore_stop_sequences: bool = True,
) -> dict[str, Any] | None:
    """Emit one linear trajectory for a history row (fragment stitching)."""
    trajectory = _longest_conversation_trajectory(history)
    if trajectory is None:
        return None
    messages = [message for message in trajectory.get("messages") or [] if message]
    if verbatim:
        messages = restore_verbatim_messages(messages, history)
    embed_reasoning_into_messages(messages, trajectory.get("reasoning_content"))
    restored = (
        attach_stop_sequences(trajectory, messages) if restore_stop_sequences else []
    )
    record = _base_record(history)
    fragment_ids = [
        fragment.get("trajectory_id")
        for fragment in history.get("trajectories") or []
        if isinstance(fragment, dict)
    ]
    record.update(
        {
            "source": "stitched",
            "stitched": True,
            "fragmented": len(fragment_ids) > 1,
            "num_fragments": len(fragment_ids),
            "trajectory_ids": fragment_ids,
            "kind": "conversation",
            "contains_compaction": any(
                bool(fragment.get("contains_compaction"))
                for fragment in history.get("trajectories") or []
                if isinstance(fragment, dict)
            ),
            "turn_indices": [
                int(entry["turn_index"])
                for entry in history.get("request_log") or []
                if isinstance(entry, dict) and isinstance(entry.get("turn_index"), int)
            ],
            "messages": messages,
            "tools": resolve_tools(trajectory, history),
            "loss_mask": [
                1 if message.get("role") == "assistant" else 0 for message in messages
            ],
        }
    )
    embed_loss_mask_into_messages(messages, record["loss_mask"])
    if restored:
        record["restored_stop_sequences"] = restored
    reasoning = _aligned(trajectory.get("reasoning_content"), len(messages))
    if reasoning is None:
        reasoning = _reasoning_from_messages(messages)
    if reasoning is not None:
        record["reasoning_content"] = reasoning
    return record


def iter_records(
    histories: Iterator[dict[str, Any]],
    *,
    mode: str = "trajectories",
    include_excluded: bool = False,
    verbatim: bool = False,
    train_excluded: bool = False,
    split_tool_sets: bool = False,
    restore_stop_sequences: bool = True,
) -> Iterator[dict[str, Any]]:
    for history in histories:
        if mode == "stitch":
            record = stitch_row(
                history,
                verbatim=verbatim,
                restore_stop_sequences=restore_stop_sequences,
            )
            if record is not None:
                yield record
            continue
        for trajectory in history.get("trajectories") or []:
            if not isinstance(trajectory, dict):
                continue
            excluded = bool(trajectory.get("exclude_from_training"))
            if excluded and not (include_excluded or train_excluded):
                continue
            if split_tool_sets:
                yield from split_trajectory_by_tool_sets(
                    history,
                    trajectory,
                    excluded=excluded,
                    train_excluded=train_excluded,
                )
                continue
            record = flatten_trajectory(
                history,
                trajectory,
                restore_stop_sequences=restore_stop_sequences,
            )
            if excluded and train_excluded:
                # The capture masked these turns; the caller explicitly wants
                # their assistant responses as training targets instead.
                record["loss_mask"] = [
                    1 if message.get("role") == "assistant" else 0
                    for message in record["messages"]
                ]
                record["loss_mask_recomputed"] = True
                embed_loss_mask_into_messages(record["messages"], record["loss_mask"])
            yield record


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


def validate_record(record: dict[str, Any]) -> list[str]:
    """Check OpenAI Chat-Completions shape and training-array alignment."""
    problems: list[str] = []
    label = f"{record.get('instance_id')}/{record.get('trajectory_id') or 'stitched'}"
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        return [f"{label}: empty messages"]
    tools = record.get("tools")
    if not isinstance(tools, list):
        problems.append(f"{label}: tools is not a list")
    elif tools:
        for tool in tools:
            if not isinstance(tool, dict) or tool.get("type") != "function":
                problems.append(f"{label}: tool is not an OpenAI function tool")
                break
            function = tool.get("function")
            if not isinstance(function, dict) or not function.get("name"):
                problems.append(f"{label}: tool has no function name")
                break
    seen_calls: set[str] = set()
    seen_results: set[str] = set()
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            problems.append(f"{label}: message[{index}] is not an object")
            continue
        role = message.get("role")
        if role not in VALID_ROLES:
            problems.append(f"{label}: message[{index}] has invalid role {role!r}")
        if "loss_mask" in message:
            folded = message.get("loss_mask")
            if role != "assistant":
                problems.append(
                    f"{label}: message[{index}] non-assistant message has loss_mask"
                )
            elif folded not in (0, 1):
                problems.append(
                    f"{label}: message[{index}] loss_mask is not 0/1 ({folded!r})"
                )
            else:
                aligned = record.get("loss_mask")
                if (
                    isinstance(aligned, list)
                    and index < len(aligned)
                    and aligned[index] != folded
                ):
                    problems.append(
                        f"{label}: message[{index}] loss_mask disagrees with "
                        "the record loss_mask"
                    )
        if role == "tool":
            call_id = message.get("tool_call_id")
            if not call_id:
                problems.append(f"{label}: message[{index}] tool message has no id")
            elif call_id not in seen_calls:
                problems.append(
                    f"{label}: message[{index}] tool result {call_id!r} has no "
                    "preceding call"
                )
            else:
                seen_results.add(str(call_id))
        for call in message.get("tool_calls") or []:
            if not isinstance(call, dict):
                problems.append(f"{label}: message[{index}] malformed tool call")
                continue
            call_id = str(call.get("id") or "")
            function = call.get("function") or {}
            if not call_id:
                problems.append(f"{label}: message[{index}] tool call has no id")
            else:
                seen_calls.add(call_id)
            if not isinstance(function, dict) or not function.get("name"):
                problems.append(f"{label}: message[{index}] tool call has no name")
            if not isinstance(function.get("arguments"), str):
                problems.append(
                    f"{label}: message[{index}] tool call arguments are not a JSON "
                    "string"
                )
    # NOTE: trailing assistant tool calls without their results are expected in
    # trajectory fragments (the continuation lives in the next fragment), so they
    # are not reported as format problems.
    for key in ("loss_mask", "reasoning_content"):
        value = record.get(key)
        if value is not None and (
            not isinstance(value, list) or len(value) != len(messages)
        ):
            problems.append(f"{label}: {key} is not aligned with messages")
    return problems


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="harness-flatten-histories",
        description=(
            "Flatten harness history JSON dumps (JSONL or cat'ed multi-line JSON) "
            "into trajectory records carrying OpenAI messages and tools."
        ),
    )
    parser.add_argument(
        "input",
        type=str,
        help=(
            "History file: JSONL or concatenated history documents, plain or "
            ".gz (e.g. *.history.streamlined-lite.json)"
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="-",
        help="Output JSONL path (default: stdout)",
    )
    parser.add_argument(
        "--classifier-output",
        type=str,
        default=None,
        help=(
            "JSONL path for auto-mode classifier samples (<block>yes/no</block>). "
            "Defaults to <output stem>.classifier.jsonl when --output is a file"
        ),
    )
    parser.add_argument(
        "--no-normalize-for-hf",
        dest="normalize_for_hf",
        action="store_false",
        help=(
            "Write raw flattened records instead of the normalized "
            "HuggingFace-friendly schema (instance_id/messages/tools/metadata)"
        ),
    )
    parser.add_argument(
        "--mode",
        choices=list(MODES),
        default="trajectories",
        help=(
            "'trajectories' emits one loss-masked record per captured fragment; "
            "'stitch' emits one linear record per history row (default: trajectories)"
        ),
    )
    parser.add_argument(
        "--include-excluded",
        action="store_true",
        help="Keep trajectories marked exclude_from_training (e.g. auxiliary calls)",
    )
    parser.add_argument(
        "--no-restore-stop-sequences",
        dest="restore_stop_sequences",
        action="store_false",
        help=(
            "Keep assistant content exactly as returned by the API instead of "
            "re-attaching a single configured stop sequence (e.g. the auto-mode "
            "classifier's </block>) that the API strips from the reply"
        ),
    )
    parser.add_argument(
        "--split-tool-sets",
        action="store_true",
        help=(
            "Split trajectories into contiguous segments per tool-set phase; "
            "each record carries the tools used for that segment and duplicates "
            "the preceding conversation as context. Requires --mode trajectories."
        ),
    )
    parser.add_argument(
        "--train-excluded",
        action="store_true",
        help=(
            "Keep trajectories masked out at capture time (e.g. auto-mode "
            "classifier calls) and recompute their loss_mask so assistant "
            "responses are trained. Implies --include-excluded."
        ),
    )
    parser.add_argument(
        "--stitch-verbatim",
        action="store_true",
        help=(
            "In --mode stitch, restore the model's original tool-call arguments "
            "when the client normalized them before replaying history"
        ),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when any format/consistency warning is found",
    )
    parser.add_argument(
        "--max-warnings",
        type=int,
        default=20,
        help="Print at most this many warning details (default: 20)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Local worker processes for JSONL input: each worker reads its own "
            "byte range of the same file (no copy) and writes part files"
        ),
    )
    parser.add_argument(
        "--total-rows",
        type=int,
        default=0,
        help=(
            "Number of rows in a JSONL corpus, used for the --workers progress "
            "bar's ETA (0 = unknown; counts/rate still shown)"
        ),
    )
    parser.add_argument(
        "--merge-parts",
        action="store_true",
        help=(
            "After --workers processing, concatenate the per-worker part files "
            "into the regular output/classifier paths instead of leaving parts"
        ),
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help=(
            "Run the per-record format/consistency checks (also implied by "
            "--strict); skipped by default for speed"
        ),
    )
    add_progress_arguments(parser)
    parser.add_argument(
        "--log-level", type=str, default="INFO", help="Logging level (default INFO)"
    )
    return parser


WRITE_BUFFER_BYTES = 1 << 20


def _write_payload(handle: Any, payload: dict[str, Any], *, binary: bool) -> None:
    """Write one JSONL record, using orjson and a large NFS-friendly buffer."""
    if binary:
        handle.write(orjson.dumps(payload) + b"\n")
    else:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def classifier_output_path(output: str) -> str:
    """Default classifier JSONL path derived from the main output path."""
    path = Path(output)
    if path.suffix == ".jsonl":
        return str(path.with_name(path.name[: -len(".jsonl")] + ".classifier.jsonl"))
    return f"{path}.classifier.jsonl"


def _record_options(args: Any) -> dict[str, Any]:
    """Picklable options shared by the serial path and worker processes."""
    return {
        "mode": args.mode,
        "include_excluded": args.include_excluded,
        "train_excluded": args.train_excluded,
        "split_tool_sets": args.split_tool_sets,
        "restore_stop_sequences": args.restore_stop_sequences,
        "verbatim": args.stitch_verbatim,
        "normalize": args.normalize_for_hf,
        "validate": args.validate or args.strict,
    }


def _flatten_history_payload_lines(
    history: dict[str, Any],
    options: dict[str, Any],
    warnings: list[str],
) -> tuple[list[bytes], list[bytes], int]:
    """Flatten one history into encoded main/classifier JSONL lines."""
    main_lines: list[bytes] = []
    classifier_lines: list[bytes] = []
    records = 0
    for record in iter_records(
        iter([history]),
        mode=options["mode"],
        # Classifier trajectories may be capture-excluded yet still belong in
        # the classifier JSONL; routing below decides.
        include_excluded=True,
        verbatim=options["verbatim"],
        train_excluded=options["train_excluded"],
        split_tool_sets=options["split_tool_sets"],
        restore_stop_sequences=options["restore_stop_sequences"],
    ):
        records += 1
        if options["validate"]:
            warnings.extend(validate_record(record))
        label = classifier_label(record)
        if label is not None:
            payload = (
                normalize_record(record, classifier=label)
                if options["normalize"]
                else {**record, "classifier": label}
            )
            classifier_lines.append(orjson.dumps(payload) + b"\n")
            continue
        if record.get("exclude_from_training") and not (
            options["include_excluded"] or options["train_excluded"]
        ):
            continue
        payload = normalize_record(record) if options["normalize"] else record
        main_lines.append(orjson.dumps(payload) + b"\n")
    if records == 0:
        warnings.append(
            f"no trajectories emitted for {history.get('instance_id')} "
            f"(attempt {history.get('attempt')})"
        )
    return main_lines, classifier_lines, records


def _part_path(base: str, index: int) -> str:
    path = Path(base)
    if path.suffix == ".jsonl":
        return str(path.with_name(f"{path.stem}.part{index:03d}.jsonl"))
    return f"{base}.part{index:03d}"


def _part_glob(base: str) -> str:
    return _part_path(base, 0).replace("part000", "part*")


def _concatenate_parts(parts: list[str], target: str) -> None:
    Path(target).parent.mkdir(parents=True, exist_ok=True)
    with open(target, "wb", buffering=WRITE_BUFFER_BYTES) as output:
        for part in parts:
            with open(part, "rb", buffering=WRITE_BUFFER_BYTES) as source:
                while True:
                    chunk = source.read(WRITE_BUFFER_BYTES)
                    if not chunk:
                        break
                    output.write(chunk)


def _flatten_byte_range(job: tuple[Any, ...]) -> dict[str, Any]:
    """Worker: read and flatten the JSONL lines in one byte range."""
    (
        input_path,
        start,
        end,
        shard_index,
        main_part,
        classifier_part,
        options,
        progress_queue,
    ) = job
    histories = 0
    emitted_main = 0
    emitted_classifier = 0
    skipped = 0
    warnings: list[str] = []
    main_handle = None
    classifier_handle = None
    try:
        with open(input_path, "rb", buffering=WRITE_BUFFER_BYTES) as handle:
            if start:
                handle.seek(start)
            while handle.tell() < end:
                raw = handle.readline()
                if not raw:
                    break
                if not raw.strip():
                    continue
                histories += 1
                try:
                    history = orjson.loads(raw)
                except orjson.JSONDecodeError as exc:
                    warnings.append(
                        f"{input_path}: invalid JSONL line in shard {shard_index}: {exc}"
                    )
                    continue
                if not isinstance(history, dict):
                    warnings.append(
                        f"{input_path}: shard {shard_index} expected JSON objects"
                    )
                    continue
                main_lines, classifier_lines, records = _flatten_history_payload_lines(
                    history, options, warnings
                )
                if main_lines:
                    if main_handle is None:
                        Path(main_part).parent.mkdir(parents=True, exist_ok=True)
                        main_handle = open(  # noqa: SIM115 - closed below
                            main_part, "wb", buffering=WRITE_BUFFER_BYTES
                        )
                    for line in main_lines:
                        main_handle.write(line)
                    emitted_main += len(main_lines)
                if classifier_lines:
                    if classifier_handle is None:
                        Path(classifier_part).parent.mkdir(parents=True, exist_ok=True)
                        classifier_handle = open(  # noqa: SIM115 - closed below
                            classifier_part, "wb", buffering=WRITE_BUFFER_BYTES
                        )
                    for line in classifier_lines:
                        classifier_handle.write(line)
                    emitted_classifier += len(classifier_lines)
                if records == 0:
                    skipped += 1
                progress_queue.put((1, len(main_lines) + len(classifier_lines)))
    finally:
        if main_handle is not None:
            main_handle.close()
        if classifier_handle is not None:
            classifier_handle.close()
    return {
        "histories": histories,
        "emitted_main": emitted_main,
        "emitted_classifier": emitted_classifier,
        "skipped": skipped,
        "warnings": warnings,
        "main_part": main_part if main_handle is not None else None,
        "classifier_part": (classifier_part if classifier_handle is not None else None),
    }


def _run_parallel(args: Any, parser: Any) -> int:
    """Shard a JSONL corpus over local worker processes without copying it."""
    if args.output == "-":
        parser.error("--workers requires a file --output")
    if not is_jsonl_file(args.input):
        parser.error(
            "--workers currently requires JSONL input (one JSON history per line)"
        )
    output_path = args.output
    classifier_path = args.classifier_output or classifier_output_path(output_path)
    if classifier_path == "-":
        parser.error("--classifier-output cannot be stdout")
    if os.path.abspath(classifier_path) == os.path.abspath(output_path):
        parser.error("--classifier-output must differ from the main --output path")

    input_path = str(args.input)
    size = Path(input_path).stat().st_size
    starts = [0]
    if size:
        with open(input_path, "rb", buffering=0) as handle:
            for worker in range(1, max(1, args.workers)):
                offset = size * worker // max(1, args.workers)
                handle.seek(offset)
                handle.readline()
                starts.append(handle.tell())
    starts.append(size)
    ranges = [
        (starts[index], starts[index + 1])
        for index in range(len(starts) - 1)
        if starts[index] < starts[index + 1]
    ]

    main_parts = [_part_path(output_path, index) for index in range(len(ranges))]
    classifier_parts = [
        _part_path(classifier_path, index) for index in range(len(ranges))
    ]
    options = _record_options(args)
    progress_enabled = (
        args.progress if args.progress is not None else sys.stderr.isatty()
    )
    manager = multiprocessing.Manager()
    progress_queue = manager.Queue()
    total_rows = args.total_rows if args.total_rows > 0 else None
    progress_bar = tqdm(
        total=total_rows,
        desc="flatten",
        unit="row",
        file=sys.stderr,
        disable=not progress_enabled,
        dynamic_ncols=True,
        mininterval=max(0.1, float(args.progress_interval)),
    )

    warnings: list[str] = []
    histories = emitted_main = emitted_classifier = skipped = failures = 0
    jobs = [
        (
            input_path,
            start,
            end,
            index,
            main_parts[index],
            classifier_parts[index],
            options,
            progress_queue,
        )
        for index, (start, end) in enumerate(ranges)
    ]
    rows_done = 0
    records_done = 0
    try:
        if jobs:
            with ProcessPoolExecutor(
                max_workers=len(jobs),
                mp_context=multiprocessing.get_context("spawn"),
            ) as pool:
                pending = {pool.submit(_flatten_byte_range, job) for job in jobs}
                while pending:
                    done, pending = wait(pending, timeout=0.5)
                    for future in done:
                        try:
                            result = future.result()
                        except Exception as exc:  # noqa: BLE001 - report and fail
                            logger.exception("flatten worker failed")
                            warnings.append(f"flatten worker failed: {exc}")
                            failures += 1
                            continue
                        histories += result["histories"]
                        emitted_main += result["emitted_main"]
                        emitted_classifier += result["emitted_classifier"]
                        skipped += result["skipped"]
                        warnings.extend(result["warnings"])
                    while True:
                        try:
                            row_increment, record_increment = (
                                progress_queue.get_nowait()
                            )
                        except Empty:
                            break
                        rows_done += int(row_increment)
                        records_done += int(record_increment)
                    progress_bar.n = (
                        rows_done if total_rows is None else min(rows_done, total_rows)
                    )
                    progress_bar.set_postfix(records=records_done, refresh=False)
                    progress_bar.refresh()
            # Drain any progress items queued right before worker exit.
            while True:
                try:
                    row_increment, record_increment = progress_queue.get_nowait()
                except Empty:
                    break
                rows_done += int(row_increment)
                records_done += int(record_increment)
    finally:
        progress_bar.n = rows_done if total_rows is None else min(rows_done, total_rows)
        progress_bar.set_postfix(records=records_done, refresh=False)
        progress_bar.close()
        manager.shutdown()

    existing_main = [path for path in main_parts if Path(path).is_file()]
    existing_classifier = [path for path in classifier_parts if Path(path).is_file()]
    if args.merge_parts:
        if existing_main:
            _concatenate_parts(existing_main, output_path)
        if existing_classifier:
            _concatenate_parts(existing_classifier, classifier_path)
    else:
        if existing_main:
            print(f"main parts: {_part_glob(output_path)}", file=sys.stderr)
        if existing_classifier:
            print(
                f"classifier parts: {_part_glob(classifier_path)}",
                file=sys.stderr,
            )
    print(
        f"workers={len(ranges)} histories={histories} "
        f"emitted={emitted_main + emitted_classifier} "
        f"emitted_main={emitted_main} emitted_classifier={emitted_classifier} "
        f"skipped_empty={skipped} warnings={len(warnings)}",
        file=sys.stderr,
    )
    for warning in warnings[: max(0, args.max_warnings)]:
        print(f"WARNING {warning}", file=sys.stderr)
    if len(warnings) > args.max_warnings:
        print(
            f"WARNING ... {len(warnings) - args.max_warnings} more warning(s)",
            file=sys.stderr,
        )
    if failures:
        return 1
    if args.strict and warnings:
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.split_tool_sets and args.mode != "trajectories":
        parser.error("--split-tool-sets requires --mode trajectories")
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    if args.workers > 1:
        return _run_parallel(args, parser)
    validate = args.validate or args.strict
    output_binary = args.output != "-"
    output = (
        sys.stdout
        if args.output == "-"
        else open(  # noqa: SIM115 - closed below
            args.output, "wb", buffering=WRITE_BUFFER_BYTES
        )
    )
    histories = 0
    emitted_main = 0
    emitted_classifier = 0
    skipped_empty = 0
    warnings: list[str] = []
    classifier_handle = None
    classifier_path = args.classifier_output
    reporter = reporter_from_args(args, "flatten")
    reporter.update(histories=0, emitted_main=0, emitted_classifier=0)
    reporter.start()

    def open_classifier() -> Any:
        nonlocal classifier_handle, classifier_path
        if classifier_handle is not None:
            return classifier_handle
        path = classifier_path
        if path is None:
            if args.output == "-":
                raise RuntimeError(
                    "auto-mode classifier samples need a file destination: pass "
                    "-o FILE (classifier JSONL is derived) or --classifier-output"
                )
            path = classifier_output_path(args.output)
        if path == "-":
            raise RuntimeError("--classifier-output cannot be stdout")
        if args.output != "-" and os.path.abspath(path) == os.path.abspath(args.output):
            raise RuntimeError(
                "--classifier-output must differ from the main --output path"
            )
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        classifier_handle = open(  # noqa: SIM115 - closed below
            path, "wb", buffering=WRITE_BUFFER_BYTES
        )
        classifier_path = path
        return classifier_handle

    try:
        for history in iter_json_documents(args.input):
            histories += 1
            reporter.update(
                histories=histories,
                emitted_main=emitted_main,
                emitted_classifier=emitted_classifier,
            )
            history_records = 0
            for record in iter_records(
                iter([history]),
                mode=args.mode,
                # Classifier trajectories may be capture-excluded yet still
                # belong in the classifier JSONL; routing below decides.
                include_excluded=True,
                verbatim=args.stitch_verbatim,
                train_excluded=args.train_excluded,
                split_tool_sets=args.split_tool_sets,
                restore_stop_sequences=args.restore_stop_sequences,
            ):
                history_records += 1
                reporter.update(
                    histories=histories,
                    emitted_main=emitted_main,
                    emitted_classifier=emitted_classifier,
                )
                label = classifier_label(record)
                if label is not None:
                    if validate:
                        warnings.extend(validate_record(record))
                    payload = (
                        normalize_record(record, classifier=label)
                        if args.normalize_for_hf
                        else {**record, "classifier": label}
                    )
                    try:
                        handle = open_classifier()
                    except (RuntimeError, OSError) as exc:
                        print(f"ERROR {exc}", file=sys.stderr)
                        return 1
                    _write_payload(handle, payload, binary=True)
                    emitted_classifier += 1
                    continue
                if record.get("exclude_from_training") and not (
                    args.include_excluded or args.train_excluded
                ):
                    continue
                if validate:
                    warnings.extend(validate_record(record))
                payload = normalize_record(record) if args.normalize_for_hf else record
                _write_payload(output, payload, binary=output_binary)
                emitted_main += 1
                reporter.update(
                    histories=histories,
                    emitted_main=emitted_main,
                    emitted_classifier=emitted_classifier,
                )
            if history_records == 0:
                skipped_empty += 1
                warnings.append(
                    f"no trajectories emitted for {history.get('instance_id')} "
                    f"(attempt {history.get('attempt')})"
                )
    finally:
        reporter.stop()
        if classifier_handle is not None:
            classifier_handle.close()
        if output is not sys.stdout:
            output.close()
    classifier_summary = (
        f" classifier_output={classifier_path}" if classifier_path is not None else ""
    )
    print(
        f"histories={histories} emitted={emitted_main + emitted_classifier} "
        f"emitted_main={emitted_main} emitted_classifier={emitted_classifier} "
        f"mode={args.mode} skipped_empty={skipped_empty} "
        f"warnings={len(warnings)}{classifier_summary}",
        file=sys.stderr,
    )
    for warning in warnings[: max(0, args.max_warnings)]:
        print(f"WARNING {warning}", file=sys.stderr)
    if len(warnings) > args.max_warnings:
        print(
            f"WARNING ... {len(warnings) - args.max_warnings} more warning(s)",
            file=sys.stderr,
        )
    if args.strict and warnings:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
