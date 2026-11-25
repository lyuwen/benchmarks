"""Streamline fragmented harness history dumps.

Claude Code replays the conversation on every turn, but normalizes the replayed
tool-call arguments first: a leading ``cd <cwd> &&`` is stripped from Bash
commands and Edit calls gain default fields such as ``replace_all``. The capture
matches prompts against previously stored assistant responses, so those
normalized replays no longer match and one logical conversation is exported as
several prefix fragments.

This tool merges such fragments back:

- fragments of the same conversation (message-prefix extensions modulo the
  normalized tool arguments) become one trajectory per conversation, including
  sub-agents and other side conversations;
- the merged messages use the tool-call arguments the model originally
  generated (recovered from the captured responses), so the ``cd <cwd> &&``
  prefix is present again;
- the training mask is the union of the fragments' trained turns;
- auxiliary requests (auto-mode classifier / security monitor) and
  compaction/summarization trajectories are kept unchanged;
- the history's root ``messages`` (and ``tools``/``tool_sets``) are taken from
  the last non-auxiliary conversation trajectory, matching the final
  conversation state;
- the result is written next to the input as ``<name>.history.streamlined.json``.

Run one directory per SLURM array task (see
``scripts/streamline_histories_slurm.sh``)::

    uv run harness-streamline-histories --dir <...>/harnesses/claude-code
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import logging
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

from benchmarks.utils.harnesses.rollout import (
    collect_tool_sets,
    resolve_trajectory_tools,
)


logger = logging.getLogger(__name__)

HISTORY_SUFFIX = ".history.json"
STREAMLINED_SUFFIX = ".history.streamlined.json"
_ATTEMPT_RE = re.compile(r"\.attempt\d+\.history\.json$")


# ---------------------------------------------------------------------------
# trajectory classification and matching
# ---------------------------------------------------------------------------


def _trajectory_turns(trajectory: dict[str, Any]) -> list[dict[str, Any]]:
    return [turn for turn in trajectory.get("turns") or [] if isinstance(turn, dict)]


def is_auxiliary_trajectory(trajectory: dict[str, Any]) -> bool:
    """Whether this trajectory holds an auxiliary request (classifier/title)."""
    if trajectory.get("exclude_from_training"):
        return True
    return any(
        str(turn.get("request_class") or "").lower() == "auxiliary"
        for turn in _trajectory_turns(trajectory)
    )


def is_compaction_trajectory(trajectory: dict[str, Any]) -> bool:
    """Standalone compaction/summarization trajectories are kept as-is."""
    return trajectory.get("kind") == "compaction"


def _is_conversation(trajectory: dict[str, Any]) -> bool:
    return trajectory.get("kind", "conversation") == "conversation"


def _first_agent(trajectory: dict[str, Any]) -> str | None:
    for turn in _trajectory_turns(trajectory):
        agent = turn.get("agent_id")
        if agent:
            return str(agent)
    return None


def _messages(trajectory: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        message
        for message in trajectory.get("messages") or []
        if isinstance(message, dict)
    ]


def _last_turn_index(trajectory: dict[str, Any]) -> int:
    leaf = trajectory.get("leaf_turn_index")
    if isinstance(leaf, int):
        return leaf
    indices = [
        int(turn["turn_index"])
        for turn in _trajectory_turns(trajectory)
        if isinstance(turn.get("turn_index"), int)
    ]
    return max(indices, default=-1)


_CD_PREFIX_RE = re.compile(r"^\s*cd\s+(?:\"[^\"]*\"|'[^']*'|[^\s;&|]+)\s*&&\s+")


def _normalized_arguments(raw: Any) -> Any:
    """Normalize the tool-argument changes Claude Code applies on replay.

    Only the normalizations seen in Claude Code replays are tolerated: a leading
    ``cd <dir> &&`` is stripped from Bash commands, Edit calls gain a default
    ``replace_all`` value, and Edit ``old_string``/``new_string`` lose trailing
    whitespace. Everything else must match for two tool calls to be the same
    step.
    """
    if isinstance(raw, dict):
        arguments = dict(raw)
    elif isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return raw
        if not isinstance(parsed, dict):
            return raw
        arguments = parsed
    else:
        return raw
    command = arguments.get("command")
    if isinstance(command, str):
        arguments["command"] = _CD_PREFIX_RE.sub("", command)
    # Edit calls: Claude Code adds a default ``replace_all`` and trims trailing
    # whitespace from the patched strings when replaying history.
    for key in ("new_string", "old_string"):
        value = arguments.get(key)
        if isinstance(value, str):
            arguments[key] = value.rstrip()
    if (
        "replace_all" in arguments
        or "file_path" in arguments
        or "old_string" in arguments
        or "new_string" in arguments
    ):
        arguments["replace_all"] = bool(arguments.get("replace_all", False))
    return arguments


def _arguments_match(left_call: dict[str, Any], right_call: dict[str, Any]) -> bool:
    left = (left_call.get("function") or {}).get("arguments")
    right = (right_call.get("function") or {}).get("arguments")
    if left == right:
        return True
    return _normalized_arguments(left) == _normalized_arguments(right)


def _messages_match(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """Whether two messages are the same step modulo tool-argument normalization.

    Claude Code only rewrites the arguments of replayed assistant tool calls
    (leading ``cd ... &&``, default Edit fields), so two tool-call messages are
    the same step when their call ids/names match and the arguments agree after
    those two normalizations; regular messages are compared verbatim.
    """
    if left.get("role") != right.get("role"):
        return False
    left_calls = left.get("tool_calls") or []
    right_calls = right.get("tool_calls") or []
    if left_calls or right_calls:
        if len(left_calls) != len(right_calls):
            return False
        for left_call, right_call in zip(left_calls, right_calls):
            if left_call.get("id") != right_call.get("id"):
                return False
            left_function = left_call.get("function") or {}
            right_function = right_call.get("function") or {}
            if left_function.get("name") != right_function.get("name"):
                return False
            if not _arguments_match(left_call, right_call):
                return False
        return True
    if left.get("role") == "tool":
        return left.get("tool_call_id") == right.get("tool_call_id") and left.get(
            "content"
        ) == right.get("content")
    return left.get("content") == right.get("content")


def _is_message_prefix(
    base_messages: list[dict[str, Any]], candidate_messages: list[dict[str, Any]]
) -> bool:
    if len(candidate_messages) > len(base_messages):
        return False
    return all(
        _messages_match(base_messages[index], candidate_messages[index])
        for index in range(len(candidate_messages))
    )


def _same_agent(base: dict[str, Any], candidate: dict[str, Any]) -> bool:
    base_agent = _first_agent(base)
    candidate_agent = _first_agent(candidate)
    if base_agent is not None and candidate_agent is not None:
        return base_agent == candidate_agent
    return True


# ---------------------------------------------------------------------------
# captured-response recovery and merging
# ---------------------------------------------------------------------------


def _original_responses(history: dict[str, Any]) -> dict[tuple[str, ...], dict]:
    """Map tool-call-id sets to the assistant message the model generated.

    Context copies of a replayed response may be client-normalized; the owner
    of a turn (``trained=True``) has the original captured response, so prefer
    trained turns when several trajectories share a call-id set.
    """
    responses: dict[tuple[str, ...], dict[str, Any]] = {}
    trained: dict[tuple[str, ...], bool] = {}
    for trajectory in history.get("trajectories") or []:
        if not isinstance(trajectory, dict):
            continue
        for turn in _trajectory_turns(trajectory):
            response = turn.get("response")
            if not isinstance(response, dict):
                continue
            calls = response.get("tool_calls") or []
            key = tuple(
                sorted(
                    str(call.get("id") or "")
                    for call in calls
                    if isinstance(call, dict)
                )
            )
            if not key:
                continue
            is_trained = bool(turn.get("trained"))
            if key not in responses or (is_trained and not trained.get(key)):
                responses[key] = response
                trained[key] = is_trained
    return responses


def _restore_original_tool_calls(
    messages: list[dict[str, Any]],
    responses: dict[tuple[str, ...], dict],
) -> int:
    """Replace replayed tool-call arguments with the model's originals in place."""
    restored = 0
    for index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        calls = message.get("tool_calls") or []
        if not calls:
            continue
        key = tuple(
            sorted(
                str(call.get("id") or "") for call in calls if isinstance(call, dict)
            )
        )
        original = responses.get(key)
        if original is None:
            continue
        original_calls = original.get("tool_calls") or []
        if original_calls == calls and original.get("content") == message.get(
            "content"
        ):
            continue
        replacement: dict[str, Any] = {
            "role": "assistant",
            "content": original.get("content"),
            "tool_calls": copy.deepcopy(original_calls),
        }
        if original.get("name") is not None:
            replacement["name"] = original["name"]
        messages[index] = replacement
        restored += 1
    return restored


def _merge_group(
    history: dict[str, Any],
    base: dict[str, Any],
    absorbed: list[dict[str, Any]],
    responses: dict[tuple[str, ...], dict],
) -> dict[str, Any]:
    """Merge prefix fragments into ``base`` (the longest fragment)."""
    messages = [copy.deepcopy(message) for message in _messages(base)]
    restored_tool_calls = _restore_original_tool_calls(messages, responses)

    merged_tool_sets: list[list[dict[str, Any]]] = []
    tool_set_index: dict[str, int] = {}

    def intern_tool_set(tools: Any) -> int | None:
        if not isinstance(tools, list) or not tools:
            return None
        key = json.dumps(tools, sort_keys=True, ensure_ascii=False)
        index = tool_set_index.get(key)
        if index is None:
            index = len(merged_tool_sets)
            tool_set_index[key] = index
            merged_tool_sets.append(tools)
        return index

    fragments = [base, *absorbed]
    turns_by_message: dict[int, dict[str, Any]] = {}
    absorbed_ids = [str(fragment.get("trajectory_id")) for fragment in absorbed]
    for fragment in fragments:
        fragment_tool_sets = fragment.get("tool_sets") or []
        for turn in _trajectory_turns(fragment):
            message_index = turn.get("message_index")
            if not isinstance(message_index, int) or not 0 <= message_index < len(
                messages
            ):
                continue
            trained = bool(turn.get("trained"))
            local_tool_set = turn.get("tool_set")
            tools = (
                fragment_tool_sets[local_tool_set]
                if isinstance(local_tool_set, int)
                and 0 <= local_tool_set < len(fragment_tool_sets)
                else None
            )
            entry = dict(turn)
            entry["message_index"] = message_index
            entry["tool_set"] = intern_tool_set(tools)
            existing = turns_by_message.get(message_index)
            if existing is None:
                entry["trained"] = trained
                turns_by_message[message_index] = entry
                continue
            # Prefer the owning (trained) turn's response/metadata, but keep a
            # target if any fragment trained this message.
            if trained and not existing.get("trained"):
                entry["trained"] = True
                turns_by_message[message_index] = entry
            elif existing.get("trained"):
                existing["trained"] = True
            elif not existing.get("response") and entry.get("response"):
                existing["response"] = entry["response"]

    merged_turns = [turns_by_message[index] for index in sorted(turns_by_message)]
    loss_mask = [0] * len(messages)
    for turn in merged_turns:
        message_index = turn["message_index"]
        if turn.get("trained"):
            loss_mask[message_index] = 1

    merged: dict[str, Any] = {
        **base,
        "messages": messages,
        "turns": merged_turns,
        "loss_mask": loss_mask,
        "tool_sets": merged_tool_sets,
        "path_turn_indices": sorted(
            {
                int(turn["turn_index"])
                for turn in merged_turns
                if isinstance(turn.get("turn_index"), int)
            }
        ),
        "trained_turns": sorted(
            {
                int(turn["turn_index"])
                for turn in merged_turns
                if turn.get("trained") and isinstance(turn.get("turn_index"), int)
            }
        ),
        "leaf_turn_index": max(
            [_last_turn_index(fragment) for fragment in fragments], default=None
        ),
        "exclude_from_training": False,
        "streamlined": {
            "merged_from_trajectory_ids": [
                str(base.get("trajectory_id")),
                *absorbed_ids,
            ],
            "absorbed_trajectory_ids": absorbed_ids,
            "merged_fragment_count": len(fragments),
            "restored_tool_call_messages": restored_tool_calls,
        },
    }
    if merged.get("parent_trajectory_id") in set(absorbed_ids):
        merged["parent_trajectory_id"] = None
    reasoning = base.get("reasoning_content")
    if isinstance(reasoning, list) and len(reasoning) == len(messages):
        merged["reasoning_content"] = reasoning
    return merged


# ---------------------------------------------------------------------------
# history-level streamlining
# ---------------------------------------------------------------------------


def _is_main_conversation(trajectory: dict[str, Any]) -> bool:
    """Whether this trajectory belongs to the main agent (not a sub-agent)."""
    first_agent = _first_agent(trajectory)
    if first_agent is None:
        return True
    for turn in _trajectory_turns(trajectory):
        if turn.get("agent_id") == first_agent:
            return turn.get("agent_type") in (None, "main")
    return True


def _latest_conversation_trajectory(
    trajectories: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """The last main-agent conversation trajectory (aux/compaction excluded).

    The history's root ``messages`` follow the final conversation state: the
    latest non-auxiliary, non-compaction trajectory of the main agent (falling
    back to any non-auxiliary conversation when the capture does not label
    agents).
    """
    candidates = [
        trajectory
        for trajectory in trajectories
        if _is_conversation(trajectory)
        and not is_auxiliary_trajectory(trajectory)
        and not is_compaction_trajectory(trajectory)
    ]
    main_candidates = [
        trajectory for trajectory in candidates if _is_main_conversation(trajectory)
    ]
    pool = main_candidates or candidates
    if not pool:
        return None
    return max(pool, key=_last_turn_index)


def streamline_history(history: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``history`` with fragmented conversations merged."""
    trajectories = [
        trajectory
        for trajectory in history.get("trajectories") or []
        if isinstance(trajectory, dict)
    ]
    responses = _original_responses(history)
    eligible = [
        trajectory
        for trajectory in trajectories
        if _is_conversation(trajectory)
        and not is_auxiliary_trajectory(trajectory)
        and not is_compaction_trajectory(trajectory)
        and _messages(trajectory)
    ]

    absorbed_ids: set[str] = set()
    groups: dict[str, list[dict[str, Any]]] = {}
    # Longest fragments are the canonical bases; shorter fragments are prefix
    # extensions of them and are absorbed into exactly one base.
    for base in sorted(
        eligible,
        key=lambda trajectory: (
            len(_messages(trajectory)),
            _last_turn_index(trajectory),
        ),
        reverse=True,
    ):
        base_id = str(base.get("trajectory_id"))
        if base_id in absorbed_ids:
            continue
        base_messages = _messages(base)
        absorbed: list[dict[str, Any]] = []
        for candidate in eligible:
            candidate_id = str(candidate.get("trajectory_id"))
            if candidate_id == base_id or candidate_id in absorbed_ids:
                continue
            candidate_messages = _messages(candidate)
            if len(candidate_messages) > len(base_messages):
                continue
            if not _is_message_prefix(base_messages, candidate_messages):
                continue
            if not _same_agent(base, candidate):
                continue
            absorbed.append(candidate)
            absorbed_ids.add(candidate_id)
        if absorbed:
            groups[base_id] = absorbed

    streamlined: list[dict[str, Any]] = []
    for trajectory in trajectories:
        trajectory_id = str(trajectory.get("trajectory_id"))
        if trajectory_id in absorbed_ids:
            continue
        group_fragments = groups.get(trajectory_id)
        if group_fragments:
            streamlined.append(
                _merge_group(history, trajectory, group_fragments, responses)
            )
        else:
            streamlined.append(trajectory)

    # Leftover fragments (divergent branch leaves, e.g. the pre-compaction leaf
    # next to an auto-compaction branch) are final trajectories too: restore the
    # model's original tool arguments in them as well.
    branch_restored = 0
    for index, trajectory in enumerate(streamlined):
        if (
            not _is_conversation(trajectory)
            or is_auxiliary_trajectory(trajectory)
            or is_compaction_trajectory(trajectory)
        ):
            continue
        messages = [copy.deepcopy(message) for message in _messages(trajectory)]
        restored = _restore_original_tool_calls(messages, responses)
        if not restored:
            continue
        updated = dict(trajectory)
        updated["messages"] = messages
        updated["restored_tool_call_messages"] = restored
        streamlined[index] = updated
        branch_restored += restored

    root = _latest_conversation_trajectory(streamlined)
    merged_summaries = [
        {
            "trajectory_id": trajectory_id,
            "absorbed_trajectory_ids": [
                str(fragment.get("trajectory_id")) for fragment in absorbed
            ],
            "restored_tool_call_messages": 0,
        }
        for trajectory_id, absorbed in groups.items()
    ]
    for summary in merged_summaries:
        merged = next(
            (
                trajectory
                for trajectory in streamlined
                if str(trajectory.get("trajectory_id")) == summary["trajectory_id"]
            ),
            None,
        )
        if merged and isinstance(merged.get("streamlined"), dict):
            summary["restored_tool_call_messages"] = merged["streamlined"].get(
                "restored_tool_call_messages", 0
            )

    result = dict(history)
    result["trajectories"] = streamlined
    if root is not None:
        result["messages"] = root.get("messages") or []
        result["tools"] = resolve_trajectory_tools(root, result)
    result["tool_sets"] = collect_tool_sets({"trajectories": streamlined})
    result["streamlined"] = {
        "fragments_before": len(trajectories),
        "trajectories_after": len(streamlined),
        "merged_groups": merged_summaries,
        "auxiliary_kept": sum(
            1 for trajectory in streamlined if is_auxiliary_trajectory(trajectory)
        ),
        "compaction_kept": sum(
            1 for trajectory in streamlined if is_compaction_trajectory(trajectory)
        ),
        "root_messages_from_trajectory_id": (
            str(root.get("trajectory_id")) if root is not None else None
        ),
        "restored_tool_call_messages": sum(
            summary["restored_tool_call_messages"] for summary in merged_summaries
        )
        + branch_restored,
        "branch_trajectories_restored": branch_restored,
    }
    return result


# ---------------------------------------------------------------------------
# files and CLI
# ---------------------------------------------------------------------------


def streamline_history_file(
    path: str | os.PathLike[str], *, overwrite: bool = False, dry_run: bool = False
) -> dict[str, Any] | None:
    """Streamline one ``*.history.json`` file next to it; return the summary."""
    source = Path(path)
    target = source.with_name(source.name[: -len(HISTORY_SUFFIX)] + STREAMLINED_SUFFIX)
    if target.exists() and not overwrite:
        logger.info("skip (exists): %s", target)
        return None
    history = json.loads(source.read_text(encoding="utf-8"))
    result = streamline_history(history)
    summary = {
        "input": str(source),
        "output": str(target),
        "fragments_before": result["streamlined"]["fragments_before"],
        "trajectories_after": result["streamlined"]["trajectories_after"],
        "merged_groups": len(result["streamlined"]["merged_groups"]),
        "restored_tool_call_messages": result["streamlined"][
            "restored_tool_call_messages"
        ],
        "root_messages_from_trajectory_id": result["streamlined"][
            "root_messages_from_trajectory_id"
        ],
        "auxiliary_kept": result["streamlined"]["auxiliary_kept"],
        "compaction_kept": result["streamlined"]["compaction_kept"],
    }
    if dry_run:
        logger.info("dry-run: %s", summary)
        return summary
    _write_json_atomic(target, result)
    logger.info(
        "streamlined %s: %d -> %d trajectories, %d merged group(s), "
        "%d restored tool-call message(s), root from %s",
        source,
        summary["fragments_before"],
        summary["trajectories_after"],
        summary["merged_groups"],
        summary["restored_tool_call_messages"],
        summary["root_messages_from_trajectory_id"],
    )
    return summary


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=path.name + ".",
        suffix=".tmp",
        delete=False,
    )
    try:
        with handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(handle.name, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(handle.name)
        raise


def _is_attempt_history(path: Path) -> bool:
    return bool(_ATTEMPT_RE.search(path.name))


def _history_files(directory: Path, *, include_attempts: bool) -> list[Path]:
    """History files directly in ``directory`` or in its claude-code dirs."""
    directory = directory.resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"not a directory: {directory}")

    def collect(candidate_dir: Path) -> list[Path]:
        files = []
        for path in sorted(candidate_dir.glob("*" + HISTORY_SUFFIX)):
            if not path.is_file() or path.name.endswith(STREAMLINED_SUFFIX):
                continue
            if not include_attempts and _is_attempt_history(path):
                continue
            files.append(path)
        return files

    files = collect(directory)
    if files:
        return files
    files = []
    for candidate in sorted(directory.glob("**/harnesses/claude-code")):
        if candidate.is_dir():
            files.extend(collect(candidate))
    return files


def _claude_code_dirs(root: Path) -> list[Path]:
    return sorted(
        {
            path.resolve()
            for path in root.glob("**/harnesses/claude-code")
            if path.is_dir()
        }
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="harness-streamline-histories",
        description=(
            "Merge fragment trajectories split by Claude Code's replayed-tool "
            "normalization (leading 'cd ... &&' and Edit defaults) and write "
            "<name>.history.streamlined.json next to each final history file."
        ),
    )
    parser.add_argument(
        "--dir",
        action="append",
        default=[],
        metavar="DIR",
        help=(
            "Directory to process (repeatable): a harnesses/claude-code dir, a "
            "job directory containing one, or a parent searched recursively."
        ),
    )
    parser.add_argument(
        "--root",
        metavar="ROOT",
        help=(
            "Root containing many job directories; with --list print every "
            "harnesses/claude-code dir below it, otherwise process them all."
        ),
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print discovered claude-code directories (requires --root) and exit.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-create *.history.streamlined.json files that already exist.",
    )
    parser.add_argument(
        "--include-attempts",
        action="store_true",
        help="Also process per-attempt (*.attemptN.history.json) files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be written without writing files.",
    )
    parser.add_argument(
        "--log-level", default="INFO", help="Logging level (default INFO)."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.list:
        if not args.root:
            parser.error("--list requires --root")
        for directory in _claude_code_dirs(Path(args.root)):
            print(directory)
        return 0

    directories: list[Path] = []
    if args.root:
        directories.extend(_claude_code_dirs(Path(args.root)))
    directories.extend(Path(value) for value in args.dir)
    if not directories:
        parser.error("pass at least one --dir, or --root")

    files: list[Path] = []
    seen: set[Path] = set()
    failures = 0
    for directory in directories:
        try:
            discovered = _history_files(
                directory, include_attempts=args.include_attempts
            )
        except FileNotFoundError as exc:
            logger.error("%s", exc)
            failures += 1
            continue
        if not discovered:
            logger.warning("no final *.history.json files under %s", directory)
        for path in discovered:
            if path in seen:
                continue
            seen.add(path)
            files.append(path)

    counters = {
        "processed": 0,
        "skipped": 0,
        "fragments_before": 0,
        "trajectories_after": 0,
    }
    summaries: list[dict[str, Any]] = []
    for path in files:
        try:
            summary = streamline_history_file(
                path, overwrite=args.overwrite, dry_run=args.dry_run
            )
        except Exception:  # noqa: BLE001 - report and continue with other files
            logger.exception("failed to streamline %s", path)
            failures += 1
            continue
        if summary is None:
            counters["skipped"] += 1
            continue
        counters["processed"] += 1
        counters["fragments_before"] += summary["fragments_before"]
        counters["trajectories_after"] += summary["trajectories_after"]
        summaries.append(summary)

    print(
        "streamline: files={processed} skipped={skipped} fragments {fragments_before}"
        " -> trajectories {trajectories_after}".format(**counters),
        file=sys.stderr,
    )
    for summary in summaries:
        print(
            "  {input}: {fragments_before} -> {trajectories_after} trajectories, "
            "{merged_groups} merged, {restored_tool_call_messages} restored".format(
                **summary
            ),
            file=sys.stderr,
        )
    return 1 if failures else 0


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())
