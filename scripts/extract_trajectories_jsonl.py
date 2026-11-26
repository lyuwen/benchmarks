#!/usr/bin/env python3
"""Extract per-trajectory rows from a JSONL of ``*.history.json`` objects.

Standalone script: only stdlib + orjson/tqdm, no imports from the benchmarks
package, so it can be copied and run anywhere (e.g. on a cluster without the
repo installed).

Input is a JSONL file where each line is one full history object (for example
``jq -c . **/*.history.json > histories.jsonl``). Output is a JSONL file with one
row per ``trajectories`` entry:

- a ``type == "trajectory"`` entry becomes a row whose ``messages`` are that
  model-view segment; ``tools`` are kept;
- a ``type == "compaction"`` entry becomes a row whose ``messages`` are the
  summarizer ``request`` messages followed by the summarizer ``response``, and
  whose ``tools`` are removed.

Message tool calls, tool schemas, annotations and metadata are normalized the
same way as ``scripts/normalize_agentic_jsonl.py`` (the normalization code is
duplicated there to keep both scripts standalone; keep them in sync if either
changes). ``trajectory_type`` and ``trajectory_id`` are root columns;
``trajectory_index`` and the remaining history entries are folded into the
serialized ``metadata`` column.

Loss masks: ``trajectory`` rows copy each message's ``loss_mask`` from the
original ``messages`` transcript (lookup keyed by role/content/tool-call ids,
default 0). A post-compaction segment re-renders the history the condensation
preserved, duplicating messages an earlier ``trajectory`` row already emitted;
those replayed messages are forced to ``loss_mask=0`` so each turn is trained
once, by the segment that produced it. ``compaction`` rows set ``loss_mask=1``
only on the last assistant message (the summary) and 0 on everything before it.

Usage:
    python extract_trajectories_jsonl.py histories.jsonl trajectories.jsonl
"""

import argparse
import os
import sys

import orjson
import tqdm


TRAJECTORY_TYPE = "trajectory"
COMPACTION_TYPE = "compaction"

_SCHEMA_KEYS = (
    "evaluation",
    "instance_id",
    "messages",
    "model",
    "temperature",
    "test_result",
    "tools",
    "top_p",
    "annotations",
    "trajectory_type",
    "trajectory_id",
)


# ---------------------------------------------------------------------------
# Normalization (kept in sync with scripts/normalize_agentic_jsonl.py)
# ---------------------------------------------------------------------------


def normalize_content(content):
    """Normalize content to string, serializing dict/list if necessary."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, (dict, list)):
        return orjson.dumps(content).decode("utf-8")
    return str(content)


def normalize_messages(messages):
    """Normalize chat messages in place (content, reasoning, tool calls, masks)."""
    for msg in messages:
        if "content" in msg:
            msg["content"] = normalize_content(msg["content"])

        if "provider_specific_fields" in msg:
            msg["provider_specific_fields"] = normalize_content(
                msg["provider_specific_fields"]
            )

        if "tool_calls" in msg:
            if isinstance(msg["tool_calls"], list):
                tool_calls = []
                for tc in msg["tool_calls"]:
                    if isinstance(tc, dict) and "arguments" in tc:
                        if isinstance(tc["arguments"], dict):
                            tc = {"function": tc}
                    if (
                        isinstance(tc, dict)
                        and "function" in tc
                        and isinstance(tc["function"], dict)
                        and "arguments" in tc["function"]
                    ):
                        tc["function"]["arguments"] = normalize_content(
                            tc["function"]["arguments"]
                        )
                    tool_calls.append(tc)
                msg["tool_calls"] = tool_calls
            elif not isinstance(msg["tool_calls"], str) and not msg["tool_calls"]:
                msg.pop("tool_calls")
                if "loss_mask" in msg:
                    msg["loss_mask"] = 0

        if "loss" in msg and isinstance(msg["loss"], (bool, int)):
            msg["loss_mask"] = int(msg["loss"])


def normalize_tools(tools):
    """Normalize tool schemas in place (wrap, required default, parameters str)."""
    normalized = []
    for tool in tools:
        if isinstance(tool, dict) and "function" not in tool and "parameters" in tool:
            tool = {"function": tool}
        if (
            isinstance(tool, dict)
            and "function" in tool
            and isinstance(tool["function"], dict)
        ):
            if "required" not in tool["function"]:
                tool["function"]["required"] = []
            if "parameters" in tool["function"]:
                tool["function"]["parameters"] = normalize_content(
                    tool["function"]["parameters"]
                )
        normalized.append(tool)
    return normalized


# ---------------------------------------------------------------------------
def normalize_metadata(data):
    """Fold non-schema keys into a serialized metadata blob (HF dataset rule).

    Mirrors ``scripts/normalize_agentic_jsonl.py``: ``annotations`` is serialized,
    and every key outside the fixed dataset schema (including the trajectory
    provenance fields) is moved into a JSON-string ``metadata`` column.
    """
    if "annotations" in data:
        data["annotations"] = orjson.dumps(data["annotations"]).decode("utf-8")
    metadata = data.pop("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {"original_metadata": metadata}
    extra_keys = [key for key in data if key not in _SCHEMA_KEYS]
    metadata |= {key: data.pop(key) for key in extra_keys}
    data["metadata"] = orjson.dumps(metadata).decode("utf-8")
    return data


# # Trajectory flattening
# ---------------------------------------------------------------------------


def _message_key(message):
    """Lookup key matching a flattened message to the original transcript."""
    content = message.get("original_content")
    if content is None:
        content = message.get("content")
    if not isinstance(content, str):
        content = normalize_content(content)
    tool_calls = message.get("tool_calls")
    call_ids = ()
    if isinstance(tool_calls, list):
        call_ids = tuple(
            tool_call.get("id")
            for tool_call in tool_calls
            if isinstance(tool_call, dict)
        )
    return (
        message.get("role"),
        content,
        message.get("tool_call_id"),
        message.get("name"),
        call_ids,
    )


def _build_loss_mask_lookup(messages):
    """Build a lookup from original transcript messages to their loss_mask."""
    lookup = {}
    if not isinstance(messages, list):
        return lookup
    for message in messages:
        if isinstance(message, dict) and "loss_mask" in message:
            lookup[_message_key(message)] = int(message["loss_mask"])
    return lookup


def _apply_trajectory_loss_masks(messages, lookup, seen_keys):
    """Copy loss masks from the original transcript, ignoring replayed messages.

    A segment that follows a compaction re-renders the history the condensation
    preserved, so some of its messages were already emitted and trained by an
    earlier segment (the one that actually produced them). Those duplicate
    copies keep ``loss_mask=0``; only messages a segment introduces itself get
    their mask from the original transcript.

    Args:
        messages: The segment's model-view messages, updated in place.
        lookup: Original-transcript loss masks, from ``_build_loss_mask_lookup``.
        seen_keys: Message keys already emitted by earlier ``trajectory`` rows.
            Updated in place with this segment's keys (including its duplicates,
            which stay ignored in later segments too).
    """
    segment_keys = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        key = _message_key(message)
        segment_keys.append(key)
        message["loss_mask"] = 0 if key in seen_keys else lookup.get(key, 0)
    seen_keys.update(segment_keys)


def _apply_compaction_loss_masks(messages):
    """Train only on the summarizer response: last assistant message mask=1."""
    last_assistant = None
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        message["loss_mask"] = 0
        if message.get("role") == "assistant":
            last_assistant = index
    if last_assistant is not None:
        messages[last_assistant]["loss_mask"] = 1


def extract_trajectory_rows(history):
    """Expand one history object into one normalized row per trajectory entry.

    Args:
        history: A parsed ``*.history.json`` object (``messages``, ``tools``,
            ``trajectories``, ...).

    Returns:
        Rows in trajectory order. Histories without a ``trajectories`` list
        (older dumps) produce a single row built from their ``messages``.
        Entries that would produce empty messages are skipped.
    """
    instance_id = history.get("instance_id", "unknown")
    loss_mask_lookup = _build_loss_mask_lookup(history.get("messages"))
    # Keys emitted by earlier trajectory rows: a post-compaction segment replays
    # the preserved history, so those messages must not be trained twice.
    seen_keys = set()
    base = {key: value for key, value in history.items() if key != "trajectories"}
    entries = history.get("trajectories") or []
    if not isinstance(entries, list) or not entries:
        messages = base.get("messages")
        if isinstance(messages, list):
            _apply_trajectory_loss_masks(messages, loss_mask_lookup, seen_keys)
        row = dict(
            base,
            trajectory_type=TRAJECTORY_TYPE,
            trajectory_index=0,
            trajectory_id=f"{instance_id}-{TRAJECTORY_TYPE}-0",
        )
        return [_normalize_row(row)]

    rows = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue
        entry_type = entry.get("type", TRAJECTORY_TYPE)
        row = dict(base)
        if entry_type == COMPACTION_TYPE:
            messages = list(entry.get("request") or [])
            response = entry.get("response")
            if response:
                messages.append(response)
            if not messages:
                continue
            _apply_compaction_loss_masks(messages)
            row["messages"] = messages
            row.pop("tools", None)
        else:
            messages = entry.get("messages")
            if not isinstance(messages, list) or not messages:
                continue
            _apply_trajectory_loss_masks(messages, loss_mask_lookup, seen_keys)
            row["messages"] = messages
            entry_type = TRAJECTORY_TYPE
        row["trajectory_type"] = entry_type
        row["trajectory_index"] = index
        row["trajectory_id"] = f"{instance_id}-{entry_type}-{index}"
        rows.append(_normalize_row(row))
    return rows


def _normalize_row(row):
    """Apply the dataset normalization to a row's messages, tools and metadata."""
    messages = row.get("messages")
    if isinstance(messages, list):
        normalize_messages(messages)
    tools = row.get("tools")
    if isinstance(tools, list):
        row["tools"] = normalize_tools(tools)
    return normalize_metadata(row)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def process_file(input_path, output_path):
    """Read a histories JSONL file and write the flattened trajectory JSONL."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    histories = 0
    rows = 0
    empty_histories = 0
    total_size = os.path.getsize(input_path)

    with (
        open(input_path, "rb") as infile,
        open(output_path, "wb") as outfile,
        tqdm.tqdm(
            total=total_size,
            unit="B",
            unit_scale=True,
            desc=os.path.basename(input_path),
        ) as pbar,
    ):
        for line in infile:
            pbar.update(len(line))
            if not line.strip():
                continue
            try:
                history = orjson.loads(line)
            except Exception as e:  # noqa: BLE001 - report and keep going
                print(f"Skipping malformed line: {e}", file=sys.stderr)
                continue
            if not isinstance(history, dict):
                print("Skipping line: history must be a JSON object", file=sys.stderr)
                continue

            histories += 1
            extracted = extract_trajectory_rows(history)
            if not extracted:
                empty_histories += 1
            for row in extracted:
                outfile.write(orjson.dumps(row) + b"\n")
                rows += 1

    print(
        f"Read {histories} histories -> wrote {rows} trajectory rows "
        f"({empty_histories} histories produced no rows)"
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Flatten the trajectories of exported history objects into a JSONL "
            "dataset with one row per trajectory entry."
        )
    )
    parser.add_argument("input", help="Input JSONL file of history objects")
    parser.add_argument("output", help="Output JSONL file")
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"Error: Input {args.input} not found.", file=sys.stderr)
        sys.exit(1)

    process_file(args.input, args.output)


if __name__ == "__main__":
    main()
