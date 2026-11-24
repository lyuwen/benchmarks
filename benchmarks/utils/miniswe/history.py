"""Normalize mini-swe-agent transcripts into the benchmark history contract.

The history contract is documented in ``docs/condenser-wiring.md`` and produced
for OpenHands runs by ``benchmarks/utils/trajectory.py``. mini has no event log
and no condenser, so the mapping is simpler:

- ``messages``: the chat transcript in order, with assistant tool calls, their
  ids, matching tool results and reasoning content preserved. mini's internal
  ``role="exit"`` records are diagnostics and stay in the raw trace only.
- ``trajectories``: one ``type: trajectory`` segment holding the exact message
  list the model saw (for mini, identical to ``messages`` because nothing is
  pruned or compacted).
- ``pruning`` is never emitted: mini does not prune the model view.

Content is encoded the way the OpenHands exporter encodes it for tool-calling
models (a list of ``{"type": "text", "text": ...}`` parts), so downstream
readers see one schema regardless of harness.
"""

from __future__ import annotations

import json
import os
from typing import Any, Sequence

from openhands.sdk import get_logger


logger = get_logger(__name__)

TRAJECTORY = "trajectory"
CHAT_ROLES = frozenset({"system", "user", "assistant", "tool"})
TOOL_NAME = "bash"
"""mini-swe-agent v2 exposes exactly one tool, always named ``bash``."""

SECRET_KEY_HINTS = (
    "api_key",
    "apikey",
    "authorization",
    "secret",
    "password",
)
"""Key fragments whose string values are redacted from serialized configs."""


def redact_secrets(value: Any, key: str | None = None) -> Any:
    """Recursively replace credential-looking values with ``"<redacted>"``."""
    if isinstance(value, dict):
        return {k: redact_secrets(v, str(k)) for k, v in value.items()}
    if isinstance(value, list):
        return [redact_secrets(item, key) for item in value]
    if isinstance(value, str) and key is not None:
        lowered = key.lower()
        if any(hint in lowered for hint in SECRET_KEY_HINTS):
            return "<redacted>"
    return value


def normalize_messages(messages: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert mini's raw chat dicts into the exported transcript schema."""
    normalized: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role not in CHAT_ROLES:
            continue
        entry: dict[str, Any] = {"role": role}

        content = message.get("content")
        if isinstance(content, str) and content.strip():
            entry["content"] = [{"type": "text", "text": content}]
        elif isinstance(content, list) and content:
            entry["content"] = content

        tool_calls = message.get("tool_calls")
        if role == "assistant" and isinstance(tool_calls, list) and tool_calls:
            entry["tool_calls"] = [_normalize_tool_call(tc) for tc in tool_calls]

        if role == "tool":
            tool_call_id = message.get("tool_call_id")
            if isinstance(tool_call_id, str) and tool_call_id:
                entry["tool_call_id"] = tool_call_id
            name = message.get("name")
            entry["name"] = name if isinstance(name, str) and name else TOOL_NAME

        reasoning = message.get("reasoning_content")
        if isinstance(reasoning, str) and reasoning:
            entry["reasoning_content"] = reasoning

        provider_fields = message.get("provider_specific_fields")
        if isinstance(provider_fields, dict) and provider_fields:
            entry["provider_specific_fields"] = provider_fields

        normalized.append(entry)
    return normalized


def build_trajectories(messages: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the single model-facing trajectory segment."""
    return [{"type": TRAJECTORY, "messages": normalize_messages(messages)}]


def build_history_dump(
    *,
    instance_id: str,
    model: str,
    temperature: float | None,
    top_p: float | None,
    messages: Sequence[dict[str, Any]],
    tools: Sequence[dict[str, Any]],
    git_patch: str,
    evaluation: Any = None,
    include_evaluation: bool = False,
    miniswe: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the ``<instance_id>.history.json`` payload for one attempt.

    Args:
        evaluation: Judge verdict. Pass ``None`` when a judge ran but returned no
            verdict.
        include_evaluation: Set when a judge is configured so the ``evaluation``
            key is written (with ``None`` for unavailable verdicts). When no judge
            is configured the key is omitted, matching the OpenHands runners.
    """
    normalized = normalize_messages(messages)
    dump: dict[str, Any] = {
        "instance_id": instance_id,
        "messages": normalized,
        "model": model,
        "tools": list(tools),
        "temperature": temperature,
        "top_p": top_p,
        "test_result": {"git_patch": git_patch},
    }
    if include_evaluation:
        dump["evaluation"] = evaluation
    dump["trajectories"] = [
        {"type": TRAJECTORY, "messages": normalize_messages(messages)}
    ]
    if miniswe is not None:
        dump["miniswe"] = dict(miniswe)
    return dump


def history_filename(instance_id: str, attempt: int | None = None) -> str:
    """Per-attempt history filename (attempt=None is the canonical name)."""
    if attempt is None:
        return f"{instance_id}.history.json"
    return f"{instance_id}.attempt{attempt}.history.json"


def write_history(
    *,
    output_dir: str,
    instance_id: str,
    attempt: int,
    dump: dict[str, Any],
) -> str:
    """Atomically write the per-attempt history file and return its path."""
    path = os.path.join(output_dir, history_filename(instance_id, attempt))
    _atomic_write_json(path, dump)
    logger.info("Dumped mini conversation history to %s", path)
    return path


def publish_canonical_histories(output_dir: str, output_path: str) -> None:
    """Publish the selected attempt's history under the canonical filename.

    ``Evaluation`` aggregates attempts outside this module, so the selected
    attempt is read back from the final JSONL (both success and error writers)
    and its per-attempt history is copied to ``<instance_id>.history.json``.
    Earlier-attempt histories and diagnostics are left untouched.
    """
    selected: dict[str, int] = {}
    for path in (output_path, output_path.replace(".jsonl", "_errors.jsonl")):
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                instance_id = row.get("instance_id")
                attempt = row.get("attempt", 1)
                if isinstance(instance_id, str) and isinstance(attempt, int):
                    selected[instance_id] = attempt

    published = 0
    for instance_id, attempt in selected.items():
        source = os.path.join(output_dir, history_filename(instance_id, attempt))
        if not os.path.exists(source):
            logger.warning(
                "Cannot publish history for %s: %s is missing", instance_id, source
            )
            continue
        with open(source, "r", encoding="utf-8") as handle:
            dump = json.load(handle)
        _atomic_write_json(
            os.path.join(output_dir, history_filename(instance_id)), dump
        )
        published += 1
    if published:
        logger.info("Published %d canonical history file(s)", published)


def _normalize_tool_call(tool_call: Any) -> dict[str, Any]:
    if not isinstance(tool_call, dict):
        return {"type": "function", "function": {}}
    normalized: dict[str, Any] = {}
    call_id = tool_call.get("id")
    if isinstance(call_id, str):
        normalized["id"] = call_id
    normalized["type"] = tool_call.get("type") or "function"
    function = tool_call.get("function")
    normalized["function"] = dict(function) if isinstance(function, dict) else {}
    return normalized


def _atomic_write_json(path: str, payload: dict[str, Any]) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    tmp_path = f"{path}.tmp.{os.getpid()}"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(tmp_path, path)
