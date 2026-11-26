"""Normalize flattened harness records for HuggingFace ``datasets`` and split
auto-mode classifier samples.

HuggingFace's JSON loader infers a pyarrow schema from the first rows and is
unhappy with heterogeneous nested records and deeply-nested optional fields, so
the flattened output is normalized here:

- the stable top-level fields are ``instance_id``, ``messages``, ``tools`` and a
  JSON-string ``metadata`` (everything else is serialized into it), plus
  ``classifier`` for classifier samples;
- message ``content`` is always a string, tool-call ``arguments`` are JSON
  strings, and empty tool-call lists are dropped;
- tools are normalized with a string ``description``, a JSON-string
  ``parameters`` and a string-list ``required`` (mirroring
  ``normalize_agentic_jsonl.py``).

Auto-mode classifier replies look like ``<block>yes</block>`` /
``<block>no</block>``; their label is extracted into ``classifier`` so those
samples can live in a separate JSONL file.
"""

from __future__ import annotations

import json
import re
from typing import Any


CLASSIFIER_RE = re.compile(r"^\s*<block>\s*(yes|no)\s*(?:</block>)?\s*$", re.IGNORECASE)

KEEP_TOP_LEVEL = frozenset({"instance_id", "messages", "tools", "classifier"})


def classifier_label(record: dict[str, Any]) -> str | None:
    """Return the auto-mode classifier verdict (``yes``/``no``) if present."""
    for message in reversed(record.get("messages") or []):
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        match = CLASSIFIER_RE.match(content)
        if match:
            return match.group(1).lower()
    return None


def normalize_content(value: Any) -> str:
    """Coerce a message/tool payload to a string (JSON for dicts/lists)."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value)


def normalize_tool_call(call: Any) -> dict[str, Any]:
    """Normalize one tool call to ``{id, type, function{name, arguments}}``."""
    if not isinstance(call, dict):
        call = {}
    if "function" not in call and ("arguments" in call or "parameters" in call):
        call = {"function": call}
    function = call.get("function")
    if not isinstance(function, dict):
        function = {}
    arguments = function.get("arguments", function.get("parameters"))
    return {
        "id": str(call.get("id") or ""),
        "type": str(call.get("type") or "function"),
        "function": {
            "name": str(function.get("name") or ""),
            "arguments": ("{}" if arguments is None else normalize_content(arguments)),
        },
    }


def normalize_tool(tool: Any) -> dict[str, Any]:
    """Normalize one tool definition for a stable HuggingFace nested schema."""
    if isinstance(tool, str):
        try:
            tool = json.loads(tool)
        except json.JSONDecodeError:
            tool = {}
    if not isinstance(tool, dict):
        tool = {}
    if "function" not in tool and "parameters" in tool:
        tool = {"function": tool}
    function = tool.get("function")
    if not isinstance(function, dict):
        function = {}
    parameters = function.get("parameters", function.get("input_schema"))
    required = function.get("required")
    if not isinstance(required, list):
        required = []
    normalized_function: dict[str, Any] = {
        "name": str(function.get("name") or ""),
        "description": normalize_content(function.get("description")),
        "parameters": ("{}" if parameters is None else normalize_content(parameters)),
        "required": [str(value) for value in required],
        "strict": bool(function.get("strict", False)),
    }
    return {"type": "function", "function": normalized_function}


def normalize_message(message: Any) -> dict[str, Any]:
    """Normalize one chat message (stable key order, JSON-safe values)."""
    if not isinstance(message, dict):
        return {"role": "", "content": ""}
    role = str(message.get("role") or "")
    normalized: dict[str, Any] = {"role": role}
    reasoning = message.get("reasoning_content")
    if reasoning:
        normalized["reasoning_content"] = normalize_content(reasoning)
    normalized["content"] = normalize_content(message.get("content"))
    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list) and tool_calls:
        normalized["tool_calls"] = [normalize_tool_call(call) for call in tool_calls]
    if message.get("tool_call_id") is not None:
        normalized["tool_call_id"] = str(message.get("tool_call_id"))
    if message.get("name") is not None:
        normalized["name"] = str(message.get("name"))
    provider_fields = message.get("provider_specific_fields")
    if provider_fields is not None:
        normalized["provider_specific_fields"] = normalize_content(provider_fields)
    if isinstance(tool_calls, list) and not tool_calls:
        # an explicitly empty tool-call list is a non-action step
        normalized["loss_mask"] = 0
    elif "loss_mask" in message:
        normalized["loss_mask"] = int(bool(message.get("loss_mask")))
    return normalized


def normalize_record(
    record: dict[str, Any], *, classifier: str | None = None
) -> dict[str, Any]:
    """Normalize a flattened record for ``datasets.load_dataset('json')``.

    Stable top-level schema: ``instance_id``, ``messages``, ``tools``,
    ``metadata`` (JSON string with every other field) and, for classifier
    samples, ``classifier``.
    """
    label = classifier
    if label is None:
        existing = record.get("classifier")
        if existing in ("yes", "no"):
            label = existing
    metadata = {
        key: value for key, value in record.items() if key not in KEEP_TOP_LEVEL
    }
    normalized: dict[str, Any] = {
        "instance_id": str(record.get("instance_id") or ""),
        "messages": [
            normalize_message(message)
            for message in record.get("messages") or []
            if isinstance(message, dict)
        ],
        "tools": [normalize_tool(tool) for tool in record.get("tools") or []],
    }
    if label in ("yes", "no"):
        normalized["classifier"] = label
    normalized["metadata"] = json.dumps(
        metadata, ensure_ascii=False, separators=(",", ":"), default=str
    )
    return normalized
