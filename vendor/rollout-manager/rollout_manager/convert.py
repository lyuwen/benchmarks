"""Wire-format conversion for the rollout-capture service.

The capture service stores every exchange in one canonical shape: OpenAI
Chat-Completions messages. Chat-Completions traffic is already in that shape and
is forwarded to the upstream unchanged. Anthropic Messages and OpenAI Responses
traffic is converted to a chat-completions request here, and the upstream
chat-completions reply is converted back into the protocol the client speaks.

Canonical messages are the *capture* shape, not the upstream shape. They drop
wire-only fields (tool-call ids, tool_call_id, response item ids) and flatten
multimodal content blocks to text, so a response we recorded compares equal to
the assistant message a client replays in a later request. That equality is what
lets :class:`~rollout_manager.trajectory.TrajectoryManager` stitch
sequential requests into one prefix tree.

Nothing here tokenizes or renders a chat template: the upstream model backend
owns that.
"""

from __future__ import annotations

import json
import re
import secrets
import time
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "ChatReply",
    "anthropic_count_tokens",
    "anthropic_to_chat_request",
    "canonicalize_message",
    "canonicalize_messages",
    "canonicalize_tools",
    "chat_to_anthropic_response",
    "chat_to_responses_response",
    "flatten_content",
    "message_identity",
    "parse_chat_reply",
    "responses_to_chat_request",
    "wire_history_message",
]


# ---------------------------------------------------------------------------
# canonical message primitives
# ---------------------------------------------------------------------------


def flatten_content(content: Any) -> str:
    """Flatten a wire content value into plain text.

    Accepts chat-completions strings/blocks, Anthropic text/tool_result blocks
    and Responses input/output items. Non-text media is replaced by a marker:
    the capture path is text-only, and the marker keeps replay equality stable.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, (int, float, bool)):
        return str(content)
    if isinstance(content, dict):
        block_type = content.get("type")
        if block_type in {"text", "input_text", "output_text"}:
            return str(content.get("text") or "")
        if block_type in {"image", "image_url", "input_image"}:
            return "[image omitted]"
        if block_type in {"file", "input_file"}:
            return "[file omitted]"
        if block_type == "tool_result":
            return flatten_content(content.get("content"))
        if "content" in content:
            return flatten_content(content.get("content"))
        if "text" in content:
            return str(content.get("text") or "")
        return ""
    if not isinstance(content, list):
        return str(content)

    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
            continue
        if not isinstance(block, dict):
            parts.append(str(block))
            continue
        block_type = block.get("type")
        if block_type in {"text", "input_text", "output_text"}:
            parts.append(str(block.get("text") or ""))
        elif block_type == "tool_result":
            parts.append(flatten_content(block.get("content")))
        elif block_type in {"image", "image_url", "input_image"}:
            parts.append("[image omitted]")
        elif block_type in {"file", "input_file"}:
            parts.append("[file omitted]")
        elif "content" in block:
            parts.append(flatten_content(block.get("content")))
        elif "text" in block:
            parts.append(str(block.get("text") or ""))
    return "\n".join(part for part in parts if part)


def arguments_to_dict(arguments: Any) -> dict[str, Any]:
    """Coerce a tool-argument payload into a dict for client-protocol rendering.

    This is only used when rendering a reply to a non-chat client (Anthropic
    ``tool_use.input`` needs an object). Captured messages keep the original
    payload string; malformed/truncated JSON is never rewritten for capture.
    """
    if isinstance(arguments, dict):
        return arguments
    if arguments is None:
        return {}
    if isinstance(arguments, str):
        text = arguments.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return {"_raw_arguments": arguments}
        return parsed if isinstance(parsed, dict) else {"_raw_arguments": arguments}
    return {"_raw_arguments": str(arguments)}


def _arguments_string(arguments: Any) -> str:
    """Return the wire ``function.arguments`` string, preserving it verbatim."""
    if arguments is None:
        return "{}"
    if isinstance(arguments, str):
        return arguments
    if isinstance(arguments, (dict, list)):
        return json.dumps(arguments, ensure_ascii=False)
    return str(arguments)


def _mint_call_id() -> str:
    return f"call_{secrets.token_hex(12)}"


def canonicalize_message(message: Any) -> dict[str, Any] | None:
    """Store one message with its native Chat-Completions role/content shape.

    Native roles (including ``developer``) and the original content shape are
    preserved; tool-call ids, ``tool_call_id`` and the original JSON-encoded
    argument string are kept so exported ``messages`` are valid
    Chat-Completions messages. ``message_identity`` normalizes these fields when
    matching replayed history, but storage does not lose them.
    """
    if not isinstance(message, dict):
        return None
    role = message.get("role")
    if role in {"system", "developer", "user"}:
        canonical: dict[str, Any] = {"role": role, "content": message.get("content")}
        if message.get("name"):
            canonical["name"] = str(message["name"])
        return canonical
    if role == "tool":
        return {
            "role": "tool",
            "tool_call_id": str(message.get("tool_call_id") or ""),
            "content": message.get("content"),
        }
    if role != "assistant":
        return None

    canonical = {"role": "assistant", "content": message.get("content")}
    if message.get("name"):
        canonical["name"] = str(message["name"])
    reasoning = message.get("reasoning_content")
    if reasoning:
        canonical["reasoning_content"] = flatten_content(reasoning)

    tool_calls: list[dict[str, Any]] = []
    for call in message.get("tool_calls") or []:
        if not isinstance(call, dict):
            continue
        function = call.get("function") if isinstance(call.get("function"), dict) else {}
        name = function.get("name") or call.get("name") or "tool"
        arguments = function.get("arguments")
        if arguments is None:
            arguments = call.get("arguments")
        tool_calls.append(
            {
                "id": str(call.get("id") or _mint_call_id()),
                "type": "function",
                "function": {"name": str(name), "arguments": _arguments_string(arguments)},
            }
        )
    if tool_calls:
        canonical["tool_calls"] = tool_calls
    return canonical


def canonicalize_messages(messages: Any) -> list[dict[str, Any]]:
    """Canonicalize a chat-completions message list, dropping unknown roles."""
    if not isinstance(messages, (list, tuple)):
        return []
    canonical: list[dict[str, Any]] = []
    for message in messages:
        converted = canonicalize_message(message)
        if converted is not None:
            canonical.append(converted)
    return canonical


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple(sorted((str(k), _freeze(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    return value


def _identity_arguments(arguments: Any) -> Any:
    """Normalize an argument payload for history matching only.

    A valid JSON string compares as its decoded value; malformed/truncated JSON
    compares as the exact raw string, so two different truncations never collide.
    """
    if arguments is None:
        return ()
    if isinstance(arguments, str):
        text = arguments.strip()
        if not text:
            return ()
        try:
            return _freeze(json.loads(text))
        except json.JSONDecodeError:
            return ("raw", arguments)
    if isinstance(arguments, (dict, list)):
        return _freeze(arguments)
    return ("raw", str(arguments))


def message_identity(message: dict[str, Any]) -> tuple[Any, ...]:
    """Hashable identity used to match a replayed message against a recorded one.

    Role, flattened content, tool-call **ids**, ``tool_call_id``, tool names and
    decoded argument values are compared. Including ids is required to keep
    duplicate tool calls distinct (otherwise a continuation of ``call_b`` could
    be mounted under an earlier, identical ``call_a`` node and export a result
    whose id matches no preceding call). Reasoning and argument JSON encoding /
    key order remain ignored.
    """
    role = message.get("role")
    content = flatten_content(message.get("content"))
    if role == "tool":
        return ("tool", message.get("tool_call_id") or "", content)
    if role != "assistant":
        return (role, content)
    tool_calls = message.get("tool_calls") or []
    calls = tuple(
        (
            str(call.get("id") or ""),
            str((call.get("function") or {}).get("name") or ""),
            _identity_arguments((call.get("function") or {}).get("arguments")),
        )
        for call in tool_calls
        if isinstance(call, dict)
    )
    return ("assistant", content, calls)


def canonicalize_tools(tools: Any) -> list[dict[str, Any]] | None:
    """Normalize a tool list to OpenAI chat-completions function tools.

    Accepts chat-completions (``{"type": "function", "function": {...}}``),
    Responses (flat ``{"type": "function", "name": ...}``) and Anthropic
    (``{"name": ..., "input_schema": ...}``) shapes. Non-function tools
    (web_search, computer_use, ...) are dropped: the capture path is text-only.
    """
    if not isinstance(tools, (list, tuple)) or not tools:
        return None
    normalized: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        tool_type = tool.get("type")
        if tool_type and tool_type != "function":
            continue
        function = tool.get("function") if isinstance(tool.get("function"), dict) else {}
        name = function.get("name") or tool.get("name")
        if not name:
            continue
        parameters = (
            function.get("parameters")
            or tool.get("parameters")
            or tool.get("input_schema")
            or {"type": "object", "properties": {}}
        )
        normalized_function: dict[str, Any] = {
            "name": str(name),
            "description": function.get("description") or tool.get("description") or "",
            "parameters": parameters,
        }
        strict = function.get("strict")
        if strict is None:
            strict = tool.get("strict")
        if strict is not None:
            normalized_function["strict"] = bool(strict)
        normalized.append({"type": "function", "function": normalized_function})
    return normalized or None


# ---------------------------------------------------------------------------
# upstream chat-completions reply
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChatReply:
    """Normalized view of one upstream chat-completions reply.

    ``message`` is the original wire message; ``wire_tool_calls`` preserves the
    original ``id`` and ``function.arguments`` string (no re-serialization), so
    the captured response is faithful to what the model produced.
    """

    message: dict[str, Any]
    content: str
    reasoning: str
    wire_tool_calls: list[dict[str, Any]] = field(default_factory=list)
    finish_reason: str = "stop"
    usage: dict[str, Any] = field(default_factory=dict)
    response_id: str | None = None
    model: str | None = None

    @property
    def cached_tokens(self) -> int:
        details = self.usage.get("prompt_tokens_details")
        if isinstance(details, dict):
            return int(details.get("cached_tokens") or 0)
        return int(self.usage.get("cached_tokens") or 0)

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.wire_tool_calls)


def parse_chat_reply(data: dict[str, Any]) -> ChatReply:
    """Pull the first choice out of an upstream chat-completions response."""
    choices = data.get("choices") or []
    if not choices:
        raise ValueError("upstream chat-completions reply has no choices")
    choice = choices[0] if isinstance(choices[0], dict) else {}
    message = choice.get("message") if isinstance(choice.get("message"), dict) else {}

    wire_tool_calls: list[dict[str, Any]] = []
    for call in message.get("tool_calls") or []:
        if not isinstance(call, dict):
            continue
        function = call.get("function") if isinstance(call.get("function"), dict) else {}
        name = str(function.get("name") or call.get("name") or "tool")
        arguments = function.get("arguments")
        if arguments is None:
            arguments = call.get("arguments")
        wire_tool_calls.append(
            {
                "id": str(call.get("id") or _mint_call_id()),
                "type": "function",
                "function": {"name": name, "arguments": _arguments_string(arguments)},
            }
        )

    usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
    return ChatReply(
        message=message,
        content=flatten_content(message.get("content")),
        reasoning=flatten_content(message.get("reasoning_content") or message.get("reasoning")),
        wire_tool_calls=wire_tool_calls,
        finish_reason=str(choice.get("finish_reason") or "stop"),
        usage=usage,
        response_id=data.get("id"),
        model=data.get("model"),
    )


def canonical_reply_message(reply: ChatReply) -> dict[str, Any]:
    """The assistant message recorded for a reply, preserving wire fields.

    Tool-call ids and the original argument strings are kept; ``reasoning_content``
    is stored internally and separated from strict exported messages later.
    """
    message: dict[str, Any] = {"role": "assistant", "content": reply.content}
    if reply.reasoning:
        message["reasoning_content"] = reply.reasoning
    if reply.wire_tool_calls:
        message["tool_calls"] = reply.wire_tool_calls
    return message


def wire_history_message(reply: ChatReply) -> dict[str, Any]:
    """The wire-shaped assistant message to replay to a stateless chat backend.

    Used for Responses ``previous_response_id`` chaining, where the client sends
    only the new items and we must rebuild the full chat history ourselves.
    """
    message: dict[str, Any] = {"role": "assistant", "content": reply.content}
    if reply.reasoning:
        message["reasoning_content"] = reply.reasoning
    if reply.wire_tool_calls:
        message["tool_calls"] = reply.wire_tool_calls
    return message


# ---------------------------------------------------------------------------
# request parameter forwarding
# ---------------------------------------------------------------------------

# Top-level fields of the inbound protocol that the converter consumes (or that
# are meaningless on the chat-completions upstream). Anything else -- including
# OpenAI-SDK ``extra_body`` extensions and provider-specific parameters -- is
# forwarded to the upstream request.
_ANTHROPIC_REQUEST_FIELDS = frozenset(
    {
        "model",
        "messages",
        "system",
        "max_tokens",
        "metadata",
        "stop_sequences",
        "stream",
        "temperature",
        "top_k",
        "top_p",
        "tools",
        "tool_choice",
        "thinking",
        "service_tier",
        "mcp_servers",
        "context_management",
        "container",
        "betas",
    }
)
_RESPONSES_REQUEST_FIELDS = frozenset(
    {
        "model",
        "input",
        "instructions",
        "max_output_tokens",
        "temperature",
        "top_p",
        "tools",
        "tool_choice",
        "parallel_tool_calls",
        "stream",
        "store",
        "metadata",
        "user",
        "service_tier",
        "previous_response_id",
        "reasoning",
        "reasoning_effort",
        "text",
        "truncation",
        "include",
        "background",
        "prompt_cache_key",
        "max_tool_calls",
        "conversation",
        "prompt",
        "top_logprobs",
    }
)


# Structural fields an ``extra_body`` object may not inject: they define the
# conversation and are built by the converter. Sampling/provider extensions
# (``top_k``, ``reasoning_effort``, ``chat_template_kwargs``, ...) are allowed.
_EXTRA_BODY_BLOCKED_FIELDS = frozenset(
    {"model", "messages", "system", "input", "instructions", "tools", "tool_choice", "stream", "previous_response_id"}
)


def _apply_extra_request_params(
    body: dict[str, Any], request: dict[str, Any], protocol_fields: frozenset[str]
) -> None:
    """Carry passthrough parameters from the inbound body onto the chat request.

    Mapped fields (``temperature``, ``tools``, ...) win; this only fills in what
    the converter did not set. An explicit ``extra_body`` object is flattened into
    the upstream body (the OpenAI SDK normally does this client-side, but raw
    clients/gateways may still send it).
    """
    extra_body = body.get("extra_body")
    if isinstance(extra_body, dict):
        for key, value in extra_body.items():
            if key not in _EXTRA_BODY_BLOCKED_FIELDS:
                request.setdefault(key, value)
    for key, value in body.items():
        if key in protocol_fields or key == "extra_body":
            continue
        request.setdefault(key, value)


def _responses_text_format_to_chat(text: Any) -> dict[str, Any] | None:
    """Map a Responses ``text.format`` to a chat-completions ``response_format``."""
    if not isinstance(text, dict):
        return None
    fmt = text.get("format")
    if not isinstance(fmt, dict):
        return None
    fmt_type = fmt.get("type")
    if fmt_type == "json_object":
        return {"type": "json_object"}
    if fmt_type == "json_schema":
        json_schema: dict[str, Any] = {
            "name": fmt.get("name") or "response",
            "schema": fmt.get("schema") or {},
        }
        if "description" in fmt:
            json_schema["description"] = fmt["description"]
        if "strict" in fmt:
            json_schema["strict"] = bool(fmt["strict"])
        return {"type": "json_schema", "json_schema": json_schema}
    return None


# ---------------------------------------------------------------------------
# Anthropic Messages <-> chat completions
# ---------------------------------------------------------------------------


def _anthropic_blocks(content: Any) -> list[Any]:
    if isinstance(content, list):
        return content
    return [{"type": "text", "text": flatten_content(content)}]


def _anthropic_system_text(system: Any) -> str:
    if isinstance(system, str):
        return system
    return flatten_content(system)


def anthropic_to_chat_request(body: dict[str, Any]) -> dict[str, Any]:
    """Convert an Anthropic ``/v1/messages`` request into a chat request.

    Tool results become ``tool`` messages (correlated by ``tool_call_id``) and
    tool uses become assistant ``tool_calls``, so the stateless chat backend sees
    the same conversation. Text blocks in a user turn are joined into one user
    message; tool results are emitted first, as clients place them.
    """
    messages: list[dict[str, Any]] = []
    system_text = _anthropic_system_text(body.get("system"))
    if system_text:
        messages.append({"role": "system", "content": system_text})

    for message in body.get("messages") or []:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")
        if role == "user":
            tool_messages: list[dict[str, Any]] = []
            text_parts: list[str] = []
            for block in _anthropic_blocks(content):
                if not isinstance(block, dict):
                    text_parts.append(str(block))
                    continue
                block_type = block.get("type")
                if block_type == "tool_result":
                    tool_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": str(block.get("tool_use_id") or ""),
                            "content": flatten_content(block.get("content")),
                        }
                    )
                elif block_type == "text":
                    text_parts.append(str(block.get("text") or ""))
                else:
                    text_parts.append(flatten_content(block))
            messages.extend(tool_messages)
            text = "\n".join(part for part in text_parts if part)
            if text or not tool_messages:
                messages.append({"role": "user", "content": text})
        elif role == "assistant":
            text_parts: list[str] = []
            thinking_parts: list[str] = []
            tool_calls: list[dict[str, Any]] = []
            for block in _anthropic_blocks(content):
                if not isinstance(block, dict):
                    continue
                block_type = block.get("type")
                if block_type == "text":
                    text_parts.append(str(block.get("text") or ""))
                elif block_type == "thinking":
                    thinking_parts.append(str(block.get("thinking") or ""))
                elif block_type == "tool_use":
                    tool_calls.append(
                        {
                            "id": str(block.get("id") or _mint_call_id()),
                            "type": "function",
                            "function": {
                                "name": str(block.get("name") or "tool"),
                                "arguments": _arguments_string(block.get("input")),
                            },
                        }
                    )
            assistant: dict[str, Any] = {"role": "assistant", "content": "".join(text_parts)}
            if thinking_parts:
                assistant["reasoning_content"] = "".join(thinking_parts)
            if tool_calls:
                assistant["tool_calls"] = tool_calls
            messages.append(assistant)
        elif role == "system":
            messages.append({"role": "system", "content": flatten_content(content)})

    request: dict[str, Any] = {"messages": messages}
    if body.get("model"):
        request["model"] = body["model"]
    for key in ("temperature", "top_p", "top_k"):
        if body.get(key) is not None:
            request[key] = body[key]
    if body.get("max_tokens") is not None:
        request["max_tokens"] = body["max_tokens"]
    if body.get("stop_sequences"):
        request["stop"] = body["stop_sequences"]

    tools = canonicalize_tools(body.get("tools"))
    if tools:
        request["tools"] = tools
        tool_choice = _anthropic_tool_choice(body.get("tool_choice"))
        if tool_choice is not None:
            request["tool_choice"] = tool_choice
    _apply_extra_request_params(body, request, _ANTHROPIC_REQUEST_FIELDS)
    return request


def _anthropic_tool_choice(tool_choice: Any) -> Any:
    if not isinstance(tool_choice, dict):
        return None
    choice_type = tool_choice.get("type")
    if choice_type == "auto":
        return "auto"
    if choice_type == "any":
        return "required"
    if choice_type == "none":
        return "none"
    if choice_type == "tool" and tool_choice.get("name"):
        return {"type": "function", "function": {"name": str(tool_choice["name"])}}
    return None


def chat_to_anthropic_response(reply: ChatReply, *, model: str | None = None) -> dict[str, Any]:
    """Convert an upstream chat-completions reply into an Anthropic message."""
    blocks: list[dict[str, Any]] = []
    if reply.reasoning:
        blocks.append({"type": "thinking", "thinking": reply.reasoning})
    if reply.content:
        blocks.append({"type": "text", "text": reply.content})
    for call in reply.wire_tool_calls:
        function = call.get("function") or {}
        blocks.append(
            {
                "type": "tool_use",
                "id": call.get("id") or _mint_call_id(),
                "name": function.get("name") or "tool",
                "input": arguments_to_dict(function.get("arguments")),
            }
        )
    if not blocks:
        blocks.append({"type": "text", "text": ""})

    if reply.has_tool_calls:
        stop_reason = "tool_use"
    elif reply.finish_reason == "length":
        stop_reason = "max_tokens"
    else:
        stop_reason = "end_turn"

    usage: dict[str, Any] = {
        "input_tokens": int(reply.usage.get("prompt_tokens") or 0),
        "output_tokens": int(reply.usage.get("completion_tokens") or 0),
    }
    if reply.cached_tokens:
        usage["cache_read_input_tokens"] = reply.cached_tokens

    return {
        "id": f"msg_{secrets.token_hex(12)}",
        "type": "message",
        "role": "assistant",
        "model": model or reply.model or "rollout-manager",
        "content": blocks,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": usage,
    }


_TOKEN_ESTIMATE_RE = re.compile(r"\s+")


def anthropic_count_tokens(body: dict[str, Any], *, chars_per_token: float = 4.0) -> int:
    """Rough input-token estimate for ``/v1/messages/count_tokens``.

    There is no tokenizer on the capture path, but returning 0 would tell the
    agent it never needs to compact -- and compaction is exactly the behaviour we
    want to capture. This is a deliberate character-based approximation; see
    ``--chars-per-token``.
    """
    parts: list[str] = []
    system_text = _anthropic_system_text(body.get("system"))
    if system_text:
        parts.append(system_text)
    for message in body.get("messages") or []:
        if not isinstance(message, dict):
            continue
        parts.append(flatten_content(message.get("content")))
    for tool in body.get("tools") or []:
        if isinstance(tool, dict):
            parts.append(json.dumps(tool, ensure_ascii=False))
    text = "\n".join(parts)
    if not text:
        return 0
    base = chars_per_token if chars_per_token > 0 else 4.0
    return max(1, int(len(_TOKEN_ESTIMATE_RE.sub(" ", text)) / base))


# ---------------------------------------------------------------------------
# OpenAI Responses <-> chat completions
# ---------------------------------------------------------------------------


def _responses_tools_to_chat(tools: Any) -> list[dict[str, Any]] | None:
    return canonicalize_tools(tools)


def _responses_tool_choice_to_chat(tool_choice: Any) -> Any:
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        return tool_choice
    if isinstance(tool_choice, dict):
        if tool_choice.get("type") == "function":
            name = tool_choice.get("name") or (tool_choice.get("function") or {}).get("name")
            if name:
                return {"type": "function", "function": {"name": str(name)}}
        if tool_choice.get("type") == "allowed_tools":
            return "auto"
    return None


def _responses_input_to_messages(items: Any) -> list[dict[str, Any]]:
    """Convert Responses ``input`` items into chat-completions messages.

    The Responses wire flattens one assistant turn into a message item plus one
    item per function call, so consecutive function-call items are accumulated
    back into a single assistant message before it is emitted.
    """
    if isinstance(items, str):
        return [{"role": "user", "content": items}]
    if not isinstance(items, (list, tuple)):
        return []

    messages: list[dict[str, Any]] = []
    pending: dict[str, Any] | None = None

    def flush() -> None:
        nonlocal pending
        if pending is not None:
            messages.append(pending)
            pending = None

    def ensure_pending() -> dict[str, Any]:
        nonlocal pending
        if pending is None:
            pending = {"role": "assistant", "content": ""}
        return pending

    for item in items:
        if isinstance(item, str):
            flush()
            messages.append({"role": "user", "content": item})
            continue
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if not item_type and item.get("role"):
            item_type = "message"

        if item_type == "message":
            role = item.get("role") or "user"
            content = flatten_content(item.get("content"))
            if role == "assistant":
                flush()
                pending = {"role": "assistant", "content": content}
            else:
                flush()
                messages.append({"role": str(role), "content": content})
        elif item_type == "reasoning":
            # Prior reasoning is not replayed: Responses clients may only carry
            # an encrypted blob, and chat backends treat it as optional context.
            continue
        elif item_type == "function_call":
            assistant = ensure_pending()
            assistant.setdefault("tool_calls", []).append(
                {
                    "id": str(item.get("call_id") or item.get("id") or _mint_call_id()),
                    "type": "function",
                    "function": {
                        "name": str(item.get("name") or "tool"),
                        "arguments": _arguments_string(item.get("arguments")),
                    },
                }
            )
        elif item_type in {"function_call_output", "computer_call_output", "local_shell_call_output"}:
            flush()
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": str(item.get("call_id") or ""),
                    "content": flatten_content(item.get("output", item.get("content"))),
                }
            )
        else:
            # Unknown item type: keep any text it carries rather than dropping
            # the turn's content silently.
            text = flatten_content(item.get("content"))
            if text:
                flush()
                messages.append({"role": "user", "content": text})
    flush()
    return messages


def responses_to_chat_request(
    body: dict[str, Any], *, previous_messages: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    """Convert an OpenAI ``/v1/responses`` request into a chat request.

    ``previous_messages`` are the wire chat messages up to and including the
    previous response, supplied by the service when the client chains through
    ``previous_response_id``. The client's current ``input`` is appended after
    them.
    """
    messages = list(previous_messages or [])
    instructions = body.get("instructions")
    if instructions:
        system_message = {"role": "system", "content": flatten_content(instructions)}
        if messages and messages[0].get("role") == "system":
            messages[0] = system_message
        else:
            messages.insert(0, system_message)
    messages.extend(_responses_input_to_messages(body.get("input")))

    request: dict[str, Any] = {"messages": messages}
    if body.get("model"):
        request["model"] = body["model"]
    if body.get("max_output_tokens") is not None:
        request["max_tokens"] = body["max_output_tokens"]
    for key in ("temperature", "top_p"):
        if body.get(key) is not None:
            request[key] = body[key]
    if body.get("parallel_tool_calls") is not None:
        request["parallel_tool_calls"] = bool(body["parallel_tool_calls"])
    reasoning = body.get("reasoning")
    if isinstance(reasoning, dict) and reasoning.get("effort") is not None:
        request["reasoning_effort"] = reasoning["effort"]
    if body.get("reasoning_effort") is not None:
        request["reasoning_effort"] = body["reasoning_effort"]
    response_format = _responses_text_format_to_chat(body.get("text"))
    if response_format is not None:
        request["response_format"] = response_format
    for key in ("store", "metadata", "user", "service_tier", "prompt_cache_key"):
        if body.get(key) is not None:
            request[key] = body[key]
    tools = _responses_tools_to_chat(body.get("tools"))
    if tools:
        request["tools"] = tools
        tool_choice = _responses_tool_choice_to_chat(body.get("tool_choice"))
        if tool_choice is not None:
            request["tool_choice"] = tool_choice
    _apply_extra_request_params(body, request, _RESPONSES_REQUEST_FIELDS)
    return request


def chat_to_responses_response(
    reply: ChatReply,
    request_body: dict[str, Any],
    *,
    previous_response_id: str | None = None,
) -> dict[str, Any]:
    """Convert an upstream chat-completions reply into a Responses object."""
    output: list[dict[str, Any]] = []
    if reply.reasoning:
        output.append(
            {
                "type": "reasoning",
                "id": f"rs_{secrets.token_hex(12)}",
                "summary": [{"type": "summary_text", "text": reply.reasoning}],
            }
        )
    if reply.content:
        output.append(
            {
                "type": "message",
                "id": f"msg_{secrets.token_hex(12)}",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "annotations": [], "text": reply.content}],
            }
        )
    for call in reply.wire_tool_calls:
        function = call.get("function") or {}
        output.append(
            {
                "type": "function_call",
                "id": f"fc_{secrets.token_hex(12)}",
                "call_id": call.get("id"),
                "name": function.get("name") or "tool",
                "arguments": function.get("arguments") or "{}",
                "status": "completed",
            }
        )

    incomplete = reply.finish_reason == "length"
    usage: dict[str, Any] = {
        "input_tokens": int(reply.usage.get("prompt_tokens") or 0),
        "output_tokens": int(reply.usage.get("completion_tokens") or 0),
        "total_tokens": int(
            reply.usage.get("total_tokens")
            or (int(reply.usage.get("prompt_tokens") or 0) + int(reply.usage.get("completion_tokens") or 0))
        ),
    }
    if reply.cached_tokens:
        usage["input_tokens_details"] = {"cached_tokens": reply.cached_tokens}

    response: dict[str, Any] = {
        "id": f"resp_{secrets.token_hex(12)}",
        "object": "response",
        "created_at": int(time.time()),
        "status": "incomplete" if incomplete else "completed",
        "error": None,
        "incomplete_details": {"reason": "max_output_tokens"} if incomplete else None,
        "instructions": request_body.get("instructions"),
        "max_output_tokens": request_body.get("max_output_tokens"),
        "model": reply.model or request_body.get("model") or "rollout-manager",
        "output": output,
        "output_text": "\n".join(
            part.get("text", "")
            for item in output
            if item.get("type") == "message"
            for part in item.get("content") or []
            if part.get("type") == "output_text"
        ),
        "parallel_tool_calls": request_body.get("parallel_tool_calls", True),
        "previous_response_id": previous_response_id,
        "reasoning": request_body.get("reasoning"),
        "store": request_body.get("store", True),
        "temperature": request_body.get("temperature"),
        "text": request_body.get("text"),
        "tool_choice": request_body.get("tool_choice"),
        "tools": request_body.get("tools"),
        "top_p": request_body.get("top_p"),
        "truncation": request_body.get("truncation"),
        "usage": usage,
        "user": request_body.get("user"),
        "metadata": request_body.get("metadata") or {},
    }
    return response
