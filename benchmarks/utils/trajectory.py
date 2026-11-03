"""Export the model-facing history for one evaluation instance.

The per-instance ``*.history.json`` written by ``run_infer.py`` has two
complementary views, both derived here:

- ``messages``: the full event-dump transcript (forgotten events included). When
  the features are on it carries two export-only annotations: tool results
  pruned by the tool-result pruner keep their full text in ``original_content``,
  and every ``Condensation`` event appears as a ``role="context"`` marker at the
  point where it was triggered. With pruning and condensation off, this is byte
  identical to the historical dump.
- ``trajectories``: the exact message lists the model saw, split at every
  condensation boundary, with the compaction request/response records in
  between:

    segment 0 (before compaction 1)
    compaction 1 (trigger events, summarizer input, summarizer output, result)
    segment 1 (after compaction 1, up to compaction 2)
    ...
    segment N (after the last compaction)

Each segment entry holds the accumulated model view at the end of that segment,
so every earlier request from the segment is a prefix of the same message list.
The pruning policy is read from the policy the agent recorded on its
``SystemPromptEvent`` at conversation start, so it cannot diverge from what the
model actually saw.

Rollout:
Only ``benchmarks/swebench/run_infer.py`` uses this module today. To adopt another
task once this has been validated, mirror its wiring: import
``resolve_pruning_config`` / ``build_messages`` / ``build_trajectories`` /
``pruning_metadata``, resolve the policy after the conversation finishes, then
replace the manual event-to-messages dump with ``build_messages`` and add
``trajectories`` and (conditionally) ``pruning`` to that task's ``dump_data``.
"""

from collections.abc import Sequence
from typing import Any

from openhands.sdk import Event, LLMConvertibleEvent, Message
from openhands.sdk.context.tool_result_pruner import (
    PRUNE_MARKER,
    ToolResultPruneConfig,
    config_from_env,
    is_enabled,
    prune_tool_result_messages,
    resolve_config,
)
from openhands.sdk.context.view import View
from openhands.sdk.event import Condensation, CondensationRequest, SystemPromptEvent


TRAJECTORY = "trajectory"
COMPACTION = "compaction"
CONTEXT_ROLE = "context"


def resolve_pruning_config(events: Sequence[Event]) -> ToolResultPruneConfig | None:
    """Resolve the tool-result pruning policy that was active for this conversation.

    The agent records the resolved policy on its ``SystemPromptEvent`` at
    conversation start, so the exported trajectory reproduces the pruning the
    model actually saw instead of trusting the exporter process's environment
    (which differs from the container's for remote workspaces).

    Args:
        events: The conversation's event log, in order.

    Returns:
        The recorded configuration, or None when pruning was disabled. Event
        logs written before the policy was recorded fall back to this process's
        ``OH_TOOL_RESULT_PRUNER_*`` environment.
    """
    for event in events:
        if (
            isinstance(event, SystemPromptEvent)
            and event.tool_result_pruning is not None
        ):
            return _config_from_record(event.tool_result_pruning)
    return config_from_env() if is_enabled() else None


def _config_from_record(record: dict[str, Any]) -> ToolResultPruneConfig | None:
    if not record.get("enabled", False):
        return None
    return resolve_config(
        ToolResultPruneConfig(
            threshold_chars=int(record["threshold_chars"]),
            head_chars=int(record["head_chars"]),
            tail_chars=int(record["tail_chars"]),
        )
    )


def pruning_metadata(config: ToolResultPruneConfig | None) -> dict[str, Any] | None:
    """Describe an active pruning configuration for the exported history.

    Args:
        config: The resolved pruning policy, or None when pruning is disabled.

    Returns:
        A JSON-serializable description of the pruner budgets, or None.
    """
    if config is None:
        return None
    return {
        "enabled": True,
        "threshold_chars": config.threshold_chars,
        "head_chars": config.head_chars,
        "tail_chars": config.tail_chars,
        "marker": PRUNE_MARKER,
    }


def build_messages(
    events: Sequence[Event],
    pruning_config: ToolResultPruneConfig | None = None,
) -> list[dict[str, Any]]:
    """Render the flat ``messages`` transcript with export annotations.

    Args:
        events: The conversation's append-only event log, in order.
        pruning_config: Resolved pruning policy. Defaults to resolving it from
            the event log.

    Returns:
        Chat-format messages for every LLM-convertible event, with pruned tool
        results carrying ``original_content`` and a ``role="context"`` entry at
        each ``Condensation`` event's position in the log.
    """
    config = (
        pruning_config if pruning_config is not None else resolve_pruning_config(events)
    )
    messages: list[dict[str, Any]] = []
    buffer: list[LLMConvertibleEvent] = []

    def flush() -> None:
        if not buffer:
            return
        messages.extend(
            _render_messages(
                LLMConvertibleEvent.events_to_messages(list(buffer)), config
            )
        )
        buffer.clear()

    for event in events:
        if isinstance(event, Condensation):
            flush()
            messages.append(_condensation_message(event))
        elif isinstance(event, LLMConvertibleEvent):
            buffer.append(event)
    flush()
    return messages


def build_trajectories(
    events: Sequence[Event],
    pruning_config: ToolResultPruneConfig | None = None,
) -> list[dict[str, Any]]:
    """Build the alternating trajectory/compaction export entries.

    Args:
        events: The conversation's append-only event log, in order.
        pruning_config: Resolved pruning policy. Defaults to resolving it from
            the event log.

    Returns:
        Ordered entries: a ``trajectory`` before each ``condensation`` event, a
        ``compaction`` record for it, and a final ``trajectory`` for the tail of
        the run. A trailing condensation request without a following
        condensation produces a final ``compaction`` entry with null
        ``request``/``response``/``result``.
    """
    config = (
        pruning_config if pruning_config is not None else resolve_pruning_config(events)
    )
    trajectories: list[dict[str, Any]] = []
    segment_start = 0

    for index, event in enumerate(events):
        if not isinstance(event, Condensation):
            continue
        trajectories.append(_trajectory_entry(events[:index], config))
        requests = [
            item
            for item in events[segment_start:index]
            if isinstance(item, CondensationRequest)
        ]
        trajectories.append(_compaction_entry(event, requests))
        segment_start = index + 1

    trajectories.append(_trajectory_entry(events, config))

    trailing_requests = [
        item for item in events[segment_start:] if isinstance(item, CondensationRequest)
    ]
    if trailing_requests:
        trajectories.append(_unmatched_compaction_entry(trailing_requests))

    return trajectories


def _trajectory_entry(
    events: Sequence[Event],
    config: ToolResultPruneConfig | None,
) -> dict[str, Any]:
    return {"type": TRAJECTORY, "messages": _render_model_view(events, config)}


def _render_model_view(
    events: Sequence[Event],
    config: ToolResultPruneConfig | None,
) -> list[dict[str, Any]]:
    """Render the model view of ``events`` as OpenAI chat messages.

    The view applies condensation (forgetting plus summary insertion) and
    unmatched-tool-call filtering, so the result matches what the agent sent to
    the LLM.
    """
    view = View.from_events(events)
    return _render_messages(LLMConvertibleEvent.events_to_messages(view.events), config)


def _render_messages(
    messages: Sequence[Message],
    config: ToolResultPruneConfig | None,
) -> list[dict[str, Any]]:
    """Serialize messages, keeping the removed text of pruned tool results."""
    if config is None:
        return [_message_to_chat_dict(message) for message in messages]

    pruned_messages = prune_tool_result_messages(messages, config)
    rendered: list[dict[str, Any]] = []
    for original, current in zip(messages, pruned_messages, strict=True):
        chat_dict = _message_to_chat_dict(current)
        if original is not current:
            chat_dict["original_content"] = _message_to_chat_dict(original)["content"]
        rendered.append(chat_dict)
    return rendered


def _condensation_message(condensation: Condensation) -> dict[str, Any]:
    """Render a condensation as a role="context" marker in the flat transcript."""
    return {
        "role": CONTEXT_ROLE,
        "content": condensation.summary,
        "kind": "Condensation",
        "id": condensation.id,
        "timestamp": condensation.timestamp,
        "source": condensation.source,
        "summary_offset": condensation.summary_offset,
        "forgotten_event_ids": condensation.forgotten_event_ids,
        "llm_response_id": condensation.llm_response_id,
    }


def _compaction_entry(
    condensation: Condensation,
    requests: Sequence[CondensationRequest],
) -> dict[str, Any]:
    return {
        "type": COMPACTION,
        "trigger": "request" if requests else "pressure",
        "trigger_events": [_event_dump(request) for request in requests],
        "request": (
            [
                _message_to_chat_dict(message)
                for message in condensation.request_messages
            ]
            if condensation.request_messages is not None
            else None
        ),
        "response": (
            _message_to_chat_dict(condensation.response_message)
            if condensation.response_message is not None
            else None
        ),
        "result": condensation.model_dump(
            mode="json", exclude={"request_messages", "response_message"}
        ),
    }


def _unmatched_compaction_entry(
    requests: Sequence[CondensationRequest],
) -> dict[str, Any]:
    return {
        "type": COMPACTION,
        "trigger": "request",
        "trigger_events": [_event_dump(request) for request in requests],
        "request": None,
        "response": None,
        "result": None,
    }


def _event_dump(event: Event) -> dict[str, Any]:
    return event.model_dump(mode="json")


def _message_to_chat_dict(message: Message) -> dict[str, Any]:
    return message.model_copy(update={"send_reasoning_content": True}).to_chat_dict()
