"""Export the model-view trajectory for one evaluation instance.

The per-instance ``*.history.json`` written by ``run_infer.py`` keeps ``messages``
as the full event-dump transcript (unpruned, pre-condensation events included).
This module derives the complementary ``trajectories`` view: the exact message
lists the model saw, split at every condensation boundary, with the compaction
request/response records in between:

    segment 0 (before compaction 1)
    compaction 1 (trigger events, summarizer input, summarizer output, result)
    segment 1 (after compaction 1, up to compaction 2)
    ...
    segment N (after the last compaction)

Each segment entry holds the accumulated model view at the end of that segment,
so every earlier request from the segment is a prefix of the same message list.
When tool-result pruning is enabled, pruned tool messages carry the removed
content in ``original_content``. The pruning policy is read from the policy the
agent recorded on its ``SystemPromptEvent`` at conversation start, so it cannot
diverge from what the model actually saw.

Rollout:
Only ``benchmarks/swebench/run_infer.py`` emits ``trajectories`` today. To adopt
another task once this has been validated, mirror its three-line wiring: import
``resolve_pruning_config`` / ``build_trajectories`` / ``pruning_metadata``, then
after the conversation finishes resolve the policy and add ``trajectories`` and
(conditionally) ``pruning`` to that task's ``dump_data`` before writing
``{instance.id}.history.json``. ``messages`` stays unchanged in every task.
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
    the LLM. Pruned tool messages keep their original text in
    ``original_content``.
    """
    view = View.from_events(events)
    messages = LLMConvertibleEvent.events_to_messages(view.events)
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
