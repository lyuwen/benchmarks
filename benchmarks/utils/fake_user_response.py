"""
Fake user response utilities for evaluation benchmarks.

This module provides functionality to automatically respond to agent messages
during evaluation, similar to the v0 OpenHands evaluation framework.

When an agent sends a message (instead of using tools), this module provides
a fake user response to keep the agent working on the task.
"""

from __future__ import annotations

import re
import time
from functools import partial
from typing import TYPE_CHECKING, Callable

from openhands.sdk import Message, TextContent, get_logger
from openhands.sdk.conversation.state import ConversationExecutionStatus
from openhands.sdk.event import ActionEvent, Event, MessageEvent
from openhands.sdk.tool.builtins.finish import FinishAction


if TYPE_CHECKING:
    from openhands.sdk import LLM
    from openhands.sdk.conversation.base import BaseConversation

logger = get_logger(__name__)

# Type alias for fake user response function
FakeUserResponseFn = Callable[["BaseConversation"], str]

# Cap the assistant message included in the classifier prompt so the extra LLM
# call stays cheap even when the agent emits a very long message.
MAX_CLASSIFIER_MESSAGE_CHARS = 12_000

_FINISH_CLASSIFIER_SYSTEM_PROMPT = (
    "You are a strict classifier used by an automated evaluation harness. "
    "Your only job is to decide whether an AI agent's latest message indicates "
    "that the agent considers the task finished and has nothing further to do.\n"
    "Answer YES only when the message clearly indicates completion: the agent "
    "reports the final result, states that the task is done, or summarizes work "
    "it considers finished.\n"
    "Answer NO when the message is an interim progress update, asks the user a "
    "question, requests clarification or help, or otherwise implies that more "
    "work remains.\n"
    "Reply with exactly YES or NO and nothing else."
)


def _encapsulation_instruction(encapsulate_solution: bool) -> str:
    """Return the optional instruction about encapsulating the final answer."""
    if not encapsulate_solution:
        return ""
    return (
        "Your final answer MUST be encapsulated within <solution> and </solution>.\n"
        "For example: The answer to the question is <solution> 42 </solution>.\n"
    )


def fake_user_response(
    conversation: "BaseConversation",
    encapsulate_solution: bool = False,
) -> str:
    """Generate a fake user response for CodeAct-style agents.

    This function is called when the agent sends a message (tries to talk to the user)
    instead of using tools. It provides a response that encourages the agent to
    continue working on the task.

    Args:
        conversation: The conversation instance containing the event history.
        encapsulate_solution: If True, instructs the agent to encapsulate the
            final answer within <solution> tags.

    Returns:
        A string message to send as a fake user response.
        Returns '/exit' if the agent has already provided an answer.
    """
    encaps_str = _encapsulation_instruction(encapsulate_solution)
    msg = (
        "Please continue working on the task on whatever approach you think is suitable.\n"
        "When you think you have solved the question, please use the finish tool and "
        "include your final answer in the message parameter of the finish tool.\n"
        f"{encaps_str}"
        "IMPORTANT: YOU SHOULD NEVER ASK FOR HUMAN HELP.\n"
    )

    events = list(conversation.state.events)
    if events:
        # Count user messages (fake responses we've sent)
        user_msgs = [
            event
            for event in events
            if isinstance(event, MessageEvent) and event.source == "user"
        ]
        if len(user_msgs) >= 2:
            # Let the agent know it can give up after multiple attempts
            return (
                msg
                + 'If you want to give up, use the "finish" tool to finish the interaction.\n'
            )
    return msg


def _message_to_text(message: Message) -> str:
    """Flatten the text parts of an LLM message into a single string."""
    return "\n".join(
        part.text for part in message.content if isinstance(part, TextContent)
    ).strip()


def _get_last_agent_message(events: list[Event]) -> str | None:
    """Return the text of the most recent agent message, if any."""
    for event in reversed(events):
        if isinstance(event, MessageEvent) and event.source == "agent":
            return _message_to_text(event.to_llm_message()) or None
    return None


def _get_conversation_llm(conversation: "BaseConversation") -> "LLM | None":
    """Best-effort lookup of the LLM backing the conversation's agent.

    Both LocalConversation and RemoteConversation expose the agent they were
    constructed with as ``.agent``; RemoteState.agent is used as a fallback.
    """
    agent = getattr(conversation, "agent", None)
    if agent is None:
        try:
            agent = conversation.state.agent
        except Exception:
            logger.debug(
                "Could not access the conversation agent for finish classification",
                exc_info=True,
            )
            return None
    llm = getattr(agent, "llm", None)
    return llm


def _parse_yes_no(text: str) -> bool | None:
    """Parse a YES/NO classifier answer, returning None when ambiguous."""
    normalized = text.strip().lower()
    if not normalized:
        return None

    bare = normalized.strip(" \t\r\n.!\"'`*_")
    if bare == "yes":
        return True
    if bare == "no":
        return False

    matches = re.findall(r"\b(yes|no)\b", normalized)
    if len(matches) == 1:
        return matches[0] == "yes"
    return None


def _llm_judges_finished(llm: "LLM", assistant_message: str) -> bool | None:
    """Ask the conversation's LLM whether the agent message implies completion.

    Returns True/False for a clear verdict, or None when the LLM call fails or
    the answer cannot be parsed. Callers treat None as "not finished" so a
    classifier failure degrades to the regular fake user response.
    """
    truncated_message = assistant_message[:MAX_CLASSIFIER_MESSAGE_CHARS]
    messages = [
        Message(
            role="system",
            content=[TextContent(text=_FINISH_CLASSIFIER_SYSTEM_PROMPT)],
        ),
        Message(
            role="user",
            content=[
                TextContent(
                    text=(
                        "The agent's latest message is delimited below.\n"
                        "<agent_message>\n"
                        f"{truncated_message}\n"
                        "</agent_message>\n\n"
                        "Does this message indicate that the job is finished? "
                        "Reply with exactly YES or NO."
                    )
                )
            ],
        ),
    ]

    try:
        response = llm.completion(messages)
    except Exception:
        logger.warning(
            "Finish classification LLM call failed; falling back to the regular "
            "fake user response",
            exc_info=True,
        )
        return None

    verdict = _parse_yes_no(_message_to_text(response.message))
    if verdict is None:
        logger.debug(
            "Finish classifier returned an unparseable answer: %r",
            _message_to_text(response.message),
        )
    return verdict


def _finish_tool_directive(encapsulate_solution: bool) -> str:
    """Build the fake user message that asks the agent to call finish directly."""
    return (
        "You indicated that the task is complete, but you have not called the "
        "finish tool. Please directly invoke the finish tool now and include your "
        "final answer in the message parameter of the finish tool. Do not ask the "
        "user any follow-up questions.\n"
        f"{_encapsulation_instruction(encapsulate_solution)}"
    )


def smart_fake_user_response(
    conversation: "BaseConversation",
    encapsulate_solution: bool = False,
) -> str:
    """Generate a fake user response that can tell when the agent is done.

    Uses the conversation's own LLM as a lightweight classifier: it inspects the
    last assistant message and asks whether that message indicates the job has
    finished. When it does, the fake user message asks the agent to directly
    invoke the finish tool (with the final answer in its ``message`` parameter).
    Otherwise the regular fake user response is returned.

    If the classification call fails or returns an unparseable answer, this
    degrades gracefully to :func:`fake_user_response`.

    Args:
        conversation: The conversation instance containing the event history.
        encapsulate_solution: If True, instructs the agent to encapsulate the
            final answer within <solution> tags.

    Returns:
        A string message to send as a fake user response.
    """
    events = list(conversation.state.events)
    assistant_message = _get_last_agent_message(events)

    if assistant_message is not None:
        llm = _get_conversation_llm(conversation)
        if llm is not None and _llm_judges_finished(llm, assistant_message):
            logger.info(
                "Last agent message indicates the job is finished; asking the "
                "agent to call the finish tool directly"
            )
            return _finish_tool_directive(encapsulate_solution)

    return fake_user_response(conversation, encapsulate_solution=encapsulate_solution)


def _agent_finished_with_finish_action(events: list[Event]) -> bool:
    """Check if the agent finished by calling the finish tool.

    Args:
        events: List of conversation events.

    Returns:
        True if the last action was a FinishAction, False otherwise.
    """
    for event in reversed(events):
        if isinstance(event, ActionEvent):
            if event.action is not None and isinstance(event.action, FinishAction):
                return True
            # Found an action that's not FinishAction
            return False
    return False


def _agent_sent_message(events: list[Event]) -> bool:
    """Check if the agent's last event was a message (not a tool call).

    Args:
        events: List of conversation events.

    Returns:
        True if the last agent event was a MessageEvent, False otherwise.
    """
    for event in reversed(events):
        if isinstance(event, MessageEvent) and event.source == "agent":
            return True
        if isinstance(event, ActionEvent):
            # Agent used a tool, not a message
            return False
    return False


def _reconcile_event_cache(event_list: object) -> None:
    """Rebuild _cached_event_ids from _cached_events under the lock.

    Works around a thread-safety bug in RemoteEventsList._do_full_sync():
    it replaces _cached_events without the lock while _cached_event_ids is
    only ever added to. This can leave event IDs marked as "seen" even though
    the corresponding event was lost from the list, permanently blocking
    re-delivery via WebSocket.
    """
    lock = getattr(event_list, "_lock", None)
    cached_events = getattr(event_list, "_cached_events", None)
    cached_event_ids = getattr(event_list, "_cached_event_ids", None)
    if lock is None or cached_events is None or cached_event_ids is None:
        return
    with lock:
        cached_event_ids.clear()
        cached_event_ids.update(e.id for e in cached_events)


def _ensure_user_message_synced(
    conversation: "BaseConversation",
    max_retries: int = 30,
    interval: float = 1.0,
) -> bool:
    """Poll until the local event cache contains at least one user MessageEvent.

    For RemoteConversation, send_message() only POSTs to the server and does
    not update the local cache. This helper forces REST syncs until the user
    message is visible locally, working around both the async WebSocket
    delivery lag and the _do_full_sync thread-safety bug.

    Returns True if the message was found, False if retries were exhausted.
    Never raises.
    """
    event_list = getattr(conversation.state, "events", None)
    sync_fn = getattr(event_list, "_do_full_sync", None)
    if sync_fn is None:
        return True

    for attempt in range(max_retries + 1):
        user_msg_count = sum(
            1
            for e in list(conversation.state.events)
            if isinstance(e, MessageEvent) and e.source == "user"
        )
        if user_msg_count >= 1:
            if attempt > 0:
                logger.info("User message synced after %d retries", attempt)
            return True

        if attempt < max_retries:
            logger.debug(
                "User message not yet in local cache, syncing (attempt %d/%d)",
                attempt + 1,
                max_retries,
            )
            sync_fn()
            _reconcile_event_cache(event_list)
            time.sleep(interval)

    logger.warning(
        "User message sync timed out after %d retries; "
        "the server-side state is correct but local history may be incomplete",
        max_retries,
    )
    return False


def run_conversation_with_fake_user_response(
    conversation: "BaseConversation",
    fake_user_response_fn: FakeUserResponseFn = fake_user_response,
    max_fake_responses: int = 10,
    timeout: int | None = None,
) -> None:
    """Run a conversation with automatic fake user responses.

    This function runs the conversation and automatically sends fake user responses
    when the agent tries to communicate with the user (sends a message instead of
    using tools). This mimics the behavior of the v0 OpenHands evaluation framework.

    The conversation continues until:
    - The agent calls the finish tool
    - The maximum number of fake responses is reached
    - The conversation enters an error or stuck state

    Args:
        conversation: The conversation instance to run.
        fake_user_response_fn: A function that generates fake user responses.
            Defaults to fake_user_response.
        max_fake_responses: Maximum number of fake responses to send before
            stopping. This prevents infinite loops.
        timeout: Timeout in seconds for each conversation.run() call.
            If None, no timeout is applied.
    """

    _ensure_user_message_synced(conversation)

    fake_response_count = 0

    while True:
        # Run the conversation
        # Only pass timeout if it's not None and the conversation supports it
        # (RemoteConversation supports timeout, LocalConversation does not)
        run = conversation.run
        if timeout is not None and hasattr(run, "__code__"):
            # Check if the run method accepts a timeout parameter
            import inspect

            sig = inspect.signature(run)
            if "timeout" in sig.parameters:
                # Unpack the kwarg so type checkers don't flag it against the
                # BaseConversation.run signature, which has no timeout parameter.
                run(**{"timeout": timeout})
            else:
                run()
        else:
            run()

        # Check the execution status
        status = conversation.state.execution_status

        # If not finished, we're done (error, stuck, paused, etc.)
        if status != ConversationExecutionStatus.FINISHED:
            logger.info(
                "Conversation ended with status: %s after %d fake responses",
                status.value,
                fake_response_count,
            )
            break

        # Check if agent finished with FinishAction (proper completion)
        events = list(conversation.state.events)
        if _agent_finished_with_finish_action(events):
            logger.info(
                "Agent finished with FinishAction after %d fake responses",
                fake_response_count,
            )
            break

        # Check if agent sent a message (needs fake response)
        if not _agent_sent_message(events):
            # Agent didn't send a message, but conversation is finished
            # This shouldn't happen normally, but handle it gracefully
            logger.warning(
                "Conversation finished without FinishAction or agent message"
            )
            break

        # Check if we've reached the maximum number of fake responses
        if fake_response_count >= max_fake_responses:
            logger.warning(
                "Reached maximum fake responses (%d), stopping conversation",
                max_fake_responses,
            )
            break

        # Generate and send fake user response
        fake_response = fake_user_response_fn(conversation)

        # Check for exit signal
        if fake_response == "/exit":
            logger.info("Fake user response function returned /exit, stopping")
            break

        logger.info(
            "Sending fake user response #%d: %s...",
            fake_response_count + 1,
            fake_response[:50],
        )
        conversation.send_message(fake_response)
        fake_response_count += 1

    logger.info(
        "Conversation completed. Total fake responses sent: %d", fake_response_count
    )


def run_conversation_with_smart_fake_user_response(
    conversation: "BaseConversation",
    max_fake_responses: int = 10,
    timeout: int | None = None,
    encapsulate_solution: bool = False,
) -> None:
    """Run a conversation with LLM-classified fake user responses.

    This is a drop-in alternative to
    :func:`run_conversation_with_fake_user_response` that uses
    :func:`smart_fake_user_response` as the response generator. When the last
    assistant message indicates the job is finished, the fake user message asks
    the agent to directly invoke the finish tool instead of simply asking it to
    keep working.

    Args:
        conversation: The conversation instance to run.
        max_fake_responses: Maximum number of fake responses to send before
            stopping. This prevents infinite loops.
        timeout: Timeout in seconds for each conversation.run() call.
            If None, no timeout is applied.
        encapsulate_solution: If True, instructs the agent to encapsulate the
            final answer within <solution> tags.
    """
    run_conversation_with_fake_user_response(
        conversation,
        fake_user_response_fn=partial(
            smart_fake_user_response, encapsulate_solution=encapsulate_solution
        ),
        max_fake_responses=max_fake_responses,
        timeout=timeout,
    )
