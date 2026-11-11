"""Tests for resyncing a RemoteConversation from the server's REST event store."""

import threading

from benchmarks.utils.fake_user_response import (
    run_conversation_with_fake_user_response,
    sync_remote_conversation_state,
)
from openhands.sdk.conversation.state import ConversationExecutionStatus
from openhands.sdk.event import MessageEvent
from openhands.sdk.event.conversation_state import ConversationStateUpdateEvent
from openhands.sdk.llm import Message, TextContent


def _state_event(key: str) -> ConversationStateUpdateEvent:
    return ConversationStateUpdateEvent(key=key, value={"value": key})


def _user_message(text: str = "do the task") -> MessageEvent:
    return MessageEvent(
        source="user",
        llm_message=Message(role="user", content=[TextContent(text=text)]),
    )


class _FakeRemoteEventsList:
    """Mimics RemoteEventsList: _do_full_sync replaces the cache from REST."""

    def __init__(self, cached, server, fail_times: int = 0):
        self._cached_events = list(cached)
        self._cached_event_ids = {event.id for event in cached}
        self._lock = threading.RLock()
        self._server_events = list(server)
        self._fail_times = fail_times
        self.sync_calls = 0

    def _do_full_sync(self) -> None:
        self.sync_calls += 1
        if self._fail_times > 0:
            self._fail_times -= 1
            raise RuntimeError("agent server unreachable")
        self._cached_events = list(self._server_events)
        self._cached_event_ids.update(event.id for event in self._server_events)

    def __iter__(self):
        return iter(self._cached_events)

    def __len__(self) -> int:
        return len(self._cached_events)


class _FakeRemoteState:
    def __init__(
        self,
        events: _FakeRemoteEventsList,
        cached_state: dict | None = None,
        execution_status: ConversationExecutionStatus = (
            ConversationExecutionStatus.RUNNING
        ),
    ):
        self.events = events
        self._cached_state = cached_state
        self._lock = threading.RLock()
        self.execution_status = execution_status


class _SimpleConversation:
    def __init__(self, state: _FakeRemoteState):
        self.state = state

    def run(self, **kwargs) -> None:
        pass


def _remote_conversation(
    cached, server, cached_state: dict | None = None
) -> _SimpleConversation:
    events = _FakeRemoteEventsList(cached, server)
    return _SimpleConversation(_FakeRemoteState(events, cached_state))


def test_resync_replaces_truncated_cache_with_server_events():
    first, second, third = _state_event("a"), _state_event("b"), _state_event("c")
    conversation = _remote_conversation(
        cached=[first, second],
        server=[first, second, third],
        cached_state={"execution_status": "running"},
    )

    assert sync_remote_conversation_state(conversation, interval=0) is True

    assert list(conversation.state.events) == [first, second, third]
    assert conversation.state.events._cached_event_ids == {
        first.id,
        second.id,
        third.id,
    }
    # RemoteState cache must be dropped so stats/status are refetched from REST.
    assert conversation.state._cached_state is None


def test_resync_is_noop_for_local_conversations():
    events = [_state_event("a")]
    conversation = _SimpleConversation(_FakeRemoteState(events))
    # A plain list has no _do_full_sync: this is a LocalConversation cache.
    conversation.state.events = events

    assert sync_remote_conversation_state(conversation) is False
    assert conversation.state.events is events


def test_resync_keeps_cache_when_server_is_unreachable():
    first = _state_event("a")
    events = _FakeRemoteEventsList(
        cached=[first], server=[first, _state_event("b")], fail_times=10
    )
    conversation = _SimpleConversation(_FakeRemoteState(events, {"status": "running"}))

    assert sync_remote_conversation_state(conversation, max_attempts=2) is False
    assert list(events) == [first]
    # Keep the previous cached state if we never managed to resync.
    assert conversation.state._cached_state == {"status": "running"}


def test_run_conversation_resyncs_after_each_run():
    cached = [_user_message()]
    server = cached + [_state_event("a"), _state_event("b")]
    events = _FakeRemoteEventsList(cached, server)
    state = _FakeRemoteState(events, execution_status=ConversationExecutionStatus.ERROR)
    conversation = _SimpleConversation(state)

    run_conversation_with_fake_user_response(conversation)

    assert events.sync_calls >= 1
    assert list(events) == server
    assert state._cached_state is None


class _SyncableConversation(_SimpleConversation):
    """RemoteConversation stand-in exposing the public sync() API."""

    def __init__(self, state: _FakeRemoteState):
        super().__init__(state)
        self.sync_calls = 0

    def sync(self) -> int:
        self.sync_calls += 1
        self.state.events._do_full_sync()
        self.state._cached_state = None
        return len(self.state.events)


def test_resync_prefers_public_conversation_sync_api():
    first, second = _state_event("a"), _state_event("b")
    events = _FakeRemoteEventsList([first], [first, second])
    state = _FakeRemoteState(events, {"execution_status": "running"})
    conversation = _SyncableConversation(state)

    assert sync_remote_conversation_state(conversation, interval=0) is True

    assert conversation.sync_calls >= 1
    assert list(events) == [first, second]
    assert state._cached_state is None
