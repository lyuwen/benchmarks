"""Incremental event persistence to disk for conversation resume.

Writes each event as it arrives (via the Conversation callback) to a per-instance
directory using the SDK's native filename format. This gives us a durable event
log that survives orchestrator or container crashes, which can later be injected
into a fresh container for conversation resumption via the SDK's native
`ConversationState.create(persistence_dir=...)` resume path.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from typing import Callable

from openhands.sdk import Event, get_logger
from openhands.sdk.conversation.persistence_const import (
    EVENT_FILE_PATTERN,
    EVENT_NAME_RE,
    EVENTS_DIR,
)


logger = get_logger(__name__)

ConversationCallback = Callable[[Event], None]

RESUME_META_FILENAME = "resume_meta.json"


def _atomic_write(path: str, content: str) -> None:
    """Write content to path atomically via tmp + rename."""
    dir_ = os.path.dirname(path)
    os.makedirs(dir_, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dir_, prefix=".tmp-", suffix=".part")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


class EventFilePersistence:
    """Writes conversation events to disk as they arrive.

    Files are written to ``{persist_dir}/events/`` using the SDK's
    ``EVENT_FILE_PATTERN`` so they can be loaded directly by the SDK's
    ``EventLog`` when injected into a container.

    A sidecar ``resume_meta.json`` tracks the conversation_id, event count,
    and tool names seen, which lets the resume orchestrator quickly decide
    whether a persisted trajectory is usable without parsing every event.
    """

    def __init__(self, persist_dir: str, instance_id: str):
        self.persist_dir = persist_dir
        self.instance_id = instance_id
        self.events_dir = os.path.join(persist_dir, EVENTS_DIR)
        self.meta_path = os.path.join(persist_dir, RESUME_META_FILENAME)

        os.makedirs(self.events_dir, exist_ok=True)

        self._counter = self._next_index()
        self._conversation_id: str | None = None
        self._tools_used: set[str] = set()
        self._load_existing_meta()

    def _next_index(self) -> int:
        """Find the highest existing event index (0-based) and return next index."""
        if not os.path.isdir(self.events_dir):
            return 0
        highest = -1
        for name in os.listdir(self.events_dir):
            m = EVENT_NAME_RE.match(name)
            if m:
                idx = int(m.group("idx"))
                if idx > highest:
                    highest = idx
        return highest + 1

    def _load_existing_meta(self) -> None:
        if not os.path.isfile(self.meta_path):
            return
        try:
            with open(self.meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            self._conversation_id = meta.get("conversation_id")
            self._tools_used = set(meta.get("tools_used", []))
        except Exception as exc:
            logger.warning(
                "Failed to read existing resume meta at %s: %s", self.meta_path, exc
            )

    def set_conversation_id(self, cid: object) -> None:
        """Record the conversation_id so resume can reuse it later."""
        try:
            cid_str = str(cid)
            if cid_str and cid_str != self._conversation_id:
                self._conversation_id = cid_str
                self._write_meta()
        except Exception as exc:
            logger.debug(
                "Failed to record conversation_id for %s: %s", self.instance_id, exc
            )

    def _write_meta(self) -> None:
        meta = {
            "instance_id": self.instance_id,
            "conversation_id": self._conversation_id,
            "event_count": self._counter,
            "tools_used": sorted(self._tools_used),
            "last_timestamp": datetime.now(timezone.utc).isoformat(),
        }
        try:
            _atomic_write(self.meta_path, json.dumps(meta, indent=2))
        except Exception as exc:
            logger.debug("Failed to write resume meta for %s: %s", self.instance_id, exc)

    def callback(self, event: Event) -> None:
        """ConversationCallbackType — writes every event to disk.

        Best-effort: any failure is logged but never raised, so persistence
        never aborts an agent run.
        """
        try:
            event_id = getattr(event, "id", None)
            if not event_id:
                return
            idx = self._counter
            filename = EVENT_FILE_PATTERN.format(idx=idx, event_id=event_id)
            path = os.path.join(self.events_dir, filename)

            serialized = event.model_dump_json(exclude_none=True)
            _atomic_write(path, serialized)

            self._counter += 1

            tool_name = getattr(event, "tool_name", None)
            if tool_name:
                self._tools_used.add(tool_name)

            # Rewrite meta every event (cheap, keeps counter durable across crashes)
            self._write_meta()
        except Exception as exc:
            logger.debug(
                "Failed to persist event for instance %s: %s", self.instance_id, exc
            )


def load_resume_metadata(persist_dir: str) -> dict | None:
    """Load the resume metadata sidecar, or None if absent/unreadable."""
    meta_path = os.path.join(persist_dir, RESUME_META_FILENAME)
    if not os.path.isfile(meta_path):
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.warning("Failed to load resume meta from %s: %s", meta_path, exc)
        return None


def load_persisted_events(persist_dir: str) -> list[Event]:
    """Load all persisted events in order, using polymorphic deserialization.

    Returns an empty list if ``persist_dir/events/`` is missing or empty.
    Silently skips files that fail to deserialize (partial writes).
    """
    events_dir = os.path.join(persist_dir, EVENTS_DIR)
    if not os.path.isdir(events_dir):
        return []

    entries: list[tuple[int, str]] = []
    for name in os.listdir(events_dir):
        m = EVENT_NAME_RE.match(name)
        if not m:
            continue
        entries.append((int(m.group("idx")), name))
    entries.sort(key=lambda t: t[0])

    events: list[Event] = []
    for _, name in entries:
        path = os.path.join(events_dir, name)
        try:
            with open(path, "r", encoding="utf-8") as f:
                txt = f.read()
            events.append(Event.model_validate_json(txt))
        except Exception as exc:
            logger.warning("Skipping unreadable event file %s: %s", path, exc)
    return events


def count_persisted_events(persist_dir: str) -> int:
    """Cheap count without deserializing events."""
    events_dir = os.path.join(persist_dir, EVENTS_DIR)
    if not os.path.isdir(events_dir):
        return 0
    return sum(1 for name in os.listdir(events_dir) if EVENT_NAME_RE.match(name))
