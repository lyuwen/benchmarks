"""Single-session capture and on-disk export for the rollout-capture service.

The manager owns exactly one agent session: it is launched just before the
harness starts, records every turn that session makes, and is stopped once the
harness finishes. There is no session registry -- the process *is* the session.

:class:`RolloutCapture` holds that session's state (in-flight turns, metadata,
the trajectory tree) and exports it as one JSON document containing *all* of its
trajectories (every branch, including the ones a context compaction produces),
so a consumer can inspect the conversation before and after a compaction
together.

Request purpose comes only from the harness's explicit gateway hint headers
(Claude Code ``x-claude-code-request-class`` / ``-compaction`` /
``-context-compacted``); request shape is never used to infer compaction. A
signalled compaction without tools is exported as its own ``kind="compaction"``
trajectory whose messages are context (``loss_mask=0``) except the summary
response. Auxiliary requests (titles/classifiers/security monitors) are labelled
per turn and can be excluded from training.

The document is rewritten atomically after every captured turn (a crash loses at
most the in-flight turn) and finalized on an explicit ``finish`` request or on
service shutdown.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from .trajectory import TrajectoryManager, TurnRecord

logger = logging.getLogger(__name__)

__all__ = ["RolloutCapture", "SessionClosed"]


class SessionClosed(Exception):
    """Raised when a request arrives after the session has been finalized."""


class RolloutCapture:
    """One session's trajectory capture plus atomic JSON export."""

    def __init__(
        self,
        output_path: str | os.PathLike[str],
        *,
        session_id: str,
        metadata: dict[str, Any] | None = None,
        snapshot_every_turn: bool = True,
        exclude_request_classes: tuple[str, ...] = (),
    ) -> None:
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id
        self.metadata = dict(metadata or {})
        self.snapshot_every_turn = snapshot_every_turn
        self.exclude_request_classes = frozenset(str(value).lower() for value in exclude_request_classes)
        self.manager = TrajectoryManager(session_id=session_id)
        self.created_at = time.time()
        self.request_count = 0
        self.compaction_count = 0
        self.excluded_count = 0
        self.inflight: set[asyncio.Task] = set()
        self.finished = False
        self.finish_reason: str | None = None
        self.model: str | None = None
        self.last_protocol: str | None = None
        self.lock = asyncio.Lock()
        self.last_summary: dict[str, Any] | None = None

    @property
    def turn_count(self) -> int:
        return self.manager.turn_count

    # -------------------- capture -----------------------------------------

    def acquire(self) -> RolloutCapture:
        """Return the live session, rejecting requests once it is finalized."""
        if self.finished:
            raise SessionClosed(self.session_id)
        return self

    def record(
        self,
        *,
        protocol: str,
        prompt_messages: list[dict[str, Any]],
        turn: TurnRecord,
    ) -> None:
        """Record one exchange and refresh the on-disk snapshot."""
        request_class = turn.client.get("request_class")
        request_class = str(request_class).lower() if request_class else None
        compaction = bool(turn.client.get("compaction_trigger")) or request_class == "compaction"
        # A compaction request is split only when it is a side call (no tools);
        # one that carries tools stays a normal continuation but is labelled.
        standalone = compaction and turn.tools is None
        excluded = request_class in self.exclude_request_classes

        # Retain the effective request verbatim (including its prompt messages)
        # so the upstream prompt can always be reconstructed, even where the
        # training-side normalization changes a role or content shape.
        effective_request = dict(turn.request)
        effective_request["message_count"] = len(turn.request.get("messages") or [])

        updates: dict[str, Any] = {
            "request": effective_request,
            "compaction": compaction,
            "standalone": standalone,
            "exclude_from_training": excluded,
        }
        if "timestamp" not in turn.metadata:
            updates["metadata"] = {**turn.metadata, "timestamp": time.time()}
        turn = dataclasses.replace(turn, **updates)
        if compaction:
            self.compaction_count += 1
        if excluded:
            self.excluded_count += 1

        self.manager.record_turn(prompt_messages=prompt_messages, turn=turn)
        self.request_count += 1
        self.model = turn.model or self.model
        self.last_protocol = protocol
        if self.snapshot_every_turn:
            self._write(finished=False, finish_reason=None)

    async def finish(
        self,
        *,
        reason: str,
        metadata: dict[str, Any] | None = None,
        wait_timeout: float = 30.0,
    ) -> dict[str, Any]:
        """Drain in-flight turns, export the final document, and close the session.

        Idempotent: calling it again returns the first summary without
        rewriting the file.
        """
        async with self.lock:
            if self.finished:
                return self.last_summary  # type: ignore[return-value]
            self.finished = True
            await self._drain(wait_timeout)
            if metadata:
                self.metadata.update(metadata)
            self.finish_reason = reason
            self.last_summary = self._write(finished=True, finish_reason=reason)
            return self.last_summary

    def status(self) -> dict[str, Any]:
        """Current session state, without touching the export file."""
        return self._summary(finished=self.finished, finish_reason=self.finish_reason)

    # -------------------- export ------------------------------------------

    def _document(self, *, finished: bool, finish_reason: str | None) -> dict[str, Any]:
        trajectories = self.manager.get_trajectories()
        request_log = self.manager.get_request_log(trajectories)
        request_classes: dict[str, int] = {}
        for entry in request_log:
            request_class = entry.get("request_class")
            if request_class:
                request_classes[request_class] = request_classes.get(request_class, 0) + 1
        return {
            "type": "rollout_session",
            "version": 2,
            "session_id": self.session_id,
            "created_at": self.created_at,
            "updated_at": time.time(),
            "finished": finished,
            "finish_reason": finish_reason,
            "model": self.model,
            "metadata": self.metadata,
            "stats": {
                "requests": self.request_count,
                "turns": self.manager.turn_count,
                "trajectories": len(trajectories),
                "compaction_requests": self.compaction_count,
                "excluded_turns": self.excluded_count,
                "request_classes": request_classes,
            },
            "request_log": request_log,
            "trajectories": trajectories,
        }

    def _summary(self, *, finished: bool, finish_reason: str | None) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "output": str(self.output_path),
            "finished": finished,
            "finish_reason": finish_reason,
            "requests": self.request_count,
            "turns": self.manager.turn_count,
            "trajectories": len(self.manager.get_trajectories()),
            "compaction_requests": self.compaction_count,
            "excluded_turns": self.excluded_count,
            "created_at": self.created_at,
            "updated_at": time.time(),
            "model": self.model,
            "protocol": self.last_protocol,
            "inflight": len(self.inflight),
        }

    def _write(self, *, finished: bool, finish_reason: str | None) -> dict[str, Any]:
        document = self._document(finished=finished, finish_reason=finish_reason)
        _write_json_atomic(self.output_path, document)
        return {
            "session_id": self.session_id,
            "output": str(self.output_path),
            "finished": finished,
            "finish_reason": finish_reason,
            "requests": self.request_count,
            "turns": self.manager.turn_count,
            "trajectories": len(document["trajectories"]),
            "compaction_requests": self.compaction_count,
            "excluded_turns": self.excluded_count,
            "created_at": self.created_at,
            "updated_at": document["updated_at"],
            "model": self.model,
            "protocol": self.last_protocol,
            "inflight": len(self.inflight),
        }

    async def _drain(self, wait_timeout: float) -> None:
        current = asyncio.current_task()
        tasks = [task for task in self.inflight if task is not current and not task.done()]
        if not tasks:
            return
        _, pending = await asyncio.wait(tasks, timeout=wait_timeout)
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)


def _write_json_atomic(path: Path, document: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(document, handle, ensure_ascii=False)
        handle.write("\n")
    os.replace(temporary, path)
