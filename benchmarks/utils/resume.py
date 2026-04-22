"""Resume manager: orchestrates full resume of an interrupted conversation.

Detects persisted events, restores workspace filesystem state via action
replay, and injects events into the container's persistence directory so
the SDK's native ``ConversationState.create()`` resume path kicks in.
"""

from __future__ import annotations

import base64
import json
import uuid
from typing import TYPE_CHECKING

from openhands.sdk import get_logger
from openhands.sdk.event import Event

from benchmarks.utils.event_persistence import (
    count_persisted_events,
    load_persisted_events,
    load_resume_metadata,
)
from benchmarks.utils.replayer import WorkspaceReplayer

if TYPE_CHECKING:
    from openhands.sdk.agent import AgentBase
    from openhands.sdk.workspace import RemoteWorkspace


logger = get_logger(__name__)

CONTAINER_CONVERSATIONS_DIR = "/workspace/conversations"


class ResumeManager:
    """Orchestrates conversation resume from persisted events.

    Usage in ``evaluate_instance`` (after testbed copy + git reset)::

        resume_mgr = instance.data.get("resume_manager")
        conversation_id = None
        if resume_mgr:
            resume_mgr.restore_workspace_state(workspace)
            resume_mgr.inject_state_into_workspace(workspace, agent)
            conversation_id = resume_mgr.conversation_id

        conversation = Conversation(
            agent=agent,
            workspace=workspace,
            conversation_id=conversation_id,
            ...
        )
        if not resume_mgr:
            conversation.send_message(instruction)
    """

    def __init__(self, persist_dir: str):
        self.persist_dir = persist_dir
        self._events: list[Event] | None = None
        self._meta: dict | None = None

    def has_resumable_state(self) -> bool:
        """True if the persist dir contains events we can resume from."""
        meta = self._get_meta()
        if meta is None:
            return False
        if not meta.get("conversation_id"):
            return False
        return count_persisted_events(self.persist_dir) > 0

    @property
    def events(self) -> list[Event]:
        if self._events is None:
            self._events = load_persisted_events(self.persist_dir)
        return self._events

    @property
    def conversation_id(self) -> uuid.UUID:
        meta = self._get_meta()
        if meta is None or not meta.get("conversation_id"):
            raise ValueError("No conversation_id in resume metadata")
        return uuid.UUID(meta["conversation_id"])

    def _get_meta(self) -> dict | None:
        if self._meta is None:
            self._meta = load_resume_metadata(self.persist_dir)
        return self._meta

    def restore_workspace_state(self, workspace: RemoteWorkspace) -> None:
        """Replay side-effecting actions into the fresh workspace.

        Must be called AFTER testbed copy + git reset so the base filesystem
        is in place before replaying modifications on top.
        """
        events = self.events
        if not events:
            logger.info("No events to replay for workspace restoration")
            return

        replayer = WorkspaceReplayer(workspace)
        report = replayer.replay(events)
        logger.info(
            "Workspace restoration: %d terminal, %d file edits, %d errors",
            report.terminal_replayed,
            report.file_edits_applied,
            len(report.errors),
        )

    def inject_state_into_workspace(
        self, workspace: RemoteWorkspace, agent: AgentBase
    ) -> None:
        """Write persisted events + base_state.json into the container.

        Writes to ``/workspace/conversations/{cid_hex}/`` so the agent
        server's ``EventService.start()`` → ``ConversationState.create()``
        will find existing state and resume the conversation.
        """
        cid = self.conversation_id
        cid_hex = cid.hex
        conv_dir = f"{CONTAINER_CONVERSATIONS_DIR}/{cid_hex}"
        events_dir = f"{conv_dir}/events"

        logger.info(
            "Injecting %d events into container at %s", len(self.events), conv_dir
        )

        workspace.execute_command(f"mkdir -p '{events_dir}'")

        base_state = self._build_base_state(cid, agent)
        self._write_to_container(workspace, f"{conv_dir}/base_state.json", base_state)

        for i, event in enumerate(self.events):
            event_id = getattr(event, "id", str(uuid.uuid4()))
            filename = f"event-{i:05d}-{event_id}.json"
            event_json = event.model_dump_json(exclude_none=True)
            self._write_to_container(workspace, f"{events_dir}/{filename}", event_json)

        logger.info("State injection complete (%d events)", len(self.events))

    def _build_base_state(self, cid: uuid.UUID, agent: AgentBase) -> str:
        """Construct a minimal base_state.json that ConversationState.create() accepts."""
        state_dict = {
            "id": str(cid),
            "agent": json.loads(agent.model_dump_json(exclude_none=True)),
            "workspace": {
                "kind": "LocalWorkspace",
                "working_dir": "/workspace",
            },
            "execution_status": "IDLE",
            "max_iterations": 500,
            "persistence_dir": CONTAINER_CONVERSATIONS_DIR,
        }
        return json.dumps(state_dict)

    @staticmethod
    def _write_to_container(
        workspace: RemoteWorkspace, path: str, content: str
    ) -> None:
        """Write a string to a file inside the container via base64 encoding."""
        encoded = base64.b64encode(content.encode("utf-8")).decode("ascii")
        cmd = (
            f"python3 -c \""
            f"import base64,sys; "
            f"sys.stdout.buffer.write(base64.b64decode('{encoded}'))"
            f"\" > '{path}'"
        )
        res = workspace.execute_command(cmd)
        if res.exit_code != 0:
            logger.warning("Failed to write %s in container: %s", path, res.stderr)
