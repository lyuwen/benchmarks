"""Workspace state restoration by replaying side-effecting actions.

Given a list of persisted events from a previous (interrupted) conversation,
replays the side-effecting actions into a fresh workspace so the filesystem
and environment match the state at the point of interruption.
"""

from __future__ import annotations

import base64
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from openhands.sdk import get_logger
from openhands.sdk.event import ActionEvent, ObservationEvent

if TYPE_CHECKING:
    from openhands.sdk.event import Event
    from openhands.sdk.workspace import RemoteWorkspace


logger = get_logger(__name__)

READONLY_COMMAND_PREFIXES = (
    "cat ",
    "ls ",
    "find ",
    "grep ",
    "egrep ",
    "fgrep ",
    "head ",
    "tail ",
    "wc ",
    "file ",
    "stat ",
    "which ",
    "type ",
    "readlink ",
    "realpath ",
    "diff ",
    "md5sum ",
    "sha256sum ",
)

READONLY_COMMAND_EXACT = frozenset({
    "ls",
    "pwd",
    "env",
    "printenv",
    "whoami",
    "id",
    "hostname",
    "uname",
    "uname -a",
    "date",
})

READONLY_GIT_PREFIXES = (
    "git log",
    "git show",
    "git diff",
    "git status",
    "git branch",
    "git remote",
    "git tag",
    "git rev-parse",
    "git describe",
    "git blame",
)

_REDIRECT_RE = re.compile(r"[12]?>")


@dataclass
class ReplayReport:
    terminal_replayed: int = 0
    terminal_skipped: int = 0
    file_edits_applied: int = 0
    file_edits_skipped: int = 0
    errors: list[str] = field(default_factory=list)


def _is_readonly_command(cmd: str) -> bool:
    """Heuristic: return True if the command is unlikely to mutate state."""
    stripped = cmd.strip()
    if not stripped:
        return True

    first_line = stripped.split("\n", 1)[0].strip()
    # Take only the first segment before && or ; to check the leading command
    first_segment = first_line.split("&&", 1)[0].split(";", 1)[0].strip()

    if first_segment in READONLY_COMMAND_EXACT:
        return True

    for prefix in READONLY_COMMAND_PREFIXES:
        if first_segment.startswith(prefix):
            if not _REDIRECT_RE.search(first_segment):
                return True

    for prefix in READONLY_GIT_PREFIXES:
        if first_segment.startswith(prefix):
            if not _REDIRECT_RE.search(first_segment):
                return True

    if first_segment.startswith("echo ") and not _REDIRECT_RE.search(first_segment):
        return True

    if (
        first_segment.startswith("python")
        and "-c" in first_segment
        and "print(" in first_segment
        and not _REDIRECT_RE.search(first_segment)
    ):
        return True

    return False


def _pair_actions_with_observations(
    events: list[Event],
) -> list[tuple[ActionEvent, ObservationEvent | None]]:
    """Walk events and pair each ActionEvent with its corresponding ObservationEvent."""
    obs_by_action_id: dict[str, ObservationEvent] = {}
    actions_in_order: list[ActionEvent] = []

    for event in events:
        if isinstance(event, ActionEvent):
            actions_in_order.append(event)
        elif isinstance(event, ObservationEvent):
            action_id = getattr(event, "action_id", None)
            if action_id:
                obs_by_action_id[str(action_id)] = event

    pairs: list[tuple[ActionEvent, ObservationEvent | None]] = []
    for action_event in actions_in_order:
        event_id = str(getattr(action_event, "id", ""))
        obs = obs_by_action_id.get(event_id)
        pairs.append((action_event, obs))
    return pairs


def _write_file_to_workspace(
    workspace: RemoteWorkspace, path: str, content: str
) -> bool:
    """Write content to a file inside the workspace using base64 to avoid escaping."""
    encoded = base64.b64encode(content.encode("utf-8")).decode("ascii")
    parent_dir = "/".join(path.split("/")[:-1])
    cmd = (
        f"mkdir -p '{parent_dir}' && "
        f"python3 -c \"import base64,sys; "
        f"sys.stdout.buffer.write(base64.b64decode('{encoded}'))\" > '{path}'"
    )
    res = workspace.execute_command(cmd)
    return res.exit_code == 0


class WorkspaceReplayer:
    """Replays side-effecting actions from a persisted event log into a workspace."""

    def __init__(self, workspace: RemoteWorkspace):
        self.workspace = workspace

    def replay(self, events: list[Any]) -> ReplayReport:
        report = ReplayReport()
        pairs = _pair_actions_with_observations(events)
        logger.info("Replaying %d action events into workspace", len(pairs))

        for action_event, obs_event in pairs:
            action = getattr(action_event, "action", None)
            if action is None:
                continue

            action_class = type(action).__name__

            if action_class == "TerminalAction":
                self._replay_terminal(action, obs_event, report)
            elif action_class == "FileEditorAction":
                self._replay_file_editor(action, obs_event, report)
            elif action_class == "ApplyPatchAction":
                self._replay_apply_patch(action, obs_event, report)
            else:
                report.file_edits_skipped += 1

        logger.info(
            "Replay complete: %d terminal cmds, %d file edits, %d skipped terminal, "
            "%d skipped edits, %d errors",
            report.terminal_replayed,
            report.file_edits_applied,
            report.terminal_skipped,
            report.file_edits_skipped,
            len(report.errors),
        )
        return report

    def _replay_terminal(
        self, action: Any, obs: ObservationEvent | None, report: ReplayReport
    ) -> None:
        if getattr(action, "is_input", False) or getattr(action, "reset", False):
            report.terminal_skipped += 1
            return

        command = getattr(action, "command", "")
        if not command or not command.strip():
            report.terminal_skipped += 1
            return

        if _is_readonly_command(command):
            report.terminal_skipped += 1
            return

        # Skip commands that failed (and are not env-modifying)
        if obs is not None:
            observation = getattr(obs, "observation", None)
            is_error = getattr(observation, "is_error", False) if observation else False
            if is_error:
                cmd_lower = command.strip().lower()
                if not cmd_lower.startswith(("cd ", "export ", "source ")):
                    logger.debug("Skipping failed command: %s", command[:100])
                    report.terminal_skipped += 1
                    return

        logger.debug("Replaying terminal: %s", command[:120])
        try:
            res = self.workspace.execute_command(command)
            if res.exit_code != 0:
                logger.debug(
                    "Replay command exited %d: %s", res.exit_code, command[:100]
                )
            report.terminal_replayed += 1
        except Exception as exc:
            msg = f"Terminal replay error for '{command[:80]}': {exc}"
            logger.warning(msg)
            report.errors.append(msg)

    def _replay_file_editor(
        self, action: Any, obs: ObservationEvent | None, report: ReplayReport
    ) -> None:
        command = getattr(action, "command", "")
        path = getattr(action, "path", "")

        if command == "view":
            report.file_edits_skipped += 1
            return

        observation = getattr(obs, "observation", None) if obs else None
        is_error = getattr(observation, "is_error", False) if observation else False
        if is_error:
            logger.debug("Skipping failed file edit: %s %s", command, path)
            report.file_edits_skipped += 1
            return

        new_content = getattr(observation, "new_content", None) if observation else None

        # Fallback for create: use action.file_text if observation lacks new_content
        if command == "create" and new_content is None:
            new_content = getattr(action, "file_text", None)

        if new_content is not None:
            logger.debug("Replaying file edit (%s): %s", command, path)
            if _write_file_to_workspace(self.workspace, path, new_content):
                report.file_edits_applied += 1
            else:
                msg = f"Failed to write file {path} during replay"
                logger.warning(msg)
                report.errors.append(msg)
        else:
            logger.debug(
                "No new_content for file edit (%s) on %s, skipping", command, path
            )
            report.file_edits_skipped += 1

    def _replay_apply_patch(
        self, action: Any, obs: ObservationEvent | None, report: ReplayReport
    ) -> None:
        patch = getattr(action, "patch", None)
        if not patch:
            report.file_edits_skipped += 1
            return

        observation = getattr(obs, "observation", None) if obs else None
        is_error = getattr(observation, "is_error", False) if observation else False
        if is_error:
            report.file_edits_skipped += 1
            return

        encoded = base64.b64encode(patch.encode("utf-8")).decode("ascii")
        cmd = (
            f"python3 -c \"import base64,sys; "
            f"sys.stdout.buffer.write(base64.b64decode('{encoded}'))\" "
            f"| git apply --allow-empty -"
        )
        logger.debug("Replaying apply_patch")
        try:
            res = self.workspace.execute_command(cmd)
            if res.exit_code != 0:
                logger.warning("Patch replay failed: %s", getattr(res, "stderr", ""))
                report.errors.append(f"Patch replay failed (exit {res.exit_code})")
            else:
                report.file_edits_applied += 1
        except Exception as exc:
            msg = f"Patch replay error: {exc}"
            logger.warning(msg)
            report.errors.append(msg)
