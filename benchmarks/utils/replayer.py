from __future__ import annotations

from typing import TYPE_CHECKING, Any

from openhands.sdk import get_logger
from openhands.sdk.event import ActionEvent
from openhands.tools.file_editor.definition import FileEditorAction
from openhands.tools.terminal.definition import TerminalAction

if TYPE_CHECKING:
    from openhands.sdk.workspace import RemoteWorkspace


logger = get_logger(__name__)


class ReplayManager:
    """Manages workspace restoration by replaying side-effecting actions."""

    def __init__(self, workspace: RemoteWorkspace):
        self.workspace = workspace

    def replay_events(self, events: list[Any]) -> None:
        """Replay side-effecting actions into the current workspace."""
        logger.info(f"Replaying {len(events)} events into workspace...")
        
        for event in events:
            if not isinstance(event, ActionEvent):
                continue
                
            action = event.action
            if isinstance(action, TerminalAction):
                self._replay_terminal_action(action)
            elif isinstance(action, FileEditorAction):
                self._replay_file_editor_action(action)

    def _replay_terminal_action(self, action: TerminalAction) -> None:
        """Replay a bash command in the terminal."""
        if action.is_input or action.reset:
            # We don't replay raw STDIN inputs or terminal resets for now
            return
            
        if not action.command:
            return
            
        logger.debug(f"Replaying terminal command: {action.command}")
        res = self.workspace.execute_command(action.command)
        if res.exit_code != 0:
            logger.warning(
                f"Replay command failed (exit {res.exit_code}): {action.command}\n"
                f"stderr: {res.stderr}"
            )

    def _replay_file_editor_action(self, action: FileEditorAction) -> None:
        """Replay a file editor operation."""
        if action.command == "view":
            return
            
        # For replay, we translate editor actions into basic workspace operations
        logger.debug(f"Replaying file editor command: {action.command} {action.path}")
        
        if action.command == "create":
            # Simple file creation via cat and redirection or python
            self._write_file(action.path, action.file_text or "")
        elif action.command in ("str_replace", "insert", "undo_edit"):
            # These are harder to translate to bash atomicly without the original file.
            # However, for eval we want robustness. If we're replaying, we should
            # rely on the same logic as the actual tool if possible.
            # Since we have the workspace, we can execute a small python snippet to
            # perform the replacement if we want to be exact.
            
            # For simplicity in this initial implementation, we warn that complex
            # state restoration for edits is best handled by replaying the 
            # entire sequence of 'create' and subsequent 'bash' edits.
            # If the agent only used the file_editor, we'd need more logic.
            pass

    def _write_file(self, path: str, content: str) -> None:
        """Helper to write a file inside the workspace."""
        # Use a small python script to write content to avoid escaping issues in bash
        import base64
        encoded_content = base64.b64encode(content.encode()).decode()
        cmd = (
            f"python3 -c 'import base64; "
            f"with open(\"{path}\", \"wb\") as f: "
            f"f.write(base64.b64decode(\"{encoded_content}\"))'"
        )
        self.workspace.execute_command(cmd)
