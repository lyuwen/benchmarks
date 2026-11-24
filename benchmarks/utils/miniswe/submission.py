"""Candidate-patch validation for mini rollouts.

mini submits a patch, so the benchmark never feeds the post-run worktree diff into
scoring (the OpenHands runner's blanket ``git add -A`` could otherwise capture
mini's scratch patch file). To still catch malformed submissions early, the
candidate is checked against the task baseline with ``git apply --check`` after
restoring the worktree. A failed check is recorded but never rewrites the
candidate bytes: the existing scorer stays the source of truth.
"""

from __future__ import annotations

import os
import shlex
import tempfile

from openhands.sdk import get_logger
from openhands.sdk.workspace import RemoteWorkspace


logger = get_logger(__name__)

REMOTE_PATCH_PATH = "/tmp/miniswe-candidate.patch"


def check_patch_applies(
    *,
    workspace: RemoteWorkspace,
    repo_path: str,
    patch: str,
    timeout: int = 180,
) -> bool | None:
    """Return whether ``patch`` applies to the task baseline.

    Args:
        workspace: Provisioned workspace containing the task checkout.
        repo_path: Repository directory inside the workspace.
        patch: Candidate patch exactly as submitted (never modified).
        timeout: Per-command timeout in seconds.

    Returns:
        True/False when the check ran, None when validation was impossible
        (transport/file-API failure). Never raises.
    """
    if not patch.strip():
        return False

    local_path: str | None = None
    try:
        handle, local_path = tempfile.mkstemp(suffix=".patch", prefix="miniswe-")
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(patch)
        upload = workspace.file_upload(local_path, REMOTE_PATCH_PATH)
        if not upload.success:
            logger.warning(
                "Patch validation skipped: upload failed for %s: %s",
                repo_path,
                upload.error,
            )
            return None

        # Mini edits the worktree while solving the task, so restore the baseline
        # before checking the submitted patch. The submitted bytes are already
        # captured; the worktree is not used for scoring afterwards.
        restore = workspace.execute_command(
            f"cd {shlex.quote(repo_path)} ; git checkout -- . ; git clean -fdq",
            timeout=timeout,
        )
        if restore.exit_code != 0:
            logger.warning(
                "Patch validation baseline restore failed in %s: %s",
                repo_path,
                restore.stderr,
            )
        check = workspace.execute_command(
            f"cd {shlex.quote(repo_path)} ; "
            f"git apply --check --whitespace=nowarn {REMOTE_PATCH_PATH}",
            timeout=timeout,
        )
        if check.exit_code != 0:
            logger.info(
                "Candidate patch does not apply cleanly to %s: %s",
                repo_path,
                check.stderr.strip()[:500],
            )
        return check.exit_code == 0
    except Exception as exc:
        logger.warning("Patch validation failed for %s: %s", repo_path, exc)
        return None
    finally:
        if local_path is not None and os.path.exists(local_path):
            os.unlink(local_path)


def get_head_commit(
    *,
    workspace: RemoteWorkspace,
    repo_path: str,
    timeout: int = 60,
) -> str | None:
    """Return the current ``HEAD`` of the task checkout, or None if unavailable."""
    try:
        result = workspace.execute_command(
            f"cd {shlex.quote(repo_path)} ; git rev-parse HEAD", timeout=timeout
        )
    except Exception as exc:
        logger.warning("Could not read HEAD of %s: %s", repo_path, exc)
        return None
    if result.exit_code != 0:
        logger.warning("Could not read HEAD of %s: %s", repo_path, result.stderr)
        return None
    head = result.stdout.strip()
    return head or None


def head_matches_commit(
    *,
    workspace: RemoteWorkspace,
    repo_path: str,
    commit: str,
    timeout: int = 60,
) -> bool | None:
    """Return whether the checkout's ``HEAD`` is ``commit``.

    Both sides are resolved with ``git rev-parse`` so abbreviated dataset commits
    (e.g. a 9-char ``parent_commit``) compare equal to the full object id.

    Returns:
        True/False when the comparison ran, None when it could not be completed.
    """
    if not commit:
        return None
    head = get_head_commit(workspace=workspace, repo_path=repo_path, timeout=timeout)
    if head is None:
        return None
    try:
        resolved = workspace.execute_command(
            f"cd {shlex.quote(repo_path)} ; "
            f"git rev-parse --verify {shlex.quote(commit)}^{{commit}}",
            timeout=timeout,
        )
    except Exception as exc:
        logger.warning(
            "Could not resolve base commit %s in %s: %s", commit, repo_path, exc
        )
        return None
    if resolved.exit_code != 0:
        logger.warning(
            "Could not resolve base commit %s in %s: %s",
            commit,
            repo_path,
            resolved.stderr,
        )
        return None
    return resolved.stdout.strip() == head
