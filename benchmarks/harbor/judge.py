"""Harbor verifier adapter: execute ``tests/test.sh`` and parse rewards.

Reference semantics (Harbor ``v0.9.0``, commit ``1b1dbc43``):

* the verifier runs **in the agent's container** (shared verification), after
  the agent is stopped;
* ``tests/`` is copied to ``/tests`` *after* the agent finishes and
  ``solution/`` is never exposed;
* ``/logs/verifier`` is cleared and the entrypoint is invoked exactly as
  ``(/tests/test.sh) > /logs/verifier/test-stdout.txt 2>&1`` through
  ``bash -c`` in the effective task working directory, with the resolved
  ``[verifier].env`` and ``[verifier].user``;
* a nonzero exit code **does not** prevent scoring -- the reward files are the
  only source of truth;
* ``/logs/verifier/reward.json`` wins over ``/logs/verifier/reward.txt``.

Deviations are deliberate and documented in ``README.md``: rewards with
non-finite numbers are rejected (the plan requires it), and an empty reward
dictionary is a verifier error rather than a zero score.
"""

from __future__ import annotations

import hashlib
import json
import math
import shlex
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol as TypingProtocol

from benchmarks.harbor.task import (
    ArtifactConfig,
    HarborTask,
    resolve_env_templates,
)
from benchmarks.harbor.workspace import ExecOutcome
from openhands.sdk.logger import get_logger


logger = get_logger(__name__)

TESTS_DIR = "/tests"
VERIFIER_DIR = "/logs/verifier"
ARTIFACTS_DIR = "/logs/artifacts"
CONVENTION_ARTIFACTS_SOURCE = ARTIFACTS_DIR
REWARD_JSON = "reward.json"
REWARD_TEXT = "reward.txt"
TEST_STDOUT = "test-stdout.txt"

STATUS_SCORED = "scored"
STATUS_VERIFIER_ERROR = "verifier_error"
STATUS_VERIFIER_TIMEOUT = "verifier_timeout"
STATUS_UNSCORED = "unscored"
STATUS_SKIPPED = "skipped"

Rewards = dict[str, float | int]


class RewardParseError(Exception):
    """A reward file exists but cannot be interpreted as numeric rewards."""


class RewardFileNotFoundError(FileNotFoundError):
    """Neither reward file was produced by the verifier."""


class WorkspaceExecutor(TypingProtocol):
    """Minimal workspace surface the verifier/artifact adapters rely on."""

    working_dir: str

    def docker_exec(
        self,
        command: str,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        user: str | int | None = None,
        timeout: float | None = None,
        kill_pattern: str | None = None,
    ) -> ExecOutcome: ...

    def upload_dir(self, source_dir: str | Path, target_dir: str) -> None: ...

    def download_dir(self, source_dir: str, target_dir: str | Path) -> None: ...

    def download_file(self, source_path: str, target_path: str | Path) -> None: ...

    def download_dir_with_exclusions(
        self, source_dir: str, target_dir: str | Path, exclude: list[str]
    ) -> None: ...

    def is_dir(self, remote_path: str) -> bool: ...

    def path_exists(self, remote_path: str) -> bool: ...


# ---------------------------------------------------------------------------
# Reward parsing
# ---------------------------------------------------------------------------


def _coerce_reward_value(value: Any, *, key: str) -> float | int:
    """Coerce one reward value the way Harbor's pydantic ``float | int`` does.

    Harbor validates ``reward.json`` through ``dict[str, float | int]``, which
    coerces numeric strings and booleans.  Values that cannot be coerced, and
    non-finite numbers (NaN/Infinity), are rejected here.
    """
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise RewardParseError(f"reward {key!r} is not a finite number: {value!r}")
        return value
    if isinstance(value, str):
        try:
            number = float(value.strip())
        except (TypeError, ValueError) as exc:
            raise RewardParseError(f"reward {key!r} is not numeric: {value!r}") from exc
        if not math.isfinite(number):
            raise RewardParseError(f"reward {key!r} is not a finite number: {value!r}")
        if number.is_integer():
            return int(number)
        return number
    raise RewardParseError(
        f"reward {key!r} has unsupported type {type(value).__name__}: {value!r}"
    )


def parse_reward_json_text(text: str) -> Rewards:
    """Parse ``reward.json`` content into numeric rewards."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RewardParseError(f"Invalid JSON reward file: {exc}") from exc
    if not isinstance(data, dict):
        raise RewardParseError(
            "reward.json must contain a JSON object of labeled numeric metrics, "
            f"got {type(data).__name__}"
        )
    rewards: Rewards = {}
    for key, value in data.items():
        if not isinstance(key, str):
            raise RewardParseError(f"reward.json keys must be strings: {key!r}")
        rewards[key] = _coerce_reward_value(value, key=key)
    if not rewards:
        raise RewardParseError(
            "reward.json contains no rewards; an empty object is not scoreable"
        )
    return rewards


def parse_reward_text_text(text: str) -> Rewards:
    """Parse ``reward.txt`` content into ``{"reward": <number>}``."""
    stripped = text.strip()
    if not stripped:
        raise RewardParseError("reward.txt is empty")
    try:
        number = float(stripped)
    except (TypeError, ValueError) as exc:
        raise RewardParseError(
            f"reward.txt is not a single number: {stripped!r}"
        ) from exc
    if not math.isfinite(number):
        raise RewardParseError(f"reward.txt is not a finite number: {stripped!r}")
    return {"reward": number}


def load_rewards(verifier_dir: Path) -> tuple[Rewards, str]:
    """Load rewards from a verifier log directory, JSON first (Harbor parity)."""
    json_path = verifier_dir / REWARD_JSON
    text_path = verifier_dir / REWARD_TEXT
    if json_path.exists():
        if json_path.stat().st_size == 0:
            raise RewardParseError(f"Reward file is empty at {json_path}")
        return parse_reward_json_text(
            json_path.read_text(encoding="utf-8")
        ), REWARD_JSON
    if text_path.exists():
        if text_path.stat().st_size == 0:
            raise RewardParseError(f"Reward file is empty at {text_path}")
        return parse_reward_text_text(
            text_path.read_text(encoding="utf-8")
        ), REWARD_TEXT
    raise RewardFileNotFoundError(f"No reward file found at {text_path} or {json_path}")


def select_primary_reward(
    rewards: Rewards | None, primary_key: str
) -> float | int | None:
    """Return the configured primary score without guessing from other keys."""
    if not rewards:
        return None
    return rewards.get(primary_key)


# ---------------------------------------------------------------------------
# Verifier execution
# ---------------------------------------------------------------------------


@dataclass
class VerifierOutcome:
    """Structured verifier result stored in ``EvalOutput.test_result['verifier']``."""

    status: str
    rewards: Rewards | None = None
    reward: float | int | None = None
    error: str | None = None
    exit_code: int | None = None
    duration_sec: float = 0.0
    reward_file: str | None = None
    log_dir: str | None = None
    stdout_file: str | None = None
    stderr_file: str | None = None
    test_stdout_file: str | None = None
    timed_out: bool = False
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "status": self.status,
            "reward": self.reward,
            "rewards": self.rewards,
            "error": self.error,
            "exit_code": self.exit_code,
            "duration_sec": round(self.duration_sec, 3),
            "log_dir": self.log_dir,
            "reward_file": self.reward_file,
            "stdout_file": self.stdout_file,
            "stderr_file": self.stderr_file,
            "test_stdout_file": self.test_stdout_file,
            "timed_out": self.timed_out,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }
        payload.update(self.extra)
        return payload


def _relative_to_run_dir(path: Path, run_dir: str | Path) -> str:
    try:
        return path.resolve().relative_to(Path(run_dir).resolve()).as_posix()
    except ValueError:
        return str(path)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _build_verifier_command(test_script: str) -> str:
    script = f"{TESTS_DIR}/{test_script}"
    stdout_path = f"{VERIFIER_DIR}/{TEST_STDOUT}"
    return f"({shlex.quote(script)}) > {shlex.quote(stdout_path)} 2>&1"


def run_shared_verifier(
    workspace: WorkspaceExecutor,
    task: HarborTask,
    *,
    trial_dir: str | Path,
    run_dir: str | Path,
    primary_reward_key: str = "reward",
    timeout_override: float | None = None,
    enabled: bool = True,
) -> dict[str, Any]:
    """Run ``tests/test.sh`` in the live workspace and return the outcome dict.

    Never raises: grading failures are structured ``verifier_error`` outcomes so
    they cannot trigger whole-rollout worker retries.
    """
    trial_path = Path(trial_dir)
    verifier_dir = trial_path / "verifier"
    verifier_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    outcome = VerifierOutcome(
        status=STATUS_SCORED,
        log_dir=_relative_to_run_dir(verifier_dir, run_dir),
    )

    if not enabled:
        outcome.status = STATUS_UNSCORED
        outcome.error = None
        outcome.finished_at = time.time()
        _write_verifier_meta(verifier_dir, outcome)
        return outcome.to_dict()

    timeout = (
        timeout_override
        if timeout_override is not None
        else task.config.verifier.timeout_sec
    )
    verifier_user = task.config.verifier.user
    command = _build_verifier_command(task.test_script)

    try:
        verifier_env = resolve_env_templates(task.config.verifier.env)
        _reset_verifier_dir_in_container(workspace)
        _install_tests(workspace, task)

        exec_timeout = _remaining_budget(started, timeout)
        if exec_timeout <= 0:
            raise TimeoutError(f"verifier exceeded its {timeout}s budget before exec")
        result = workspace.docker_exec(
            command,
            cwd=workspace.working_dir,
            env=verifier_env or None,
            user=verifier_user,
            timeout=exec_timeout,
            kill_pattern=f"{TESTS_DIR}/{task.test_script}",
        )
        outcome.exit_code = result.exit_code
        outcome.timed_out = result.timed_out
        outcome.duration_sec = result.duration_sec
        _write_text(verifier_dir / "exec-stdout.txt", result.stdout)
        _write_text(verifier_dir / "exec-stderr.txt", result.stderr)
        outcome.stdout_file = _relative_to_run_dir(
            verifier_dir / "exec-stdout.txt", run_dir
        )
        outcome.stderr_file = _relative_to_run_dir(
            verifier_dir / "exec-stderr.txt", run_dir
        )

        _download_verifier_dir(workspace, verifier_dir)
        test_stdout = verifier_dir / TEST_STDOUT
        if test_stdout.exists():
            outcome.test_stdout_file = _relative_to_run_dir(test_stdout, run_dir)

        if result.timed_out:
            outcome.status = STATUS_VERIFIER_TIMEOUT
            outcome.error = f"Verifier timed out after {timeout:g}s"
            _preserve_raw_rewards(outcome, verifier_dir, primary_reward_key)
        else:
            try:
                rewards, reward_file = load_rewards(verifier_dir)
            except RewardParseError as exc:
                outcome.status = STATUS_VERIFIER_ERROR
                outcome.error = str(exc)
            except RewardFileNotFoundError as exc:
                outcome.status = STATUS_VERIFIER_ERROR
                outcome.error = str(exc)
            else:
                outcome.status = STATUS_SCORED
                outcome.rewards = rewards
                outcome.reward = select_primary_reward(rewards, primary_reward_key)
                outcome.reward_file = reward_file
                if outcome.reward is None:
                    outcome.error = (
                        f"Scored rewards have no primary key "
                        f"{primary_reward_key!r}; available keys: "
                        f"{sorted(rewards)}"
                    )
    except Exception as exc:  # grading must never retry the rollout
        outcome.status = (
            STATUS_VERIFIER_TIMEOUT
            if isinstance(exc, TimeoutError)
            else STATUS_VERIFIER_ERROR
        )
        outcome.error = f"{type(exc).__name__}: {exc}"
        outcome.timed_out = outcome.status == STATUS_VERIFIER_TIMEOUT
        logger.warning(
            "Verifier failed for %s: %s", task.id, outcome.error, exc_info=False
        )
        # Best effort: capture whatever logs/rewards the verifier produced
        # before the failure, including on infra errors.
        _download_verifier_dir(workspace, verifier_dir)
        _preserve_raw_rewards(outcome, verifier_dir, primary_reward_key)
    finally:
        outcome.duration_sec = outcome.duration_sec or (time.monotonic() - started)
        outcome.finished_at = time.time()
        _write_verifier_meta(verifier_dir, outcome)

    return outcome.to_dict()


def _remaining_budget(started: float, timeout: float) -> float:
    remaining = timeout - (time.monotonic() - started)
    return max(0.0, remaining)


def _reset_verifier_dir_in_container(workspace: WorkspaceExecutor) -> None:
    command = (
        f"rm -rf {VERIFIER_DIR} && mkdir -p {VERIFIER_DIR} && chmod 777 {VERIFIER_DIR}"
    )
    result = workspace.docker_exec(command, user="root", timeout=120)
    if not result.ok:
        raise RuntimeError(
            f"Failed to reset {VERIFIER_DIR}: {result.stderr.strip()[:1000]}"
        )


def _install_tests(workspace: WorkspaceExecutor, task: HarborTask) -> None:
    removed = workspace.docker_exec(f"rm -rf {TESTS_DIR}", user="root", timeout=120)
    if not removed.ok:
        raise RuntimeError(
            f"Failed to clear {TESTS_DIR}: {removed.stderr.strip()[:1000]}"
        )
    workspace.upload_dir(task.tests_dir, TESTS_DIR)
    chmod = workspace.docker_exec(
        f"chmod +x {shlex.quote(f'{TESTS_DIR}/{task.test_script}')}",
        user="root",
        timeout=120,
    )
    if not chmod.ok:
        raise RuntimeError(
            "Failed to make the verifier entrypoint executable: "
            f"{chmod.stderr.strip()[:1000]}"
        )


def _download_verifier_dir(workspace: WorkspaceExecutor, verifier_dir: Path) -> None:
    try:
        workspace.download_dir(VERIFIER_DIR, verifier_dir)
    except Exception as exc:
        # The reward files may still be missing; the caller turns this into a
        # structured verifier error. Keep the partial logs we already have.
        logger.warning("Failed to download %s: %s", VERIFIER_DIR, exc)


def _preserve_raw_rewards(
    outcome: VerifierOutcome, verifier_dir: Path, primary_reward_key: str
) -> None:
    """Keep a raw diagnostic reward even when the verifier errored/timed out."""
    try:
        rewards, reward_file = load_rewards(verifier_dir)
    except (RewardParseError, RewardFileNotFoundError):
        return
    outcome.rewards = rewards
    outcome.reward_file = reward_file
    outcome.reward = select_primary_reward(rewards, primary_reward_key)


def _write_verifier_meta(verifier_dir: Path, outcome: VerifierOutcome) -> None:
    try:
        _write_text(
            verifier_dir / "meta.json",
            json.dumps(outcome.to_dict(), indent=2, sort_keys=True) + "\n",
        )
    except OSError as exc:  # pragma: no cover - best effort diagnostics
        logger.warning("Failed to write verifier meta.json: %s", exc)


# ---------------------------------------------------------------------------
# Artifact collection (Harbor order: after the agent, before the verifier)
# ---------------------------------------------------------------------------


def _relative_host_destination(destination: str) -> Path:
    from pathlib import PurePosixPath

    parts = [part for part in PurePosixPath(destination).parts if part not in ("", "/")]
    return Path(*parts) if parts else Path(".")


def _is_artifacts_dir_source(source: str) -> bool:
    return source.rstrip("/") == ARTIFACTS_DIR


def _artifact_host_path(artifacts_dir: Path, artifact: ArtifactConfig) -> Path:
    if _is_artifacts_dir_source(artifact.source):
        destination = artifact.destination or artifact.source
        if _is_artifacts_dir_source(destination) or destination == ".":
            return artifacts_dir
        return artifacts_dir / _relative_host_destination(destination)
    destination = artifact.destination or Path(artifact.source).name
    return artifacts_dir / _relative_host_destination(destination)


def _file_checksums(root: Path, *, limit: int = 5000) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    if not root.exists():
        return entries
    single_file = root.is_file()
    candidates = [root] if single_file else sorted(root.rglob("*"))
    for path in candidates:
        if not path.is_file():
            continue
        digest = hashlib.sha256()
        try:
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
        except OSError:
            continue
        if single_file:
            relative = Path(path.name)
        else:
            try:
                relative = path.relative_to(root)
            except ValueError:
                relative = Path(path.name)
        entries.append(
            {
                "path": relative.as_posix(),
                "size": path.stat().st_size,
                "sha256": digest.hexdigest(),
            }
        )
        if len(entries) >= limit:
            break
    return entries


def _normalized_artifacts(task: HarborTask) -> list[ArtifactConfig]:
    declared = task.config.normalized_artifacts
    if not any(_is_artifacts_dir_source(item.source) for item in declared):
        declared.insert(0, ArtifactConfig(source=CONVENTION_ARTIFACTS_SOURCE))
    return declared


def collect_artifacts(
    workspace: WorkspaceExecutor,
    task: HarborTask,
    *,
    trial_dir: str | Path,
    run_dir: str | Path,
) -> dict[str, Any]:
    """Collect declared artifacts into the trial directory (best effort).

    Mirrors Harbor's ``ArtifactHandler``: the ``/logs/artifacts`` convention
    source is always included, host destinations follow source/destination
    rules, directory exclusions use ``tar --exclude``, and failures are
    recorded rather than raised.
    """
    trial_path = Path(trial_dir)
    artifacts_dir = trial_path / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    started = time.monotonic()

    for artifact in _normalized_artifacts(task):
        source = artifact.source
        target = _artifact_host_path(artifacts_dir, artifact)
        entry: dict[str, Any] = {
            "source": source,
            "destination": _relative_to_run_dir(target, run_dir),
        }
        try:
            if not workspace.path_exists(source):
                entry.update({"type": "unknown", "status": "failed"})
            elif workspace.is_dir(source):
                entry["type"] = "directory"
                if artifact.exclude:
                    workspace.download_dir_with_exclusions(
                        source, target, artifact.exclude
                    )
                else:
                    workspace.download_dir(source, target)
                has_contents = target.exists() and any(target.iterdir())
                entry["status"] = "ok" if has_contents else "empty"
            else:
                entry["type"] = "file"
                target.parent.mkdir(parents=True, exist_ok=True)
                workspace.download_file(source, target)
                entry["status"] = "ok"
            entry["files"] = _file_checksums(target)
        except Exception as exc:  # best effort, like Harbor
            entry["status"] = "failed"
            entry["error"] = f"{type(exc).__name__}: {exc}"
            logger.debug("Artifact collection failed for %s", source, exc_info=True)
        entries.append(entry)

    manifest = {"artifacts": entries}
    try:
        _write_text(
            artifacts_dir / "manifest.json",
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        )
    except OSError as exc:  # pragma: no cover - best effort
        logger.warning("Failed to write artifact manifest: %s", exc)

    return {
        "entries": entries,
        "manifest": _relative_to_run_dir(artifacts_dir / "manifest.json", run_dir),
        "duration_sec": round(time.monotonic() - started, 3),
        "failed_count": sum(1 for entry in entries if entry["status"] == "failed"),
    }


__all__ = [
    "RewardFileNotFoundError",
    "RewardParseError",
    "STATUS_SCORED",
    "STATUS_SKIPPED",
    "STATUS_UNSCORED",
    "STATUS_VERIFIER_ERROR",
    "STATUS_VERIFIER_TIMEOUT",
    "VerifierOutcome",
    "WorkspaceExecutor",
    "collect_artifacts",
    "load_rewards",
    "parse_reward_json_text",
    "parse_reward_text_text",
    "run_shared_verifier",
    "select_primary_reward",
]
