"""Harbor-format task loading, validation, and fingerprinting.

This module is the single source of truth for how this repository interprets a
Harbor task directory.  It targets the task format of the *pinned* Harbor
reference release (``v0.9.0``, commit ``1b1dbc43``) and does not depend on the
``harbor`` package: the workspace only needs ``tomllib`` (stdlib) and
``pydantic`` (already a runtime dependency), so loading a task cannot conflict
with the SDK dependency tree.

Supported subset (Phase 1)
--------------------------
* single-step tasks: ``task.toml`` + ``instruction.md`` + ``environment/`` +
  ``tests/`` (no ``[[steps]]`` and no ``solution/`` exposure);
* schema versions ``1.0`` through ``1.4`` (``version = "1.0"`` is accepted as
  the pre-rename spelling, exactly like Harbor's ``handle_version_rename``);
* prebuilt ``[environment].docker_image`` or a single-container
  ``environment/Dockerfile`` build;
* shared verifier (the verifier runs in the agent's container);
* top-level ``artifacts`` collection.

Everything whose execution semantics this adapter cannot honor is rejected at
load time with :class:`UnsupportedTaskError` instead of being silently ignored.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


# The Harbor release this adapter is written against.  Documented in the
# module README; bumping it requires re-validating the parity fixtures.
HARBOR_REFERENCE_VERSION = "0.9.0"
HARBOR_REFERENCE_COMMIT = "1b1dbc43729edff705071a797bb0f98a90cf62f5"

# Task schema versions this adapter claims to load.  The nub of v0.9.0 is the
# `1.2` schema; `1.0`/`1.1` are the pre-workdir/pre-separate-verifier dialects
# still used by CalibForge and Terminal-Bench, and `1.3`/`1.4` only add fields
# this adapter audits explicitly (see `validate_supported`).
SUPPORTED_SCHEMA_VERSIONS: tuple[str, ...] = ("1.0", "1.1", "1.2", "1.3", "1.4")

FINGERPRINT_ALGORITHM = "harbor-task-fingerprint-v1"

_TASK_MARKER = "task.toml"
_INSTRUCTION_FILE = "instruction.md"
_ENVIRONMENT_DIR = "environment"
_TESTS_DIR = "tests"
_DOCKERFILE = "Dockerfile"
_DOCKER_COMPOSE_FILES = ("docker-compose.yaml", "docker-compose.yml")
_LINUX_TEST_SCRIPT = "test.sh"
_WINDOWS_TEST_SCRIPT = "test.bat"

_MANIFEST_FILENAME = "harbor_tasks.json"

_WARNED_LEGACY_FIELDS: set[str] = set()
_SAFE_ID_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")
_SAFE_ID_TRAILING = re.compile(r"^[._-]+|[._-]+$")


class HarborTaskError(Exception):
    """Base error for Harbor task loading/validation failures."""


class TaskLoadError(HarborTaskError):
    """A task directory is malformed or missing required files."""


class UnsupportedTaskError(HarborTaskError):
    """A task uses an execution feature this adapter cannot honor."""


# ---------------------------------------------------------------------------
# Task configuration models (mirroring Harbor v0.9.0 models/task/config.py)
# ---------------------------------------------------------------------------


class Author(BaseModel):
    name: str
    email: str | None = None


class PackageInfo(BaseModel):
    """The optional ``[task]`` registry-metadata section."""

    name: str
    description: str = ""
    authors: list[Author] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        if "/" not in value or ".." in value:
            raise ValueError(
                "task.name must be in 'org/name' format and cannot contain '..', "
                f"got {value!r}"
            )
        return value


class SolutionConfig(BaseModel):
    env: dict[str, str] = Field(default_factory=dict)


class AgentConfig(BaseModel):
    timeout_sec: float | None = None
    user: str | int | None = Field(
        default=None,
        description=(
            "Username or UID to run the agent as. This adapter supports only the "
            "container default (typically root) because the injected agent server "
            "cannot switch users for tool execution."
        ),
    )
    # Post-v0.9.0 extension: parsed only so it can be rejected explicitly.
    network_mode: str | None = None
    allowed_hosts: list[str] | None = None


class HealthcheckConfig(BaseModel):
    """Mirror of Harbor's ``[environment.healthcheck]`` block."""

    command: str
    interval_sec: float = 5.0
    timeout_sec: float = 30.0
    start_period_sec: float = 0.0
    start_interval_sec: float = 5.0
    retries: int = 3


def _parse_size_to_mb(size: str) -> int:
    """Parse a legacy ``"2G"``/``"512M"`` resource string into megabytes."""
    normalized = size.strip().upper()
    if normalized.endswith("G"):
        return int(float(normalized[:-1]) * 1024)
    if normalized.endswith("M"):
        return int(float(normalized[:-1]))
    if normalized.endswith("K"):
        return int(float(normalized[:-1]) / 1024)
    raise ValueError(
        f"Invalid size format: {size!r}. Expected format like '1G', '512M', etc."
    )


class HarborEnvironmentConfig(BaseModel):
    """Mirror of Harbor v0.9.0's ``[environment]`` / ``[verifier.environment]``."""

    build_timeout_sec: float = 600.0
    docker_image: str | None = None
    os: str = "linux"
    cpus: int | None = None
    memory_mb: int | None = None
    storage_mb: int | None = None
    gpus: int | None = None
    gpu_types: list[str] | None = None
    tpu: Any | None = None
    allow_internet: bool = True
    mcp_servers: list[dict[str, Any]] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    skills_dir: str | None = None
    healthcheck: HealthcheckConfig | None = None
    workdir: str | None = None
    # Post-v0.9.0 extension: parsed only so it can be rejected explicitly.
    network_mode: str | None = None
    allowed_hosts: list[str] | None = None
    mounts_json: Any | None = None

    @field_validator("os", mode="before")
    @classmethod
    def _normalize_os(cls, value: Any) -> Any:
        return value.lower() if isinstance(value, str) else value

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_resource_fields(cls, data: Any) -> Any:
        """Map Harbor's deprecated ``memory``/``storage`` strings to ``*_mb``."""
        if not isinstance(data, dict):
            return data
        for legacy, modern in (("memory", "memory_mb"), ("storage", "storage_mb")):
            if legacy not in data:
                continue
            if legacy not in _WARNED_LEGACY_FIELDS:
                _WARNED_LEGACY_FIELDS.add(legacy)
                warnings.warn(
                    f"The '{legacy}' field is deprecated in Harbor; use '{modern}'.",
                    DeprecationWarning,
                    stacklevel=2,
                )
            value = data.pop(legacy)
            if isinstance(value, str):
                converted = _parse_size_to_mb(value)
                if modern in data and data[modern] != converted:
                    raise ValueError(
                        f"Conflicting '{legacy}' and '{modern}' values: "
                        f"{legacy}={value!r} ({converted} MB) != "
                        f"{modern}={data[modern]!r}."
                    )
                data.setdefault(modern, converted)
        return data


class VerifierConfig(BaseModel):
    """Mirror of Harbor v0.9.0's ``[verifier]`` section."""

    timeout_sec: float = 600.0
    env: dict[str, str] = Field(default_factory=dict)
    user: str | int | None = None
    environment_mode: str | None = None
    environment: HarborEnvironmentConfig | None = None
    # Post-v0.9.0 extensions (Pier/deep-swe dialect): parsed to reject explicitly.
    network_mode: str | None = None
    allowed_hosts: list[str] | None = None
    collect: list[Any] | None = None


class ArtifactConfig(BaseModel):
    source: str
    destination: str | None = None
    exclude: list[str] = Field(default_factory=list)


class HarborTaskConfig(BaseModel):
    """Normalized ``task.toml`` for the supported Harbor subset."""

    schema_version: str = "1.2"
    task: PackageInfo | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    verifier: VerifierConfig = Field(default_factory=VerifierConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    environment: HarborEnvironmentConfig = Field(
        default_factory=HarborEnvironmentConfig
    )
    solution: SolutionConfig = Field(default_factory=SolutionConfig)
    source: str | None = None
    multi_step_reward_strategy: str | None = None
    steps: list[Any] | None = None
    artifacts: list[str | ArtifactConfig] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _handle_version_rename(cls, data: Any) -> Any:
        if isinstance(data, dict) and "version" in data:
            data.setdefault("schema_version", data.pop("version"))
        return data

    @field_validator("schema_version", mode="before")
    @classmethod
    def _normalize_schema_version(cls, value: Any) -> str:
        if isinstance(value, bool):  # defensive: bool is an int
            raise ValueError(f"Invalid schema_version: {value!r}")
        if isinstance(value, (int, float)):
            return f"{value:g}"
        return str(value).strip()

    @property
    def normalized_artifacts(self) -> list[ArtifactConfig]:
        return [
            ArtifactConfig(source=item) if isinstance(item, str) else item
            for item in self.artifacts
        ]


# ---------------------------------------------------------------------------
# Supported-subset validation
# ---------------------------------------------------------------------------


def _reject_network_mode(mode: str | None, where: str, task_id: str) -> None:
    if mode is None or mode.lower() in ("public", "allow-internet"):
        return
    raise UnsupportedTaskError(
        f"Task {task_id!r} declares {where}.network_mode={mode!r}. This adapter "
        "only supports the default public network, because the injected agent "
        "server must stay reachable over a published port."
    )


def validate_supported(config: HarborTaskConfig, task_hint: str) -> None:
    """Reject task configurations whose semantics this adapter cannot honor."""
    if config.schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        raise UnsupportedTaskError(
            f"Task {task_hint!r} uses schema_version={config.schema_version!r}; "
            f"supported versions are {', '.join(SUPPORTED_SCHEMA_VERSIONS)}."
        )

    if config.steps:
        raise UnsupportedTaskError(
            f"Task {task_hint!r} is a multi-step task ([[steps]]); multi-step "
            "tasks are not supported by this adapter."
        )

    environment = config.environment
    if environment.os != "linux":
        raise UnsupportedTaskError(
            f"Task {task_hint!r} targets environment.os={environment.os!r}; only "
            "linux containers are supported."
        )
    _reject_network_mode(environment.network_mode, "[environment]", task_hint)
    if environment.allowed_hosts:
        raise UnsupportedTaskError(
            f"Task {task_hint!r} declares [environment].allowed_hosts; network "
            "allowlists are not supported."
        )
    if not environment.allow_internet:
        raise UnsupportedTaskError(
            f"Task {task_hint!r} sets [environment].allow_internet=false; "
            "no-network execution is not supported."
        )
    if environment.gpus or environment.gpu_types or environment.tpu is not None:
        raise UnsupportedTaskError(
            f"Task {task_hint!r} requests GPU/TPU resources; only CPU tasks are "
            "supported."
        )
    if environment.mcp_servers:
        raise UnsupportedTaskError(
            f"Task {task_hint!r} declares [environment].mcp_servers; MCP setup is "
            "not supported."
        )
    if environment.skills_dir:
        raise UnsupportedTaskError(
            f"Task {task_hint!r} declares [environment].skills_dir; injecting "
            "skills is not supported."
        )
    if environment.mounts_json:
        raise UnsupportedTaskError(
            f"Task {task_hint!r} declares [environment].mounts_json; custom "
            "mounts are not supported."
        )
    if environment.workdir is not None and not environment.workdir.startswith("/"):
        raise UnsupportedTaskError(
            f"Task {task_hint!r} sets a relative [environment].workdir="
            f"{environment.workdir!r}; an absolute container path is required."
        )

    agent = config.agent
    _reject_network_mode(agent.network_mode, "[agent]", task_hint)
    if agent.allowed_hosts:
        raise UnsupportedTaskError(
            f"Task {task_hint!r} declares [agent].allowed_hosts; network "
            "allowlists are not supported."
        )
    if agent.user not in (None, "root", 0):
        raise UnsupportedTaskError(
            f"Task {task_hint!r} sets [agent].user={agent.user!r}; this adapter "
            "can only run the agent as the container default user (root), "
            "because the injected agent server cannot switch users."
        )

    verifier = config.verifier
    _reject_network_mode(verifier.network_mode, "[verifier]", task_hint)
    if verifier.allowed_hosts:
        raise UnsupportedTaskError(
            f"Task {task_hint!r} declares [verifier].allowed_hosts; network "
            "allowlists are not supported."
        )
    if verifier.collect:
        raise UnsupportedTaskError(
            f"Task {task_hint!r} declares [[verifier.collect]]; collection hooks "
            "are a post-v0.9.0 dialect and are not supported."
        )
    if verifier.environment_mode not in (None, "shared"):
        raise UnsupportedTaskError(
            f"Task {task_hint!r} uses [verifier].environment_mode="
            f"{verifier.environment_mode!r}; only shared verification is "
            "supported in Phase 1."
        )
    if verifier.environment is not None:
        raise UnsupportedTaskError(
            f"Task {task_hint!r} declares [verifier.environment]; a separate "
            "verifier sandbox is not supported in Phase 1."
        )


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _read_toml(path: Path) -> dict[str, Any]:
    import tomllib

    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise TaskLoadError(f"Failed to parse {path}: {exc}") from exc


def _discover_test_script(tests_dir: Path) -> str:
    candidate = tests_dir / _LINUX_TEST_SCRIPT
    if candidate.is_file():
        return _LINUX_TEST_SCRIPT
    if (tests_dir / _WINDOWS_TEST_SCRIPT).is_file():
        raise UnsupportedTaskError(
            f"Task at {tests_dir.parent} provides only {_WINDOWS_TEST_SCRIPT}; "
            "Windows tasks are not supported."
        )
    raise TaskLoadError(
        f"Task at {tests_dir.parent} has no executable verifier script "
        f"({_TESTS_DIR}/{_LINUX_TEST_SCRIPT})."
    )


def should_upload_environment_dir(
    environment_dir: Path, *, docker_image: str | None
) -> bool:
    """True when Harbor uploads ``environment/`` into the workdir after start.

    Harbor does this only for prebuilt-image tasks whose runtime files live in
    ``environment/`` instead of a Dockerfile/compose build spec.
    """
    if not docker_image:
        return False
    if (environment_dir / _DOCKERFILE).is_file():
        return False
    if any((environment_dir / name).is_file() for name in _DOCKER_COMPOSE_FILES):
        return False
    if not environment_dir.is_dir():
        return False
    return any(environment_dir.iterdir())


def _validate_environment_definition(
    task_dir: Path, config: HarborTaskConfig, task_hint: str
) -> None:
    environment_dir = task_dir / _ENVIRONMENT_DIR
    compose_files = [
        name for name in _DOCKER_COMPOSE_FILES if (environment_dir / name).is_file()
    ]
    has_dockerfile = (environment_dir / _DOCKERFILE).is_file()
    if compose_files:
        raise UnsupportedTaskError(
            f"Task {task_hint!r} defines environment/{compose_files[0]}; "
            "compose-based (multi-container) environments are not supported."
        )
    if not config.environment.docker_image and not has_dockerfile:
        raise TaskLoadError(
            f"Task {task_hint!r} has no environment definition: set "
            "[environment].docker_image or add environment/Dockerfile."
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_context_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    files: list[Path] = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        if any(part in (".git", "__pycache__") for part in path.parts):
            continue
        files.append(path)
    return files


def _canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def compute_task_fingerprint(
    task_dir: Path, config: HarborTaskConfig, instruction: str
) -> str:
    """Hash task settings, instruction, build context, and verifier inputs.

    The hash is deterministic across hosts: file names are dataset-relative and
    contents are hashed in sorted order.  It deliberately excludes
    ``solution/`` (never executed by the agent) and any local-only files.
    """
    digest = hashlib.sha256()
    digest.update(FINGERPRINT_ALGORITHM.encode())
    digest.update(b"\0")
    digest.update(HARBOR_REFERENCE_VERSION.encode())
    digest.update(b"\0")
    digest.update(
        _canonical_json(config.model_dump(mode="json", exclude_none=True)).encode()
    )
    digest.update(b"\0")
    digest.update(instruction.encode())
    for directory_name in (_ENVIRONMENT_DIR, _TESTS_DIR):
        directory = task_dir / directory_name
        digest.update(b"\0" + directory_name.encode() + b"\0")
        for path in _iter_context_files(directory):
            relative = path.relative_to(task_dir).as_posix()
            digest.update(relative.encode())
            digest.update(b"\0")
            digest.update(_sha256_file(path).encode())
            digest.update(b"\0")
    return digest.hexdigest()


def _git_revision(path: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    revision = completed.stdout.strip()
    return revision if completed.returncode == 0 and revision else None


@dataclass(frozen=True)
class HarborTask:
    """One loadable Harbor task directory."""

    id: str
    relative_path: str
    path: Path
    dataset_root: Path
    config: HarborTaskConfig
    instruction: str
    fingerprint: str
    test_script: str

    @property
    def docker_image(self) -> str | None:
        return self.config.environment.docker_image

    @property
    def tests_dir(self) -> Path:
        return self.path / _TESTS_DIR

    @property
    def environment_dir(self) -> Path:
        return self.path / _ENVIRONMENT_DIR

    def to_instance_data(self) -> dict[str, Any]:
        """Normalized task data stored in ``EvalInstance.data``/``EvalOutput``."""
        return {
            "task_id": self.id,
            "relative_path": self.relative_path,
            "task_dir": str(self.path),
            "dataset_root": str(self.dataset_root),
            "fingerprint": self.fingerprint,
            "instruction": self.instruction,
            "test_script": self.test_script,
            "schema_version": self.config.schema_version,
            "config": self.config.model_dump(mode="json", exclude_none=True),
        }


@dataclass(frozen=True)
class HarborTaskSet:
    """The selected tasks plus the provenance used for resume validation."""

    tasks: list[HarborTask]
    tasks_path: Path
    dataset_revision: str | None
    manifest_entries: list[dict[str, str]]
    manifest_hash: str

    @property
    def fingerprints(self) -> dict[str, str]:
        return {entry["id"]: entry["fingerprint"] for entry in self.manifest_entries}


def derive_instance_id(relative_path: str) -> str:
    """Build a filesystem-safe, deterministic instance id from a task path.

    Subset identity is preserved by joining path components with ``__``, so
    ``multi_solver/foo`` and ``contrastive_solver/foo`` stay distinct.
    """
    raw = relative_path.strip("/").replace("/", "__")
    safe = _SAFE_ID_PATTERN.sub("-", raw)
    safe = _SAFE_ID_TRAILING.sub("", safe)
    if not safe:
        safe = hashlib.sha1(raw.encode()).hexdigest()[:12]
    if len(safe) > 160:
        suffix = hashlib.sha1(raw.encode()).hexdigest()[:12]
        safe = f"{safe[:140]}-{suffix}"
    return safe


def normalize_relative_path(path: Path, root: Path, tasks_path_is_task: bool) -> str:
    if tasks_path_is_task:
        return path.name
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name


def load_task_dir(task_dir: Path, dataset_root: Path, *, is_single: bool) -> HarborTask:
    """Load and validate a single Harbor task directory."""
    task_dir = task_dir.resolve()
    marker = task_dir / _TASK_MARKER
    if not marker.is_file():
        raise TaskLoadError(
            f"{task_dir} is not a Harbor task directory (no {_TASK_MARKER})"
        )

    relative_path = normalize_relative_path(task_dir, dataset_root.resolve(), is_single)
    task_hint = relative_path

    raw = _read_toml(marker)
    try:
        config = HarborTaskConfig.model_validate(raw)
    except Exception as exc:
        raise TaskLoadError(
            f"Invalid {_TASK_MARKER} for task {task_hint!r}: {exc}"
        ) from exc

    validate_supported(config, task_hint)
    _validate_environment_definition(task_dir, config, task_hint)

    instruction_path = task_dir / _INSTRUCTION_FILE
    if not instruction_path.is_file():
        raise TaskLoadError(
            f"Task {task_hint!r} is missing required file {_INSTRUCTION_FILE}."
        )
    instruction = instruction_path.read_text(encoding="utf-8")

    tests_dir = task_dir / _TESTS_DIR
    if not tests_dir.is_dir():
        raise TaskLoadError(
            f"Task {task_hint!r} is missing the {_TESTS_DIR}/ directory."
        )
    test_script = _discover_test_script(tests_dir)

    fingerprint = compute_task_fingerprint(task_dir, config, instruction)
    return HarborTask(
        id=derive_instance_id(relative_path),
        relative_path=relative_path,
        path=task_dir,
        dataset_root=dataset_root.resolve(),
        config=config,
        instruction=instruction,
        fingerprint=fingerprint,
        test_script=test_script,
    )


def _is_hidden(path: Path, root: Path) -> bool:
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        parts = path.parts
    return any(part.startswith(".") or part == "__pycache__" for part in parts)


def discover_task_dirs(tasks_path: Path) -> tuple[list[Path], Path, bool]:
    """Return ``(task_dirs, dataset_root, is_single)`` for a path.

    ``tasks_path`` may point at an individual task directory or at a dataset
    root.  Task directories are discovered recursively so nested subsets
    (CalibForge's ``multi_solver/`` and ``contrastive_solver/``) are supported.
    """
    tasks_path = tasks_path.resolve()
    if not tasks_path.exists():
        raise TaskLoadError(f"Tasks path does not exist: {tasks_path}")
    if (tasks_path / _TASK_MARKER).is_file():
        return [tasks_path], tasks_path.parent, True
    if not tasks_path.is_dir():
        raise TaskLoadError(f"Tasks path is not a directory: {tasks_path}")
    task_dirs = [
        marker.parent
        for marker in sorted(tasks_path.rglob(_TASK_MARKER))
        if not _is_hidden(marker, tasks_path)
    ]
    if not task_dirs:
        raise TaskLoadError(
            f"No Harbor tasks ({_TASK_MARKER}) found under {tasks_path}"
        )
    return task_dirs, tasks_path, False


def read_selection_file(path: str) -> list[str]:
    """Read instance ids (one per line, ``#`` comments allowed) from a file."""
    selection_path = Path(path)
    if not selection_path.is_file():
        raise TaskLoadError(f"Selection file does not exist: {selection_path}")
    entries: list[str] = []
    for line in selection_path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        entries.append(value)
    return entries


def build_manifest(tasks: list[HarborTask]) -> tuple[list[dict[str, str]], str]:
    entries = [
        {
            "id": task.id,
            "relative_path": task.relative_path,
            "fingerprint": task.fingerprint,
        }
        for task in tasks
    ]
    entries.sort(key=lambda entry: entry["relative_path"])
    manifest_hash = hashlib.sha256(_canonical_json(entries).encode()).hexdigest()
    return entries, manifest_hash


def load_tasks(
    tasks_path: str | Path,
    *,
    selected_instances_file: str | None = None,
    eval_limit: int = 0,
) -> HarborTaskSet:
    """Discover, validate, select, and hash all tasks under ``tasks_path``.

    Ordering is deterministic: tasks are sorted by their dataset-relative path
    *before* ``--select`` and ``--n-limit`` are applied.
    """
    task_dirs, dataset_root, is_single = discover_task_dirs(Path(tasks_path))
    tasks = [
        load_task_dir(task_dir, dataset_root, is_single=is_single)
        for task_dir in task_dirs
    ]
    tasks.sort(key=lambda task: task.relative_path)

    seen_ids: dict[str, str] = {}
    for task in tasks:
        previous = seen_ids.get(task.id)
        if previous is not None:
            raise TaskLoadError(
                f"Task id collision for {task.id!r}: {previous!r} and "
                f"{task.relative_path!r} produce the same filesystem-safe id. "
                "Rename one of the task directories."
            )
        seen_ids[task.id] = task.relative_path

    if selected_instances_file:
        selected = set(read_selection_file(selected_instances_file))
        by_id = {task.id: task for task in tasks}
        by_path = {task.relative_path: task for task in tasks}
        unresolved = [
            entry for entry in selected if entry not in by_id and entry not in by_path
        ]
        if unresolved:
            raise TaskLoadError(
                "Selection file entries did not match any task id or relative "
                f"path: {', '.join(sorted(unresolved)[:10])}"
            )
        tasks = [
            task
            for task in tasks
            if task.id in selected or task.relative_path in selected
        ]

    if eval_limit > 0:
        tasks = tasks[:eval_limit]

    entries, manifest_hash = build_manifest(tasks)
    return HarborTaskSet(
        tasks=tasks,
        tasks_path=Path(tasks_path).resolve(),
        dataset_revision=_git_revision(Path(tasks_path).resolve())
        or _git_revision(dataset_root),
        manifest_entries=entries,
        manifest_hash=manifest_hash,
    )


def manifest_path_for(output_dir: str | Path) -> Path:
    return Path(output_dir) / _MANIFEST_FILENAME


def write_task_manifest(
    output_dir: str | Path,
    task_set: HarborTaskSet,
    reference_version: str = HARBOR_REFERENCE_VERSION,
) -> Path:
    """Persist the selected-task manifest for audit and resume validation."""
    path = manifest_path_for(output_dir)
    payload = {
        "reference_version": reference_version,
        "tasks_path": str(task_set.tasks_path),
        "dataset_revision": task_set.dataset_revision,
        "task_count": len(task_set.tasks),
        "manifest_hash": task_set.manifest_hash,
        "tasks": task_set.manifest_entries,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def compute_run_settings_hash(settings: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(settings).encode()).hexdigest()


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def check_resume_compatible(
    output_dir: str | Path,
    *,
    manifest_hash: str,
    run_settings_hash: str,
) -> None:
    """Refuse to resume a run whose task set or run settings changed.

    Called before ``metadata.json`` is written, so a mismatch never clobbers the
    provenance of the original run.  A fresh directory (or a directory without
    Harbor metadata) is always accepted.
    """
    output_path = Path(output_dir)
    metadata = _load_json(output_path / "metadata.json")
    if metadata is None:
        return
    details = metadata.get("details")
    harbor = details.get("harbor") if isinstance(details, dict) else None
    if not isinstance(harbor, dict):
        raise HarborTaskError(
            f"Output directory {output_path} already contains a run without "
            "Harbor provenance. Refusing to resume; choose a new --output-dir."
        )

    stored_manifest = harbor.get("manifest_hash")
    if stored_manifest != manifest_hash:
        raise HarborTaskError(
            f"Task set/content changed since the run in {output_path} was created "
            f"(stored manifest {stored_manifest!r}, current {manifest_hash!r}). "
            "Resume is rejected; choose a new --output-dir or restore the task "
            "files."
        )
    stored_settings = harbor.get("run_settings_hash")
    if stored_settings != run_settings_hash:
        raise HarborTaskError(
            f"Harbor run settings changed since the run in {output_path} was "
            f"created (stored {stored_settings!r}, current {run_settings_hash!r}). "
            "Resume is rejected; choose a new --output-dir."
        )


_ENV_TEMPLATE = re.compile(r"\$\{([^}:]+)(?::-(.*))?\}")


def resolve_env_templates(env: dict[str, str]) -> dict[str, str]:
    """Resolve Harbor's ``${VAR}`` / ``${VAR:-default}`` env templating.

    Mirrors ``harbor.utils.env.resolve_env_vars``: a value that is exactly one
    template is replaced from the host environment (or its default); any other
    value is passed through literally.
    """
    resolved: dict[str, str] = {}
    for key, value in env.items():
        match = _ENV_TEMPLATE.fullmatch(value)
        if match is None:
            resolved[key] = value
            continue
        variable, default = match.group(1), match.group(2)
        if variable in os.environ:
            resolved[key] = os.environ[variable]
        elif default is not None:
            resolved[key] = default
        else:
            raise TaskLoadError(
                f"Environment variable '{variable}' referenced by task config is "
                "not set on the host and has no default."
            )
    return resolved


__all__ = [
    "HARBOR_REFERENCE_VERSION",
    "HARBOR_REFERENCE_COMMIT",
    "SUPPORTED_SCHEMA_VERSIONS",
    "ArtifactConfig",
    "HarborEnvironmentConfig",
    "HarborTask",
    "HarborTaskConfig",
    "HarborTaskError",
    "HarborTaskSet",
    "TaskLoadError",
    "UnsupportedTaskError",
    "build_manifest",
    "check_resume_compatible",
    "compute_run_settings_hash",
    "compute_task_fingerprint",
    "derive_instance_id",
    "discover_task_dirs",
    "load_task_dir",
    "load_tasks",
    "manifest_path_for",
    "read_selection_file",
    "resolve_env_templates",
    "should_upload_environment_dir",
    "validate_supported",
    "write_task_manifest",
]
