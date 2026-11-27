"""Loader, validation, id derivation, fingerprint, and resume tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.harbor.task import (
    HARBOR_REFERENCE_VERSION,
    HarborTaskError,
    TaskLoadError,
    UnsupportedTaskError,
    build_manifest,
    check_resume_compatible,
    compute_run_settings_hash,
    compute_task_fingerprint,
    derive_instance_id,
    discover_task_dirs,
    load_task_dir,
    load_tasks,
    resolve_env_templates,
)
from benchmarks.harbor.tests.conftest import write_task


def test_load_individual_task(tasks_dir: Path) -> None:
    task_dir = write_task(tasks_dir, "solo")
    result = load_tasks(task_dir)

    assert len(result.tasks) == 1
    task = result.tasks[0]
    assert task.id == "solo"
    assert task.relative_path == "solo"
    assert task.config.schema_version == "1.2"
    assert task.config.environment.docker_image == "example/task:latest"
    assert task.instruction == "Do the thing."
    assert task.test_script == "test.sh"
    assert len(task.fingerprint) == 64
    assert result.manifest_hash == build_manifest(result.tasks)[1]


def test_load_nested_dataset_subsets(dataset_root: Path) -> None:
    write_task(dataset_root / "multi_solver", "task-1")
    write_task(dataset_root / "multi_solver", "task-2")
    write_task(dataset_root / "contrastive_solver", "task-1")

    result = load_tasks(dataset_root)

    assert [task.id for task in result.tasks] == [
        "contrastive_solver__task-1",
        "multi_solver__task-1",
        "multi_solver__task-2",
    ]
    assert [task.relative_path for task in result.tasks] == [
        "contrastive_solver/task-1",
        "multi_solver/task-1",
        "multi_solver/task-2",
    ]
    assert {entry["fingerprint"] for entry in result.manifest_entries} == {
        task.fingerprint for task in result.tasks
    }


def test_missing_instruction_rejected(tasks_dir: Path) -> None:
    write_task(tasks_dir, "broken", instruction=None)
    with pytest.raises(TaskLoadError, match="instruction.md"):
        load_tasks(tasks_dir)


def test_missing_tests_rejected(tasks_dir: Path) -> None:
    task_dir = write_task(tasks_dir, "broken")
    (task_dir / "tests" / "test.sh").unlink()
    with pytest.raises(TaskLoadError, match="verifier script"):
        load_tasks(tasks_dir)


def test_missing_environment_definition_rejected(tasks_dir: Path) -> None:
    write_task(
        tasks_dir,
        "broken",
        task_toml=(
            'schema_version = "1.2"\n\n'
            "[verifier]\ntimeout_sec = 60.0\n\n"
            "[environment]\ncpus = 1\n"
        ),
    )
    with pytest.raises(TaskLoadError, match="environment definition"):
        load_tasks(tasks_dir)


def test_invalid_toml_rejected(tasks_dir: Path) -> None:
    write_task(tasks_dir, "broken", task_toml="this is not = valid = toml")
    with pytest.raises(TaskLoadError, match="Failed to parse"):
        load_tasks(tasks_dir)


def test_empty_dataset_rejected(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(TaskLoadError, match="No Harbor tasks"):
        discover_task_dirs(empty)


UNSUPPORTED_TASKS = {
    "multi-step": (
        'schema_version = "1.2"\n[environment]\ndocker_image = "img"\n'
        '[[steps]]\nname = "one"\n',
        "multi-step",
    ),
    "separate-verifier": (
        'schema_version = "1.2"\n[environment]\ndocker_image = "img"\n'
        '[verifier]\nenvironment_mode = "separate"\n',
        "shared verification",
    ),
    "separate-verifier-env": (
        'schema_version = "1.2"\n[environment]\ndocker_image = "img"\n'
        "[verifier.environment]\ncpus = 1\n",
        "separate verifier sandbox",
    ),
    "gpu": (
        'schema_version = "1.2"\n[environment]\ndocker_image = "img"\ngpus = 1\n',
        "GPU/TPU",
    ),
    "mcp": (
        'schema_version = "1.2"\n[environment]\ndocker_image = "img"\n'
        '[[environment.mcp_servers]]\nname = "m"\ntransport = "sse"\n'
        'url = "http://localhost:1"\n',
        "MCP",
    ),
    "windows": (
        'schema_version = "1.2"\n[environment]\ndocker_image = "img"\nos = "windows"\n',
        "linux containers",
    ),
    "skills": (
        'schema_version = "1.2"\n[environment]\ndocker_image = "img"\n'
        'skills_dir = "/skills"\n',
        "skills",
    ),
    "no-network": (
        'schema_version = "1.3"\n[environment]\ndocker_image = "img"\n'
        'network_mode = "no-network"\n',
        "public network",
    ),
    "agent-network": (
        'schema_version = "1.3"\n[environment]\ndocker_image = "img"\n'
        '[agent]\nnetwork_mode = "no-network"\n',
        "public network",
    ),
    "no-internet": (
        'schema_version = "1.2"\n[environment]\ndocker_image = "img"\n'
        "allow_internet = false\n",
        "no-network execution",
    ),
    "agent-user": (
        'schema_version = "1.2"\n[environment]\ndocker_image = "img"\n'
        '[agent]\nuser = "agent"\n',
        "container default user",
    ),
    "collect-hook": (
        'schema_version = "1.3"\n[environment]\ndocker_image = "img"\n'
        '[[verifier.collect]]\ncommand = "true"\n',
        "collection hooks",
    ),
    "unknown-schema": (
        'schema_version = "9.9"\n[environment]\ndocker_image = "img"\n',
        "schema_version",
    ),
    "relative-workdir": (
        'schema_version = "1.2"\n[environment]\ndocker_image = "img"\n'
        'workdir = "app"\n',
        "absolute container path",
    ),
}


@pytest.mark.parametrize(
    ("task_toml", "match"),
    UNSUPPORTED_TASKS.values(),
    ids=list(UNSUPPORTED_TASKS),
)
def test_unsupported_tasks_rejected(
    tasks_dir: Path, task_toml: str, match: str
) -> None:
    write_task(tasks_dir, "unsupported", task_toml=task_toml)
    with pytest.raises(UnsupportedTaskError, match=match):
        load_tasks(tasks_dir)


def test_compose_environment_rejected(tasks_dir: Path) -> None:
    task_dir = write_task(tasks_dir, "compose")
    (task_dir / "environment" / "docker-compose.yaml").write_text(
        "services:\n  main:\n    image: busybox\n", encoding="utf-8"
    )
    with pytest.raises(UnsupportedTaskError, match="compose"):
        load_tasks(tasks_dir)


@pytest.mark.filterwarnings("ignore:The 'memory' field is deprecated")
@pytest.mark.filterwarnings("ignore:The 'storage' field is deprecated")
def test_legacy_schema_and_resource_migration(dataset_root: Path) -> None:
    legacy_toml = (
        'version = "1.0"\n\n'
        "[verifier]\ntimeout_sec = 600.0\n\n"
        "[agent]\ntimeout_sec = 600.0\n\n"
        "[environment]\n"
        'docker_image = "aweaiteam/calibforge:foo"\n'
        "cpus = 1\n"
        'memory = "2G"\n'
        'storage = "5G"\n'
    )
    write_task(dataset_root, "legacy", task_toml=legacy_toml)

    result = load_tasks(dataset_root)
    config = result.tasks[0].config

    assert config.schema_version == "1.0"
    assert config.environment.memory_mb == 2048
    assert config.environment.storage_mb == 5 * 1024
    assert config.environment.cpus == 1


def test_conflicting_legacy_resources_rejected(dataset_root: Path) -> None:
    conflicting = (
        'schema_version = "1.0"\n\n'
        "[environment]\n"
        'docker_image = "img"\n'
        'memory = "2G"\n'
        "memory_mb = 4096\n"
    )
    write_task(dataset_root, "legacy", task_toml=conflicting)
    with pytest.raises(TaskLoadError, match="Conflicting"):
        load_tasks(dataset_root)


def test_id_collisions_rejected(dataset_root: Path) -> None:
    write_task(dataset_root / "a b", "task")
    write_task(dataset_root / "a-b", "task")
    with pytest.raises(TaskLoadError, match="collision"):
        load_tasks(dataset_root)


def test_derive_instance_id_is_deterministic_and_safe() -> None:
    assert derive_instance_id("multi_solver/task-1") == "multi_solver__task-1"
    assert derive_instance_id("weird dir/with spaces") == "weird-dir__with-spaces"
    assert derive_instance_id("a/b c/d") == "a__b-c__d"
    long_id = derive_instance_id("x" * 300)
    assert len(long_id) <= 160


def test_selection_applies_before_limit(dataset_root: Path, tmp_path: Path) -> None:
    for name in ("task-a", "task-b", "task-c"):
        write_task(dataset_root, name)
    selection_file = tmp_path / "select.txt"
    selection_file.write_text("# comment\ntask-b\ntask-c\n", encoding="utf-8")

    result = load_tasks(
        dataset_root,
        selected_instances_file=str(selection_file),
        eval_limit=1,
    )
    assert [task.id for task in result.tasks] == ["task-b"]


def test_selection_by_relative_path(dataset_root: Path, tmp_path: Path) -> None:
    write_task(dataset_root / "subset", "task-a")
    write_task(dataset_root / "subset", "task-b")
    selection_file = tmp_path / "select.txt"
    selection_file.write_text("subset/task-b\n", encoding="utf-8")

    result = load_tasks(dataset_root, selected_instances_file=str(selection_file))
    assert [task.id for task in result.tasks] == ["subset__task-b"]


def test_unknown_selection_entry_rejected(dataset_root: Path, tmp_path: Path) -> None:
    write_task(dataset_root, "task-a")
    selection_file = tmp_path / "select.txt"
    selection_file.write_text("does-not-exist\n", encoding="utf-8")
    with pytest.raises(TaskLoadError, match="did not match"):
        load_tasks(dataset_root, selected_instances_file=str(selection_file))


def test_n_limit_zero_means_all(dataset_root: Path) -> None:
    for name in ("task-a", "task-b"):
        write_task(dataset_root, name)
    assert len(load_tasks(dataset_root, eval_limit=0).tasks) == 2


def test_fingerprint_tracks_relevant_inputs(dataset_root: Path) -> None:
    task_dir = write_task(
        dataset_root,
        "task",
        environment_files={"notes/setup.txt": "one"},
        tests={"test.sh": "#!/bin/bash\necho 1 > /logs/verifier/reward.txt\n"},
        extra_files={"README.md": "unrelated\n"},
    )
    first = load_tasks(dataset_root).tasks[0].fingerprint
    assert first == load_tasks(dataset_root).tasks[0].fingerprint

    (task_dir / "instruction.md").write_text("Do something else.", encoding="utf-8")
    second = load_tasks(dataset_root).tasks[0].fingerprint
    assert second != first

    (task_dir / "tests" / "test.sh").write_text(
        "#!/bin/bash\necho 0 > /logs/verifier/reward.txt\n", encoding="utf-8"
    )
    third = load_tasks(dataset_root).tasks[0].fingerprint
    assert third != second

    (task_dir / "environment" / "notes" / "setup.txt").write_text(
        "two", encoding="utf-8"
    )
    fourth = load_tasks(dataset_root).tasks[0].fingerprint
    assert fourth != third

    (task_dir / "README.md").write_text("different unrelated\n", encoding="utf-8")
    fifth = load_tasks(dataset_root).tasks[0].fingerprint
    assert fifth == fourth


def test_compute_task_fingerprint_excludes_solution(dataset_root: Path) -> None:
    task_dir = write_task(dataset_root, "task")
    config = load_task_dir(task_dir, dataset_root, is_single=False).config
    base = compute_task_fingerprint(task_dir, config, "Do the thing.")

    (task_dir / "solution").mkdir()
    (task_dir / "solution" / "solve.sh").write_text("#!/bin/bash\n", encoding="utf-8")
    assert compute_task_fingerprint(task_dir, config, "Do the thing.") == base


def test_resume_compatible_roundtrip(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    manifest_hash = "manifest-abc"
    settings_hash = compute_run_settings_hash({"workspace_backend": "docker"})

    # Fresh directory: always accepted.
    check_resume_compatible(
        output_dir, manifest_hash=manifest_hash, run_settings_hash=settings_hash
    )

    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "details": {
                    "harbor": {
                        "manifest_hash": manifest_hash,
                        "run_settings_hash": settings_hash,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    # Same task set/settings: accepted.
    check_resume_compatible(
        output_dir, manifest_hash=manifest_hash, run_settings_hash=settings_hash
    )

    with pytest.raises(HarborTaskError, match="Task set/content changed"):
        check_resume_compatible(
            output_dir,
            manifest_hash="different",
            run_settings_hash=settings_hash,
        )
    with pytest.raises(HarborTaskError, match="run settings changed"):
        check_resume_compatible(
            output_dir,
            manifest_hash=manifest_hash,
            run_settings_hash="different",
        )


def test_resume_rejects_non_harbor_metadata(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    (output_dir / "metadata.json").write_text(
        json.dumps({"details": {}}), encoding="utf-8"
    )
    with pytest.raises(HarborTaskError, match="without Harbor provenance"):
        check_resume_compatible(
            output_dir,
            manifest_hash="x",
            run_settings_hash="y",
        )


def test_reference_version_is_pinned() -> None:
    assert HARBOR_REFERENCE_VERSION == "0.9.0"


def test_resolve_env_templates(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HARBOR_UNIT_VAR", "from-host")
    resolved = resolve_env_templates(
        {
            "FROM_HOST": "${HARBOR_UNIT_VAR}",
            "WITH_DEFAULT": "${HARBOR_UNSET_VAR:-fallback}",
            "LITERAL": "plain-value",
            "NOT_ONLY_TEMPLATE": "prefix-${HARBOR_UNIT_VAR}",
        }
    )
    assert resolved == {
        "FROM_HOST": "from-host",
        "WITH_DEFAULT": "fallback",
        "LITERAL": "plain-value",
        "NOT_ONLY_TEMPLATE": "prefix-${HARBOR_UNIT_VAR}",
    }


def test_resolve_env_templates_missing_required() -> None:
    with pytest.raises(TaskLoadError, match="HARBOR_DEFINITELY_UNSET"):
        resolve_env_templates({"X": "${HARBOR_DEFINITELY_UNSET}"})


def test_should_upload_environment_dir(tmp_path: Path) -> None:
    from benchmarks.harbor.task import should_upload_environment_dir

    env_dir = tmp_path / "environment"
    env_dir.mkdir()
    # Empty dir: nothing to upload.
    assert not should_upload_environment_dir(env_dir, docker_image="img")
    (env_dir / "setup.sh").write_text("#!/bin/bash\n", encoding="utf-8")
    # Prebuilt image without a build spec: Harbor uploads the runtime files.
    assert should_upload_environment_dir(env_dir, docker_image="img")
    # Built tasks already contain their environment/ contents in the image.
    (env_dir / "Dockerfile").write_text("FROM debian\n", encoding="utf-8")
    assert not should_upload_environment_dir(env_dir, docker_image="img")
    (env_dir / "Dockerfile").unlink()
    (env_dir / "docker-compose.yaml").write_text(
        "services:\n  main:\n    image: busybox\n", encoding="utf-8"
    )
    assert not should_upload_environment_dir(env_dir, docker_image="img")
    # No prebuilt image at all: nothing to upload over.
    assert not should_upload_environment_dir(env_dir, docker_image=None)


def test_example_fixture_loads() -> None:
    from benchmarks.harbor.tests.conftest import REPO_ROOT

    example = REPO_ROOT / "benchmarks" / "harbor" / "examples" / "hello-task"
    result = load_tasks(example)
    assert len(result.tasks) == 1
    task = result.tasks[0]
    assert task.id == "hello-task"
    assert task.config.schema_version == "1.2"
    assert task.config.environment.docker_image == "debian:bookworm-slim"
    assert task.test_script == "test.sh"
    assert "hello harbor" in task.instruction
