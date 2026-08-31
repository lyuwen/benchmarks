"""
z021data execution-based judge.

Evaluates an agent patch by reproducing the validation procedure implemented in
``benchmarks/z021data/core/validate_test_images_complete.py``:

  1. Start the instance Docker image (``sleep infinity`` CMD, entrypoint runs).
  2. Checkout ``parent_commit`` and clean the tree.
  3. Apply the ground-truth test patch (test-file part of ``patch``) and
     ``f2p_patch``, and write the generated test file from ``test_script``.
  4. Run the test command (``setup`` / pre-command first, then ``test_command``)
     BEFORE the agent's fix -> Result 1.
  5. Apply the agent's git patch (with test-file edits stripped) -> the "fix".
  6. Run the same test command AFTER the fix -> Result 2.
  7. FAIL_TO_PASS = tests failing/erroring before that pass after;
     PASS_TO_FAIL = tests passing before that fail/error after.
  8. Resolved iff FAIL_TO_PASS > 0 and PASS_TO_FAIL == 0.

All patch-application, command-building and log-parsing logic is reused
verbatim from ``core.validate_test_images_complete`` and ``core.python_logparse``
so the judge stays in lockstep with the offline validator. The heavy Docker
dependency (docker-py) is imported lazily; if it is unavailable the judge is
skipped (returns ``None``) rather than failing the run.
"""

from __future__ import annotations

import io
import logging
import tarfile
import traceback
from pathlib import Path
from typing import Any

from pydantic import Field

from benchmarks.utils.execution_judge import ExecutionBasedJudge, register_judge

logger = logging.getLogger(__name__)

_CORE_DIR = Path(__file__).resolve().parent / "core"

# Generated test file name (mirrors core.F2P_TEST_FILE).
F2P_TEST_FILE = "test_f2p_generated.py"


def _has_registry_host(image_url: str) -> bool:
    """True if ``image_url`` already carries a registry host component.

    Docker's rule: the first '/'-separated component is a registry host when it
    contains a '.' or ':' or equals 'localhost'. Otherwise the reference is a
    bare ``namespace/repo`` on the default registry and a prefix may be added.
    """
    head, sep, _ = image_url.partition("/")
    if not sep:
        return False
    return "." in head or ":" in head or head == "localhost"


def _resolve_image_url(image_url: str, prefix: str | None = None) -> str:
    """Prepend a registry/namespace prefix to a bare ``image_url``.

    When no prefix is given, or when ``image_url`` already carries a registry
    host (e.g. ``myregistry.com/ns/repo:tag``), the URL is returned unchanged
    (still lowercased, since the images are published lowercase). Otherwise the
    prefix is prepended, matching the offline validator's ``f"{prefix}/{url}"``.
    """
    if not prefix or _has_registry_host(image_url):
        return image_url.lower()
    return f"{prefix.rstrip('/')}/{image_url}".lower()


def _load_core():
    """Lazy-import the validator core (which imports docker-py at module top).

    Returns the ``validate_test_images_complete`` module, or raises
    ImportError if docker-py / the core module is unavailable.
    """
    import sys

    if str(_CORE_DIR) not in sys.path:
        sys.path.insert(0, str(_CORE_DIR))
    from benchmarks.z021data.core import validate_test_images_complete as core

    return core


@register_judge("z021data")
class Z021DataJudge(ExecutionBasedJudge):
    """Judge that runs the z021data test procedure in the instance image.

    Reuses the offline validator's patch-apply / run / parse helpers so a
    passing judge here means the same thing as a passing offline validation.
    """

    docker_image_prefix: str | None = Field(
        default=None,
        description="Registry/namespace prefix prepended to a bare image_url",
    )
    rm_image: bool = Field(
        default=False,
        description="Remove Docker image after evaluation",
    )
    force_rebuild: bool = Field(
        default=False,
        description="Unused; accepted for CLI compatibility",
    )

    def judge(
        self,
        instance_id: str,
        git_patch: str,
        instance_data: dict[str, Any],
    ) -> bool | None:
        if not git_patch or not git_patch.strip():
            logger.warning("Empty or missing git patch for %s", instance_id)
            return False

        try:
            return self._run_eval(instance_id, git_patch, instance_data)
        except Exception as e:
            logger.error(
                "Z021DataJudge failed for %s: %s\n%s",
                instance_id,
                e,
                traceback.format_exc(),
            )
            return None

    def _run_eval(
        self,
        instance_id: str,
        git_patch: str,
        instance_data: dict[str, Any],
    ) -> bool | None:
        try:
            core = _load_core()
        except (ImportError, ModuleNotFoundError) as e:
            logger.warning(
                "z021data core (docker-py) is not available (%s). Skipping judge.",
                e,
            )
            return None

        import docker

        # ---- Resolve inputs (mirror core.test_image field reads) --------------
        image_url = instance_data.get("image_url", "").strip()
        if not image_url:
            logger.error("Instance %s has no image_url", instance_id)
            return None
        image_name = _resolve_image_url(image_url, self.docker_image_prefix)

        base_commit = instance_data.get("parent_commit", "") or instance_data.get(
            "base_commit", ""
        )
        workdir = instance_data.get("workdir", "/workspace") or "/workspace"

        # Ground-truth test artifacts.
        gold_patch = core.normalize_patch(instance_data.get("patch", "") or "")
        f2p_patch = core.normalize_patch(instance_data.get("f2p_patch", "") or "")
        # Generated test body: `test_script` (same content as `f2p_script`).
        f2p_script = instance_data.get("test_script", "") or ""

        # The test-file part of the gold patch is applied as the ground-truth
        # test patch; the source part is the gold fix (NOT used here — the
        # agent's patch replaces it).
        _gold_source, gold_test_patch = core.separate_test_and_source_patch(gold_patch)

        # The agent's patch is the candidate fix. Strip any test-file edits so a
        # candidate cannot smuggle changes to the ground-truth test files.
        agent_source, _agent_test = core.separate_test_and_source_patch(
            core.normalize_patch(git_patch)
        )

        # ---- Test command: setup (pre-command) then test_command --------------
        parsed = instance_data.get("pre_commands_parsed") or {}
        if isinstance(parsed, str):
            import json

            try:
                parsed = json.loads(parsed)
            except Exception:
                parsed = {}
        setup_cmd = (parsed.get("setup") or "").strip() if isinstance(parsed, dict) else ""

        # Bare test command, preferring the split-out `test_command`.
        test_command = (
            instance_data.get("test_command")
            or (parsed.get("command") if isinstance(parsed, dict) else "")
            # or instance_data.get("pre_commands")
            or ""
        ).strip()
        runner = core.detect_runner(test_command)

        logger.info(
            "Z021DataJudge %s: image=%s workdir=%s runner=%s base=%s",
            instance_id,
            image_name,
            workdir,
            runner,
            base_commit,
        )

        client = docker.from_env(timeout=300)
        container = None
        log_content: list[str] = []

        try:
            container = client.containers.run(
                image_name,
                "sleep infinity",
                detach=True,
                remove=False,
                mem_limit="24g",
                memswap_limit="24g",
                stop_signal="SIGKILL",
            )
            container.reload()
            if container.status != "running":
                raise RuntimeError(
                    f"Container failed to start, status: {container.status}"
                )

            if not core.wait_for_container_ready(container, workdir):
                logger.warning(
                    "Z021DataJudge %s: /opt/.container_ready not seen before timeout; "
                    "proceeding",
                    instance_id,
                )

            import shlex

            def exec_cmd(cmd: str, description: str, timeout: int = 600):
                result = container.exec_run(
                    [
                        "bash",
                        "-c",
                        f"cd {workdir} && timeout {timeout} bash -c {shlex.quote(cmd)}",
                    ],
                    demux=False,
                    workdir=workdir,
                )
                output = (
                    result.output.decode("utf-8", errors="replace")
                    if result.output
                    else ""
                )
                if result.exit_code == 124:
                    output += f"\n[TIMEOUT] Command timed out after {timeout}s"
                return result.exit_code, output

            # Step 1: checkout base commit and clean tree.
            if base_commit:
                exec_cmd(f"git checkout {base_commit} -f", "Checkout base commit")
                exec_cmd("git checkout .", "Clean working directory")
                exec_cmd("git clean -fd", "Clean working directory")

            # Step 2/3: apply ground-truth test patch + f2p_patch.
            core.apply_patch(
                container, gold_test_patch, "test_patch.diff", exec_cmd, log_content, "Step 2"
            )
            core.apply_patch(
                container, f2p_patch, "f2p_patch.diff", exec_cmd, log_content, "Step 3"
            )

            # Step 4: write the generated test file into the workdir, the deepest
            # `cd` target, and the repo's own test dir (so its conftest applies).
            dest_dirs = [workdir]
            cd_dir = core.deepest_cd_dir(test_command)
            if cd_dir and cd_dir not in dest_dirs:
                dest_dirs.append(cd_dir)

            _, repo_test_dir_out = exec_cmd(
                "ls -d tests test Tests Test testing 2>/dev/null | head -1 || echo ''",
                "Detect repo test directory",
            )
            repo_test_dir = repo_test_dir_out.strip()
            if repo_test_dir and repo_test_dir != ".":
                test_dest = f"{workdir.rstrip('/')}/{repo_test_dir}"
                if test_dest not in dest_dirs:
                    dest_dirs.append(test_dest)

            if f2p_script.strip():
                for dest in dest_dirs:
                    script_bytes = f2p_script.encode("utf-8")
                    tar_stream = io.BytesIO()
                    with tarfile.open(fileobj=tar_stream, mode="w") as tar:
                        info = tarfile.TarInfo(name=F2P_TEST_FILE)
                        info.size = len(script_bytes)
                        tar.addfile(info, io.BytesIO(script_bytes))
                    tar_stream.seek(0)
                    try:
                        container.put_archive(dest, tar_stream)
                    except Exception as e:
                        logger.debug(
                            "Failed to write %s to %s: %s", F2P_TEST_FILE, dest, e
                        )

            if runner in ("pytest", "django") and repo_test_dir and repo_test_dir != ".":
                test_file_name = f"{repo_test_dir.rstrip('/')}/{F2P_TEST_FILE}"
            else:
                test_file_name = F2P_TEST_FILE

            _, test_dir_output = exec_cmd(
                "ls -d tests test Tests Test testing 2>/dev/null | head -1 || echo ''",
                "Detect test directory",
            )
            test_dir = test_dir_output.strip() if test_dir_output.strip() else "."

            run_timeout = 1800 if runner != "none" else 1200

            def run_tests(label: str):
                run_cmd = core.build_run_command(
                    runner, test_command, test_file_name, test_dir, run_all=False
                )
                # Honor user guidance: run the pre-command (setup) before the
                # test command in the same shell so services/env are ready.
                if setup_cmd:
                    run_cmd = f"{setup_cmd} ; {run_cmd}"
                exit_code, output = exec_cmd(run_cmd, label, timeout=run_timeout)
                return exit_code, core.parse_run_output(runner, output)

            # Step 5: run BEFORE the agent's fix (Result 1).
            _ec1, result1 = run_tests("Run tests BEFORE agent patch (Result 1)")

            # Step 6: apply the agent's fix (source-only).
            core.apply_patch(
                container, agent_source, "source_patch.diff", exec_cmd, log_content, "Step 6"
            )

            # Step 7: run AFTER the agent's fix (Result 2).
            _ec2, result2 = run_tests("Run tests AFTER agent patch (Result 2)")

            # Step 8: transitions.
            if runner == "unsupported":
                logger.warning(
                    "Z021DataJudge %s: unsupported runner (opaque test_command); "
                    "cannot compute F2P",
                    instance_id,
                )
                return None

            before_failed = set(result1["failed"] + result1["errors"])
            before_passed = set(result1["passed"])
            after_failed = set(result2["failed"] + result2["errors"])
            after_passed = set(result2["passed"])

            fail_to_pass = before_failed & after_passed
            pass_to_fail = before_passed & after_failed

            resolved = len(fail_to_pass) > 0 and len(pass_to_fail) == 0
            logger.info(
                "Z021DataJudge %s: resolved=%s F2P=%d P2F=%d "
                "(before p/f/e=%d/%d/%d, after p/f/e=%d/%d/%d)",
                instance_id,
                resolved,
                len(fail_to_pass),
                len(pass_to_fail),
                result1["passed_count"],
                result1["failed_count"],
                result1["error_count"],
                result2["passed_count"],
                result2["failed_count"],
                result2["error_count"],
            )
            return resolved

        finally:
            if self.rm_image:
                # Full teardown: stop/remove the container AND the image.
                core.cleanup_container_and_image(
                    container, client, image_name, log_content
                )
            elif container is not None:
                # Keep the image (cheaper reruns); just remove the container.
                try:
                    container.remove(force=True)
                except Exception as e:
                    logger.debug(
                        "Z021DataJudge %s: container cleanup failed: %s",
                        instance_id,
                        e,
                    )
