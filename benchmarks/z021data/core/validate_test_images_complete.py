#!/usr/bin/env python3
"""
Validate test images by running the complete test collection workflow.

This script follows the same workflow as test_collector:
1. Apply f2p_patch and f2p_script (if exists)
2. Run pytest -> Result 1 (before fix)
3. Apply source patch (the fix)
4. Run pytest -> Result 2 (after fix)
5. Calculate PASS_TO_PASS and FAIL_TO_PASS
6. Compare with expected results from JSONL
"""

import argparse
import docker
import io
import json
import os
import re
import shlex
import sys
import tarfile
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

# Import unified log parsing module. The canonical copy lives alongside this
# file in core/; older layouts kept it under a sibling gen_test/ directory, so
# add both to sys.path and import whichever resolves first.
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'gen_test'))
import python_logparse


# ============================================================================
# Generated test file name
# ============================================================================
F2P_TEST_FILE = "test_f2p_generated.py"


# ============================================================================
# Runner detection / command building
# ----------------------------------------------------------------------------
# These helpers are shared with gen_test/f2p_validation.py (the subprocess-based
# mirror of this workflow). They detect how a repo runs its tests from
# `pre_commands`, build the concrete run command with the generated test target
# injected, and parse the resulting output (delegated to python_logparse).
# ============================================================================

# Matches pytest / py.test / python -m pytest
_PYTEST_INVOCATION_RE = re.compile(
    r'(?P<inv>(?:python[0-9.]*\s+-m\s+)?(?:py\.test|pytest))\b'
)
_DJANGO_INVOCATION_RE = re.compile(r'manage\.py\s+test\b')
_DJANGO_FULL_INVOCATION_RE = re.compile(
    r'(?:coverage\s+run(?:\s+[^\s]+)*\s+|python[0-9.]*\s+)?'
    r'(?:\./)?manage\.py\s+test\b[^\n;&|<>]*'
)
_PYTEST_REPORT_FLAGS = "-rA -v --tb=short --continue-on-collection-errors"
# Options that take a following value; their value must NOT be stripped as a
# positional test target when we replace the repo's pytest targets.
_PYTEST_VALUE_OPTS = {
    '-p', '-o', '-m', '-k', '-c', '-W', '-n',
    '--deselect', '--ignore', '--ignore-glob', '--rootdir', '--confcutdir',
    '--junitxml', '--junit-xml', '--override-ini', '--maxfail', '--durations',
    '--ds', '--cov', '--cov-report', '--basetemp', '--import-mode',
    '--dist', '--tx', '--reruns', '--reruns-delay',
}


def _contains_heredoc(cmd: str) -> bool:
    return bool(re.search(r'<<-?\s*[\'"]?\w+', cmd))


def detect_runner(pre_commands: str) -> str:
    """Classify pre_commands as none/pytest/django/unittest/unsupported."""
    if not pre_commands or not pre_commands.strip():
        return 'none'
    if _contains_heredoc(pre_commands):
        return 'unsupported'
    if _DJANGO_INVOCATION_RE.search(pre_commands):
        return 'django'
    if _PYTEST_INVOCATION_RE.search(pre_commands):
        return 'pytest'
    if re.search(r'python[0-9.]*\s+-m\s+unittest\b', pre_commands):
        return 'unittest'
    return 'unsupported'


def deepest_cd_dir(pre_commands: str) -> Optional[str]:
    """Return the last absolute `cd` target in pre_commands, or None."""
    matches = re.findall(r'\bcd\s+(/[^\s;&|]+)', pre_commands or "")
    return matches[-1] if matches else None


def _strip_conda_run(cmd: str) -> str:
    """Drop any `conda run [-opts] -n <env>` prefix so the command runs bare.

    Migrated images bake the correct testbed ``PATH``/``CONDA_PREFIX`` into the
    image config, so a bare ``python``/``pytest`` already resolves to the testbed
    interpreter under ``docker exec``. Keeping ``conda run`` on top only adds a
    wrapper that swallows output and diverges from how the build project runs
    tests, so we strip it and rely on the image ENV instead.
    """
    if 'conda run' not in cmd:
        return cmd
    return re.sub(
        r'conda\s+run\s+(?:--no-capture-output\s+)?(?:-n|--name)\s+\S+\s+',
        '',
        cmd,
    )


def _replace_django_invocation_with_pytest(cmd: str, pytest_tail: str) -> str:
    replacement = f"python -m pytest {pytest_tail}"
    m = _DJANGO_FULL_INVOCATION_RE.search(cmd)
    if not m:
        return f"{cmd} && {replacement}"
    return cmd[:m.start()] + replacement + cmd[m.end():]


def _iter_arg_tokens(s: str):
    """Yield (unquoted_value, start, end) for each whitespace-separated token."""
    i, n = 0, len(s)
    while i < n:
        while i < n and s[i] in ' \t\n':
            i += 1
        if i >= n:
            break
        start = i
        buf = []
        while i < n and s[i] not in ' \t\n':
            c = s[i]
            if c == '\\' and i + 1 < n:
                buf.append(s[i + 1]); i += 2; continue
            if c == "'":
                i += 1
                while i < n and s[i] != "'":
                    buf.append(s[i]); i += 1
                i += 1; continue
            if c == '"':
                i += 1
                while i < n and s[i] != '"':
                    if s[i] == '\\' and i + 1 < n:
                        buf.append(s[i + 1]); i += 2; continue
                    buf.append(s[i]); i += 1
                i += 1; continue
            buf.append(c); i += 1
        yield ''.join(buf), start, i


def _looks_like_test_path(token: str) -> bool:
    return (
        '/' in token
        or token.endswith('.py')
        or '::' in token
        or token in {'tests', 'test', 'Tests', 'Test', 'testing'}
    )


def build_pytest_command(cmd: str, insertion: str) -> str:
    """Inject `insertion` after the pytest invocation, stripping the repo's own
    positional test targets (e.g. `tests/`) while keeping options + shell tail."""
    matches = list(_PYTEST_INVOCATION_RE.finditer(cmd))
    if not matches:
        return cmd + insertion
    p_end = matches[-1].end()
    args = cmd[p_end:]

    stop_at = None
    removal = []
    prev_was_value_opt = False
    for val, s, e in _iter_arg_tokens(args):
        raw = args[s:e]
        if val in ('&&', '||', ';', '|', '&') or re.search(r'[<>]', raw):
            stop_at = s
            break
        if prev_was_value_opt:
            prev_was_value_opt = False
            continue
        if val.startswith('-'):
            if val in _PYTEST_VALUE_OPTS:
                prev_was_value_opt = True
            continue
        if _looks_like_test_path(val):
            removal.append((s, e))

    head_args = args if stop_at is None else args[:stop_at]
    tail = '' if stop_at is None else args[stop_at:]

    kept = []
    last = 0
    for s, e in removal:
        kept.append(head_args[last:s])
        last = e
    kept.append(head_args[last:])
    new_args = ''.join(kept)
    return cmd[:p_end] + insertion + new_args + tail


def _run_all_target(test_file_name: str, test_dir: str, run_all: bool) -> str:
    """Extra pytest target (the repo test dir) appended when run_all is set."""
    if run_all and test_dir and test_dir not in ('.', '', test_file_name):
        return f" {test_dir}"
    return ""


def build_run_command(runner: str, pre_commands: str, test_file_name: str,
                      test_dir: str, run_all: bool = False,
                      collect_only: bool = False) -> str:
    """Build the concrete test run command for a detected runner.

    The generated test file (``test_file_name``) is injected as the pytest
    target. When ``run_all`` is True the repo's own ``test_dir`` is appended so
    the full suite runs alongside the generated tests.
    """
    extra = _run_all_target(test_file_name, test_dir, run_all)

    if runner == 'none':
        flags = ("--collect-only -q --continue-on-collection-errors" if collect_only
                 else "-rA -v --tb=short --maxfail=999 --continue-on-collection-errors")
        return f"python -m pytest {test_file_name}{extra} {flags}"

    cmd = _strip_conda_run(pre_commands)

    if runner == 'pytest':
        if collect_only:
            insertion = f" {test_file_name}{extra} --collect-only -q --continue-on-collection-errors"
        else:
            insertion = f" {test_file_name}{extra} {_PYTEST_REPORT_FLAGS}"
        return build_pytest_command(cmd, insertion)

    if runner == 'django':
        if collect_only:
            pytest_tail = f"{test_file_name}{extra} --collect-only -q --continue-on-collection-errors"
        else:
            pytest_tail = f"{test_file_name}{extra} {_PYTEST_REPORT_FLAGS} --maxfail=999"
        return _replace_django_invocation_with_pytest(cmd, pytest_tail)

    # unsupported: run verbatim
    return cmd


def parse_pytest_output(output: str) -> dict:
    return python_logparse.parse_pytest_output(output)


def parse_unittest_output(output: str) -> dict:
    return python_logparse.parse_unittest_output(output)


def parse_run_output(runner: str, output: str) -> dict:
    return python_logparse.parse_run_output(runner, output)


# ============================================================================
# Container Readiness
# ============================================================================
#
# Test commands are exec'd bare (no `conda activate`, no `conda run`): the image
# config carries the correct testbed PATH/CONDA_PREFIX and `docker exec` injects
# it, exactly like the build project's validation flow. The only thing we must
# wait for is the entrypoint finishing service startup.


def wait_for_container_ready(container, workdir: str = "/app", timeout: int = 120) -> bool:
    """Poll for the entrypoint's readiness marker (/opt/.container_ready).

    Migrated images (build=static/commit=dynamic) ship an ENTRYPOINT that starts
    per-repo services (postgres/redis) via /opt/setup.sh and only then writes
    /opt/.container_ready. Because ``sleep infinity`` is passed as the CMD (not
    an entrypoint override), that ENTRYPOINT still runs at container start; we
    must wait for the marker before exec-ing test commands so we don't race a
    slow-starting service.

    Legacy images without an entrypoint never create the marker. To stay
    backward compatible we treat "no /opt/setup.sh present" as "nothing to wait
    for" and return immediately; otherwise we poll until the marker appears or
    the timeout elapses (returning False on timeout, callers proceed anyway).
    """
    # If the image has no setup.sh, there is no service startup to wait for.
    probe = container.exec_run(["bash", "-c", "test -f /opt/setup.sh"], workdir=workdir)
    if probe.exit_code != 0:
        return True

    deadline = time.time() + timeout
    while time.time() < deadline:
        res = container.exec_run(
            ["bash", "-c", "test -f /opt/.container_ready"], workdir=workdir
        )
        if res.exit_code == 0:
            return True
        time.sleep(2)
    return False


# ============================================================================
# Patch Application Management
# ============================================================================

# Comprehensive patch application strategies
PATCH_STRATEGIES = [
    ("patch --batch --fuzz=5 -p1 -i {patch_file}", False),
    ("git apply --verbose {patch_file}", False),
    ("git apply --verbose --ignore-space-change --ignore-whitespace {patch_file}", False),
    ("git apply --verbose --reject {patch_file}", True),
    ("git apply --verbose --reject --ignore-space-change --ignore-whitespace {patch_file}", True),
    ("git apply --verbose --reject --ignore-space-change --ignore-whitespace --allow-empty {patch_file}", True),
]


def write_patch_to_container(container, patch_content: str, patch_filename: str) -> None:
    """Write a patch file to the container's /tmp directory."""
    patch_bytes = patch_content.encode('utf-8')
    tar_stream = io.BytesIO()
    with tarfile.open(fileobj=tar_stream, mode='w') as tar:
        info = tarfile.TarInfo(name=patch_filename)
        info.size = len(patch_bytes)
        tar.addfile(info, io.BytesIO(patch_bytes))
    tar_stream.seek(0)
    container.put_archive('/tmp', tar_stream)


def apply_patch(container, patch_content: str, patch_name: str, exec_cmd_func, log_content: List[str], step_name: str) -> bool:
    """
    Apply a patch using multiple strategies with proper cleanup.

    IMPORTANT: This function preserves previously applied patches and only reverts
    the current patch attempt if it fails.

    Strategy:
    1. Save current working tree state to a patch file
    2. Try to apply the new patch with different strategies
    3. If a strategy fails, restore the saved state and try next strategy
    4. If all strategies fail, restore the saved state (keeping previous patches)

    Args:
        container: Docker container object
        patch_content: The patch content as string
        patch_name: Name for the patch file (e.g., 'test_patch.diff')
        exec_cmd_func: Function to execute commands in container
        log_content: List to append log messages
        step_name: Name of the step for logging (e.g., "Step 2")

    Returns:
        bool: True if patch was applied successfully, False otherwise
    """
    if not patch_content.strip():
        return True  # Empty patch is considered successful

    # Write patch to container
    patch_file = f"/tmp/{patch_name}"
    write_patch_to_container(container, patch_content, patch_name)

    # Save current state (with all previously applied patches) to a patch file
    state_file = f"/tmp/state_before_{patch_name}"
    exec_cmd_func(
        f"git diff HEAD > {state_file}",
        f"{step_name}: Save current state before applying {patch_name}"
    )

    applied = False
    reversed_patch_detected = False
    for idx, (strategy_template, allow_reject) in enumerate(PATCH_STRATEGIES):
        # If this is not the first attempt, restore to the state before this patch
        if idx > 0:
            # First, reset to HEAD to clear any changes
            exec_cmd_func(
                "git reset --hard HEAD",
                f"{step_name}: Reset to HEAD before retry {idx + 1}"
            )
            # Then re-apply the saved state (all previous patches)
            exec_cmd_func(
                f"if [ -s {state_file} ]; then git apply {state_file} 2>/dev/null || true; fi",
                f"{step_name}: Restore previous patches (attempt {idx + 1})"
            )

        # Format strategy command with actual patch file path
        strategy_cmd = strategy_template.format(patch_file=patch_file)

        exit_code, output = exec_cmd_func(
            f"{strategy_cmd} 2>&1",
            f"{step_name}: Apply {patch_name} (strategy {idx + 1}: {strategy_cmd.split()[0:3]})"
        )

        # Check if application was successful
        if exit_code == 0:
            # Check for reversed patch detection
            if "Reversed (or previously applied) patch detected" in output:
                log_content.append(f"\n{'='*80}")
                log_content.append(f"WARNING: {patch_name} is a REVERSED PATCH!")
                log_content.append(f"{'='*80}")
                log_content.append(f"This patch undoes previous changes instead of applying new ones.")
                log_content.append(f"This is a DATA ISSUE - the patch file content is reversed.")
                log_content.append(f"\nReverting this patch application to preserve previous changes...")
                log_content.append(f"{'='*80}\n")

                # Revert the reversed patch by resetting and restoring previous state
                exec_cmd_func(
                    "git reset --hard HEAD",
                    f"{step_name}: Reset to HEAD to undo reversed patch"
                )
                exec_cmd_func(
                    f"if [ -s {state_file} ]; then git apply {state_file}; fi",
                    f"{step_name}: Restore previous patches after detecting reversed patch"
                )

                # Mark as reversed patch detected and break to skip this patch
                reversed_patch_detected = True
                log_content.append(f"{patch_name} was SKIPPED due to reversed patch detection")
                break

            if allow_reject:
                # For --reject strategies, check if there are .rej files
                check_exit, check_output = exec_cmd_func(
                    "find /app -name '*.rej' -type f 2>/dev/null",
                    f"{step_name}: Check for .rej files"
                )
                if check_output.strip():
                    log_content.append(f"{patch_name} has .rej files, trying next strategy")
                    # Clean up .rej files
                    exec_cmd_func(
                        "find /app -name '*.rej' -type f -delete 2>/dev/null || true",
                        f"{step_name}: Clean up .rej files"
                    )
                    continue  # Try next strategy
            applied = True
            log_content.append(f"{patch_name} applied successfully with strategy {idx + 1}: {strategy_cmd.split()[0:3]}")

            # Debug: Show patch details after successful application
            # Show git diff statistics
            _, diff_stat = exec_cmd_func(
                "git diff HEAD --stat",
                f"{step_name}: Show diff statistics after applying {patch_name}"
            )
            log_content.append(f"=== Patch Statistics for {patch_name} ===")
            log_content.append(diff_stat.strip() if diff_stat.strip() else "No changes detected")

            # Show modified files
            _, modified_files = exec_cmd_func(
                "git diff HEAD --name-only",
                f"{step_name}: List modified files after applying {patch_name}"
            )
            if modified_files.strip():
                log_content.append(f"=== Modified Files ({patch_name}) ===")
                for file in modified_files.strip().split('\n'):
                    log_content.append(f"  - {file}")

            # Show brief diff summary
            _, diff_summary = exec_cmd_func(
                "git diff HEAD --numstat | head -20",
                f"{step_name}: Show diff numstat after applying {patch_name}"
            )
            if diff_summary.strip():
                log_content.append(f"=== Diff Summary ({patch_name}) ===")
                log_content.append(diff_summary.strip())

            log_content.append(f"=== End of {patch_name} Debug Info ===")
            break

    if not applied and not reversed_patch_detected:
        log_content.append(f"Warning: {patch_name} application failed with all strategies")
        # Restore to the state before attempting this patch (keep previous patches intact)
        exec_cmd_func(
            "git reset --hard HEAD",
            f"{step_name}: Reset to HEAD after all strategies failed"
        )
        exec_cmd_func(
            f"if [ -s {state_file} ]; then git apply {state_file} 2>/dev/null || true; fi",
            f"{step_name}: Restore previous patches after failure"
        )

    # Clean up: delete the state file after use
    exec_cmd_func(
        f"rm -f {state_file}",
        f"{step_name}: Clean up state file {state_file}"
    )

    return applied


# ============================================================================
# Utility Functions
# ============================================================================

def load_test_data_from_jsonl(jsonl_file: Path) -> List[Dict]:
    """Load all test data from JSONL file.

    Args:
        jsonl_file: Path to JSONL file

    Returns:
        List of test data dictionaries, each containing instance_id and test metadata.
    """
    if not jsonl_file.exists():
        print(f"Error: JSONL file not found: {jsonl_file}")
        return []

    test_data_list = []
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                if 'instance_id' not in data:
                    print(f"Warning: Line {line_num} missing 'instance_id', skipping")
                    continue
                test_data_list.append(data)
            except json.JSONDecodeError as e:
                print(f"Warning: Line {line_num} invalid JSON: {e}")
                continue

    return test_data_list


def pull_docker_image(client: docker.DockerClient, image_name: str, registry_prefix: str = "", max_retries: int = 3) -> bool:
    """Pull Docker image from remote registry with retry mechanism.

    Args:
        client: Docker client
        image_name: Full image name (e.g., harbor.zhejianglab.com/zj021/scaleswe:instance_id)
        registry_prefix: Registry prefix for logging
        max_retries: Maximum number of retry attempts (default: 3)

    Returns:
        True if pull successful, False otherwise
    """
    import time

    for attempt in range(1, max_retries + 1):
        try:
            if attempt == 1:
                print(f"  Pulling image: {image_name}")
            else:
                print(f"  Retry {attempt}/{max_retries}: {image_name}")

            # 创建临时客户端，设置更长的超时（10 分钟）
            temp_client = docker.from_env(timeout=600)

            # Pull image
            start_time = time.time()
            temp_client.images.pull(image_name)
            elapsed = time.time() - start_time

            print(f"  ✓ Successfully pulled: {image_name} (took {elapsed:.1f}s)")
            return True

        except docker.errors.ImageNotFound:
            print(f"  ✗ Image not found: {image_name}")
            return False  # No point retrying if image doesn't exist

        except docker.errors.APIError as e:
            error_msg = str(e).lower()
            if "timeout" in error_msg or "timed out" in error_msg:
                print(f"  ⚠ Pull timeout (attempt {attempt}): {e}")
                if attempt < max_retries:
                    wait_time = 5
                    print(f"  Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                else:
                    print(f"  ✗ Failed after {max_retries} attempts: timeout")
                    return False
            else:
                if attempt < max_retries:
                    wait_time = 2 ** attempt  # Exponential backoff: 2, 4, 8 seconds
                    print(f"  ⚠ Failed (attempt {attempt}): {e}")
                    print(f"  Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                else:
                    print(f"  ✗ Failed after {max_retries} attempts: {e}")
                    return False

        except Exception as e:
            error_msg = str(e).lower()
            if "timeout" in error_msg or "timed out" in error_msg:
                print(f"  ⚠ Pull timeout (attempt {attempt}): {e}")
            else:
                print(f"  ⚠ Unexpected error (attempt {attempt}): {e}")

            if attempt < max_retries:
                wait_time = 2 ** attempt
                print(f"  Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
            else:
                print(f"  ✗ Unexpected error after {max_retries} attempts: {e}")
                return False

    return False


def normalize_patch(patch: str) -> str:
    """Normalize patch format to ensure it can be applied correctly.

    This function:
    1. Ensures the patch ends with a newline
    2. Normalizes line endings to Unix style (\n)
    3. Ensures proper spacing between diff sections
    """
    if not patch:
        return ""

    # Normalize line endings to Unix style
    patch = patch.replace('\r\n', '\n').replace('\r', '\n')

    # Ensure patch ends with newline
    if not patch.endswith('\n'):
        patch += '\n'

    # Ensure each diff section starts on a new line
    # Split by diff --git and rejoin with proper newlines
    lines = patch.split('\n')
    normalized_lines = []

    for i, line in enumerate(lines):
        if line.startswith('diff --git'):
            # Ensure there's a blank line before each diff section (except the first)
            if normalized_lines and normalized_lines[-1] != '':
                normalized_lines.append('')
        normalized_lines.append(line)

    return '\n'.join(normalized_lines)


def separate_test_and_source_patch(patch: str) -> tuple:
    """Separate patch into test-related and source-code-only parts."""
    if not patch:
        return "", ""

    test_patterns = [
        r'test_.*\.py',
        r'.*_test\.py',
        r'tests/.*\.py',
        r'test/.*\.py',
        r'.*/tests/.*\.py',
        r'.*/test/.*\.py',
        r'conftest\.py',
    ]

    def is_test_file(filepath: str) -> bool:
        for pattern in test_patterns:
            if re.search(pattern, filepath):
                return True
        return False

    diff_sections = re.split(r'(?=diff --git)', patch)
    source_diffs = []
    test_diffs = []

    for section in diff_sections:
        if not section.strip():
            continue
        match = re.search(r'diff --git a/(.+?) b/(.+?)(?:\n|$)', section)
        if match:
            filepath = match.group(2)
            if is_test_file(filepath):
                test_diffs.append(section)
            else:
                source_diffs.append(section)
        else:
            source_diffs.append(section)

    source_patch = ''.join(source_diffs)
    test_patch = ''.join(test_diffs)

    # Normalize both patches
    return normalize_patch(source_patch), normalize_patch(test_patch)


def cleanup_container_and_image(container, client, image_name, log_content):
    """完善的容器和镜像清理函数

    确保容器被停止、删除，然后删除镜像
    """

    # Step 1: 停止并删除容器
    if container:
        try:
            log_content.append("\n" + "=" * 80)
            log_content.append("Cleanup: Stopping and removing container")
            log_content.append("=" * 80)
            container.stop(timeout=300)  # 增加超时到 30 秒
            log_content.append("✓ Container stopped")
        except docker.errors.NotFound:
            log_content.append("✓ Container already removed")
        except Exception as e:
            error_msg = str(e).lower()
            if "timeout" in error_msg or "timed out" in error_msg:
                log_content.append(f"⚠ Stop timeout: {e}")
                log_content.append("Trying to kill container...")
                try:
                    container.kill()
                    log_content.append("✓ Container killed")
                except Exception as e2:
                    log_content.append(f"⚠ Failed to kill container: {e2}")
            else:
                log_content.append(f"⚠ Failed to stop container: {e}")
                # 如果 stop 失败，尝试 kill
                try:
                    log_content.append("Trying to kill container...")
                    container.kill()
                    log_content.append("✓ Container killed")
                except Exception as e2:
                    log_content.append(f"⚠ Failed to kill container: {e2}")

        # 确保容器被删除
        try:
            container.remove(force=True)
            log_content.append("✓ Container removed")
        except docker.errors.NotFound:
            log_content.append("✓ Container already removed")
        except Exception as e:
            log_content.append(f"⚠ Failed to remove container: {e}")

    # Step 2: 删除镜像
    try:
        log_content.append("\n" + "=" * 80)
        log_content.append("Cleanup: Removing Docker image")
        log_content.append("=" * 80)
        log_content.append(f"Removing image: {image_name}")
        client.images.remove(image_name, force=True)
        log_content.append(f"✓ Successfully removed image: {image_name}")
    except docker.errors.ImageNotFound:
        log_content.append(f"✓ Image {image_name} not found (already removed?)")
    except docker.errors.APIError as e:
        log_content.append(f"⚠ Failed to remove image {image_name}: {e}")
        # 如果是因为容器引用，尝试清理所有相关容器
        if "container" in str(e).lower() or "reference" in str(e).lower():
            log_content.append("Attempting to remove all containers using this image...")
            try:
                containers = client.containers.list(all=True, filters={"ancestor": image_name})
                for c in containers:
                    try:
                        c.remove(force=True)
                        log_content.append(f"  ✓ Removed container {c.short_id}")
                    except Exception as e2:
                        log_content.append(f"  ⚠ Failed to remove container {c.short_id}: {e2}")

                # 再次尝试删除镜像
                client.images.remove(image_name, force=True)
                log_content.append(f"✓ Successfully removed image after cleaning containers")
            except Exception as e3:
                log_content.append(f"⚠ Final attempt to remove image failed: {e3}")
    except Exception as e:
        log_content.append(f"⚠ Unexpected error removing image {image_name}: {e}")


def test_image(
    image_name: str,
    instance_id: str,
    test_data: Optional[Dict],
    idx: int,
    total: int,
    log_dir: Path,
    force: bool = False,
    run_all: bool = False
) -> Tuple[bool, str, Dict]:
    """
    Test a single Docker image using the complete test collection workflow.

    Args:
        run_all: If True, run the full test suite (f2p_script file + test dir).
                 If False (default), only run the dataset's f2p_script file.

    Returns:
        Tuple of (success: bool, message: str, results: dict)
    """
    # 创建 Docker 客户端，设置更长的超时（5 分钟）
    client = docker.from_env(timeout=300)

    log_file = log_dir / f"{instance_id}.log"
    result_file = log_dir / f"{instance_id}_result.json"

    if log_file.exists() and not force:
        return True, f"[{idx}/{total}] ⊙ {instance_id} (skipped, log exists)", {}

    if not test_data:
        return False, f"[{idx}/{total}] ✗ {instance_id}: No test data found", {}

    # Extract required data
    patch = test_data.get('patch', '')
    f2p_patch = test_data.get('f2p_patch', '')
    # Generated test cases: F2P from `test_script` only.
    # This is written to F2P_TEST_FILE and run via pre_commands.
    f2p_script = test_data.get('test_script', '') or ''
    base_commit = test_data.get('parent_commit', '')
    workdir = test_data.get('workdir', '/workspace')
    # Read the bare test command from `test_command`; fall back to legacy
    # `pre_commands` for datasets that predate the split-out field.
    pre_commands = (test_data.get('test_command') or test_data.get('pre_commands') or '').strip()

    # Detect how this repo runs its tests from pre_commands. Empty pre_commands keeps
    # the legacy standalone-pytest behavior; otherwise we run the authored program and
    # inject the generated test target/flags for pytest, or verbose output for Django.
    runner = detect_runner(pre_commands)

    # Normalize patches
    patch = normalize_patch(patch)
    f2p_patch = normalize_patch(f2p_patch)

    # Separate source and test patches
    source_patch, test_patch = separate_test_and_source_patch(patch)

    container = None
    log_content = []

    try:
        # Start container the same way the build project does: pass `sleep
        # infinity` as the CMD (never override the ENTRYPOINT) so the image's
        # /opt/entrypoint.sh runs, starts services via /opt/setup.sh and writes
        # the readiness marker, while the image's baked ENV (testbed PATH,
        # CONDA_PREFIX, project vars) is injected into every later `docker exec`.
        container = client.containers.run(
            image_name,
            "sleep infinity",
            detach=True,
            remove=False,  # 手动控制删除，确保清理逻辑可靠
            mem_limit="24g",  # Limit container memory to 4GB
            memswap_limit="24g",  # Disable swap (same as mem_limit = no swap)
            stop_signal="SIGKILL",
        )
        # Wait briefly to confirm container started successfully
        container.reload()
        if container.status != "running":
            raise RuntimeError(f"Container failed to start, status: {container.status}")

        log_content.append(f"Instance ID: {instance_id}")
        log_content.append(f"Image: {image_name}")
        log_content.append(f"Base Commit: {base_commit}")
        log_content.append(f"Workdir: {workdir}")
        log_content.append(f"Test Runner: {runner}")

        # Wait for the image entrypoint to finish starting services (postgres/
        # redis) before running any test command. No-op for legacy images.
        if not wait_for_container_ready(container, workdir):
            log_content.append(
                "[WARN] /opt/.container_ready not seen before timeout; "
                "proceeding (services may not be fully up)"
            )

        log_content.append("=" * 80)

        # Helper function to execute commands with shell-level timeout
        def exec_cmd(cmd: str, description: str, timeout: int = 600) -> Tuple[int, str]:
            log_content.append(f"\n[{description}]")
            log_content.append(f"Command: {cmd}")
            log_content.append(f"Timeout: {timeout}s")
            result = container.exec_run(
                ["bash", "-c", f"cd {workdir} && timeout {timeout} bash -c {shlex.quote(cmd)}"],
                demux=False,
                workdir=workdir
            )
            output = result.output.decode('utf-8', errors='replace') if result.output else ""
            if result.exit_code == 124:
                output += f"\n[TIMEOUT] Command timed out after {timeout}s"
                log_content.append(f"[TIMEOUT] Command timed out after {timeout}s")
            log_content.append(f"Exit Code: {result.exit_code}")
            log_content.append(f"Output:\n{output}")
            return result.exit_code, output

        # Step 1: Checkout base commit
        if base_commit:
            exec_cmd(
                f"git checkout {base_commit} -f",
                "Step 1: Checkout base commit"
            )
            exec_cmd(
                f"git checkout .",
                "Step 1: Clean working directory"
            )

        # Step 2: Apply test patch from original patch (if exists)
        apply_patch(container, test_patch, "test_patch.diff", exec_cmd, log_content, "Step 2")

        # Step 3: Apply f2p_patch (if exists)
        apply_patch(container, f2p_patch, "f2p_patch.diff", exec_cmd, log_content, "Step 3")

        # Step 4: Write generated test file (F2P from `test_script`).
        # Place it in the workdir AND in the deepest `cd` target from pre_commands
        # (e.g. /app/cfgov) so the runner — which may run from that subdir —
        # discovers/imports it with the expected basename.
        dest_dirs = [workdir]
        cd_dir = deepest_cd_dir(pre_commands)
        if cd_dir and cd_dir not in dest_dirs:
            dest_dirs.append(cd_dir)

        # Also place the generated file inside the repo's own test directory (e.g.
        # `tests/`). Many repos register pytest options and the DJANGO_SETTINGS_MODULE
        # in a `tests/conftest.py`, which pytest only loads for test files living
        # UNDER that directory. Writing there makes repo-specific options (e.g.
        # `--sqlite`) and settings resolve so the generated tests actually collect/run.
        _, repo_test_dir_out = exec_cmd(
            "ls -d tests test Tests Test testing 2>/dev/null | head -1 || echo ''",
            "Step 4: Detect repo test directory for file placement",
            timeout=600,
        )
        repo_test_dir = repo_test_dir_out.strip()
        if repo_test_dir and repo_test_dir != ".":
            test_dest = f"{workdir.rstrip('/')}/{repo_test_dir}"
            if test_dest not in dest_dirs:
                dest_dirs.append(test_dest)

        wrote_f2p = False
        if f2p_script.strip():
            for dest in dest_dirs:
                script_bytes = f2p_script.encode('utf-8')
                tar_stream = io.BytesIO()
                with tarfile.open(fileobj=tar_stream, mode='w') as tar:
                    info = tarfile.TarInfo(name=F2P_TEST_FILE)
                    info.size = len(script_bytes)
                    tar.addfile(info, io.BytesIO(script_bytes))
                tar_stream.seek(0)
                try:
                    container.put_archive(dest, tar_stream)
                    log_content.append(f"\n[Step 4: Write {F2P_TEST_FILE}]")
                    log_content.append(f"Wrote {F2P_TEST_FILE} to {dest}")
                    wrote_f2p = True
                except Exception as e:
                    log_content.append(f"⚠ Failed to write {F2P_TEST_FILE} to {dest}: {e}")

        # For pytest/django runners, reference the copy placed UNDER the repo test dir
        # (e.g. `tests/test_f2p_generated.py`) when one exists, so pytest loads that
        # dir's conftest.py (repo options like `--sqlite` + settings). The legacy 'none'
        # path keeps bare basenames (its wrapper runs from workdir).
        if runner in ('pytest', 'django') and repo_test_dir and repo_test_dir != ".":
            test_file_name = f"{repo_test_dir.rstrip('/')}/{F2P_TEST_FILE}"
        else:
            test_file_name = F2P_TEST_FILE

        # Step 5: Run pytest BEFORE applying source patch (Result 1)
        log_content.append("\n" + "=" * 80)
        log_content.append("Step 5: Run pytest BEFORE applying source patch (Result 1)")
        log_content.append("=" * 80)

        # First check if pytest can collect tests
        # Detect common test directory names to avoid scanning problematic directories like Snippets/
        # Common test directory names: tests, test, Tests, Test, testing
        test_dirs_check_cmd = "ls -d tests test Tests Test testing 2>/dev/null | head -1 || echo ''"
        _, test_dir_output = exec_cmd(test_dirs_check_cmd, "Detect test directory", timeout=600)
        test_dir = test_dir_output.strip() if test_dir_output.strip() else "."

        # Build the run command from the detected runner. Non-empty pre_commands run
        # the authored program (services/DB/env init) with the generated test target
        # injected; empty pre_commands keep the legacy standalone-pytest path.
        # Non-'none' runs are heavier (init + full suite), so allow more time.
        run_timeout = 1800 if runner != 'none' else 1200

        # Initialize exit_code1 and result1
        exit_code1 = -1
        result1 = {'passed': [], 'failed': [], 'errors': [], 'passed_count': 0, 'failed_count': 0, 'error_count': 0}

        # Collection gate uses pytest's --collect-only, so it applies only to the
        # runners that actually run through pytest: the pytest runner and the legacy
        # 'none' path. Django's unittest runner has no cheap collect-only (a
        # collect pass would run the whole suite twice and emit no "collected N"
        # line), so it is NOT gated — Step 5 runs it directly.
        skip_run_before = False
        if runner in ('pytest', 'none'):
            collect_cmd = build_run_command(
                runner, pre_commands, test_file_name, test_dir, run_all, collect_only=True
            )
            exit_code_collect, output_collect = exec_cmd(
                collect_cmd,
                "Step 5: Check pytest collection",
                timeout=900
            )

            collected_patterns = [
                r'(\d+)\s+tests?\s+collected',  # "388 tests collected"
                r'collected\s+(\d+)\s+items?',   # "collected 388 items"
            ]
            tests_collected = 0
            for pattern in collected_patterns:
                match = re.search(pattern, output_collect)
                if match:
                    tests_collected = int(match.group(1))
                    break

            # Only skip the run if NO tests were collected at all.
            if tests_collected == 0 and exit_code_collect != 0:
                log_content.append(f"\nWarning: Pytest collection failed completely (0 tests collected), skipping pytest run before patch")
                exit_code1 = exit_code_collect
                skip_run_before = True
            elif tests_collected > 0 and exit_code_collect != 0:
                log_content.append(f"\nNote: Pytest collected {tests_collected} tests with some collection errors, proceeding with pytest run")

        if not skip_run_before:
            run_cmd = build_run_command(
                runner, pre_commands, test_file_name, test_dir, run_all, collect_only=False
            )
            exit_code1, output1 = exec_cmd(
                run_cmd,
                f"Step 5: Run tests BEFORE source patch (Result 1, runner={runner})",
                timeout=run_timeout
            )
            result1 = parse_run_output(runner, output1)

            log_content.append(f"\nStep 5 Results:")
            log_content.append(f"  Passed: {result1['passed_count']}")
            log_content.append(f"  Failed: {result1['failed_count']}")
            log_content.append(f"  Errors: {result1['error_count']}")
            log_content.append(f"  Exit Code: {exit_code1}")

        # Step 6: Apply source patch (the fix)
        log_content.append("\n" + "=" * 80)
        log_content.append("Step 6: Apply source patch (the fix)")
        log_content.append("=" * 80)

        apply_patch(container, source_patch, "source_patch.diff", exec_cmd, log_content, "Step 6")

        # Step 7: Run pytest AFTER applying source patch (Result 2)
        log_content.append("\n" + "=" * 80)
        log_content.append("Step 7: Run pytest AFTER applying source patch (Result 2)")
        log_content.append("=" * 80)

        run_cmd = build_run_command(
            runner, pre_commands, test_file_name, test_dir, run_all, collect_only=False
        )
        exit_code2, output2 = exec_cmd(
            run_cmd,
            f"Step 7: Run tests AFTER source patch (Result 2, runner={runner})",
            timeout=run_timeout
        )
        result2 = parse_run_output(runner, output2)

        log_content.append(f"\nStep 7 Results:")
        log_content.append(f"  Passed: {result2['passed_count']}")
        log_content.append(f"  Failed: {result2['failed_count']}")
        log_content.append(f"  Errors: {result2['error_count']}")
        log_content.append(f"  Exit Code: {exit_code2}")

        # Step 8: Calculate FAIL_TO_PASS
        log_content.append("\n" + "=" * 80)
        log_content.append("Step 8: Calculate test transitions")
        log_content.append("=" * 80)

        # Opaque runners produce no per-test node ids, so we cannot compute reliable
        # transitions. Skip rather than fabricate F2P from unparsed output.
        runner_unsupported = (runner == 'unsupported')

        if runner_unsupported:
            fail_to_pass = []
            pass_to_fail = []
            log_content.append(
                "Runner is 'unsupported' (opaque pre_commands); skipping F2P "
                "computation — no per-test node ids available."
            )
        else:
            before_failed = set(result1['failed'] + result1['errors'])
            before_passed = set(result1['passed'])
            after_failed = set(result2['failed'] + result2['errors'])
            after_passed = set(result2['passed'])

            fail_to_pass = list(before_failed & after_passed)  # Failed before, passed after
            pass_to_fail = list(before_passed & after_failed)  # Regression!

        log_content.append(f"FAIL_TO_PASS: {len(fail_to_pass)} tests")
        log_content.append(f"PASS_TO_FAIL: {len(pass_to_fail)} tests")

        # Debug: Show detailed comparison with expected results
        expected_f2p = test_data.get('FAIL_TO_PASS', [])

        if isinstance(expected_f2p, str):
            expected_f2p = [t.strip() for t in expected_f2p.split('\n') if t.strip()]

        log_content.append("\n" + "=" * 80)
        log_content.append("DEBUG: Expected vs Calculated Comparison")
        log_content.append("=" * 80)
        log_content.append(f"Expected FAIL_TO_PASS count: {len(expected_f2p)}")
        log_content.append(f"Calculated FAIL_TO_PASS count: {len(fail_to_pass)}")
        log_content.append(f"Difference: {len(fail_to_pass) - len(expected_f2p)}")

        if fail_to_pass:
            log_content.append("\n=== Calculated FAIL_TO_PASS tests (first 20) ===")
            for test in fail_to_pass[:20]:
                log_content.append(f"  - {test}")
            if len(fail_to_pass) > 20:
                log_content.append(f"  ... and {len(fail_to_pass) - 20} more")

        if expected_f2p:
            log_content.append("\n=== Expected FAIL_TO_PASS tests (first 20) ===")
            for test in expected_f2p[:20]:
                log_content.append(f"  - {test}")
            if len(expected_f2p) > 20:
                log_content.append(f"  ... and {len(expected_f2p) - 20} more")

        # Show tests that are in calculated but not in expected
        expected_f2p_set = set(expected_f2p)
        calculated_f2p_set = set(fail_to_pass)
        extra_f2p = calculated_f2p_set - expected_f2p_set
        missing_f2p = expected_f2p_set - calculated_f2p_set

        if extra_f2p:
            log_content.append(f"\n=== Extra FAIL_TO_PASS (in calculated but not in expected): {len(extra_f2p)} ===")
            for test in list(extra_f2p)[:20]:
                log_content.append(f"  + {test}")
            if len(extra_f2p) > 20:
                log_content.append(f"  ... and {len(extra_f2p) - 20} more")

        if missing_f2p:
            log_content.append(f"\n=== Missing FAIL_TO_PASS (in expected but not in calculated): {len(missing_f2p)} ===")
            for test in list(missing_f2p)[:20]:
                log_content.append(f"  - {test}")
            if len(missing_f2p) > 20:
                log_content.append(f"  ... and {len(missing_f2p) - 20} more")

        if pass_to_fail:
            log_content.append("\n=== PASS_TO_FAIL tests (REGRESSION) ===")
            for test in pass_to_fail:
                log_content.append(f"  ! {test}")

        results = {
            'instance_id': instance_id,
            'runner': runner,
            'before_patch': {
                'passed': result1['passed'],
                'failed': result1['failed'],
                'errors': result1['errors'],
                'exit_code': exit_code1
            },
            'after_patch': {
                'passed': result2['passed'],
                'failed': result2['failed'],
                'errors': result2['errors'],
                'exit_code': exit_code2
            },
            'calculated': {
                'fail_to_pass': fail_to_pass,
                'pass_to_fail': pass_to_fail,
                'fail_to_pass_count': len(fail_to_pass),
                'pass_to_fail_count': len(pass_to_fail)
            },
            'expected': {
                'fail_to_pass': expected_f2p,
                'fail_to_pass_count': len(expected_f2p)
            },
            'validation': {
                'f2p_match': len(fail_to_pass) == len(expected_f2p),
                'has_regression': len(pass_to_fail) > 0,
                'runner_unsupported': runner_unsupported
            }
        }

        # Summary
        log_content.append("\n" + "=" * 80)
        log_content.append("SUMMARY")
        log_content.append("=" * 80)
        log_content.append(f"FAIL_TO_PASS: {len(fail_to_pass)} (expected: {len(expected_f2p)})")
        log_content.append(f"PASS_TO_FAIL: {len(pass_to_fail)} (REGRESSION!)" if pass_to_fail else "PASS_TO_FAIL: 0")

        success = (len(fail_to_pass) > 0 and len(pass_to_fail) == 0)

        # Save log
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(log_content))

        # Save results
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        if runner_unsupported:
            return False, f"[{idx}/{total}] ✗ {instance_id}: unsupported runner (opaque pre_commands), F2P not computed", results
        if success:
            return True, f"[{idx}/{total}] ✓ {instance_id} (F2P={len(fail_to_pass)})", results
        else:
            return False, f"[{idx}/{total}] ✗ {instance_id} (F2P={len(fail_to_pass)}, P2F={len(pass_to_fail)})", results

    except Exception as e:
        import traceback
        log_content.append(f"\nERROR: {str(e)}")
        log_content.append(f"Traceback: {traceback.format_exc()}")

        # 保存错误日志
        try:
            with open(log_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(log_content))
        except Exception as log_error:
            print(f"Failed to save error log: {log_error}")

        return False, f"[{idx}/{total}] ✗ {instance_id}: {e}", {}

    finally:
        # 无论成功还是失败，都清理容器和镜像
        cleanup_container_and_image(container, client, image_name, log_content)

        # 保存完整日志（包括清理日志）
        try:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write('\n' + '\n'.join(log_content[-20:]) + '\n')  # 只追加最后的清理日志
        except Exception as e:
            print(f"Failed to save cleanup log: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Validate test images using complete test collection workflow"
    )
    parser.add_argument(
        "--jsonl-file",
        type=str,
        required=True,
        help="Path to input JSONL file containing test data"
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs/validation",
        help="Directory to save test logs"
    )
    parser.add_argument(
        "-j", "--jobs",
        type=int,
        default=4,
        help="Number of parallel jobs"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force overwrite existing log files"
    )
    parser.add_argument(
        "--docker-image-prefix",
        type=str,
        default="",
        help="Docker registry prefix prepended to bare image_url values"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run the full test suite. Default runs only the dataset's f2p_script file."
    )

    args = parser.parse_args()

    # Resolve paths
    jsonl_file = Path(args.jsonl_file).resolve()
    log_dir = Path(args.log_dir).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    # Check JSONL file exists
    if not jsonl_file.exists():
        print(f"Error: JSONL file not found: {jsonl_file}")
        return

    print(f"Loading test data from: {jsonl_file}")

    # Load all test data from JSONL
    test_data_list = load_test_data_from_jsonl(jsonl_file)

    if not test_data_list:
        print("No test data found in JSONL file. Exiting.")
        return

    print(f"Loaded {len(test_data_list)} test instances from JSONL")

    # Initialize Docker client
    client = docker.from_env(timeout=300)  # 设置 5 分钟超时

    # Prepare test tasks (without pulling images)
    print("\nPreparing test tasks...")
    test_tasks = []

    for idx, test_data in enumerate(test_data_list, 1):
        instance_id = test_data.get('instance_id')
        if not instance_id:
            print(f"  [{idx}/{len(test_data_list)}] Warning: Missing instance_id, skipping")
            continue

        # Build image name: prefer image_url from data; fall back to constructed name
        image_url = test_data.get('image_url', '').strip()
        if image_url:
            image_name = image_url.lower()
            image_name = f"{args.docker_image_prefix}/{image_url}".lower()
        else:
            image_name = f"{args.docker_image_prefix}/swe_pr_agent/python_images:{instance_id}".lower()
        test_tasks.append((image_name, instance_id, test_data))

    total = len(test_tasks)
    if total == 0:
        print("\nNo test tasks to process. Exiting.")
        return

    print(f"Prepared {total} test tasks")
    print(f"\nProcessing {total} instances with {args.jobs} parallel jobs")
    print("Each instance: pull image → test → delete image\n")

    # Run tests with pull-test-delete for each instance
    success_count = 0
    failed_count = 0
    pull_failed = []
    all_results = []

    def process_single_instance(image_name, instance_id, test_data, idx, total, log_dir, force, docker_image_prefix, run_all):
        """Pull image, test, and delete for a single instance.

        关键流程：
        1. Pull image
        2. Test image
        3. finally: 无论成功、失败还是异常，都删除镜像

        这个函数是每个线程的入口点，必须保证执行完成后镜像被删除
        """
        # 创建 Docker 客户端，设置更长的超时（5 分钟）
        client = docker.from_env(timeout=300)
        image_pulled = False
        result = (False, f"[{idx}/{total}] ✗ {instance_id}: Unknown error", {}, None)
        log_file = log_dir / f"{instance_id}.log"
        if log_file.exists() and not force:
            return True, f"[{idx}/{total}] ⊙ {instance_id} (skipped, log exists)", {}, None

        try:
            # Step 1: Pull image
            print(f"[{idx}/{total}] Pulling {instance_id}...")
            if not pull_docker_image(client, image_name, docker_image_prefix):
                # Pull 失败，不需要删除镜像（因为没有成功拉取）
                result = (False, f"[{idx}/{total}] ✗ {instance_id}: Failed to pull image", {}, instance_id)
                return result

            image_pulled = True  # 标记镜像已拉取

            # Step 2: Test image
            # test_image() 内部的 finally 块会删除容器和镜像
            success, message, results = test_image(
                image_name,
                instance_id,
                test_data,
                idx,
                total,
                log_dir,
                force,
                run_all
            )
            result = (success, message, results, None)
            return result

        except Exception as e:
            # 测试过程中发生异常
            print(f"[{idx}/{total}] ✗ {instance_id}: Exception in process_single_instance: {e}")
            result = (False, f"[{idx}/{total}] ✗ {instance_id}: {e}", {}, None)
            return result

        finally:
            # 关键：无论成功、失败还是异常，都确保镜像被删除
            # 只有在镜像已拉取的情况下才需要删除
            if image_pulled:
                try:
                    print(f"[{idx}/{total}] Finally: Cleaning up image {image_name}")
                    client.images.remove(image_name, force=True)
                    print(f"[{idx}/{total}] ✓ Image removed in finally block")
                except docker.errors.ImageNotFound:
                    print(f"[{idx}/{total}] ✓ Image already removed (by test_image)")
                except Exception as cleanup_error:
                    print(f"[{idx}/{total}] ⚠ Failed to remove image in finally: {cleanup_error}")
                    # 尝试清理所有相关容器后重试
                    try:
                        containers = client.containers.list(all=True, filters={"ancestor": image_name})
                        if containers:
                            print(f"[{idx}/{total}] Found {len(containers)} container(s) using this image, removing...")
                            for c in containers:
                                try:
                                    c.remove(force=True)
                                    print(f"[{idx}/{total}]   ✓ Removed container {c.short_id}")
                                except Exception as e:
                                    print(f"[{idx}/{total}]   ⚠ Failed to remove container {c.short_id}: {e}")

                            # 重试删除镜像
                            client.images.remove(image_name, force=True)
                            print(f"[{idx}/{total}] ✓ Image removed after cleaning containers")
                    except Exception as final_error:
                        print(f"[{idx}/{total}] ⚠ Final cleanup attempt failed: {final_error}")

            return result

    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = {
            executor.submit(
                process_single_instance,
                image_name,
                instance_id,
                test_data,
                idx,
                total,
                log_dir,
                args.force,
                args.docker_image_prefix,
                args.all
            ): (instance_id, idx)
            for idx, (image_name, instance_id, test_data) in enumerate(test_tasks, 1)
        }

        for future in as_completed(futures):
            instance_id, idx = futures[future]
            try:
                success, message, results, failed_instance = future.result()
                print(message)
                if failed_instance:
                    pull_failed.append(failed_instance)
                    failed_count += 1
                elif success:
                    success_count += 1
                else:
                    failed_count += 1
                if results:
                    all_results.append(results)
            except Exception as e:
                print(f"[{idx}/{total}] ✗ {instance_id}: Unexpected error: {e}")
                failed_count += 1

    print(f"\n{'=' * 80}")
    print(f"Validation completed!")
    print(f"  Success: {success_count}/{total}")
    print(f"  Failed:  {failed_count}/{total}")
    if pull_failed:
        print(f"  Images not pulled: {len(pull_failed)}")
    print(f"  Logs saved to: {log_dir}")
    print(f"{'=' * 80}")

    # Clean up dangling images
    print("\nCleaning up dangling Docker images...")
    try:
        pruned = client.images.prune(filters={'dangling': True})
        if pruned.get('ImagesDeleted'):
            deleted_count = len(pruned['ImagesDeleted'])
            print(f"  ✓ Removed {deleted_count} dangling image(s)")
            if pruned.get('SpaceReclaimed'):
                space_mb = pruned['SpaceReclaimed'] / (1024 * 1024)
                print(f"  ✓ Reclaimed {space_mb:.2f} MB of disk space")
        else:
            print("  ✓ No dangling images to remove")
    except Exception as e:
        print(f"  ⚠ Warning: Failed to prune dangling images: {e}")

    # Final cleanup: only clean up images from this task
    print("\nFinal cleanup: checking for remaining task images...")
    task_images = {image_name for image_name, _, _ in test_tasks}
    cleanup_task_images(client, task_images)

    print(f"\n{'=' * 80}")
    print("All tasks completed!")
    print(f"{'=' * 80}\n")


def cleanup_task_images(client, task_images):
    """只清理本次任务的镜像

    Args:
        client: Docker client
        task_images: 本次任务的镜像名称集合
    """
    if not task_images:
        print("  ✓ No task images to clean up")
        return

    print(f"  Checking {len(task_images)} task image(s)...")
    cleaned = 0
    skipped = 0

    for image_name in task_images:
        try:
            # 检查镜像是否存在
            client.images.get(image_name)

            # 镜像存在，尝试删除
            try:
                client.images.remove(image_name, force=True)
                print(f"    ✓ Removed remaining image: {image_name.split('/')[-1]}")
                cleaned += 1
            except docker.errors.APIError as e:
                if "conflict" in str(e).lower() or "reference" in str(e).lower():
                    # 镜像被容器引用，清理容器后重试
                    try:
                        containers = client.containers.list(all=True, filters={"ancestor": image_name})
                        for c in containers:
                            c.remove(force=True)
                        client.images.remove(image_name, force=True)
                        print(f"    ✓ Removed image after cleaning containers: {image_name.split('/')[-1]}")
                        cleaned += 1
                    except Exception as e2:
                        print(f"    ⚠ Failed to remove {image_name.split('/')[-1]}: {e2}")
                else:
                    print(f"    ⚠ Failed to remove {image_name.split('/')[-1]}: {e}")
            except Exception as e:
                print(f"    ⚠ Failed to remove {image_name.split('/')[-1]}: {e}")

        except docker.errors.ImageNotFound:
            # 镜像不存在，已经被清理
            skipped += 1
        except Exception as e:
            print(f"    ⚠ Error checking {image_name.split('/')[-1]}: {e}")

    if cleaned > 0:
        print(f"  ✓ Cleaned up {cleaned} remaining image(s)")
    if skipped == len(task_images):
        print("  ✓ All task images already cleaned up")


if __name__ == "__main__":
    main()
