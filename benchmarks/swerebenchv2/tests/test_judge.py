"""Judge-level regression tests for ``SWERebenchV2Judge`` (no Docker needed)."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

from benchmarks.swerebenchv2.judge import SWERebenchV2Judge, _normalize_test_name


GIT_PATCH = """diff --git a/iolib.go b/iolib.go
--- a/iolib.go
+++ b/iolib.go
@@ -1,1 +1,1 @@
-old
+new
"""

TEST_PATCH = """diff --git a/iolib_test.go b/iolib_test.go
new file mode 100644
--- /dev/null
+++ b/iolib_test.go
@@ -0,0 +1,1 @@
+package lua
"""


def _instance_data() -> dict[str, Any]:
    return {
        "repo": "yuin/gopher-lua",
        "image_name": "example/gopher-lua:test",
        "test_patch": TEST_PATCH,
        "install_config": {"test_cmd": "go test -v", "log_parser": "parse_log_gotest"},
        "PASS_TO_PASS": ["TestExisting"],
        "FAIL_TO_PASS": ["TestFix"],
    }


def _go_test_output(*test_names: str) -> str:
    return "".join(f"--- PASS: {name} (0.00s)\n" for name in test_names)


def _mock_test_output(monkeypatch: pytest.MonkeyPatch, *test_names: str) -> None:
    def fake_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=_go_test_output(*test_names),
            stderr="",
        )

    monkeypatch.setattr("benchmarks.swerebenchv2.judge.subprocess.run", fake_run)


def test_judge_resolves_when_all_expected_tests_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_test_output(monkeypatch, "TestExisting", "TestFix")

    judge = SWERebenchV2Judge()

    assert judge.judge("yuin__gopher-lua-427", GIT_PATCH, _instance_data()) is True


def test_judge_resolves_when_agent_added_tests_also_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Extra PASSED tests from the agent must not make the instance fail."""
    _mock_test_output(monkeypatch, "TestExisting", "TestFix", "TestAgentOwnRegression")

    judge = SWERebenchV2Judge()

    assert judge.judge("yuin__gopher-lua-427", GIT_PATCH, _instance_data()) is True


def test_judge_does_not_resolve_when_pass_to_pass_test_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_test_output(monkeypatch, "TestFix", "TestAgentOwnRegression")

    judge = SWERebenchV2Judge()

    assert judge.judge("yuin__gopher-lua-427", GIT_PATCH, _instance_data()) is False


BINARY_JUNK_PATCH = (
    GIT_PATCH
    + """diff --git a/go1.23.0.linux-amd64.tar.gz b/go1.23.0.linux-amd64.tar.gz
new file mode 100644
index 0000000..1234567
Binary files /dev/null and b/go1.23.0.linux-amd64.tar.gz differ
"""
)


def test_judge_strips_binary_files_before_applying_the_patch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A binary artifact in the patch must not stop the text changes from applying."""
    captured_patch: dict[str, str] = {}

    def fake_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        cmd = args[0]
        mount = cmd[cmd.index("-v") + 1].split(":")[0]
        captured_patch["text"] = (Path(mount) / "patch.diff").read_text()
        return subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=_go_test_output("TestExisting", "TestFix"),
            stderr="",
        )

    monkeypatch.setattr("benchmarks.swerebenchv2.judge.subprocess.run", fake_run)

    judge = SWERebenchV2Judge()

    assert (
        judge.judge("yuin__gopher-lua-427", BINARY_JUNK_PATCH, _instance_data()) is True
    )
    assert "iolib.go" in captured_patch["text"]
    assert "go1.23.0.linux-amd64.tar.gz" not in captured_patch["text"]
    assert captured_patch["text"].endswith("\n")


def test_normalize_test_name_strips_embedded_durations() -> None:
    assert _normalize_test_name("Compiled in 0.065s") == "Compiled"
