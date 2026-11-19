"""Judge-level regression tests for ``SWERebenchV2Judge`` (no Docker needed)."""

from __future__ import annotations

import subprocess
from typing import Any

import pytest

from benchmarks.swerebenchv2.judge import SWERebenchV2Judge


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
