"""Tests for the live mini rollout progress output."""

import sys
from pathlib import Path
from typing import Any, cast

import pytest

from benchmarks.utils.miniswe.environment import SUBMISSION_MARKER
from benchmarks.utils.miniswe.model import MiniModelSpec
from benchmarks.utils.miniswe.progress import MiniProgress
from benchmarks.utils.miniswe.rollout import MiniRolloutSettings, run_mini_rollout
from benchmarks.utils.tests.fakes import ScriptedModel, submission_workspace
from openhands.sdk import LLM
from openhands.sdk.workspace import RemoteWorkspace


pytest.importorskip("minisweagent")

from minisweagent.exceptions import FormatError  # noqa: E402


def _settings(tmp_path: Path, **overrides: Any) -> MiniRolloutSettings:
    path = tmp_path / "logs" / "instance_test__instance-1.mini.log"
    settings: dict[str, Any] = {
        "cwd": "/workspace/repo",
        "step_limit": 5,
        "trace_path": str(tmp_path / "trace.json"),
        "progress_path": str(path),
        "progress_console": False,
        "instance_id": "test__instance-1",
        "attempt": 2,
    }
    settings.update(overrides)
    return MiniRolloutSettings(**settings)


def _llm() -> LLM:
    return LLM.model_validate(
        {
            "model": "openai/gpt-4o-mini",
            "api_key": "sk-test",
            "base_url": "http://localhost:1/v1",
        }
    )


def _spec() -> MiniModelSpec:
    return MiniModelSpec(model_name="scripted/test", temperature=0.0, top_p=1.0)


def _run(model: Any, tmp_path: Path, **overrides: Any):
    return run_mini_rollout(
        llm=_llm(),
        workspace=cast(RemoteWorkspace, submission_workspace()),
        instruction="fix the bug",
        settings=_settings(tmp_path, **overrides),
        model=model,
        model_spec=_spec(),
    )


def _progress_file(tmp_path: Path) -> Path:
    return tmp_path / "logs" / "instance_test__instance-1.mini.log"


def test_progress_log_records_rollout_events(tmp_path) -> None:
    long_command = "echo " + "x" * 800
    result = _run(
        ScriptedModel(
            [
                long_command,
                f"echo {SUBMISSION_MARKER} && cat /tmp/patch.txt",
            ]
        ),
        tmp_path,
    )

    assert result.status == "Submitted"
    lines = _progress_file(tmp_path).read_text().splitlines()
    text = "\n".join(lines)
    for event in (
        "rollout_start",
        "model_call",
        "action",
        "observation",
        "interrupt",
        "rollout_end",
    ):
        assert event in text, event
    assert "[miniswe instance=test__instance-1 attempt=2]" in text
    assert "rollout_end status=Submitted" in text

    # One line per event: previews are newline-escaped and truncated.
    action_lines = [line for line in lines if " action " in line]
    assert action_lines, lines
    assert all(len(line) <= 700 for line in action_lines), action_lines
    assert "…(+" in text

    model_lines = [line for line in lines if " model_call " in line]
    assert "tokens=10+5" in model_lines[0]
    assert "tools=1" in model_lines[0]

    interrupt_lines = [line for line in lines if " interrupt " in line]
    assert interrupt_lines and "type=Submitted" in interrupt_lines[0]


def test_format_errors_are_visible_in_progress(tmp_path) -> None:
    class FormatErrorModel(ScriptedModel):
        def query(self, messages: list[dict], **kwargs: Any) -> dict:
            raise FormatError(
                {
                    "role": "user",
                    "content": "bad tool call",
                    "extra": {"interrupt_type": "FormatError"},
                }
            )

    result = _run(FormatErrorModel(["ls"]), tmp_path)

    assert result.status == "RepeatedFormatError"
    text = _progress_file(tmp_path).read_text()
    assert "model_error" in text
    assert "type=FormatError" in text
    assert "rollout_end status=RepeatedFormatError" in text


def test_progress_is_disabled_without_handlers(tmp_path) -> None:
    result = _run(
        ScriptedModel(["ls"]),
        tmp_path,
        progress_path=None,
        progress_console=False,
    )

    assert result.status == "LimitsExceeded"  # step_limit=5: answered normally
    assert not _progress_file(tmp_path).exists()


def test_console_mirror_uses_real_stdout(monkeypatch) -> None:
    import io

    stream = io.StringIO()
    monkeypatch.setattr(sys, "__stdout__", stream)
    progress = MiniProgress(console=True, label="instance=x attempt=1")
    progress.rollout_start(
        model="openai/test",
        cwd="/testbed",
        step_limit=3,
        command_timeout=60,
        timeout_seconds=None,
    )
    progress.close()

    assert "rollout_start" in stream.getvalue()
    assert "[miniswe instance=x attempt=1]" in stream.getvalue()


def test_progress_file_is_written_even_on_exception(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="provider exploded"):
        _run(ScriptedModel(["ls"], fail=True), tmp_path)

    lines = _progress_file(tmp_path).read_text().splitlines()
    assert any(
        " model_error " in line and "type=RuntimeError" in line for line in lines
    ), lines
    assert any(
        " rollout_error " in line and "type=RuntimeError" in line for line in lines
    ), lines
