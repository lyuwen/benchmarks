"""End-to-end mini rollout tests with a scripted model and a fake workspace."""

import json
from types import SimpleNamespace
from typing import Any, cast

import pytest

from benchmarks.utils.miniswe.environment import SUBMISSION_MARKER
from benchmarks.utils.miniswe.model import MiniModelSpec
from benchmarks.utils.miniswe.rollout import MiniRolloutSettings, run_mini_rollout
from benchmarks.utils.tests.fakes import FakeWorkspace, ok_result
from openhands.sdk import LLM
from openhands.sdk.workspace import RemoteWorkspace


pytest.importorskip("minisweagent")

PATCH = "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-x\n+y\n"


class ScriptedModel:
    """Minimal mini ``Model`` protocol implementation driven by commands."""

    def __init__(self, commands: list[str], *, fail: bool = False) -> None:
        self.config = SimpleNamespace(model_name="scripted/test")
        self._commands = list(commands)
        self._calls = 0
        self._fail = fail

    def query(self, messages: list[dict], **kwargs: Any) -> dict:
        if self._fail:
            raise RuntimeError("provider exploded")
        index = min(self._calls, len(self._commands) - 1)
        command = self._commands[index]
        self._calls += 1
        call_id = f"call_{self._calls}"
        return {
            "role": "assistant",
            "content": f"step {self._calls}",
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "arguments": json.dumps({"command": command}),
                    },
                }
            ],
            "extra": {
                "actions": [{"command": command, "tool_call_id": call_id}],
                "response": {"usage": {"prompt_tokens": 10, "completion_tokens": 5}},
                "cost": 0.001,
            },
        }

    def format_message(self, **kwargs: Any) -> dict:
        return dict(kwargs)

    def format_observation_messages(
        self, message: dict, outputs: list[dict], template_vars: dict | None = None
    ) -> list[dict]:
        actions = message.get("extra", {}).get("actions", [])
        return [
            {
                "role": "tool",
                "content": output.get("output", ""),
                "tool_call_id": action["tool_call_id"],
            }
            for action, output in zip(actions, outputs)
        ]

    def get_template_vars(self, **kwargs: Any) -> dict:
        return {}

    def serialize(self) -> dict:
        return {"info": {"config": {"model": {"model_name": self.config.model_name}}}}


def _workspace() -> FakeWorkspace:
    def responder(command: str, cwd: Any):
        if SUBMISSION_MARKER in command:
            return ok_result(command, stdout=f"{SUBMISSION_MARKER}\n{PATCH}")
        return ok_result(command, stdout="ok")

    return FakeWorkspace(responder=responder)


def _settings(tmp_path, **overrides: Any) -> MiniRolloutSettings:
    settings: dict[str, Any] = {
        "cwd": "/workspace/repo",
        "step_limit": 5,
        "trace_path": str(tmp_path / "trace.json"),
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
    return MiniModelSpec(
        model_name="scripted/test",
        model_kwargs={"api_key": "sk-test"},
        temperature=0.3,
        top_p=0.9,
    )


def test_rollout_returns_submitted_patch_and_metrics(tmp_path) -> None:
    workspace = _workspace()
    result = run_mini_rollout(
        llm=_llm(),
        workspace=cast(RemoteWorkspace, workspace),
        instruction="fix the bug",
        settings=_settings(tmp_path),
        model=ScriptedModel(["ls", f"echo {SUBMISSION_MARKER} && cat /tmp/patch.txt"]),
        model_spec=_spec(),
    )

    assert result.status == "Submitted"
    assert result.submission == PATCH
    assert result.n_calls == 2
    assert result.cost == pytest.approx(0.002)
    assert result.model.temperature == 0.3
    assert [message["role"] for message in result.messages] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
        "exit",
    ]
    assert result.tools[0]["function"]["name"] == "bash"

    assert result.metrics is not None
    assert result.metrics.accumulated_token_usage is not None
    assert result.metrics.accumulated_token_usage.prompt_tokens == 20
    assert result.metrics.accumulated_token_usage.completion_tokens == 10
    assert result.metrics.accumulated_cost == pytest.approx(0.002)


def test_rollout_trace_is_saved_and_redacted(tmp_path) -> None:
    trace_path = tmp_path / "traces" / "trace.json"
    run_mini_rollout(
        llm=_llm(),
        workspace=cast(RemoteWorkspace, _workspace()),
        instruction="fix the bug",
        settings=_settings(tmp_path, trace_path=str(trace_path)),
        model=ScriptedModel(["ls", f"echo {SUBMISSION_MARKER} && cat /tmp/patch.txt"]),
        model_spec=_spec(),
    )

    raw = trace_path.read_text()
    assert "sk-test" not in raw
    trace = json.loads(raw)
    assert trace["info"]["exit_status"] == "Submitted"
    assert trace["info"]["mini_version"] == "2.4.6"
    assert trace["model"]["model_kwargs"]["api_key"] == "<redacted>"
    assert trace["trajectory_format"] == "mini-swe-agent-1.1"


def test_step_limit_is_a_terminal_status_without_submission(tmp_path) -> None:
    result = run_mini_rollout(
        llm=_llm(),
        workspace=cast(RemoteWorkspace, _workspace()),
        instruction="fix the bug",
        settings=_settings(tmp_path, step_limit=1),
        model=ScriptedModel(["ls", "pwd"]),
        model_spec=_spec(),
    )

    assert result.status == "LimitsExceeded"
    assert result.submission == ""


def test_unhandled_model_error_is_reraised_after_trace(tmp_path) -> None:
    trace_path = tmp_path / "trace.json"

    with pytest.raises(RuntimeError, match="provider exploded"):
        run_mini_rollout(
            llm=_llm(),
            workspace=cast(RemoteWorkspace, _workspace()),
            instruction="fix the bug",
            settings=_settings(tmp_path, trace_path=str(trace_path)),
            model=ScriptedModel(["ls"], fail=True),
            model_spec=_spec(),
        )

    trace = json.loads(trace_path.read_text())
    assert trace["info"]["exit_status"] == "Exception"
    assert "provider exploded" in trace["info"]["error"]
