from hashlib import sha256
from pathlib import Path
from unittest.mock import Mock

from swebench.harness import run_evaluation

from benchmarks.swebench.judge import SWEBenchJudge


def test_judge_uses_configured_log_dir(monkeypatch, tmp_path: Path) -> None:
    run_instance = Mock(return_value={"completed": True, "resolved": True})
    monkeypatch.setattr(run_evaluation, "run_instance", run_instance)
    monkeypatch.setattr(
        "swebench.harness.test_spec.test_spec.make_test_spec",
        Mock(return_value=Mock()),
    )
    monkeypatch.setattr("benchmarks.swebench.judge.docker.from_env", Mock())

    patch = "diff --git a/example.py b/example.py\n"
    log_dir = tmp_path / "judge_logs"
    judge = SWEBenchJudge(evaluation_log_dir=str(log_dir))

    assert judge.judge("owner__repo-1", patch, {}) is True

    assert run_evaluation.RUN_EVALUATION_LOG_DIR == log_dir.resolve()
    assert run_instance.call_args.kwargs["run_id"] == (
        f"judge-{sha256(patch.encode()).hexdigest()[:16]}"
    )
