"""mini-swe-agent inference entry point for SWE-bench.

The benchmark ``Evaluation`` keeps instance loading, workspace provisioning,
retries/resume, aggregation and the judge. Only the per-instance agent loop is
delegated to mini-swe-agent; the candidate patch is mini's own submission rather
than the post-run worktree diff.

Usage:
    uv run python -m benchmarks.swebench.run_infer_miniswe model.json \\
      --dataset princeton-nlp/SWE-bench_Verified --split test \\
      --workspace docker --n-limit 1 --num-workers 1 --max-attempts 1 \\
      --judge swebench
"""

import os
import uuid
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import List, Optional

from omegaconf import OmegaConf
from pydantic import Field

from benchmarks.swebench.judge import SWEBenchJudge

# Import judge modules to trigger registration.
from benchmarks.swebench.run_infer import SWEBenchEvaluation, get_instruction
from benchmarks.utils.args_parser import get_parser
from benchmarks.utils.critics import create_critic
from benchmarks.utils.evaluation import OnResult
from benchmarks.utils.evaluation_utils import (
    construct_eval_output_dir,
    get_default_on_result_writer,
)
from benchmarks.utils.execution_judge import (
    add_judge_args,
    create_judge,
)
from benchmarks.utils.miniswe import (
    MiniRolloutSettings,
    build_history_dump,
    check_patch_applies,
    get_head_commit,
    head_matches_commit,
    publish_canonical_histories,
    run_mini_rollout,
    write_history,
)
from benchmarks.utils.models import EvalInstance, EvalMetadata, EvalOutput
from openhands.sdk import LLM, get_logger
from openhands.sdk.workspace import RemoteWorkspace


logger = get_logger(__name__)

MINISWE_SUPPORTED_CRITICS = ("pass", "empty_patch_critic")


class SWEBenchMiniEvaluation(SWEBenchEvaluation):
    """SWE-bench evaluator whose per-instance agent loop is mini-swe-agent."""

    mini_cost_limit: float = Field(
        default=0.0,
        description="Stop a rollout after this cost in USD (0 disables the limit)",
    )
    mini_command_timeout: int = Field(
        default=60, description="Per-command timeout inside the task container"
    )
    mini_bash_env_file: str | None = Field(
        default=None,
        description=(
            "Optional rc file exported as BASH_ENV for non-login bash. Off by "
            "default: agent-server images already put the task environment on "
            "PATH, and /root/.bashrc is unreadable to the non-root agent user"
        ),
    )

    # The OpenHands archive hook snapshots /workspace/conversations, which a mini
    # rollout never creates; skip the remote command entirely.
    def _capture_conversation_archive(
        self, workspace: RemoteWorkspace, instance: EvalInstance
    ) -> None:
        logger.debug("[child] mini rollout has no conversation archive to capture")

    def run(self, *, on_result: Optional[OnResult] = None) -> List[EvalOutput]:
        outputs = super().run(on_result=on_result)
        # Evaluation picks the winning attempt during aggregation; publish that
        # attempt's history under the canonical filename afterwards.
        publish_canonical_histories(self.metadata.eval_output_dir, self.output_path)
        return outputs

    def evaluate_instance(
        self, instance: EvalInstance, workspace: RemoteWorkspace
    ) -> EvalOutput:
        assert self.metadata.prompt_path is not None
        base_commit = instance.data["base_commit"]
        # Agent-server images can sit at an environment-setup commit instead of
        # the dataset base commit. Reset first so the agent's `git diff` and the
        # judge (which applies the patch to base_commit) share one baseline.
        repo_path = self.prepare_repo(workspace, instance, checkout_commit=base_commit)

        matches = head_matches_commit(
            workspace=workspace, repo_path=repo_path, commit=base_commit
        )
        if matches is False:
            head = get_head_commit(workspace=workspace, repo_path=repo_path)
            raise RuntimeError(
                f"Prepared checkout at {head} does not match base commit "
                f"{base_commit} for {instance.id}; the image may be built for a "
                "different dataset version"
            )

        instruction = get_instruction(
            instance=instance.data,
            metadata=self.metadata,
            workspace_path=workspace.working_dir,
        )

        trace_path = os.path.join(
            self.metadata.eval_output_dir,
            "miniswe_traces",
            f"{instance.id}.attempt{self.current_attempt}.{uuid.uuid4().hex[:8]}.json",
        )
        settings = MiniRolloutSettings(
            cwd=repo_path,
            step_limit=self.metadata.max_iterations,
            cost_limit=self.mini_cost_limit,
            timeout_seconds=self.metadata.conversation_timeout,
            command_timeout=self.mini_command_timeout,
            bash_env_file=self.mini_bash_env_file or None,
            trace_path=trace_path,
        )
        rollout = run_mini_rollout(
            llm=self.metadata.llm,
            workspace=workspace,
            instruction=instruction,
            settings=settings,
            rollout_key=str(uuid.uuid4()),
        )

        # The submitted bytes are the candidate; the worktree diff is deliberately
        # not used (mini's scratch patch file would pollute `git add -A`).
        git_patch = rollout.submission
        patch_applies = check_patch_applies(
            workspace=workspace,
            repo_path=repo_path,
            patch=git_patch,
        )

        evaluation_result = None
        if self.judge is not None:
            try:
                evaluation_result = self.judge.judge(
                    instance_id=instance.id,
                    git_patch=git_patch,
                    instance_data=instance.data,
                )
                logger.info("Judge result for %s: %s", instance.id, evaluation_result)
            except Exception as e:
                logger.error("Judge failed for %s: %s", instance.id, e)
                evaluation_result = None

        dump = build_history_dump(
            instance_id=instance.id,
            model=rollout.model.model_name,
            temperature=rollout.model.temperature,
            top_p=rollout.model.top_p,
            messages=rollout.messages,
            tools=rollout.tools,
            git_patch=git_patch,
            evaluation=evaluation_result,
            include_evaluation=self.judge is not None,
            miniswe={
                "exit_status": rollout.status,
                "n_calls": rollout.n_calls,
                "cost": rollout.cost,
                "duration_seconds": rollout.duration_seconds,
                "trace_path": trace_path,
                "patch_applies_cleanly": patch_applies,
                "step_limit": self.metadata.max_iterations,
                "command_timeout": self.mini_command_timeout,
                "timeout_seconds": self.metadata.conversation_timeout,
                "bash_env_file": self.mini_bash_env_file or None,
            },
        )
        write_history(
            output_dir=self.metadata.eval_output_dir,
            instance_id=instance.id,
            attempt=self.current_attempt,
            dump=dump,
        )

        return EvalOutput(
            instance_id=instance.id,
            attempt=self.current_attempt,
            test_result={"git_patch": git_patch},
            instruction=instruction,
            error=None,
            history=[],
            metrics=rollout.metrics,
        )


def _validate_mini_args(parser: ArgumentParser, args: Namespace) -> None:
    if args.critic not in MINISWE_SUPPORTED_CRITICS:
        parser.error(
            f"--critic {args.critic!r} relies on OpenHands SDK events, which mini "
            f"rollouts do not produce. Supported critics: "
            f"{', '.join(MINISWE_SUPPORTED_CRITICS)}."
        )
    if args.condenser != "none":
        parser.error(
            "--condenser is an OpenHands-only feature and is not supported by "
            "mini-swe-agent runs."
        )


def main() -> None:
    prompt_dir = (Path(__file__).parent / "prompts").resolve()
    default_prompt_path = prompt_dir / "miniswe-default.j2"
    assert default_prompt_path.exists(), (
        f"Default mini prompt {default_prompt_path} not found"
    )

    parser = get_parser()
    parser.add_argument(
        "--prompt-path",
        type=str,
        default=str(default_prompt_path),
        help="Path to the mini-swe-agent instruction template",
    )
    parser.add_argument(
        "--mini-cost-limit",
        type=float,
        default=0.0,
        help="Stop a rollout after this cost in USD (default: 0, no limit)",
    )
    parser.add_argument(
        "--mini-command-timeout",
        type=int,
        default=60,
        help="Per-command timeout in seconds (default: 60)",
    )
    parser.add_argument(
        "--mini-bash-env-file",
        type=str,
        default=None,
        help=(
            "Optional rc file exported as BASH_ENV for non-login bash. Off by "
            "default: agent-server images already put the task environment on "
            "PATH and /root/.bashrc is unreadable to the non-root agent user. "
            "Example: --mini-bash-env-file /home/openhands/.bashrc"
        ),
    )
    add_judge_args(parser, default_judge="swebench")
    args = parser.parse_args()
    _validate_mini_args(parser, args)

    if args.max_attempts < 1:
        raise ValueError(f"max_attempts must be >= 1, got {args.max_attempts}")

    llm_config_path = args.llm_config_path
    if not os.path.isfile(llm_config_path):
        raise ValueError(f"LLM config file {llm_config_path} does not exist")
    llm_config = OmegaConf.to_container(OmegaConf.load(llm_config_path), resolve=True)
    assert isinstance(llm_config, dict), (
        f"LLM config {llm_config_path} must resolve to a JSON object"
    )
    llm = LLM.model_validate(llm_config)
    logger.info("Using LLM config: %s", llm.model_dump_json(indent=2))

    dataset_description = (
        args.dataset.replace("/", "__") + "-" + args.split.replace("/", "__")
    )
    # Harness-specific suffix: mini runs with identical model/dataset/note resume
    # independently from OpenHands runs.
    structured_output_dir = os.path.join(
        construct_eval_output_dir(
            base_dir=args.output_dir,
            dataset_name=dataset_description,
            model_name=llm.model,
            max_iterations=args.max_iterations,
            eval_note=args.note,
        ),
        "miniswe",
    )

    critic = create_critic(args)
    logger.info("Using critic: %s", type(critic).__name__)

    judge = create_judge(args)
    if judge is not None:
        if isinstance(judge, SWEBenchJudge):
            judge.evaluation_log_dir = str(Path(structured_output_dir) / "judge_logs")
        logger.info("Using judge: %s", type(judge).__name__)

    metadata = EvalMetadata(
        llm=llm,
        dataset=args.dataset,
        dataset_split=args.split,
        max_iterations=args.max_iterations,
        eval_output_dir=structured_output_dir,
        details={},
        prompt_path=args.prompt_path,
        eval_limit=args.n_limit,
        env_setup_commands=["export PIP_CACHE_DIR=~/.cache/pip"],
        max_attempts=args.max_attempts,
        critic=critic,
        selected_instances_file=args.select,
        max_retries=args.max_retries,
        workspace_type=args.workspace,
        conversation_timeout=args.conversation_timeout,
        condenser="none",
    )

    bash_env_file = args.mini_bash_env_file
    if isinstance(bash_env_file, str) and bash_env_file.lower() in ("", "none"):
        bash_env_file = None

    evaluator = SWEBenchMiniEvaluation(
        metadata=metadata,
        num_workers=args.num_workers,
        judge=judge,
        mini_cost_limit=args.mini_cost_limit,
        mini_command_timeout=args.mini_command_timeout,
        mini_bash_env_file=bash_env_file,
    )

    evaluator.run(on_result=get_default_on_result_writer(evaluator.output_path))

    logger.info("Evaluation completed!")


if __name__ == "__main__":
    main()
