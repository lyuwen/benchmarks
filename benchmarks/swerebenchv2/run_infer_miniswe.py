"""mini-swe-agent inference entry point for SWE-rebench-V2.

Same structure as the SWE-bench/Scale-SWE mini entry points: the rebench-v2
``Evaluation`` keeps dataset loading, image resolution, workspace provisioning,
retries, history publication and the judge; only the per-instance agent loop is
mini-swe-agent. The submitted patch is the candidate and is passed through the
same baseline-untracked-artifact filtering as the OpenHands runner.

Usage:
    uv run python -m benchmarks.swerebenchv2.run_infer_miniswe model.json \\
      --dataset thirdparty/SWE-rebench-V2/SWE-rebench-V2-py.jsonl \\
      --workspace flex --n-limit 1 --num-workers 1 --max-attempts 1 \\
      --judge swerebenchv2
"""

import os
import uuid
from pathlib import Path
from typing import List, Literal, Optional

from omegaconf import OmegaConf
from pydantic import Field

import benchmarks.swerebenchv2.judge  # noqa: F401  (register the judge)
from benchmarks.swerebenchv2.run_infer import (
    SWERebenchV2Evaluation,
    get_instruction,
)
from benchmarks.utils.args_parser import get_parser
from benchmarks.utils.critics import create_critic
from benchmarks.utils.evaluation import OnResult
from benchmarks.utils.evaluation_utils import (
    construct_eval_output_dir,
    get_default_on_result_writer,
)
from benchmarks.utils.execution_judge import add_judge_args, create_judge
from benchmarks.utils.miniswe import (
    MiniRolloutSettings,
    add_mini_common_args,
    build_history_dump,
    check_patch_applies,
    get_head_commit,
    head_matches_commit,
    mini_progress_console,
    mini_progress_path,
    publish_canonical_histories,
    resolve_bash_env_file,
    run_mini_rollout,
    validate_mini_args,
    write_history,
)
from benchmarks.utils.mirror_config import get_mirror_env_commands
from benchmarks.utils.models import EvalInstance, EvalMetadata, EvalOutput
from benchmarks.utils.patch_utils import remove_files_from_patch
from openhands.sdk import LLM, get_logger
from openhands.sdk.workspace import RemoteWorkspace


logger = get_logger(__name__)


class SWERebenchV2MiniEvaluation(SWERebenchV2Evaluation):
    """SWE-rebench-V2 evaluator whose per-instance agent loop is mini-swe-agent."""

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
            "PATH and /root/.bashrc is unreadable to the non-root agent user"
        ),
    )
    mini_progress: Literal["console", "file", "none"] = Field(
        default="console",
        description=(
            "Live rollout progress: 'console' writes a per-instance progress file "
            "and mirrors lines to the console, 'file' writes the file only, "
            "'none' disables progress output"
        ),
    )

    def _capture_conversation_archive(
        self, workspace: RemoteWorkspace, instance: EvalInstance
    ) -> None:
        logger.debug("[child] mini rollout has no conversation archive to capture")

    def run(self, *, on_result: Optional[OnResult] = None) -> List[EvalOutput]:
        outputs = super().run(on_result=on_result)
        publish_canonical_histories(self.metadata.eval_output_dir, self.output_path)
        return outputs

    def evaluate_instance(
        self, instance: EvalInstance, workspace: RemoteWorkspace
    ) -> EvalOutput:
        assert self.metadata.prompt_path is not None
        base_commit = str(instance.data["base_commit"])
        # Agent-server images are not guaranteed to sit at base_commit; reset so
        # the agent's diff and the judge (which applies the patch to base_commit)
        # share one baseline.
        repo_path = self.prepare_repo(workspace, instance, checkout_commit=base_commit)
        baseline_untracked_files = self.baseline_untracked_files(workspace, repo_path)

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
            progress_path=mini_progress_path(
                self.mini_progress, self.metadata.eval_output_dir, instance.id
            ),
            progress_console=mini_progress_console(self.mini_progress),
            instance_id=instance.id,
            attempt=self.current_attempt,
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
        if baseline_untracked_files:
            # Keep the legacy runner's postprocessing boundary: image build
            # artifacts must never reach the judge's `git apply`.
            filtered_patch = remove_files_from_patch(
                git_patch, baseline_untracked_files
            )
            if filtered_patch != git_patch:
                logger.info(
                    "Removed pre-existing untracked files from the submitted patch"
                )
            git_patch = filtered_patch
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
                "baseline_untracked_files": len(baseline_untracked_files),
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


def main() -> None:
    prompt_dir = (Path(__file__).parent / "prompts").resolve()
    default_prompt_path = prompt_dir / "miniswe-default.j2"
    assert default_prompt_path.exists(), (
        f"Default mini prompt {default_prompt_path} not found"
    )

    parser = get_parser()
    parser.set_defaults(
        dataset="thirdparty/SWE-rebench-V2/SWE-rebench-V2-py.jsonl",
        workspace="flex",
    )
    parser.add_argument(
        "--prompt-path",
        type=str,
        default=str(default_prompt_path),
        help="Path to the mini-swe-agent instruction template",
    )
    parser.add_argument(
        "--docker-image-prefix",
        type=str,
        default=None,
        help=(
            "Override image namespace/registry. Replaces everything before the "
            "last '/' in the dataset image_name."
        ),
    )
    add_mini_common_args(parser)
    add_judge_args(parser, default_judge="swerebenchv2")
    args = parser.parse_args()
    validate_mini_args(parser, args)

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
        logger.info("Using judge: %s", type(judge).__name__)

    env_vars = [
        f'export {k}="{v}"' for k, v in os.environ.items() if k.startswith("OH_")
    ]

    metadata = EvalMetadata(
        llm=llm,
        dataset=args.dataset,
        dataset_split=args.split,
        max_iterations=args.max_iterations,
        eval_output_dir=structured_output_dir,
        details={},
        prompt_path=args.prompt_path,
        eval_limit=args.n_limit,
        env_setup_commands=get_mirror_env_commands()
        + ["export PIP_CACHE_DIR=~/.cache/pip"]
        + env_vars,
        max_attempts=args.max_attempts,
        critic=critic,
        selected_instances_file=args.select,
        max_retries=args.max_retries,
        workspace_type=args.workspace,
        conversation_timeout=args.conversation_timeout,
        condenser="none",
    )

    evaluator = SWERebenchV2MiniEvaluation(
        metadata=metadata,
        num_workers=args.num_workers,
        docker_image_prefix=args.docker_image_prefix,
        judge=judge,
        mini_cost_limit=args.mini_cost_limit,
        mini_command_timeout=args.mini_command_timeout,
        mini_bash_env_file=resolve_bash_env_file(args.mini_bash_env_file),
        mini_progress=args.mini_progress,
    )

    evaluator.run(on_result=get_default_on_result_writer(evaluator.output_path))

    logger.info("Evaluation completed!")


if __name__ == "__main__":
    main()
