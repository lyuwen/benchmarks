"""Blackbox-harness inference entry point for SWE-rebench-V2.

The harness runs inside the dataset's official multi-language task image. Its
model traffic is captured by a per-instance ``rollout-manager`` proxy, whose
trajectory export is written next to the evaluation outputs.

Usage:
    uv run swerebenchv2-infer-harnesses .llm_config/model.json \\
      --harness claude-code --n-limit 1 --num-workers 1 --max-attempts 1 \\
      --judge swerebenchv2
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, cast

from pydantic import Field

from benchmarks.swerebenchv2.build_images import get_official_docker_image

# Import judge to trigger registration.
from benchmarks.swerebenchv2.judge import SWERebenchV2Judge  # noqa: F401
from benchmarks.swerebenchv2.run_infer import SWERebenchV2Evaluation, get_instruction
from benchmarks.utils.args_parser import get_parser
from benchmarks.utils.critics import create_critic
from benchmarks.utils.evaluation import OnResult
from benchmarks.utils.evaluation_utils import (
    construct_eval_output_dir,
    get_default_on_result_writer,
)
from benchmarks.utils.execution_judge import (
    ExecutionBasedJudge,
    add_judge_args,
    create_judge,
)
from benchmarks.utils.harnesses import (
    HarnessConfig,
    HarnessRunRequest,
    add_harness_args,
    ensure_distribution,
    harness_config_from_args,
    publish_canonical_histories,
    run_harness_instance,
    validate_harness_args,
)
from benchmarks.utils.harnesses.claude_code import build_distribution_bind_volumes
from benchmarks.utils.harnesses.llm_config import load_llm_config
from benchmarks.utils.harnesses.runner import env_from_setup_commands
from benchmarks.utils.harnesses.workspace import DockerContainerWorkspace
from benchmarks.utils.mirror_config import get_mirror_env_commands
from benchmarks.utils.models import EvalInstance, EvalMetadata, EvalOutput
from openhands.sdk import get_logger
from openhands.sdk.workspace import RemoteWorkspace


logger = get_logger(__name__)


class SWERebenchV2HarnessEvaluation(SWERebenchV2Evaluation):
    """SWE-rebench-V2 evaluator whose per-instance loop is a blackbox harness."""

    harness_config: HarnessConfig = Field(default_factory=HarnessConfig)

    def _capture_conversation_archive(
        self, workspace: RemoteWorkspace, instance: EvalInstance
    ) -> None:
        logger.debug("[child] harness run has no conversation archive to capture")

    def run(self, *, on_result: OnResult | None = None) -> List[EvalOutput]:
        outputs = super().run(on_result=on_result)
        publish_canonical_histories(self.metadata.eval_output_dir, self.output_path)
        return outputs

    # ---- Hook: prepare a workspace per instance ----------------------------------
    def prepare_workspace(
        self,
        instance: EvalInstance,
        resource_factor: int = 1,
        forward_env: list[str] | None = None,
    ) -> RemoteWorkspace:
        if self.metadata.workspace_type != "docker":
            raise ValueError(
                "Harness inference supports --workspace docker only: the harness "
                "must execute inside the task container"
            )
        image = get_official_docker_image(
            str(instance.data["image_name"]), self.docker_image_prefix
        )
        distribution = ensure_distribution(self.harness_config)
        bind_volumes = build_distribution_bind_volumes(
            self.harness_config, distribution
        )
        workspace = DockerContainerWorkspace.start(
            image=image,
            working_dir="/",
            pull_missing=not self.harness_config.skip_image_pull,
            bind_volumes=bind_volumes,
        )
        for cmd in self.metadata.env_setup_commands or []:
            result = workspace.execute_command(cmd)
            if result.exit_code != 0:
                logger.warning(
                    "Harness env setup command failed (continuing): %r: %s",
                    cmd,
                    result.stderr,
                )
        return cast(RemoteWorkspace, workspace)

    # ---- Hook: evaluate one instance ---------------------------------------------
    def evaluate_instance(
        self, instance: EvalInstance, workspace: RemoteWorkspace
    ) -> EvalOutput:
        assert isinstance(workspace, DockerContainerWorkspace)
        # SWE-rebench-V2 images place the repo at /<repo_name> (not /testbed).
        repo_name = str(instance.data["repo"]).split("/")[-1]
        repo_path = f"/{repo_name}"
        base_commit = str(instance.data["base_commit"])
        instance.data["repo_path"] = repo_path
        instruction = get_instruction(
            instance=instance.data,
            metadata=self.metadata,
            workspace_path=workspace.working_dir,
        )
        result = run_harness_instance(
            HarnessRunRequest(
                instance=instance,
                attempt=self.current_attempt,
                repo_path=repo_path,
                base_commit=base_commit,
                instruction=instruction,
                output_dir=self.metadata.eval_output_dir,
                max_iterations=self.metadata.max_iterations,
                conversation_timeout=(
                    float(self.metadata.conversation_timeout)
                    if self.metadata.conversation_timeout
                    else None
                ),
                extra_env=env_from_setup_commands(self.metadata.env_setup_commands),
            ),
            workspace=workspace,
            config=self.harness_config,
            llm=self.metadata.llm,
            judge=self.judge,
        )
        return EvalOutput(
            instance_id=instance.id,
            attempt=self.current_attempt,
            test_result={"git_patch": result.git_patch},
            instruction=result.instruction,
            error=None,
            history=[],
            metrics=result.metrics,
        )


def main() -> None:
    prompt_dir = (Path(__file__).parent / "prompts").resolve()
    default_prompt_path = prompt_dir / "harness-default.j2"
    assert default_prompt_path.exists(), (
        f"Default harness prompt {default_prompt_path} not found"
    )
    choices = [str(path) for path in prompt_dir.glob("*.j2")]

    parser = get_parser()
    parser.set_defaults(
        dataset="thirdparty/SWE-rebench-V2/SWE-rebench-V2-py.jsonl",
        workspace="docker",
    )
    parser.add_argument(
        "--prompt-path",
        type=str,
        default=str(default_prompt_path),
        choices=choices,
        help="Path to the prompt template rendered for the harness",
    )
    parser.add_argument(
        "--docker-image-prefix",
        type=str,
        default=None,
        help=(
            "Override image namespace/registry. Replaces everything before "
            "the last '/' in the dataset image_name, e.g. 'myregistry.com/myorg'."
        ),
    )
    add_judge_args(parser, default_judge="swerebenchv2")
    add_harness_args(parser)
    args = parser.parse_args()
    validate_harness_args(parser, args)

    llm, config_headers = load_llm_config(args)
    harness_config = harness_config_from_args(args)
    if config_headers:
        merged = {**config_headers, **harness_config.custom_headers}
        harness_config = harness_config.model_copy(update={"custom_headers": merged})

    distribution = ensure_distribution(harness_config)
    logger.info(
        "Harness distribution: %s@%s (%s) at %s",
        harness_config.name,
        harness_config.version,
        distribution.platform,
        distribution.binary_path,
    )

    dataset_description = (
        args.dataset.replace("/", "__") + "-" + args.split.replace("/", "__")
    )
    base_output_dir = construct_eval_output_dir(
        base_dir=args.output_dir,
        dataset_name=dataset_description,
        model_name=llm.model,
        max_iterations=args.max_iterations,
        eval_note=args.note,
    )
    structured_output_dir = os.path.join(
        base_output_dir, harness_config.harness_output_subdir
    )

    critic = create_critic(args)
    logger.info("Using critic: %s", type(critic).__name__)
    judge = create_judge(args)
    if isinstance(judge, SWERebenchV2Judge):
        logger.info("Using judge: %s", type(judge).__name__)

    metadata = EvalMetadata(
        llm=llm,
        dataset=args.dataset,
        dataset_split=args.split,
        max_iterations=args.max_iterations,
        eval_output_dir=structured_output_dir,
        details={"harness": harness_config.for_metadata()},
        prompt_path=args.prompt_path,
        eval_limit=args.n_limit,
        env_setup_commands=get_mirror_env_commands()
        + ["export PIP_CACHE_DIR=~/.cache/pip"],
        max_attempts=args.max_attempts,
        critic=critic,
        selected_instances_file=args.select,
        max_retries=args.max_retries,
        workspace_type=args.workspace,
        conversation_timeout=args.conversation_timeout,
        condenser="none",
    )

    evaluator = SWERebenchV2HarnessEvaluation(
        metadata=metadata,
        num_workers=args.num_workers,
        judge=cast(ExecutionBasedJudge | None, judge),
        harness_config=harness_config,
        docker_image_prefix=args.docker_image_prefix,
    )
    evaluator.run(on_result=get_default_on_result_writer(evaluator.output_path))
    logger.info("Harness evaluation completed!")


if __name__ == "__main__":
    main()
