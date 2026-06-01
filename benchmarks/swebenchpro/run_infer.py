import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, List

RUNTIME_IMPORT_ERROR: ModuleNotFoundError | None = None

try:
    from omegaconf import OmegaConf
    from pydantic import Field

    from benchmarks.swebenchpro.build_images import (
        extract_custom_tag,
        get_official_docker_image,
    )
    from benchmarks.swebenchpro.judge import SWEBenchProJudge  # noqa: F401
    from benchmarks.utils.args_parser import get_parser
    from benchmarks.utils.build_utils import build_image
    from benchmarks.utils.constants import EVAL_AGENT_SERVER_IMAGE
    from benchmarks.utils.conversation import build_event_persistence_callback
    from benchmarks.utils.critics import create_critic
    from benchmarks.utils.dataset import get_dataset
    from benchmarks.utils.evaluation import Evaluation
    from benchmarks.utils.evaluation_utils import (
        construct_eval_output_dir,
        get_default_on_result_writer,
    )
    from benchmarks.utils.execution_judge import (
        ExecutionBasedJudge,
        add_judge_args,
        create_judge,
    )
    from benchmarks.utils.fake_user_response import run_conversation_with_fake_user_response
    from benchmarks.utils.image_utils import image_exists
    from benchmarks.utils.models import EvalInstance, EvalMetadata, EvalOutput
    from benchmarks.utils.version import SDK_SHORT_SHA
    from openhands.sdk import LLM, Agent, Conversation, get_logger
    from openhands.sdk.event.base import LLMConvertibleEvent
    from openhands.sdk.event.llm_convertible.system import SystemPromptEvent
    from openhands.sdk.tool.tool import ToolDefinition
    from openhands.sdk.workspace import RemoteWorkspace
    from openhands.tools.preset.default import get_default_tools
    from openhands.tools.preset.legacy import get_legacy_tools
    from openhands.workspace import APIRemoteWorkspace, DockerWorkspace, FlexWorkspace

    logger = get_logger(__name__)
except ModuleNotFoundError as exc:
    RUNTIME_IMPORT_ERROR = exc
    OmegaConf = None

    def Field(*args, **kwargs):
        return kwargs.get("default")

    class Evaluation:  # type: ignore[override]
        pass

    class ExecutionBasedJudge:
        pass

    class EvalInstance:
        id: str
        data: dict[str, Any]

    class EvalMetadata:
        details: dict[str, Any] | None
        prompt_path: str | None
        llm: Any
        eval_output_dir: str
        max_iterations: int
        conversation_timeout: int | None
        dataset: str
        dataset_split: str
        eval_limit: int
        selected_instances_file: str | None
        env_setup_commands: list[str] | None
        workspace_type: str

    class EvalOutput:
        pass

    class RemoteWorkspace:
        pass

    class LLMConvertibleEvent:
        pass

    class SystemPromptEvent:
        pass

    class ToolDefinition:
        pass

    class APIRemoteWorkspace:
        pass

    class DockerWorkspace:
        pass

    class FlexWorkspace:
        pass

    LLM = Agent = Conversation = object
    SDK_SHORT_SHA = ""
    EVAL_AGENT_SERVER_IMAGE = ""
    logger = logging.getLogger(__name__)


def _build_bootstrap_help_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Evaluation inference")
    parser.add_argument("llm_config_path", type=str, nargs="?", help="Path to JSON LLM configuration")
    parser.add_argument("--dataset", type=str, default="ScaleAI/SWE-bench_Pro", help="Dataset name")
    parser.add_argument("--split", type=str, default="test", help="Dataset split")
    parser.add_argument(
        "--workspace",
        type=str,
        default="flex",
        choices=["docker", "flex", "remote"],
        help="Type of workspace to use (default: docker)",
    )
    parser.add_argument("--max-iterations", type=int, default=100, help="Maximum iterations")
    parser.add_argument("--num-workers", type=int, default=1, help="Number of evaluation workers")
    parser.add_argument("--note", type=str, default="initial", help="Evaluation note")
    parser.add_argument("--output-dir", type=str, default="./eval_outputs", help="Evaluation output directory")
    parser.add_argument("--n-limit", type=int, default=0, help="Limit number of instances to evaluate")
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="Maximum number of attempts for iterative mode (default: 3, min: 1)",
    )
    parser.add_argument("--critic", type=str, default="pass", help="Name of the critic to use for evaluation")
    parser.add_argument("--critic-config", type=str, default=None, help="Path to JSON critic config")
    parser.add_argument("--select", type=str, default=None, help="Path to selected instance IDs file")
    parser.add_argument("--max-retries", type=int, default=3, help="Maximum retries for instance failures")
    parser.add_argument(
        "--conversation-timeout",
        type=int,
        default=None,
        help="Timeout in seconds for each conversation.run() call (default: None, no timeout)",
    )

    prompt_dir = (Path(__file__).parent / "prompts").resolve()
    choices = [str(p.relative_to(Path.cwd())) for p in prompt_dir.glob("*.j2")]
    default_prompt_path = prompt_dir / "default.j2"
    assert default_prompt_path.exists(), f"Default prompt {default_prompt_path} not found"
    parser.add_argument(
        "--prompt-path",
        type=str,
        default=str(default_prompt_path),
        choices=choices,
        help="Path to prompt template file",
    )
    parser.add_argument("--use-legacy-tools", action="store_true", help="Use legacy tools")
    parser.add_argument("--bind-dev-sdk", action="store_true", help="Bind SDK paths for dev features")
    parser.add_argument(
        "--judge",
        nargs="?",
        const="swebenchpro",
        default=None,
        help="Enable execution-based judge after the agent finishes.",
    )
    parser.add_argument("--judge-timeout", type=int, default=1800, help="Per-instance judge timeout in seconds (default: 1800).")
    parser.add_argument("--judge-rm-image", action="store_true", help="Remove Docker image after each judge evaluation (default: False).")
    parser.add_argument("--judge-force-rebuild", action="store_true", help="Force rebuild Docker images for judge evaluation (default: False).")
    return parser


def get_instruction(
    instance: dict,
    metadata: EvalMetadata,
    workspace_path: str,
) -> str:
    """Generate instruction for the agent."""
    from jinja2 import Environment, FileSystemLoader

    workspace_dir_name = instance["repo"].split("/")[-1]
    assert metadata.details is not None
    assert metadata.prompt_path is not None

    prompts_dir = os.path.dirname(metadata.prompt_path)
    template_name = os.path.basename(metadata.prompt_path)
    env = Environment(loader=FileSystemLoader(prompts_dir))
    template = env.get_template(template_name)

    context = {
        "instance": instance,
        "workspace_dir_name": workspace_dir_name,
        "actual_workspace_path": workspace_path,
        "metadata": metadata,
        "test_instructions": "",
    }
    return template.render(context)


class SWEBenchProEvaluation(Evaluation):
    use_legacy_tools: int = Field(
        default=False, description="Use legacy CodeActAgent tools"
    )
    bind_dev_sdk: int = Field(
        default=False, description="Bind SDK paths for dev features"
    )
    judge: ExecutionBasedJudge | None = Field(
        default=None, description="Optional execution-based judge"
    )

    def prepare_instances(self) -> List[EvalInstance]:
        logger.info("Setting up SWE-bench Pro evaluation data")

        df = get_dataset(
            dataset_name=self.metadata.dataset,
            split=self.metadata.dataset_split,
            eval_limit=self.metadata.eval_limit,
            selected_instances_file=self.metadata.selected_instances_file,
        )

        instances: List[EvalInstance] = []
        for _, row in df.iterrows():
            instance_id = str(row["instance_id"])
            instances.append(EvalInstance(id=instance_id, data=row.to_dict()))

        logger.info("Total instances to process: %d", len(instances))
        return instances

    def prepare_workspace(
        self,
        instance: EvalInstance,
        resource_factor: int = 1,
        forward_env: list[str] | None = None,
    ) -> RemoteWorkspace:
        official_docker_image = get_official_docker_image(
            str(instance.data["dockerhub_tag"])
        )
        build_target = "source-minimal"
        custom_tag = extract_custom_tag(official_docker_image)
        suffix = f"-{build_target}" if build_target != "binary" else ""

        if self.metadata.workspace_type == "flex":
            agent_plugin_image = os.getenv(
                "AGENT_PLUGIN_IMAGE", "openhands/agent-plugin"
            )
            bind_volumes = []
            if self.bind_dev_sdk:
                sdk_base = (
                    Path(__file__).parent.parent.parent / "vendor/software-agent-sdk"
                )
                for module in ["tools", "sdk", "agent-server", "workspace"]:
                    bind_volumes.append(
                        f"{sdk_base}/openhands-{module}/openhands/{module}:"
                        f"/agent-server/.venv/lib/python3.12/site-packages/openhands/{module}"
                        ":ro"
                    )
            workspace = FlexWorkspace(
                base_image=official_docker_image,
                agent_plugin_image=agent_plugin_image,
                working_dir="/workspace",
                forward_env=forward_env or [],
                bind_volumes=bind_volumes,
            )
        elif self.metadata.workspace_type == "docker":
            base_agent_image = (
                f"{EVAL_AGENT_SERVER_IMAGE}:{SDK_SHORT_SHA}-{custom_tag}{suffix}"
            )
            agent_server_image = base_agent_image
            skip_build = os.getenv("SKIP_BUILD", "1").lower() in ("1", "true", "yes")
            logger.info("SKIP_BUILD=%s", skip_build)
            if not skip_build:
                logger.info(
                    "Building workspace from %s for instance %s. This may take a while...\n"
                    "You can run benchmarks/swebenchpro/build_images.py and set SKIP_BUILD=1 to skip building and use pre-built agent-server image.",
                    official_docker_image,
                    instance.id,
                )
                output = build_image(
                    base_image=official_docker_image,
                    target_image=EVAL_AGENT_SERVER_IMAGE,
                    custom_tag=custom_tag,
                    target=build_target,
                    push=False,
                )
                logger.info("Image build output: %s", output)
                assert output.error is None, f"Image build failed: {output.error}"
                if base_agent_image not in output.tags:
                    raise RuntimeError(
                        f"Built image tags {output.tags} do not include expected tag "
                        f"{base_agent_image}"
                    )

            bind_volumes = []
            if self.bind_dev_sdk:
                sdk_base = (
                    Path(__file__).parent.parent.parent / "vendor/software-agent-sdk"
                )
                for module in ["tools", "sdk", "agent-server", "workspace"]:
                    bind_volumes.append(
                        f"{sdk_base}/openhands-{module}/openhands/{module}:"
                        f"/agent-server/.venv/lib/python3.12/site-packages/openhands/{module}"
                    )
            workspace = DockerWorkspace(
                server_image=agent_server_image,
                working_dir="/workspace",
                forward_env=forward_env or [],
                bind_volumes=bind_volumes,
            )
        elif self.metadata.workspace_type == "remote":
            runtime_api_key = os.getenv("RUNTIME_API_KEY")
            sdk_short_sha = os.getenv("SDK_SHORT_SHA", SDK_SHORT_SHA)
            if not runtime_api_key:
                raise ValueError(
                    "RUNTIME_API_KEY environment variable is not set for remote workspace"
                )

            agent_server_image = (
                f"{EVAL_AGENT_SERVER_IMAGE}:{sdk_short_sha}-{custom_tag}{suffix}"
            )
            if not image_exists(agent_server_image):
                raise RuntimeError(
                    f"Agent server image {agent_server_image} does not exist in container registry, "
                    "make sure to build, push it, and make it public accessible before using remote workspace."
                )
            logger.info(
                "Using remote workspace with image %s (sdk sha: %s, resource_factor: %s)",
                agent_server_image,
                sdk_short_sha,
                resource_factor,
            )
            startup_timeout = float(
                os.getenv("REMOTE_RUNTIME_STARTUP_TIMEOUT", "600")
            )
            workspace = APIRemoteWorkspace(
                runtime_api_url=os.getenv(
                    "RUNTIME_API_URL", "https://runtime.eval.all-hands.dev"
                ),
                runtime_api_key=runtime_api_key,
                server_image=agent_server_image,
                target_type="source" if "source" in build_target else "binary",
                forward_env=forward_env or [],
                resource_factor=resource_factor,
                init_timeout=startup_timeout,
                startup_wait_timeout=startup_timeout,
            )
        else:
            raise ValueError(
                f"Unsupported workspace_type: {self.metadata.workspace_type}"
            )

        for cmd in self.metadata.env_setup_commands or []:
            res = workspace.execute_command(cmd)
            if res.exit_code != 0:
                raise RuntimeError(
                    f"Failed to run env setup command '{cmd}': {res.stderr}"
                )
            logger.debug("Ran env setup command '%s': %s", cmd, res.stdout)
        return workspace

    def evaluate_instance(
        self, instance: EvalInstance, workspace: RemoteWorkspace
    ) -> EvalOutput:
        if self.use_legacy_tools:
            tools = get_legacy_tools(enable_browser=False)
        else:
            tools = get_default_tools(enable_browser=False)

        agent = Agent(
            llm=self.metadata.llm,
            tools=tools,
            system_prompt_kwargs={"cli_mode": True},
        )

        assert isinstance(workspace, RemoteWorkspace)

        repo_path = "/workspace/repo/"
        instance.data["repo_path"] = repo_path

        persist_callback = build_event_persistence_callback(
            run_id=self.metadata.eval_output_dir,
            instance_id=instance.id,
            attempt=self.current_attempt,
        )

        conversation = Conversation(
            agent=agent,
            workspace=workspace,
            callbacks=[persist_callback],
            max_iteration_per_run=self.metadata.max_iterations,
        )

        logger.info("repo_path: %s (source: /app)", repo_path)
        cp_repo = workspace.execute_command(
            f"mkdir -p {repo_path} ; cp -r /app/. {repo_path}"
        )
        assert cp_repo.exit_code == 0, f"cp repo failed: {cp_repo.stderr}"

        git_reset = workspace.execute_command(f"cd {repo_path} ; git reset --hard")
        assert git_reset.exit_code == 0, f"git reset failed: {git_reset.stderr}"

        base_commit = str(instance.data["base_commit"])
        brief_history = workspace.execute_command(
            f"cd {repo_path} ; git --no-pager log --oneline -10"
        )
        logger.info(
            "Repo status:\n* Current commit: %s\n* Top 10 history:\n%s",
            base_commit,
            brief_history.stdout.strip(),
        )

        instruction = get_instruction(
            instance=instance.data,
            metadata=self.metadata,
            workspace_path=workspace.working_dir,
        )
        conversation.send_message(instruction)
        run_conversation_with_fake_user_response(
            conversation, timeout=self.metadata.conversation_timeout
        )

        git_status_diff = workspace.execute_command(
            f"cd {repo_path} ; git status ; git --no-pager diff"
        )
        logger.info("Repo status:\n%s", git_status_diff.stdout.strip())

        workspace.execute_command(f"cd {repo_path} ; git add -A")
        workspace.execute_command(
            f"cd {repo_path} && "
            "git config --global user.email 'evaluation@openhands.dev' && "
            "git config --global user.name 'OpenHands Evaluation' && "
            "git commit -m 'patch'"
        )

        git_patch_result = workspace.execute_command(
            f"cd {repo_path} ; git --no-pager diff --no-color {base_commit} HEAD"
        )
        assert git_patch_result.exit_code == 0, (
            f"git diff failed: {git_patch_result.stderr}"
        )
        git_patch = git_patch_result.stdout

        evaluation_result = None
        if self.judge is not None:
            try:
                evaluation_result = self.judge.judge(
                    instance_id=instance.id,
                    git_patch=git_patch,
                    instance_data=instance.data,
                )
                logger.info("Judge result for %s: %s", instance.id, evaluation_result)
            except Exception as exc:
                logger.error("Judge failed for %s: %s", instance.id, exc)
                evaluation_result = None

        messages = []
        tools_list = []

        convertible_events = [
            event
            for event in conversation.state.events
            if isinstance(event, LLMConvertibleEvent)
        ]
        msgs = LLMConvertibleEvent.events_to_messages(convertible_events)

        for msg in msgs:
            msg_copy = msg.model_copy(update={"send_reasoning_content": True})
            messages.append(msg_copy.to_chat_dict())

        for event in conversation.state.events:
            if isinstance(event, SystemPromptEvent):
                for tool in event.tools:
                    if isinstance(tool, ToolDefinition):
                        tools_list.append(tool.to_openai_tool())

        if not tools_list and tools:
            for tool in tools:
                if isinstance(tool, ToolDefinition):
                    tools_list.append(tool.to_openai_tool())

        dump_data = {
            "instance_id": instance.id,
            "messages": messages,
            "model": self.metadata.llm.model,
            "tools": tools_list,
            "temperature": self.metadata.llm.temperature,
            "top_p": self.metadata.llm.top_p,
            "test_result": {"git_patch": git_patch},
        }
        if self.judge is not None:
            dump_data["evaluation"] = evaluation_result

        history_file = os.path.join(
            self.metadata.eval_output_dir, f"{instance.id}.history.json"
        )
        with open(history_file, "w") as file:
            json.dump(dump_data, file, indent=2)
        logger.info("Dumped conversation history to %s", history_file)

        return EvalOutput(
            instance_id=instance.id,
            attempt=self.current_attempt,
            test_result={"git_patch": git_patch},
            instruction=instruction,
            error=None,
            history=list(conversation.state.events),
            metrics=conversation.conversation_stats.get_combined_metrics(),
        )


def main() -> None:
    if RUNTIME_IMPORT_ERROR is not None and any(arg in sys.argv[1:] for arg in ["-h", "--help"]):
        _build_bootstrap_help_parser().parse_args(["--help"])
        return

    if RUNTIME_IMPORT_ERROR is not None:
        raise RUNTIME_IMPORT_ERROR

    prompt_dir = (Path(__file__).parent / "prompts").resolve()
    choices = [str(p.relative_to(Path.cwd())) for p in prompt_dir.glob("*.j2")]
    default_prompt_path = prompt_dir / "default.j2"
    assert default_prompt_path.exists(), (
        f"Default prompt {default_prompt_path} not found"
    )

    parser = get_parser()
    parser.set_defaults(dataset="ScaleAI/SWE-bench_Pro", split="test", workspace="flex")
    parser.add_argument(
        "--prompt-path",
        type=str,
        default=str(default_prompt_path),
        choices=choices,
        help="Path to prompt template file",
    )
    parser.add_argument(
        "--use-legacy-tools",
        action="store_true",
        help="Use legacy tools",
    )
    parser.add_argument(
        "--bind-dev-sdk",
        action="store_true",
        help="Bind SDK paths for dev features",
    )
    add_judge_args(parser, default_judge="swebenchpro")
    args = parser.parse_args()

    if args.max_attempts < 1:
        raise ValueError(f"max_attempts must be >= 1, got {args.max_attempts}")

    llm_config_path = args.llm_config_path
    if not os.path.isfile(llm_config_path):
        raise ValueError(f"LLM config file {llm_config_path} does not exist")
    llm_config = json.dumps(
        OmegaConf.to_container(OmegaConf.load(llm_config_path), resolve=True)
    )
    llm = LLM.model_validate_json(llm_config)
    logger.info("Using LLM config: %s", llm.model_dump_json(indent=2))

    dataset_description = (
        args.dataset.replace("/", "__") + "-" + args.split.replace("/", "__")
    )
    structured_output_dir = construct_eval_output_dir(
        base_dir=args.output_dir,
        dataset_name=dataset_description,
        model_name=llm.model,
        max_iterations=args.max_iterations,
        eval_note=args.note,
    )

    critic = create_critic(args)
    logger.info("Using critic: %s", type(critic).__name__)

    judge = create_judge(args)
    if judge is not None:
        logger.info("Using judge: %s", type(judge).__name__)

    env_vars = [
        f'export {key}="{value}"'
        for key, value in os.environ.items()
        if key.startswith("OH_")
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
        env_setup_commands=["export PIP_CACHE_DIR=~/.cache/pip"] + env_vars,
        max_attempts=args.max_attempts,
        critic=critic,
        selected_instances_file=args.select,
        max_retries=args.max_retries,
        workspace_type=args.workspace,
        conversation_timeout=args.conversation_timeout,
    )

    evaluator = SWEBenchProEvaluation(
        metadata=metadata,
        num_workers=args.num_workers,
        use_legacy_tools=args.use_legacy_tools,
        bind_dev_sdk=args.bind_dev_sdk,
        judge=judge,
    )

    evaluator.run(on_result=get_default_on_result_writer(evaluator.output_path))

    logger.info("Evaluation completed!")


if __name__ == "__main__":
    main()
