import os
import json
import inspect
from pathlib import Path
from typing import List

from jinja2 import Environment, FileSystemLoader
from pydantic import Field
from omegaconf import OmegaConf

from benchmarks.utils.args_parser import get_parser
from benchmarks.utils.mirror_config import get_mirror_env_commands
from benchmarks.utils.constants import EVAL_AGENT_SERVER_IMAGE
from benchmarks.utils.conversation import build_event_persistence_callback
from benchmarks.utils.critics import create_critic
from benchmarks.utils.execution_judge import (
    ExecutionBasedJudge,
    add_judge_args,
    create_judge,
)
from benchmarks.utils.dataset import get_dataset
from benchmarks.utils.evaluation import Evaluation
from benchmarks.utils.evaluation_utils import (
    construct_eval_output_dir,
    get_default_on_result_writer,
)
from benchmarks.utils.image_utils import image_exists
from benchmarks.utils.models import (
    EvalInstance,
    EvalMetadata,
    EvalOutput,
)
from benchmarks.utils.version import SDK_SHORT_SHA

# Import judge to trigger registration
from benchmarks.scaleswe.judge import ScaleSWEJudge  # noqa: F401
from openhands.sdk import LLM, Agent, Conversation, get_logger
from openhands.sdk.event.base import LLMConvertibleEvent
from openhands.sdk.event.llm_convertible.system import SystemPromptEvent
from openhands.sdk.tool.tool import ToolDefinition
from openhands.sdk.workspace import RemoteWorkspace
from openhands.workspace import APIRemoteWorkspace, DockerWorkspace, FlexWorkspace

# Import DeepSeek minimal preset — also triggers tool registration.
from openhands.tools.deepseek.preset import get_deepseek_minimal_tools
from openhands.tools.deepseek.bash.definition import DeepSeekBashTool  # noqa: F401
from openhands.tools.deepseek.str_replace_editor.definition import DeepSeekStrReplaceEditorTool  # noqa: F401


logger = get_logger(__name__)


def resolve_image_url(image_url: str, prefix: str | None = None) -> str:
    """Resolve Docker image URL with optional namespace/registry override.

    If prefix is provided, replaces everything before the last '/' in image_url.

    Examples:
        resolve_image_url("aweaiteam/scaleswe:tag", None)
            -> "aweaiteam/scaleswe:tag"
        resolve_image_url("aweaiteam/scaleswe:tag", "myregistry.com/myorg")
            -> "myregistry.com/myorg/scaleswe:tag"
    """
    if not prefix:
        return image_url
    _, _, image_name = image_url.rpartition("/")
    return f"{prefix.rstrip('/')}/{image_name}"


def extract_custom_tag(image_url: str) -> str:
    """Extract image name (without registry and tag) for use as a custom tag."""
    name_tag = image_url.split("/")[-1]
    name = name_tag.split(":")[0]
    return name


def get_instruction(
    instance: dict,
    metadata: EvalMetadata,
    workspace_path: str,
) -> str:
    """Generate instruction for the agent."""
    assert metadata.details is not None
    assert metadata.prompt_path is not None

    prompts_dir = os.path.dirname(metadata.prompt_path)
    template_name = os.path.basename(metadata.prompt_path)
    env = Environment(loader=FileSystemLoader(prompts_dir))
    template = env.get_template(template_name)

    context = {
        "instance": instance,
        "workspace_dir_name": instance.get("repo", "").split("/")[-1],
        "actual_workspace_path": workspace_path,
        "metadata": metadata,
        "test_instructions": "",
    }
    return template.render(context)


class ScaleSWEDSHEvaluation(Evaluation):
    """
    Scale-SWE evaluation using the DeepSeek-harness minimum tool set
    (bash + str_replace_editor, no think/task/finish, no fake user messages).

    Key differences from ScaleSWEEvaluation:
    - Uses DeepSeek-compatible bash and str_replace_editor tools only.
    - Conversation runs once to completion; no fake user responses are injected.
    - No use_legacy_tools option.
    """

    bind_dev_sdk: int = Field(
        default=False, description="Bind SDK paths for dev features"
    )
    docker_image_prefix: str | None = Field(
        default=None,
        description="Override image namespace/registry (e.g., 'myregistry.com/myorg')",
    )
    judge: ExecutionBasedJudge | None = Field(
        default=None, description="Optional execution-based judge"
    )

    def prepare_instances(self) -> List[EvalInstance]:
        logger.info("Setting up Scale-SWE evaluation data")

        df = get_dataset(
            dataset_name=self.metadata.dataset,
            split=self.metadata.dataset_split,
            eval_limit=self.metadata.eval_limit,
            selected_instances_file=self.metadata.selected_instances_file,
        )

        instances: List[EvalInstance] = []
        for _, row in df.iterrows():
            inst_id = str(row["instance_id"])
            instances.append(EvalInstance(id=inst_id, data=row.to_dict()))

        logger.info("Total instances to process: %d", len(instances))
        return instances

    def prepare_workspace(
        self,
        instance: EvalInstance,
        resource_factor: int = 1,
        forward_env: list[str] | None = None,
    ) -> RemoteWorkspace:
        """Prepare workspace from the dataset's image_url field."""
        image_url = instance.data.get("image_url", "")
        if not image_url:
            raise ValueError(
                f"Instance {instance.id} has no 'image_url' field in dataset"
            )

        base_image = resolve_image_url(image_url, self.docker_image_prefix)
        custom_tag = extract_custom_tag(base_image)
        build_target = "source-minimal"
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
                base_image=base_image,
                agent_plugin_image=agent_plugin_image,
                working_dir="/workspace",
                forward_env=forward_env or [],
                bind_volumes=bind_volumes,
            )
        elif self.metadata.workspace_type == "docker":
            agent_server_image = (
                f"{EVAL_AGENT_SERVER_IMAGE}:{SDK_SHORT_SHA}-{custom_tag}{suffix}"
            )
            SKIP_BUILD = os.getenv("SKIP_BUILD", "1").lower() in ("1", "true", "yes")
            logger.info(f"SKIP_BUILD={SKIP_BUILD}")
            if not SKIP_BUILD:
                from benchmarks.utils.build_utils import build_image

                logger.info(
                    f"Building workspace from {base_image} "
                    f"for instance {instance.id}. "
                    "This may take a while..."
                )
                output = build_image(
                    base_image=base_image,
                    target_image=EVAL_AGENT_SERVER_IMAGE,
                    custom_tag=custom_tag,
                    target=build_target,
                    push=False,
                )
                logger.info(f"Image build output: {output}")
                assert output.error is None, f"Image build failed: {output.error}"

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
                f"Using remote workspace with image {agent_server_image} "
                f"(sdk sha: {sdk_short_sha}, resource_factor: {resource_factor})"
            )
            startup_timeout = float(os.getenv("REMOTE_RUNTIME_STARTUP_TIMEOUT", "600"))
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
            logger.debug(f"Ran env setup command '{cmd}': {res.stdout}")
        return workspace

    def evaluate_instance(
        self, instance: EvalInstance, workspace: RemoteWorkspace
    ) -> EvalOutput:
        """
        Create conversation, run agent to completion, collect history and git patch.

        Uses the DeepSeek minimum tool set (bash + str_replace_editor).
        The conversation runs once without fake user message injection.
        """
        tools = get_deepseek_minimal_tools()

        agent = Agent(
            llm=self.metadata.llm,
            tools=tools,
            system_prompt_kwargs={"cli_mode": True},
        )

        assert isinstance(workspace, RemoteWorkspace)

        # Instantiate Conversation early to give the WebSocket event client time
        # to establish its subscription before we send the first message.
        # (See CLAUDE.local.md — remote conversation event capture gotcha.)
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

        # Scale-SWE: repo is already at workdir (no /testbed copy needed).
        repo_path = instance.data.get("workdir", "/workspace")
        if not repo_path.endswith("/"):
            repo_path += "/"

        # Normalize fields for prompt template compatibility.
        instance.data["repo_path"] = repo_path
        instance.data["base_commit"] = instance.data.get(
            "parent_commit", instance.data.get("base_commit", "")
        )
        if "repo" not in instance.data or "/" not in instance.data.get("repo", ""):
            user = instance.data.get("user", "")
            repo = instance.data.get("repo", "")
            if user and repo:
                instance.data["repo"] = f"{user}/{repo}"

        # Run pre_commands from dataset (git checkout, branch setup, etc.).
        # These also serve as the timing gap between Conversation() and send_message().
        pre_commands = instance.data.get("pre_commands", "")
        if pre_commands and pre_commands.strip():
            pre_cmd = pre_commands.strip().removesuffix("\\n")
            logger.info("Running pre_commands for %s", instance.id)
            pre_result = workspace.execute_command(f"cd {repo_path} && {pre_cmd}")
            if pre_result.exit_code != 0:
                logger.warning(
                    "pre_commands returned non-zero exit code %d: %s",
                    pre_result.exit_code,
                    pre_result.stderr,
                )

        breif_history = workspace.execute_command(
            (f"cd {repo_path} ; git --no-pager log --oneline -10")
        )
        logger.info(f"Repo status:\n* Current commit: {instance.data['base_commit']}\n* Top 10 history:\n{breif_history.stdout.strip()}")

        instruction = get_instruction(
            instance=instance.data,
            metadata=self.metadata,
            workspace_path=workspace.working_dir,
        )
        conversation.send_message(instruction)

        # Run the agent to completion without fake user responses.
        timeout = self.metadata.conversation_timeout
        sig = inspect.signature(conversation.run)
        if timeout is not None and "timeout" in sig.parameters:
            conversation.run(timeout=timeout)
        else:
            conversation.run()

        # git add
        workspace.execute_command(f"cd {repo_path} ; git add -A")

        # git commit
        workspace.execute_command(
            f"cd {repo_path} && "
            "git config --global user.email 'evaluation@openhands.dev' && "
            "git config --global user.name 'OpenHands Evaluation' && "
            "git commit -m 'patch'"
        )

        # Get git patch
        base_commit = instance.data["base_commit"]
        git_patch_result = workspace.execute_command(
            f"cd {repo_path} ; git --no-pager diff --no-color {base_commit} HEAD"
        )
        assert git_patch_result.exit_code == 0, (
            f"git diff failed: {git_patch_result.stderr}"
        )
        git_patch = git_patch_result.stdout

        # Run execution-based judge if configured.
        evaluation_result = None
        if self.judge is not None:
            try:
                evaluation_result = self.judge.judge(
                    instance_id=instance.id,
                    git_patch=git_patch,
                    instance_data=instance.data,
                )
                logger.info(
                    "Judge result for %s: %s", instance.id, evaluation_result
                )
            except Exception as e:
                logger.error("Judge failed for %s: %s", instance.id, e)
                evaluation_result = None

        # Dump conversation history
        messages = []
        tools_list = []

        convertible_events = [
            e for e in conversation.state.events if isinstance(e, LLMConvertibleEvent)
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
        with open(history_file, "w") as f:
            json.dump(dump_data, f, indent=2)
        logger.info(f"Dumped conversation history to {history_file}")

        out = EvalOutput(
            instance_id=instance.id,
            attempt=self.current_attempt,
            test_result={"git_patch": git_patch},
            instruction=instruction,
            error=None,
            history=list(conversation.state.events),
            metrics=conversation.conversation_stats.get_combined_metrics(),
        )
        return out


def main() -> None:
    prompt_dir = (Path(__file__).parent / "prompts").resolve()
    choices = [str(p.relative_to(Path.cwd())) for p in prompt_dir.glob("*.j2")]
    default_prompt_path = prompt_dir / "default.j2"
    assert default_prompt_path.exists(), (
        f"Default prompt {default_prompt_path} not found"
    )

    parser = get_parser()
    parser.set_defaults(
        dataset="thirdparty/Scale-SWE/scale-swe-batch1.jsonl",
        workspace="flex",
    )
    parser.add_argument(
        "--prompt-path",
        type=str,
        default=str(default_prompt_path),
        choices=choices,
        help="Path to prompt template file",
    )
    parser.add_argument(
        "--bind-dev-sdk",
        action="store_true",
        help="Bind SDK paths for dev features",
    )
    parser.add_argument(
        "--docker-image-prefix",
        type=str,
        default=None,
        help=(
            "Override image namespace/registry. Replaces everything before "
            "the last '/' in image_url. "
            "E.g., 'myregistry.com/myorg' turns 'aweaiteam/scaleswe:tag' "
            "into 'myregistry.com/myorg/scaleswe:tag'."
        ),
    )
    add_judge_args(parser, default_judge="scaleswe")
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
    logger.info(f"Using critic: {type(critic).__name__}")

    judge = create_judge(args)
    if judge is not None:
        logger.info(f"Using judge: {type(judge).__name__}")

    metadata = EvalMetadata(
        llm=llm,
        dataset=args.dataset,
        dataset_split=args.split,
        max_iterations=args.max_iterations,
        eval_output_dir=structured_output_dir,
        details={},
        prompt_path=args.prompt_path,
        eval_limit=args.n_limit,
        env_setup_commands=get_mirror_env_commands() + ["export PIP_CACHE_DIR=~/.cache/pip"],
        max_attempts=args.max_attempts,
        critic=critic,
        selected_instances_file=args.select,
        max_retries=args.max_retries,
        workspace_type=args.workspace,
        conversation_timeout=args.conversation_timeout,
    )

    evaluator = ScaleSWEDSHEvaluation(
        metadata=metadata,
        num_workers=args.num_workers,
        bind_dev_sdk=args.bind_dev_sdk,
        docker_image_prefix=args.docker_image_prefix,
        judge=judge,
    )

    evaluator.run(on_result=get_default_on_result_writer(evaluator.output_path))

    logger.info("Evaluation completed!")


if __name__ == "__main__":
    main()
