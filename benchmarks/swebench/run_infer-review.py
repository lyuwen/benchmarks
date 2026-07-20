import os
import json
import base64
import re
from pathlib import Path
from typing import List

from jinja2 import Environment, FileSystemLoader
from pydantic import Field
from omegaconf import OmegaConf

from benchmarks.swebench.build_images import (
    extract_custom_tag,
    get_official_docker_image,
    should_wrap_instance_id,
    wrap_image,
)
from benchmarks.utils.args_parser import get_parser
from benchmarks.utils.mirror_config import get_mirror_env_commands
from benchmarks.utils.build_utils import build_image
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
from benchmarks.utils.fake_user_response import run_conversation_with_fake_user_response
from benchmarks.utils.image_utils import image_exists
from benchmarks.utils.models import (
    EvalInstance,
    EvalMetadata,
    EvalOutput,
)
from benchmarks.utils.version import SDK_SHORT_SHA

# Import judge to trigger registration
from benchmarks.swebench.judge import SWEBenchJudge  # noqa: F401

from openhands.sdk import LLM, Agent, Conversation, get_logger
from openhands.sdk.event.base import LLMConvertibleEvent
from openhands.sdk.event.llm_convertible.system import SystemPromptEvent
from openhands.sdk.tool.tool import ToolDefinition
from openhands.sdk.workspace import RemoteWorkspace
from openhands.tools.preset.legacy import (
    register_legacy_tools,
    StrReplaceEditorTool,
    ExecuteBashTool,
    Tool,
)
from openhands.workspace import APIRemoteWorkspace, DockerWorkspace, FlexWorkspace


logger = get_logger(__name__)


def fake_user_response_review(
    conversation: "BaseConversation",
    encapsulate_solution: bool = False,
) -> str:
    return (
        "Please continue reviewing the applied patch. When you have reached a "
        "judgement, submit it with the `finish` tool, including the machine-parseable "
        "### VERDICT block (decision: pass|fail, confidence, reasoning) in the message."
    )


def get_instruction(
    instance: dict,
    metadata: EvalMetadata,
    workspace_path: str,
    review_mode: str = "test",
) -> str:
    """Generate instruction for the agent."""
    workspace_dir_name = instance["repo"].split("/")[-1]
    assert metadata.details is not None

    # Set up Jinja2 environment
    assert metadata.prompt_path is not None
    prompts_dir = os.path.dirname(metadata.prompt_path)
    template_name = os.path.basename(metadata.prompt_path)
    env = Environment(loader=FileSystemLoader(prompts_dir))
    template = env.get_template(template_name)

    # Prepare context for rendering
    context = {
        "instance": instance,
        "workspace_dir_name": workspace_dir_name,
        "actual_workspace_path": workspace_path,
        "metadata": metadata,
        "review_mode": review_mode,
    }
    context["test_instructions"] = ""

    # Render the instruction
    instruction = template.render(context)
    return instruction


def load_patch_from_history(patch_dir: str, instance_id: str) -> str:
    """Extract the fix patch for an instance from a prior fix run.

    The fixing pass dumps its result to ``{patch_dir}/{instance_id}.history.json``
    with the produced diff at ``test_result.git_patch``. This reads it back so
    the review pass can apply it to a fresh environment.

    Raises FileNotFoundError / ValueError if no patch can be recovered, so the
    run fails fast rather than reviewing an empty diff.
    """
    history_file = os.path.join(patch_dir, f"{instance_id}.history.json")
    if not os.path.isfile(history_file):
        raise FileNotFoundError(
            f"Fix history not found for instance {instance_id}: {history_file}"
        )

    with open(history_file) as f:
        data = json.load(f)

    patch = (data.get("test_result") or {}).get("git_patch")
    if not patch or not patch.strip():
        raise ValueError(
            f"No non-empty test_result.git_patch found in {history_file} for "
            f"instance {instance_id}"
        )
    return patch


def extract_finish_message(conversation) -> str:
    """Return the text of the reviewer's last finish tool call, or ''.

    The FinishAction carries the verdict in its ``message`` field. We scan the
    conversation events for the most recent finish action.
    """
    message = ""
    for event in conversation.state.events:
        action = getattr(event, "action", None)
        if action is not None and type(action).__name__ == "FinishAction":
            msg = getattr(action, "message", None)
            if msg:
                message = msg
    return message


def parse_verdict(review_message: str) -> dict:
    """Parse the machine-readable ### VERDICT block from a review message.

    Returns a dict with keys: decision ('pass'|'fail'|None), confidence, reasoning,
    and the raw review_message. Missing/malformed blocks yield decision=None.
    """
    result = {
        "decision": None,
        "confidence": None,
        "reasoning": None,
        "review_message": review_message,
    }
    if not review_message:
        return result

    def _field(name: str) -> str | None:
        m = re.search(rf"^\s*{name}\s*:\s*(.+?)\s*$", review_message, re.MULTILINE)
        return m.group(1).strip() if m else None

    decision = _field("decision")
    if decision:
        decision = decision.lower()
        if decision in ("pass", "fail"):
            result["decision"] = decision
    result["confidence"] = _field("confidence")
    result["reasoning"] = _field("reasoning")
    return result


class SWEBenchEvaluation(Evaluation):
    """
    Process-based SWE-bench evaluation implemented as a child of the
    abstract Evaluation orchestrator.

    Implements:
      - prepare_instances()
      - prepare_workspace(instance)
      - evaluate_instance(instance, workspace)
    """

    use_legacy_tools: int = Field(
        default=False, description="Use legacy CodeActAgent tools"
    )
    bind_dev_sdk: int = Field(
        default=False, description="Bind SDK paths for dev features"
    )
    patch_dir: str = Field(
        default="",
        description=(
            "Directory of a prior fix run; the fix patch for each instance is "
            "read from test_result.git_patch in "
            "{patch_dir}/{instance_id}.history.json and applied before review"
        ),
    )
    review_mode: str = Field(
        default="test",
        description=(
            "'test': reviewer may read and run tests (no edits); "
            "'readonly': reviewer may only read/inspect, no shell/test execution"
        ),
    )
    judge: ExecutionBasedJudge | None = Field(
        default=None, description="Optional execution-based judge"
    )

    def prepare_instances(self) -> List[EvalInstance]:
        logger.info("Setting up SWE-bench evaluation data")

        df = get_dataset(
            dataset_name=self.metadata.dataset,
            split=self.metadata.dataset_split,
            eval_limit=self.metadata.eval_limit,
            selected_instances_file=self.metadata.selected_instances_file,
        )

        instances: List[EvalInstance] = []
        for _, row in df.iterrows():
            inst_id = str(row["instance_id"])
            data = row.to_dict()
            data["patch"] = load_patch_from_history(self.patch_dir, inst_id)
            instances.append(EvalInstance(id=inst_id, data=data))

        logger.info("Total instances to process: %d", len(instances))
        return instances

    # ---- Hook: prepare a workspace per instance ----------------------------------
    def prepare_workspace(
        self,
        instance: EvalInstance,
        resource_factor: int = 1,
        forward_env: list[str] | None = None,
    ) -> RemoteWorkspace:
        """
        Use DockerWorkspace by default.

        Args:
            instance: The evaluation instance to prepare workspace for.
            resource_factor: Resource factor for runtime allocation (default: 1).
                           Higher values allocate more CPU/memory resources.
                           Used by APIRemoteWorkspace for remote runtime allocation.
        """
        if "image_name" in instance.data:
            official_docker_image = instance.data["image_name"]
        else:
            official_docker_image = get_official_docker_image(instance.id)
        build_target = "source-minimal"
        custom_tag = extract_custom_tag(get_official_docker_image(instance.id))
        # For non-binary targets, append target suffix
        suffix = f"-{build_target}" if build_target != "binary" else ""
        base_agent_image = (
            f"{EVAL_AGENT_SERVER_IMAGE}:{SDK_SHORT_SHA}-{custom_tag}{suffix}"
        )
        wrap_needed = should_wrap_instance_id(instance.id)
        agent_server_image = base_agent_image

        if self.metadata.workspace_type == "docker":
            SKIP_BUILD = os.getenv("SKIP_BUILD", "1").lower() in ("1", "true", "yes")
            logger.info(f"SKIP_BUILD={SKIP_BUILD}")
            if not SKIP_BUILD:
                logger.info(
                    f"Building workspace from {official_docker_image} "
                    f"for instance {instance.id}. "
                    "This may take a while...\n"
                    "You can run benchmarks/swebench/build_images.py and set "
                    "SWE_BENCH_SKIP_BUILD=1 to skip building and use pre-built "
                    "agent-server image."
                )
                output = build_image(
                    base_image=official_docker_image,
                    target_image=EVAL_AGENT_SERVER_IMAGE,
                    custom_tag=custom_tag,
                    target=build_target,
                    push=False,
                )
                logger.info(f"Image build output: {output}")
                assert output.error is None, f"Image build failed: {output.error}"
                if base_agent_image not in output.tags:
                    raise RuntimeError(
                        f"Built image tags {output.tags} do not include expected tag "
                        f"{base_agent_image}"
                    )
                if wrap_needed:
                    wrapped_result = wrap_image(base_agent_image, push=False)
                    if wrapped_result.error:
                        raise RuntimeError(
                            "Wrapped image build failed: "
                            f"{wrapped_result.error}; log={wrapped_result.log_path}"
                        )

            bind_volumes = []
            if self.bind_dev_sdk:
                sdk_base = Path(__file__).parent.parent.parent / "vendor/software-agent-sdk"
                for module in ["tools", "sdk", "agent-server", "workspace"]:
                    bind_volumes.append(
                        f"{sdk_base}/openhands-{module}/openhands/{module.replace('-', '_')}:"
                        f"/agent-server/.venv/lib/python3.12/site-packages/openhands/{module.replace('-', '_')}"
                        ":ro"
                        )
            workspace = DockerWorkspace(
                server_image=agent_server_image,
                working_dir="/workspace",
                forward_env=forward_env or [],
                bind_volumes=bind_volumes,
            )
        elif self.metadata.workspace_type == "flex":
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
                        f"{sdk_base}/openhands-{module}/openhands/{module.replace('-', '_')}:"
                        f"/agent-server/.venv/lib/python3.12/site-packages/openhands/{module.replace('-', '_')}"
                        ":ro"
                    )
            workspace = FlexWorkspace(
                base_image=official_docker_image,
                agent_plugin_image=agent_plugin_image,
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
            logger.debug(f"Ran env setup command '{cmd}': {res.stdout}")
        return workspace

    # ---- Hook: evaluate one instance ---------------------------------------------
    def evaluate_instance(
        self, instance: EvalInstance, workspace: RemoteWorkspace
    ) -> EvalOutput:
        """
        Create conversation, apply the fix patch, run the reviewer agent, and
        collect the judgement. Do not write files here; just return EvalOutput.
        """
        # Reviewer toolset: read/view always; bash (for running tests) only in
        # 'test' mode. Never the file-editor's edit commands — the reviewer must
        # not modify the patched code. FinishTool carries the verdict.
        register_legacy_tools(enable_browser=False)
        tools = [Tool(name=StrReplaceEditorTool.name)]
        if self.review_mode != "readonly":
            tools.append(Tool(name=ExecuteBashTool.name))
        agent = Agent(
            llm=self.metadata.llm,
            tools=tools,
            system_prompt_kwargs={"cli_mode": True},
            include_default_tools=['FinishTool'],
        )

        assert isinstance(workspace, RemoteWorkspace)

        repo_path = f"/workspace/{instance.data['repo'].split('/')[-1]}/"
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

        logger.info("repo_path: %s", repo_path)
        cp_testebed_repo = workspace.execute_command(
            (f"mkdir -p {repo_path} ; cp -r /testbed/. {repo_path}")
        )
        assert cp_testebed_repo.exit_code == 0, (
            f"cp_testebed_repo failed: {cp_testebed_repo.stderr}"
        )

        # git reset to a clean base environment
        git_reset = workspace.execute_command(f"cd {repo_path} ; git reset --hard")
        assert git_reset.exit_code == 0, f"git reset failed: {git_reset.stderr}"

        # Apply the fix patch under review into the fresh environment.
        # Transfer via base64 to avoid quoting/heredoc issues with the diff text.
        patch_text = instance.data["patch"]
        patch_remote = "/tmp/fix_under_review.patch"
        patch_b64 = base64.b64encode(patch_text.encode()).decode()
        write_res = workspace.execute_command(
            f"echo {patch_b64} | base64 -d > {patch_remote}"
        )
        assert write_res.exit_code == 0, (
            f"failed to write patch file: {write_res.stderr}"
        )
        apply_res = workspace.execute_command(
            f"cd {repo_path} ; git apply --verbose {patch_remote}"
        )
        if apply_res.exit_code != 0:
            # Fall back to a more lenient apply before giving up.
            apply_res = workspace.execute_command(
                f"cd {repo_path} ; git apply --3way --verbose {patch_remote} || "
                f"patch -p1 -i {patch_remote}"
            )
        patch_applied = apply_res.exit_code == 0
        if not patch_applied:
            logger.error(
                "Failed to apply fix patch for %s: %s",
                instance.id,
                apply_res.stderr,
            )

        instruction = get_instruction(
            instance=instance.data,
            metadata=self.metadata,
            workspace_path=workspace.working_dir,
            review_mode=self.review_mode,
        )
        conversation.send_message(instruction)
        # Run conversation with a review-specific fake user response: the default
        # response nudges the agent to keep *fixing* the task, which is wrong for
        # a reviewer. This one nudges it to finish with a verdict instead.
        run_conversation_with_fake_user_response(
            conversation,
            timeout=self.metadata.conversation_timeout,
            fake_user_response_fn=fake_user_response_review,
        )

        # Extract the reviewer's verdict from the final finish message.
        review_message = extract_finish_message(conversation)
        verdict = parse_verdict(review_message)
        # Dump conversation history
        messages = []
        tools_list = []
        
        # Convert events to messages
        convertible_events = [e for e in conversation.state.events if isinstance(e, LLMConvertibleEvent)]
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
            # Fallback to initial tools if not found in events
            for tool in tools:
                 if isinstance(tool, ToolDefinition):
                     tools_list.append(tool.to_openai_tool())

        review_result = {
            "decision": verdict["decision"],
            "confidence": verdict["confidence"],
            "reasoning": verdict["reasoning"],
            "review_message": review_message,
            "patch_applied": patch_applied,
            "review_mode": self.review_mode,
        }
        logger.info(
            "Review verdict for %s: decision=%s confidence=%s patch_applied=%s",
            instance.id,
            verdict["decision"],
            verdict["confidence"],
            patch_applied,
        )

        dump_data = {
            "instance_id": instance.id,
            "messages": messages,
            "model": self.metadata.llm.model,
            "tools": tools_list,
            "temperature": self.metadata.llm.temperature,
            "top_p": self.metadata.llm.top_p,
            "test_result": {
                "git_patch": instance.data["patch"],
                "review": review_result,
            },
        }

        history_file = os.path.join(self.metadata.eval_output_dir, f"{instance.id}.history.json")
        with open(history_file, "w") as f:
            json.dump(dump_data, f, indent=2)
        logger.info(f"Dumped conversation history to {history_file}")

        # EvalOutput is your model; keep fields consistent with prior JSONL
        out = EvalOutput(
            instance_id=instance.id,
            attempt=self.current_attempt,
            test_result={
                "git_patch": instance.data["patch"],
                "review": review_result,
            },
            instruction=instruction,
            error=None,
            history=list(conversation.state.events),
            metrics=conversation.conversation_stats.get_combined_metrics(),
        )
        return out


def main() -> None:
    prompt_dir = (Path(__file__).parent / "prompts").resolve()
    choices = [str(p.relative_to(Path.cwd())) for p in prompt_dir.glob("*.j2")]
    default_prompt_path = prompt_dir / "review.j2"
    assert default_prompt_path.exists(), (
        f"Default prompt {default_prompt_path} not found"
    )

    parser = get_parser()
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
    parser.add_argument(
        "--patch-dir",
        type=str,
        default=None,
        required=True,
        help=(
            "Directory of a prior fix run. The fix patch for each instance is "
            "read from test_result.git_patch in "
            "{patch_dir}/{instance_id}.history.json, applied to a fresh "
            "environment, and reviewed."
        ),
    )
    parser.add_argument(
        "--review-mode",
        type=str,
        default="test",
        choices=["test", "readonly"],
        help=(
            "'test' (default): reviewer may read code and run tests to verify "
            "the patch (no edits). 'readonly': reviewer may only inspect code, "
            "no shell/test execution."
        ),
    )
    add_judge_args(parser, default_judge="swebench")
    args = parser.parse_args()

    # Validate max_attempts
    if args.max_attempts < 1:
        raise ValueError(f"max_attempts must be >= 1, got {args.max_attempts}")

    llm_config_path = args.llm_config_path
    if not os.path.isfile(llm_config_path):
        raise ValueError(f"LLM config file {llm_config_path} does not exist")
    with open(llm_config_path, "r") as f:
        llm_config = f.read()
    # use omegaconf to resolve environment variables, and then serialize back to JSON
    llm_config = json.dumps(OmegaConf.to_container(OmegaConf.load(llm_config_path), resolve=True))
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

    # Create critic instance from parsed arguments
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

    # Run orchestrator with a simple JSONL writer
    evaluator = SWEBenchEvaluation(
        metadata=metadata,
        num_workers=args.num_workers,
        use_legacy_tools=args.use_legacy_tools,
        bind_dev_sdk=args.bind_dev_sdk,
        patch_dir=args.patch_dir,
        review_mode=args.review_mode,
        judge=judge,
    )

    evaluator.run(on_result=get_default_on_result_writer(evaluator.output_path))

    logger.info("Evaluation completed!")


if __name__ == "__main__":
    main()
