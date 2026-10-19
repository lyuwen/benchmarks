"""Two-agent interactive inference for Scale-SWE (trajectory generation)."""
import json
import os
from pathlib import Path

from omegaconf import OmegaConf

from benchmarks.scaleswe.run_infer import ScaleSWEEvaluation, get_instruction
from benchmarks.scaleswe_interactive.driver import run_interactive_session
from benchmarks.scaleswe_interactive.transcript import build_interactive_transcript
from benchmarks.scaleswe_interactive.user_agent import UserAgent
from benchmarks.utils.args_parser import get_parser
from benchmarks.utils.conversation import build_event_persistence_callback
from benchmarks.utils.critics import create_critic
from benchmarks.utils.evaluation_utils import (
    construct_eval_output_dir,
    get_default_on_result_writer,
)
from benchmarks.utils.mirror_config import get_mirror_env_commands
from benchmarks.utils.models import EvalInstance, EvalMetadata, EvalOutput
from openhands.sdk import LLM, Agent, Conversation, get_logger
from openhands.sdk.event.base import LLMConvertibleEvent
from openhands.sdk.event.llm_convertible.system import SystemPromptEvent
from openhands.sdk.tool.tool import ToolDefinition
from openhands.sdk.workspace import RemoteWorkspace
from openhands.tools.preset.default import get_default_tools

logger = get_logger(__name__)

PROMPTS_DIR = str((Path(__file__).parent / "prompts").resolve())


def load_llm_from_config(path: str) -> LLM:
    if not os.path.isfile(path):
        raise ValueError(f"LLM config file {path} does not exist")
    cfg = json.dumps(OmegaConf.to_container(OmegaConf.load(path), resolve=True))
    return LLM.model_validate_json(cfg)


class ScaleSWEInteractiveEvaluation(ScaleSWEEvaluation):
    """Reuses scaleswe instance/workspace setup; swaps in the two-agent loop."""

    def evaluate_instance(self, instance: EvalInstance,
                          workspace: RemoteWorkspace) -> EvalOutput:
        details = self.metadata.details or {}
        mode = details.get("mode", "plan")
        user_tools = details.get("user_tools", "none")
        max_user_turns = details.get("max_user_turns", 20)
        user_llm: LLM = details.get("user_llm") or self.metadata.llm

        coding_template = ("coding_system_plan.j2" if mode == "plan"
                           else "coding_system_auto.j2")
        from jinja2 import Environment, FileSystemLoader
        env = Environment(loader=FileSystemLoader(PROMPTS_DIR))
        coding_system_suffix = env.get_template(coding_template).render()

        tools = get_default_tools(enable_browser=False)
        agent = Agent(
            llm=self.metadata.llm,
            tools=tools,
            system_prompt_kwargs={"cli_mode": True},
        )

        assert isinstance(workspace, RemoteWorkspace)

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

        # --- Repo prep (also provides the Conversation->send_message gap) ---
        repo_path = instance.data.get("workdir", "/workspace")
        if not repo_path.endswith("/"):
            repo_path += "/"
        instance.data["repo_path"] = repo_path
        instance.data["base_commit"] = instance.data.get(
            "parent_commit", instance.data.get("base_commit", ""))
        if "repo" not in instance.data or "/" not in instance.data.get("repo", ""):
            user = instance.data.get("user", "")
            repo = instance.data.get("repo", "")
            if user and repo:
                instance.data["repo"] = f"{user}/{repo}"
        pre_commands = instance.data.get("pre_commands", "")
        if pre_commands and pre_commands.strip():
            pre_cmd = pre_commands.strip().removesuffix("\\n")
            workspace.execute_command(f"cd {repo_path} && {pre_cmd}")

        problem_statement = instance.data.get("problem_statement", "")
        instruction = get_instruction(
            instance=instance.data, metadata=self.metadata,
            workspace_path=workspace.working_dir)
        # Append the mode-specific coding-agent guidance to the instruction.
        instruction = f"{instruction}\n\n{coding_system_suffix}"

        user_agent = UserAgent(
            user_llm=user_llm, workspace=workspace, repo_path=repo_path,
            mode=mode, user_tools=user_tools, prompts_dir=PROMPTS_DIR,
            problem_statement=problem_statement, max_user_turns=max_user_turns)

        session = run_interactive_session(
            conversation, user_agent, initial_instruction=instruction,
            max_user_turns=max_user_turns,
            timeout=self.metadata.conversation_timeout)

        # --- git add/commit/diff (same as scaleswe) ---
        workspace.execute_command(f"cd {repo_path} ; git add -A")
        workspace.execute_command(
            f"cd {repo_path} && "
            "git config --global user.email 'evaluation@openhands.dev' && "
            "git config --global user.name 'OpenHands Evaluation' && "
            "git commit -m 'patch'")
        base_commit = instance.data["base_commit"]
        git_patch_result = workspace.execute_command(
            f"cd {repo_path} ; git --no-pager diff --no-color {base_commit} HEAD")
        git_patch = git_patch_result.stdout

        # --- dump scaleswe-compatible history.json ---
        messages = []
        tools_list = []
        convertible = [e for e in conversation.state.events
                       if isinstance(e, LLMConvertibleEvent)]
        for msg in LLMConvertibleEvent.events_to_messages(convertible):
            messages.append(
                msg.model_copy(update={"send_reasoning_content": True}).to_chat_dict())
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
        history_file = os.path.join(
            self.metadata.eval_output_dir, f"{instance.id}.history.json")
        with open(history_file, "w") as f:
            json.dump(dump_data, f, indent=2)

        # --- dump richer interactive side file ---
        transcript = build_interactive_transcript(
            instance_id=instance.id, mode=mode, user_tools=user_tools,
            coding_model=self.metadata.llm.model, user_model=user_llm.model,
            session=session)
        interactive_file = os.path.join(
            self.metadata.eval_output_dir, f"{instance.id}.interactive.json")
        with open(interactive_file, "w") as f:
            json.dump(transcript, f, indent=2)
        logger.info("Dumped interactive transcript to %s (termination=%s)",
                    interactive_file, session.termination_reason)

        return EvalOutput(
            instance_id=instance.id,
            attempt=self.current_attempt,
            test_result={"git_patch": git_patch},
            instruction=instruction,
            error=None,
            history=list(conversation.state.events),
            metrics=conversation.conversation_stats.get_combined_metrics(),
        )


def build_arg_parser():
    prompt_dir = (Path(__file__).parent.parent / "scaleswe" / "prompts").resolve()
    default_prompt_path = prompt_dir / "default.j2"
    parser = get_parser()
    parser.set_defaults(
        dataset="thirdparty/Scale-SWE/scale-swe-batch1.jsonl",
        workspace="flex",
    )
    parser.add_argument("--prompt-path", type=str,
                        default=str(default_prompt_path),
                        help="Coding-agent instruction template")
    parser.add_argument("--user-llm-config-path", type=str, default=None,
                        help="LLM config for the user agent (defaults to coding LLM)")
    parser.add_argument("--mode", choices=["plan", "auto"], default="plan",
                        help="Collaboration mode")
    parser.add_argument("--user-tools", choices=["none", "readonly"],
                        default="none", help="User agent repo access")
    parser.add_argument("--max-user-turns", type=int, default=20,
                        help="Cap on total user turns")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    llm = load_llm_from_config(args.llm_config_path)
    user_llm = (load_llm_from_config(args.user_llm_config_path)
                if args.user_llm_config_path else llm)
    logger.info("Coding LLM: %s | User LLM: %s", llm.model, user_llm.model)

    dataset_description = (
        args.dataset.replace("/", "__") + "-" + args.split.replace("/", "__"))
    structured_output_dir = construct_eval_output_dir(
        base_dir=args.output_dir, dataset_name=dataset_description,
        model_name=llm.model, max_iterations=args.max_iterations,
        eval_note=args.note)

    critic = create_critic(args)
    logger.info("Using critic: %s", type(critic).__name__)

    metadata = EvalMetadata(
        llm=llm,
        dataset=args.dataset,
        dataset_split=args.split,
        max_iterations=args.max_iterations,
        eval_output_dir=structured_output_dir,
        details={
            "mode": args.mode,
            "user_tools": args.user_tools,
            "max_user_turns": args.max_user_turns,
            "user_llm": user_llm,
        },
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
    )

    evaluator = ScaleSWEInteractiveEvaluation(
        metadata=metadata, num_workers=args.num_workers)
    evaluator.run(on_result=get_default_on_result_writer(evaluator.output_path))
    logger.info("Interactive evaluation completed!")


if __name__ == "__main__":
    main()
