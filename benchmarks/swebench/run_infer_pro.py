"""
SWE-bench evaluation using agent-server-pro orchestrator.

Instead of the SDK's Agent/Conversation loop, this module spins up a Docker
container with the agent-server-pro orchestrator bound in, then calls its
``/run`` endpoint with the rendered prompt.  The orchestrator launches
Claude Code (via a local proxy) and returns the full trace.

The existing ``prepare_instances`` and git-patch logic are reused from the
standard ``run_infer`` scaffold.

Orchestrator configuration
--------------------------
All orchestrator-specific settings — including secrets — come from a single
JSON config file passed via ``--orchestrator-config``.  Example::

    {
        "anthropic_api_key": "sk-ant-...",
        "anthropic_base_url": "https://api.anthropic.com",
        "anthropic_model": "claude-sonnet-4-20250514",
        "claude_binary": "/usr/local/bin/claude",
        "claude_code_path": "/host/path/to/claude",
        "nodejs_path": "/host/path/to/node",
        "claude_timeout": 1800,
        "keep_logs": true
    }

Only ``anthropic_api_key`` is required; all other fields have defaults.
``claude_code_path`` is the *host-side* path to the Claude Code binary (or a
directory containing it) which is bind-mounted read-only into the container.
"""

import json
import os
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, List
from urllib.request import urlopen

import httpx
from jinja2 import Environment, FileSystemLoader
from pydantic import BaseModel, Field

from benchmarks.swebench.build_images import (
    extract_custom_tag,
    get_official_docker_image,
    should_wrap_instance_id,
    wrap_image,
)
from benchmarks.utils.args_parser import get_parser
from benchmarks.utils.build_utils import build_image
from benchmarks.utils.constants import EVAL_AGENT_SERVER_IMAGE
from benchmarks.utils.critics import create_critic
from benchmarks.utils.dataset import get_dataset
from benchmarks.utils.evaluation import Evaluation
from benchmarks.utils.evaluation_utils import (
    construct_eval_output_dir,
    get_default_on_result_writer,
)
from benchmarks.utils.models import (
    EvalInstance,
    EvalMetadata,
    EvalOutput,
)
from benchmarks.utils.version import SDK_SHORT_SHA
from openhands.sdk import LLM, get_logger
from openhands.sdk.utils.command import execute_command
from openhands.sdk.workspace import RemoteWorkspace

# Re-use the trace_to_chat converter from agent-server-pro
VENDOR_DIR = Path(__file__).resolve().parent.parent.parent / "vendor" / "agent-server-pro"
sys.path.insert(0, str(VENDOR_DIR))
from trace_to_chat import extract_chat_history  # noqa: E402


logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Container timeout for the full /run call (seconds).  Claude Code itself
# is already bounded by CLAUDE_TIMEOUT inside the orchestrator; this is an
# outer safety net.
# ---------------------------------------------------------------------------
CONTAINER_RUN_TIMEOUT = float(os.environ.get("CONTAINER_RUN_TIMEOUT", "1800"))

# Port the orchestrator listens on inside the container.
ORCHESTRATOR_CONTAINER_PORT = 8000


# ---------------------------------------------------------------------------
# Orchestrator config (loaded from external JSON file)
# ---------------------------------------------------------------------------
class OrchestratorConfig(BaseModel):
    """
    All settings required to start and drive the agent-server-pro orchestrator
    inside the evaluation container.  Loaded from a JSON file so that secrets
    (API keys) never appear on the command line.
    """

    # --- Anthropic / Claude Code credentials ---------------------------------
    anthropic_api_key: str = Field(
        description="Anthropic API key forwarded to the orchestrator as ANTHROPIC_API_KEY."
    )
    anthropic_base_url: str = Field(
        default="https://api.anthropic.com",
        description="Anthropic API base URL (ANTHROPIC_BASE_URL inside the container).",
    )
    anthropic_model: str = Field(
        default="",
        description="Anthropic model to use (ANTHROPIC_MODEL inside the container).",
    )

    # --- Claude Code binary --------------------------------------------------
    claude_binary: str = Field(
        default="claude",
        description=(
            "Path to the Claude Code binary *inside* the container.  "
            "Set this when the binary is not on PATH (e.g. after bind-mounting "
            "via claude_code_path).  Maps to CLAUDE_BINARY."
        ),
    )
    claude_code_path: str = Field(
        default="",
        description=(
            "Host-side path to the Claude Code binary or a directory "
            "containing it.  Bind-mounted read-only into the container.  "
            "When a file is given it is mounted directly at claude_binary; "
            "when a directory is given the directory is mounted at "
            "/opt/claude-code and claude_binary is set accordingly if it "
            "still holds its default value."
        ),
    )

    # --- Node.js binding -----------------------------------------------------
    nodejs_path: str = Field(
        default="",
        description=(
            "Host-side path to a Node.js installation directory.  "
            "Bind-mounted read-only into the container at /opt/nodejs.  "
            "The bin subdirectory is prepended to PATH."
        ),
    )

    # --- Orchestrator tunables -----------------------------------------------
    claude_timeout: float = Field(
        default=1800,
        description="Seconds Claude Code is allowed to run per job (CLAUDE_TIMEOUT).",
    )
    keep_logs: bool = Field(
        default=True,
        description="Keep per-job JSONL trace files after the job completes (KEEP_LOGS).",
    )
    pip_index_url: str = Field(
        default="",
        description="PyPI mirror URL passed to pip install -i (e.g. https://mirrors.ustc.edu.cn/pypi/simple).",
    )

    def to_container_env(self) -> dict[str, str]:
        """
        Return the env-var dict that must be injected into the Docker container.

        Does *not* include CLAUDE_BINARY when the binary path was derived from
        a directory mount — that is handled separately by ``_ProContainer.start``
        after the volume flags are assembled.
        """
        env = {
            "ANTHROPIC_API_KEY": self.anthropic_api_key,
            "ANTHROPIC_BASE_URL": self.anthropic_base_url,
            "CLAUDE_BINARY": self.claude_binary,
            "CLAUDE_TIMEOUT": str(self.claude_timeout),
            "KEEP_LOGS": "true" if self.keep_logs else "false",
            # orchestrator resolves work_dir relative to BASE_DIR; using "/"
            # means absolute paths (e.g. /workspace/repo/) are passed through.
            "BASE_DIR": "/",
        }
        if self.anthropic_model:
            env["ANTHROPIC_MODEL"] = self.anthropic_model
        return env


# ---------------------------------------------------------------------------
# Instruction builder (same as run_infer.py)
# ---------------------------------------------------------------------------
def get_instruction(
    instance: dict,
    metadata: EvalMetadata,
    workspace_path: str,
) -> str:
    """Generate instruction for the agent."""
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
    }
    context["test_instructions"] = ""
    return template.render(context)


# ---------------------------------------------------------------------------
# Thin Docker-container wrapper (no RemoteWorkspace / agent-server)
# ---------------------------------------------------------------------------
class _ProContainer:
    """
    Manages a Docker container running the agent-server-pro orchestrator.

    The container uses the same SWE-bench evaluation image as ``run_infer.py``
    but overrides its entrypoint so only the orchestrator starts.  The host-side
    ``vendor/agent-server-pro`` directory is bind-mounted in at
    ``/agent-server-pro`` (read-only); the orchestrator is then copied to a
    writable location inside the container before being launched, so that its
    ``logs/`` directory can be created.
    """

    AGENT_SERVER_PRO_MOUNT = "/agent-server-pro"
    WRITABLE_DIR = "/tmp/agent-server-pro"

    # Default mount point for a single Claude Code binary.
    CLAUDE_CODE_MOUNT = "/usr/local/bin/claude"
    # Mount point used when claude_code_path points to a directory.
    CLAUDE_CODE_DIR_MOUNT = "/opt/claude-code"
    # Mount point for Node.js installation.
    NODEJS_DIR_MOUNT = "/opt/nodejs"

    def __init__(
        self,
        image: str,
        host_port: int,
        orchestrator_config: OrchestratorConfig,
        bind_volumes: list[str] | None = None,
        platform: str = "linux/amd64",
    ):
        self.image = image
        self.host_port = host_port
        self.orchestrator_config = orchestrator_config
        self.bind_volumes = bind_volumes or []
        self.platform = platform
        self.container_id: str | None = None
        self._logs_thread: threading.Thread | None = None
        self._stop_logs = threading.Event()

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        """``docker run`` the container with the orchestrator entrypoint."""
        flags: list[str] = []
        cfg = self.orchestrator_config

        # --- Inject orchestrator env vars from config (no host env lookup) ---
        container_env = cfg.to_container_env()
        path_prepend: list[str] = []  # Directories to prepend to PATH

        # --- Bind-mount Claude Code binary / directory -----------------------
        # Resolve the host path and adjust CLAUDE_BINARY inside the container
        # if the config points to a directory rather than a single file.
        if cfg.claude_code_path:
            host_path = Path(cfg.claude_code_path).resolve()
            if host_path.is_file():
                # Mount the binary directly to the in-container path that
                # claude_binary already points at (default: /usr/local/bin/claude).
                flags += ["-v", f"{host_path}:{cfg.claude_binary}:ro"]
            elif host_path.is_dir():
                # Mount the directory; adjust CLAUDE_BINARY to match.
                flags += ["-v", f"{host_path}:{self.CLAUDE_CODE_DIR_MOUNT}:ro"]
                container_env["CLAUDE_BINARY"] = (
                    f"{self.CLAUDE_CODE_DIR_MOUNT}/{Path(cfg.claude_binary).name}"
                )
            else:
                logger.warning(
                    "claude_code_path %s does not exist, skipping bind", host_path
                )

        # --- Bind-mount Node.js installation ---------------------------------
        if cfg.nodejs_path:
            host_path = Path(cfg.nodejs_path).resolve()
            if host_path.is_dir():
                flags += ["-v", f"{host_path}:{self.NODEJS_DIR_MOUNT}:ro"]
                path_prepend.append(f"{self.NODEJS_DIR_MOUNT}/bin")
            else:
                logger.warning(
                    "nodejs_path %s does not exist or is not a directory, skipping bind",
                    host_path,
                )

        # Write the fully-resolved env vars into docker flags.
        for key, value in container_env.items():
            flags += ["-e", f"{key}={value}"]

        # --- Bind-mount agent-server-pro (read-only source) ------------------
        flags += ["-v", f"{VENDOR_DIR}:{self.AGENT_SERVER_PRO_MOUNT}:ro"]

        # --- Extra user-requested volumes ------------------------------------
        for vol in self.bind_volumes:
            flags += ["-v", vol]

        # --- Port mapping ----------------------------------------------------
        flags += ["-p", f"{self.host_port}:{ORCHESTRATOR_CONTAINER_PORT}"]

        # --- Entrypoint: copy to writable dir, install deps, launch ----------
        # LOGS_DIR in orchestrator.py is Path(__file__).parent / "logs".
        # Since the bind-mount is :ro, we copy to /tmp first so the logs
        # directory (and the JSONL trace files) can be created.
        path_export = ""
        if path_prepend:
            # Export PATH inside the shell so $PATH expands correctly
            path_export = f"export PATH={':'.join(path_prepend)}:$PATH && "
        pip_index = ""
        if cfg.pip_index_url:
            pip_index = f" -i {cfg.pip_index_url}"
        entrypoint_cmd = (
            f"{path_export}"
            f"cp -r {self.AGENT_SERVER_PRO_MOUNT} {self.WRITABLE_DIR} && "
            f"cd {self.WRITABLE_DIR} && "
            f"pip install -q{pip_index} -r requirements.txt && "
            f"python -m uvicorn orchestrator:app "
            f"--host 0.0.0.0 --port {ORCHESTRATOR_CONTAINER_PORT}"
        )

        run_cmd = [
            "docker", "run", "-d",
            "--platform", self.platform,
            "--rm",
            "--name", f"agent-pro-{uuid.uuid4()}",
            *flags,
            "--entrypoint", "/bin/bash",
            self.image,
            "-c", entrypoint_cmd,
        ]

        proc = execute_command(run_cmd)
        if proc.returncode != 0:
            raise RuntimeError(f"Failed to start container: {proc.stderr}")
        self.container_id = proc.stdout.strip()
        logger.info("Started pro container: %s", self.container_id)

        # Stream docker logs in background for real-time visibility
        self._logs_thread = threading.Thread(target=self._stream_logs, daemon=True)
        self._logs_thread.start()

        # Wait for orchestrator to be healthy before returning
        self._wait_healthy()

    def stop(self) -> None:
        if self.container_id:
            self._stop_logs.set()
            if self._logs_thread and self._logs_thread.is_alive():
                self._logs_thread.join(timeout=2)
            logger.info("Stopping container: %s", self.container_id)
            execute_command(["docker", "stop", self.container_id])
            self.container_id = None

    def __enter__(self) -> "_ProContainer":
        return self

    def __exit__(self, *_: Any) -> None:
        self.stop()

    # -- helpers -------------------------------------------------------------

    def exec(self, cmd: str, timeout: float = 120) -> subprocess.CompletedProcess[str]:
        """Run a command inside the container via ``docker exec``."""
        assert self.container_id, "Container not started"
        return subprocess.run(
            ["docker", "exec", self.container_id, "/bin/bash", "-c", cmd],
            capture_output=True, text=True, timeout=timeout,
        )

    @property
    def orchestrator_url(self) -> str:
        return f"http://127.0.0.1:{self.host_port}"

    def _wait_healthy(self, timeout: float = 180) -> None:
        """Poll the orchestrator's /docs endpoint until it responds."""
        start = time.time()
        url = f"{self.orchestrator_url}/docs"
        while time.time() - start < timeout:
            try:
                with urlopen(url, timeout=2) as resp:
                    if 200 <= getattr(resp, "status", 200) < 400:
                        logger.info("Orchestrator healthy at %s", self.orchestrator_url)
                        return
            except Exception:
                pass

            # Verify the container is still running
            if self.container_id:
                ps = execute_command(
                    ["docker", "inspect", "-f", "{{.State.Running}}", self.container_id]
                )
                if ps.stdout.strip() != "true":
                    logs = execute_command(["docker", "logs", self.container_id])
                    raise RuntimeError(
                        f"Container exited. Logs:\n{logs.stdout}\n{logs.stderr}"
                    )
            time.sleep(2)
        # Capture container logs to help diagnose the failure
        if self.container_id:
            logs = execute_command(["docker", "logs", "--tail", "50", self.container_id])
            raise RuntimeError(
                f"Orchestrator did not become healthy in time. "
                f"Container logs:\n{logs.stdout}\n{logs.stderr}"
            )
        raise RuntimeError("Orchestrator did not become healthy in time")

    def _stream_logs(self) -> None:
        """Stream ``docker logs -f`` to stdout with a [PRO] prefix."""
        if not self.container_id:
            return
        try:
            p = subprocess.Popen(
                ["docker", "logs", "-f", self.container_id],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            )
            if p.stdout is None:
                return
            for line in iter(p.stdout.readline, ""):
                if self._stop_logs.is_set():
                    break
                if line:
                    sys.stdout.write(f"[PRO] {line}")
                    sys.stdout.flush()
        except Exception as e:
            sys.stderr.write(f"Error streaming pro container logs: {e}\n")


# ---------------------------------------------------------------------------
# Find available port (same helper as DockerWorkspace)
# ---------------------------------------------------------------------------
def _find_port(min_port: int = 30000, max_port: int = 39999) -> int:
    import random, socket  # noqa: E401
    rng = random.SystemRandom()
    ports = list(range(min_port, max_port + 1))
    rng.shuffle(ports)
    for port in ports[:50]:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("0.0.0.0", port))
                return port
            except OSError:
                continue
    raise RuntimeError("No available port found")


# ---------------------------------------------------------------------------
# Evaluation class
# ---------------------------------------------------------------------------
class ProSWEBenchEvaluation(Evaluation):
    """
    SWE-bench evaluation driven by the agent-server-pro orchestrator.

    Lifecycle per instance:
      1. Spin up Docker container with overridden entrypoint → orchestrator.
      2. ``docker exec`` to prepare testbed (copy /testbed → /workspace/<repo>).
      3. POST /run with the rendered prompt.
      4. Stream / collect logs in real time.
      5. ``docker exec`` to generate the git patch.
      6. Return ``EvalOutput``.
    """

    orchestrator_config: OrchestratorConfig = Field(
        description="Orchestrator settings loaded from the external config file.",
    )

    # -- prepare_instances (reused from run_infer) ----------------------------

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
            instances.append(EvalInstance(id=str(row["instance_id"]), data=row.to_dict()))
        logger.info("Total instances to process: %d", len(instances))
        return instances

    # -- prepare_workspace ----------------------------------------------------

    def prepare_workspace(
        self,
        instance: EvalInstance,
        resource_factor: int = 1,  # noqa: ARG002 — required by abstract interface
        forward_env: list[str] | None = None,  # noqa: ARG002 — not used for pro
    ) -> RemoteWorkspace:
        """
        Build the Docker image (if needed) and return a ``_ProWorkspace``
        wrapping a running ``_ProContainer``.
        """
        official_docker_image = get_official_docker_image(instance.id)
        build_target = "source-minimal"
        custom_tag = extract_custom_tag(official_docker_image)
        suffix = f"-{build_target}" if build_target != "binary" else ""
        agent_server_image = (
            f"{EVAL_AGENT_SERVER_IMAGE}:{SDK_SHORT_SHA}-{custom_tag}{suffix}"
        )

        SKIP_BUILD = os.getenv("SKIP_BUILD", "1").lower() in ("1", "true", "yes")
        if not SKIP_BUILD:
            logger.info(
                "Building workspace from %s for instance %s",
                official_docker_image, instance.id,
            )
            output = build_image(
                base_image=official_docker_image,
                target_image=EVAL_AGENT_SERVER_IMAGE,
                custom_tag=custom_tag,
                target=build_target,
                push=False,
            )
            assert output.error is None, f"Image build failed: {output.error}"
            if should_wrap_instance_id(instance.id):
                wrapped = wrap_image(agent_server_image, push=False)
                if wrapped.error:
                    raise RuntimeError(f"Wrap failed: {wrapped.error}")

        host_port = _find_port()
        container = _ProContainer(
            image=agent_server_image,
            host_port=host_port,
            orchestrator_config=self.orchestrator_config,
        )
        container.start()
        return _ProWorkspace(container=container)

    # -- evaluate_instance ----------------------------------------------------

    def evaluate_instance(
        self,
        instance: EvalInstance,
        workspace: RemoteWorkspace,
    ) -> EvalOutput:
        assert isinstance(workspace, _ProWorkspace)
        container = workspace.container

        repo_path = f"/workspace/{instance.data['repo'].split('/')[-1]}/"
        instance.data["repo_path"] = repo_path

        # ---- 1. Prepare testbed inside container ----------------------------
        logger.info("Preparing testbed at %s", repo_path)
        r = container.exec(f"mkdir -p {repo_path} && cp -r /testbed/. {repo_path}")
        assert r.returncode == 0, f"cp testbed failed: {r.stderr}"

        r = container.exec(f"cd {repo_path} && git reset --hard")
        assert r.returncode == 0, f"git reset failed: {r.stderr}"

        # ---- 2. Render prompt -----------------------------------------------
        instruction = get_instruction(
            instance=instance.data,
            metadata=self.metadata,
            workspace_path="/workspace",
        )

        # ---- 3. POST to /run ------------------------------------------------
        logger.info("Posting to orchestrator /run for instance %s", instance.id)

        trace: list[dict[str, Any]] = []
        claude_stdout: str | None = None
        claude_stderr: str | None = None
        claude_exit_code: int | None = None
        run_status = "error"

        try:
            # HTTP timeout must exceed claude_timeout to allow orchestrator to respond
            http_timeout = self.orchestrator_config.claude_timeout + 60
            with httpx.Client(timeout=httpx.Timeout(http_timeout)) as client:
                resp = client.post(
                    f"{container.orchestrator_url}/run",
                    json={"prompt": instruction, "work_dir": repo_path},
                )
                resp.raise_for_status()
                result = resp.json()

            run_status = result.get("status", "unknown")
            claude_stdout = result.get("claude_stdout")
            claude_stderr = result.get("claude_stderr")
            claude_exit_code = result.get("claude_exit_code")
            trace = result.get("trace") or []

            logger.info(
                "Orchestrator returned status=%s exit_code=%s trace_entries=%d",
                run_status, claude_exit_code, len(trace),
            )
            if claude_stderr:
                logger.info("Claude stderr:\n%s", claude_stderr[:2000])

        except httpx.TimeoutException:
            logger.error("Orchestrator /run timed out for %s", instance.id)
            run_status = "timed_out"
        except Exception as e:
            logger.error("Orchestrator /run failed for %s: %s", instance.id, e)
            run_status = "error"

        # ---- 4. Generate git patch ------------------------------------------
        container.exec(f"cd {repo_path} && git add -A")
        container.exec(
            f"cd {repo_path} && "
            "git config --global user.email 'evaluation@openhands.dev' && "
            "git config --global user.name 'OpenHands Evaluation' && "
            "git commit -m 'patch' || true"
        )

        base_commit = instance.data["base_commit"]
        diff_result = container.exec(
            f"cd {repo_path} && git --no-pager diff --no-color {base_commit} HEAD"
        )
        git_patch = diff_result.stdout if diff_result.returncode == 0 else ""

        # ---- 5. Build conversation history from trace -----------------------
        messages: list[dict[str, Any]] = []
        tools_list: list[dict[str, Any]] = []

        if trace:
            chat = extract_chat_history(trace)
            if chat.get("system"):
                messages.insert(0, {"role": "system", "content": chat["system"]})
            messages.extend(chat.get("messages", []))
            tools_list = chat.get("tools", [])

        # Dump converted history
        dump_data = {
            "instance_id": instance.id,
            "orchestrator_status": run_status,
            "claude_exit_code": claude_exit_code,
            "messages": messages,
            "model": self.metadata.llm.model,
            "tools": tools_list,
            "temperature": self.metadata.llm.temperature,
            "top_p": self.metadata.llm.top_p,
            "test_result": {"git_patch": git_patch},
        }

        history_file = os.path.join(
            self.metadata.eval_output_dir, f"{instance.id}.history.json"
        )
        with open(history_file, "w") as f:
            json.dump(dump_data, f, indent=2)
        logger.info("Dumped conversation history to %s", history_file)

        # Dump raw orchestrator response
        raw_file = os.path.join(
            self.metadata.eval_output_dir, f"{instance.id}.orchestrator_response.json"
        )
        with open(raw_file, "w") as f:
            json.dump(
                {
                    "status": run_status,
                    "claude_stdout": claude_stdout,
                    "claude_stderr": claude_stderr,
                    "claude_exit_code": claude_exit_code,
                    "trace": trace,
                },
                f, indent=2,
            )

        # ---- 6. Build EvalOutput -------------------------------------------
        error_msg = None
        if run_status != "completed":
            error_msg = f"Orchestrator status: {run_status}"
            if claude_stderr:
                error_msg += f" | stderr: {claude_stderr[:200]}"

        return EvalOutput(
            instance_id=instance.id,
            attempt=self.current_attempt,
            test_result={"git_patch": git_patch},
            instruction=instruction,
            error=error_msg,
            history=[],
            metrics=None,
        )


# ---------------------------------------------------------------------------
# _ProWorkspace: thin adapter so Evaluation.run() can manage lifecycle
# ---------------------------------------------------------------------------
class _ProWorkspace(RemoteWorkspace):
    """
    Minimal ``RemoteWorkspace`` subclass that wraps a ``_ProContainer``.

    The ``Evaluation`` base class calls ``workspace.__exit__()`` for cleanup,
    and ``evaluate_instance`` accesses ``workspace.container`` for docker-exec
    and HTTP calls.
    """

    class Config:
        arbitrary_types_allowed = True

    container: Any = Field(default=None, exclude=True)  # _ProContainer (not serialisable)

    def __init__(self, container: _ProContainer, **kwargs: Any):
        super().__init__(
            host=container.orchestrator_url,
            working_dir="/workspace",
            container=container,
            **kwargs,
        )

    def model_post_init(self, context: Any) -> None:  # noqa: ARG002
        """Skip parent's model_post_init which tries to connect to agent-server."""
        pass

    def __enter__(self) -> "_ProWorkspace":
        return self

    def __exit__(self, *_: Any) -> None:
        if self.container is not None:
            self.container.stop()

    def execute_command(self, cmd: str, **_kwargs: Any) -> Any:  # type: ignore[override]
        """Execute a command inside the container via docker exec."""
        result = self.container.exec(cmd)

        class _CmdResult:
            def __init__(self, stdout: str, stderr: str, exit_code: int):
                self.stdout = stdout
                self.stderr = stderr
                self.exit_code = exit_code

        return _CmdResult(
            stdout=result.stdout, stderr=result.stderr, exit_code=result.returncode,
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main() -> None:
    prompt_dir = (Path(__file__).parent / "prompts").resolve()
    choices = [str(p.relative_to(Path.cwd())) for p in prompt_dir.glob("*.j2")]
    default_prompt_path = prompt_dir / "default.j2"
    assert default_prompt_path.exists(), f"Default prompt {default_prompt_path} not found"

    parser = get_parser()
    parser.add_argument(
        "--prompt-path",
        type=str,
        default=str(default_prompt_path),
        choices=choices,
        help="Path to prompt template file",
    )
    parser.add_argument(
        "--orchestrator-config",
        type=str,
        required=True,
        help=(
            "Path to the orchestrator JSON config file.  "
            "Must contain at minimum {'anthropic_api_key': '...'}; "
            "see OrchestratorConfig for all supported fields."
        ),
    )
    args = parser.parse_args()

    if args.max_attempts < 1:
        raise ValueError(f"max_attempts must be >= 1, got {args.max_attempts}")

    # --- Load LLM config -----------------------------------------------------
    llm_config_path = args.llm_config_path
    if not os.path.isfile(llm_config_path):
        raise ValueError(f"LLM config file {llm_config_path} does not exist")
    with open(llm_config_path) as f:
        llm_config = f.read()
    llm = LLM.model_validate_json(llm_config)
    logger.info("Using LLM config: %s", llm.model_dump_json(indent=2))

    # --- Load orchestrator config --------------------------------------------
    orchestrator_config_path = args.orchestrator_config
    if not os.path.isfile(orchestrator_config_path):
        raise ValueError(
            f"Orchestrator config file {orchestrator_config_path} does not exist"
        )
    with open(orchestrator_config_path) as f:
        orchestrator_config = OrchestratorConfig.model_validate_json(f.read())
    # Log config without the API key
    safe = orchestrator_config.model_dump()
    safe["anthropic_api_key"] = "***"
    logger.info("Using orchestrator config: %s", json.dumps(safe, indent=2))

    # --- Build eval metadata -------------------------------------------------
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
        workspace_type="docker",  # Pro always uses docker
    )

    evaluator = ProSWEBenchEvaluation(
        metadata=metadata,
        num_workers=args.num_workers,
        orchestrator_config=orchestrator_config,
    )

    evaluator.run(on_result=get_default_on_result_writer(evaluator.output_path))
    logger.info("Evaluation completed!")


if __name__ == "__main__":
    main()
