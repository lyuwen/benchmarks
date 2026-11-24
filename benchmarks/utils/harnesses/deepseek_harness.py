"""The DeepSeek Harness (dsh) blackbox harness.

DeepSeek Harness speaks OpenAI Chat-Completions through its ``llm-pi-ai``
provider adapter. We point a benchmark-only provider route at the rollout-manager
proxy, which records every exchange and forwards it to the real backend. The
proxy is the only party that sees the upstream credential; the container gets a
dummy capture key.

The harness runs from a single native executable (the Python SDK runtime carrier
built as a Node single-executable application), so task containers do not need
Node.js. ``dsh --profile headless`` runs one task, prints newline-delimited JSON
events (``--json``) and exits.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from benchmarks.utils.harnesses.base import HarnessRun, InstalledHarness
from benchmarks.utils.harnesses.config import HarnessConfig
from benchmarks.utils.harnesses.distribution import (
    HarnessDistribution,
    payload_paths,
)
from benchmarks.utils.harnesses.rollout import (
    RolloutManagerError,
    parse_dsh_json_result,
)
from benchmarks.utils.harnesses.workspace import DockerContainerWorkspace


logger = logging.getLogger(__name__)

HARNESS_NAME = "deepseek-harness"

# The proxy ignores client auth; the real upstream credential never enters the
# container.
CAPTURE_AUTH_TOKEN = "rollout-manager-capture"

WORK_DIR = "/tmp/harness"
HOME_DIR = f"{WORK_DIR}/home"
DSH_HOME = f"{WORK_DIR}/dsh-home"
PATCH_PATH = f"{WORK_DIR}/dsh-patch.json"
TASK_PATH = f"{WORK_DIR}/task.txt"

# Benchmark-only provider route configured through the --patch overlay.
PROVIDER_ID = "benchmark"
API_KEY_ENV = "DSH_CAPTURE_API_KEY"

# Map the shared --harness-permission-mode vocabulary onto dsh permission
# presets. dsh presets bundle a sandbox mode with an approval policy:
# danger-full-access (never ask), workspace-write (ask) and read-only (ask).
# Headless has no live approval responder, so the runner always passes
# --approval-policy never; workspace-write therefore stays confined without
# waiting for a human.
_PERMISSION_MODES = {
    "bypassPermissions": "danger-full-access",
    "danger-full-access": "danger-full-access",
    "auto": "danger-full-access",
    "workspace-write": "workspace-write",
    "acceptEdits": "workspace-write",
    "default": "workspace-write",
    "plan": "read-only",
    "read-only": "read-only",
}


class DeepSeekHarness:
    """Install and drive DeepSeek Harness inside the task container."""

    name = HARNESS_NAME

    # -------------------- installation ------------------------------------

    def install(
        self,
        workspace: DockerContainerWorkspace,
        distribution: HarnessDistribution,
        config: HarnessConfig,
    ) -> InstalledHarness:
        """Copy (or mount) the portable runtime into the container and verify it."""
        install_dir = config.install_dir.rstrip("/") or "/opt/harness"
        binary_path = f"{install_dir}/{distribution.binary_path.name}"
        if config.dist_mode == "copy":
            parents = {
                f"{install_dir}/{relative}".rsplit("/", 1)[0]
                for relative in distribution.payload_files()
            }
            for parent in sorted(parents):
                mkdir = workspace.execute_command(f"mkdir -p {parent}", timeout=30)
                if not mkdir.ok:
                    raise RolloutManagerError(
                        f"Cannot create harness install dirs in the container: "
                        f"{mkdir.stderr}"
                    )
            for local_path, relative in zip(
                payload_paths(distribution), distribution.payload_files()
            ):
                workspace.copy_in(local_path, f"{install_dir}/{relative}")
            for relative in distribution.payload_files():
                if (
                    relative.endswith("-rg")
                    or relative == distribution.binary_path.name
                ):
                    chmod = workspace.execute_command(
                        f"chmod 0755 {install_dir}/{relative}", timeout=30
                    )
                    if not chmod.ok:
                        raise RolloutManagerError(
                            "Cannot chmod harness payload in the container: "
                            f"{chmod.stderr}"
                        )
        else:
            binary_name = distribution.binary_path.name
            if not workspace.path_exists(f"{install_dir}/{binary_name}"):
                raise RolloutManagerError(
                    f"Harness binary {install_dir}/{binary_name} is not present; "
                    "the distribution mount is missing (dist_mode=mount)"
                )

        setup = workspace.execute_command(
            f"mkdir -p {HOME_DIR} {DSH_HOME} {WORK_DIR}", timeout=30
        )
        if not setup.ok:
            raise RolloutManagerError(
                f"Cannot create harness work dirs in the container: {setup.stderr}"
            )

        version = workspace.execute_command(f"{binary_path} --version", timeout=120)
        if not version.ok:
            raise RolloutManagerError(
                f"Harness binary {binary_path} does not run in the task container: "
                f"{(version.stderr or version.stdout).strip()[:2000]}"
            )
        version_text = (version.stdout or version.stderr or "").strip()
        logger.info("Installed %s in container: %s", self.name, version_text)
        return InstalledHarness(
            binary_path=binary_path,
            install_dir=install_dir,
            work_dir=WORK_DIR,
            settings_path=PATCH_PATH,
            instruction_path=TASK_PATH,
            version=version_text,
        )

    # -------------------- configuration -----------------------------------

    def permission_mode(self, config: HarnessConfig) -> str:
        """Resolve the dsh permission preset for this run."""
        requested = config.permission_mode or (
            "danger-full-access" if config.skip_permissions else "workspace-write"
        )
        try:
            return _PERMISSION_MODES[requested]
        except KeyError as exc:
            supported = ", ".join(sorted(_PERMISSION_MODES))
            raise ValueError(
                f"Unsupported permission mode {requested!r} for "
                f"{HARNESS_NAME}; supported: {supported}"
            ) from exc

    def build_patch(
        self,
        config: HarnessConfig,
        *,
        proxy_url: str,
        harness_model: str,
    ) -> list[dict[str, Any]]:
        """The ``--patch`` overlay configuring the benchmark provider route."""
        model: dict[str, Any] = {"id": harness_model}
        if config.context_window > 0:
            model["contextWindow"] = config.context_window
        if config.max_output_tokens:
            model["maxTokens"] = config.max_output_tokens
        provider: dict[str, Any] = {
            "apiKeyEnv": API_KEY_ENV,
            # The proxy converts nothing: dsh talks OpenAI Chat-Completions to it.
            "api": "openai-completions",
            "baseURL": f"{proxy_url.rstrip('/')}/v1",
            # The proxy's backend is an OpenAI-compatible gateway of unknown
            # flavor; stop pi-ai from assuming OpenAI's own wire details.
            "compat": {
                "supportsStore": False,
                "supportsDeveloperRole": False,
                "supportsReasoningEffort": False,
                "maxTokensField": "max_tokens",
            },
            "models": [model],
        }
        if config.custom_headers:
            provider["headers"] = dict(config.custom_headers)
        if config.api_timeout_ms:
            provider["timeoutMs"] = config.api_timeout_ms
        patch: list[dict[str, Any]] = [
            {
                "id": "agent-default-model",
                "config": {"provider": PROVIDER_ID, "model": harness_model},
            },
            {
                "id": "llm-pi-ai",
                "config": {"providers": {PROVIDER_ID: provider}},
            },
            # The session-title model call is auxiliary traffic that would
            # otherwise be captured as another training trajectory; benchmarks
            # never need it.
            {"id": "session-title-llm", "disabled": True},
        ]
        if config.compaction_threshold > 0:
            # dsh expresses compaction pressure as a fraction of the context
            # window, matching the percentage the benchmark exposes.
            ratio = min(config.compaction_threshold / 100.0, 0.99)
            patch.append(
                {
                    "id": "compaction-basic",
                    "config": {"thresholdRatio": ratio, "auto": True},
                }
            )
        return patch

    def build_env(
        self,
        config: HarnessConfig,
        *,
        proxy_url: str,
        harness_model: str,
    ) -> dict[str, str]:
        """Environment for the in-container dsh process."""
        env: dict[str, str] = {
            "HOME": HOME_DIR,
            "DSH_HOME": DSH_HOME,
            API_KEY_ENV: CAPTURE_AUTH_TOKEN,
            # Some base features resolve the DeepSeek route's key for web tools;
            # a dummy keeps them from using an ambient credential.
            "DEEPSEEK_API_KEY": CAPTURE_AUTH_TOKEN,
            "DSH_TELEMETRY_DISABLED": "1",
            "NO_PROXY": "127.0.0.1,localhost,host.docker.internal",
            "no_proxy": "127.0.0.1,localhost,host.docker.internal",
        }
        env.update(config.extra_env)
        return env

    def build_command(
        self,
        config: HarnessConfig,
        installed: InstalledHarness,
        *,
        instruction: str,
    ) -> list[str]:
        """Argv for one headless dsh run."""
        argv = [
            installed.binary_path,
            "--patch",
            installed.settings_path,
            "--profile",
            "headless",
            "--json",
            "--permission-mode",
            self.permission_mode(config),
            "--approval-policy",
            "never",
        ]
        argv.extend(config.extra_args)
        # dsh takes the task as the positional app argument; the launcher stops
        # parsing its own flags at the first unrecognized token.
        argv.append(instruction)
        return argv

    # -------------------- execution ---------------------------------------

    def run(
        self,
        workspace: DockerContainerWorkspace,
        config: HarnessConfig,
        installed: InstalledHarness,
        *,
        proxy_url: str,
        harness_model: str,
        instruction: str,
        max_turns: int,
        repo_path: str,
        log_path: str,
        timeout_sec: float | None,
        extra_env: dict[str, str] | None = None,
    ) -> HarnessRun:
        """Write the provider patch and run dsh headless."""
        if max_turns > 0:
            logger.debug(
                "deepseek-harness has no max-turn flag; ignoring max_turns=%d "
                "(the conversation timeout bounds the run)",
                max_turns,
            )
        workspace.write_text(instruction, installed.instruction_path)
        patch = self.build_patch(
            config, proxy_url=proxy_url, harness_model=harness_model
        )
        workspace.write_text(
            json.dumps(patch, indent=2) + "\n", installed.settings_path
        )
        env = self.build_env(config, proxy_url=proxy_url, harness_model=harness_model)
        if extra_env:
            env.update(extra_env)
        argv = self.build_command(config, installed, instruction=instruction)
        outcome = workspace.run_streaming(
            argv,
            cwd=repo_path,
            env=env,
            log_path=log_path,
            timeout_sec=timeout_sec,
        )
        return HarnessRun(
            argv=argv,
            env=env,
            outcome=outcome,
            result_event=parse_dsh_json_result(log_path),
        )


__all__ = ["DeepSeekHarness"]
