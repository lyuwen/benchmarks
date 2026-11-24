"""The Claude Code blackbox harness.

Claude Code speaks the Anthropic Messages API. We point
``ANTHROPIC_BASE_URL`` at the rollout-manager proxy, which converts the traffic
to OpenAI Chat-Completions for the real backend and records every exchange.
The proxy is the only party that sees credentials; the container gets a dummy
token.

Everything below is specific to Claude Code; the generic pipeline lives in
``runner.py``/``rollout.py`` so other harnesses can reuse it.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from benchmarks.utils.harnesses.config import HarnessConfig
from benchmarks.utils.harnesses.distribution import HarnessDistribution
from benchmarks.utils.harnesses.rollout import (
    RolloutManagerError,
    parse_stream_json_result,
)
from benchmarks.utils.harnesses.workspace import (
    DockerContainerWorkspace,
    ProcessOutcome,
)


logger = logging.getLogger(__name__)

HARNESS_NAME = "claude-code"

# The proxy ignores auth; the real upstream credential never enters the container.
CAPTURE_AUTH_TOKEN = "rollout-manager-capture"

# Claude Code relocates its config (settings, transcripts, ~/.claude.json) here.
WORK_DIR = "/tmp/harness"
HOME_DIR = f"{WORK_DIR}/home"
CONFIG_DIR = f"{WORK_DIR}/claude-config"
SETTINGS_PATH = f"{WORK_DIR}/settings.json"
INSTRUCTION_PATH = f"{WORK_DIR}/instruction.txt"

# A `-p` run may wait up to 10 minutes after the final turn for background work.
# Benchmark rollouts should not.
PRINT_BG_WAIT_CEILING_MS = "120000"

# Gateways often reject Claude Code's experimental beta request fields once they
# are converted to Chat-Completions by the proxy.
DISABLE_EXPERIMENTAL_BETAS = "1"

# Values are treated as "on" when non-empty, so these are the canonical switches.
_DISABLED_FEATURES = {
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    "DISABLE_TELEMETRY": "1",
    "DISABLE_ERROR_REPORTING": "1",
    "DISABLE_AUTOUPDATER": "1",
    "DISABLE_UPDATES": "1",
    "CLAUDE_CODE_SKIP_PROMPT_HISTORY": "1",
    "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": DISABLE_EXPERIMENTAL_BETAS,
    "CLAUDE_CODE_PRINT_BG_WAIT_CEILING_MS": PRINT_BG_WAIT_CEILING_MS,
}

# Make sure a shell that inherits a gateway environment does not try a cloud
# provider (empty value means unset for provider selection).
_PROVIDER_OVERRIDES = {
    "CLAUDE_CODE_USE_BEDROCK": "",
    "CLAUDE_CODE_USE_VERTEX": "",
    "CLAUDE_CODE_USE_FOUNDRY": "",
}


@dataclass(frozen=True)
class InstalledHarness:
    """Where the harness binary lives inside a task container."""

    binary_path: str
    install_dir: str
    work_dir: str
    settings_path: str
    instruction_path: str
    version: str


@dataclass(frozen=True)
class HarnessRun:
    """Outcome of one headless harness invocation inside the container."""

    argv: list[str]
    env: dict[str, str]
    outcome: ProcessOutcome
    result_event: dict[str, Any] | None

    @property
    def exit_code(self) -> int:
        return self.outcome.exit_code

    @property
    def timed_out(self) -> bool:
        return self.outcome.timed_out


class ClaudeCodeHarness:
    """Install and drive Claude Code inside the task container."""

    name = HARNESS_NAME

    # -------------------- installation ------------------------------------

    def install(
        self,
        workspace: DockerContainerWorkspace,
        distribution: HarnessDistribution,
        config: HarnessConfig,
    ) -> InstalledHarness:
        """Make the portable harness usable in the container and verify it."""
        install_dir = config.install_dir.rstrip("/") or "/opt/harness"
        binary_path = f"{install_dir}/{distribution.binary_path.name}"
        if config.dist_mode == "copy":
            workspace.copy_in(distribution.binary_path, binary_path)
            chmod = workspace.execute_command(f"chmod 0755 {binary_path}", timeout=30)
            if not chmod.ok:
                raise RolloutManagerError(
                    f"Cannot chmod harness binary in the container: {chmod.stderr}"
                )
        elif not workspace.path_exists(binary_path):
            raise RolloutManagerError(
                f"Harness binary {binary_path} is not present; the distribution "
                "mount is missing (dist_mode=mount)"
            )

        setup = workspace.execute_command(
            f"mkdir -p {HOME_DIR} {CONFIG_DIR} {WORK_DIR}", timeout=30
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
            settings_path=SETTINGS_PATH,
            instruction_path=INSTRUCTION_PATH,
            version=version_text,
        )

    # -------------------- configuration -----------------------------------

    def build_env(
        self,
        config: HarnessConfig,
        *,
        proxy_url: str,
        harness_model: str,
    ) -> dict[str, str]:
        """Environment for the in-container harness process."""
        env: dict[str, str] = {
            "HOME": HOME_DIR,
            "CLAUDE_CONFIG_DIR": CONFIG_DIR,
            "ANTHROPIC_BASE_URL": proxy_url.rstrip("/"),
            "ANTHROPIC_AUTH_TOKEN": CAPTURE_AUTH_TOKEN,
            "ANTHROPIC_MODEL": harness_model,
            # Background/auxiliary requests (titles, classifiers, compaction)
            # otherwise ask for Claude model aliases the proxy backend lacks.
            "ANTHROPIC_DEFAULT_HAIKU_MODEL": harness_model,
            "ANTHROPIC_DEFAULT_SONNET_MODEL": harness_model,
            "ANTHROPIC_DEFAULT_OPUS_MODEL": harness_model,
            "ANTHROPIC_SMALL_FAST_MODEL": harness_model,
            "CLAUDE_CODE_GATEWAY_HINT_HEADERS": "1",
            "IS_SANDBOX": "1",
            "API_TIMEOUT_MS": str(config.api_timeout_ms),
            "BASH_DEFAULT_TIMEOUT_MS": str(config.bash_timeout_ms),
            "BASH_MAX_TIMEOUT_MS": str(config.bash_timeout_ms),
            "NO_PROXY": "127.0.0.1,localhost,host.docker.internal",
            "no_proxy": "127.0.0.1,localhost,host.docker.internal",
        }
        env.update(_DISABLED_FEATURES)
        env.update(_PROVIDER_OVERRIDES)

        if config.context_window > 0:
            env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] = str(config.context_window)
            # The auto-compact window key/flag floor is 100k tokens.
            if config.context_window >= 100_000:
                env["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] = str(config.context_window)
        if config.context_window <= 200_000:
            # Keep a gateway model ID on the 200K assumptions even if Claude
            # Code would otherwise treat it as a 1M-context model.
            env["CLAUDE_CODE_DISABLE_1M_CONTEXT"] = "1"
        if config.compaction_threshold > 0:
            env["CLAUDE_AUTOCOMPACT_PCT_OVERRIDE"] = str(config.compaction_threshold)
        if config.max_output_tokens is not None:
            env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = str(config.max_output_tokens)
        if config.custom_headers:
            # Claude Code forwards these on every request; the proxy forwards
            # them to the upstream model backend.
            env["ANTHROPIC_CUSTOM_HEADERS"] = "\n".join(
                f"{name}: {value}" for name, value in config.custom_headers.items()
            )
        env.update(config.extra_env)
        return env

    def build_settings(
        self, config: HarnessConfig, *, compaction_window: int | None = None
    ) -> dict[str, Any]:
        """The ``--settings`` JSON passed to the harness for this session."""
        settings: dict[str, Any] = {
            "includeCoAuthoredBy": False,
            "cleanupPeriodDays": 1,
            "autoCompactEnabled": True,
        }
        if compaction_window is not None and compaction_window >= 100_000:
            settings["autoCompactWindow"] = compaction_window
        if config.skip_permissions:
            settings["permissions"] = {"defaultMode": "bypassPermissions"}
        return settings

    def build_command(
        self,
        config: HarnessConfig,
        installed: InstalledHarness,
        *,
        instruction: str,
        max_turns: int,
        settings_path: str,
        harness_model: str,
    ) -> list[str]:
        """Argv for one headless Claude Code run."""
        argv = [
            installed.binary_path,
            "-p",
            instruction,
            "--output-format",
            "stream-json",
            # stream-json requires --verbose in print mode.
            "--verbose",
            "--settings",
            settings_path,
            "--model",
            harness_model,
            "--no-session-persistence",
        ]
        if config.skip_permissions:
            argv.append("--dangerously-skip-permissions")
        if config.bare:
            argv.append("--bare")
        if max_turns > 0:
            argv.extend(["--max-turns", str(max_turns)])
        argv.extend(config.extra_args)
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
        """Write settings/instruction and run the harness headless."""
        workspace.write_text(
            instruction,
            installed.instruction_path,
        )
        settings = self.build_settings(
            config,
            compaction_window=(
                config.context_window if config.context_window >= 100_000 else None
            ),
        )
        workspace.write_text(
            json.dumps(settings, indent=2) + "\n",
            installed.settings_path,
        )
        env = self.build_env(config, proxy_url=proxy_url, harness_model=harness_model)
        if extra_env:
            env.update(extra_env)
        argv = self.build_command(
            config,
            installed,
            instruction=instruction,
            max_turns=max_turns,
            settings_path=installed.settings_path,
            harness_model=harness_model,
        )
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
            result_event=parse_stream_json_result(log_path),
        )


def build_distribution_bind_volumes(
    config: HarnessConfig, distribution: HarnessDistribution
) -> list[str]:
    """Bind mounts needed to expose a portable distribution in the container."""
    if config.dist_mode != "mount":
        return []
    install_dir = config.install_dir.rstrip("/") or "/opt/harness"
    return [f"{distribution.directory}:{install_dir}:ro"]
