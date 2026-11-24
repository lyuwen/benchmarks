#!/usr/bin/env python3
"""Manual end-to-end smoke test for the blackbox harness pipeline.

Runs a real harness binary (Claude Code or DeepSeek Harness) inside a local
Docker container, with a fake OpenAI-compatible upstream behind the
rollout-manager capture proxy, and checks that the patch and the trajectory
export are produced.

It needs Docker and a container image with ``git`` and ``bash``::

    uv run python scripts/harness_smoke.py \\
        --image 021harbor.zero2x.org/swe_pr_agent/python_images:quran-tafseer__tafseer_api-14 \\
        --claude-binary /path/to/downloaded/claude \\
        --dist-mode mount

``--claude-binary`` lets you reuse a locally downloaded native binary instead of
downloading the ~380 MiB release again; without it the normal distribution
cache/download path is exercised.

DeepSeek Harness can be smoke-tested the same way; ``--dsh-runtime`` seeds the
cache from a local runtime directory (as installed by the Python SDK), the
extracted executable or the published runtime wheel::

    uv run python scripts/harness_smoke.py --harness deepseek-harness \
        --image <task-image> --dsh-runtime /path/to/deepseek_harness_runtime/runtime

To check that Claude Code's auto-mode classifier requests are captured, run with
``--permission-mode auto`` and a Bash command the classifier has to review, e.g.
``--tool-command 'curl -sS https://example.com -o /testbed/out.html'``. The
scripted upstream answers classifier prompts with plain text, so the classifier
may deny the command (no patch is produced) — the captured ``request_log`` then
shows the ``auxiliary`` classifier turns, which is what the experiment checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from benchmarks.utils.harnesses.config import (
    DEFAULT_CLAUDE_CODE_VERSION,
    DEFAULT_DEEPSEEK_HARNESS_VERSION,
    HarnessConfig,
)
from benchmarks.utils.harnesses.distribution import (
    BINARY_NAME,
    build_distribution_bind_volumes,
    distribution_dir,
    ensure_distribution,
    seed_deepseek_distribution,
)
from benchmarks.utils.harnesses.runner import HarnessRunRequest, run_harness_instance
from benchmarks.utils.harnesses.workspace import DockerContainerWorkspace
from benchmarks.utils.models import EvalInstance
from openhands.sdk import LLM


TOOL_COMMAND = "echo smoke-ok > /testbed/smoke.txt"
FINAL_TEXT = "Done: the smoke file has been created."
FORCED_REQUEST_PARAMS = {
    "temperature": 0.1,
    "top_p": 0.9,
    "reasoning_effort": "high",
    "max_tokens": 2048,
}


class FakeUpstream:
    """Minimal OpenAI Chat-Completions backend with a scripted tool call."""

    def __init__(self, tool_command: str, *, shell_tool: str = "Bash") -> None:
        self.tool_command = tool_command
        self.shell_tool = shell_tool
        self.requests: list[dict[str, Any]] = []
        self.server: ThreadingHTTPServer | None = None

    def start(self) -> int:
        upstream = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                pass

            def _json(self, payload: dict[str, Any], status: int = 200) -> None:
                body = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:  # noqa: N802 - stdlib API
                if self.path.endswith("/models"):
                    self._json({"object": "list", "data": [{"id": "smoke-model"}]})
                    return
                self._json({"error": "not found"}, status=404)

            def do_POST(self) -> None:  # noqa: N802 - stdlib API
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
                upstream.requests.append(body)
                request_class = (
                    self.headers.get("x-claude-code-request-class") or "main"
                )
                messages = body.get("messages") or []
                has_tool_result = any(
                    message.get("role") == "tool" for message in messages
                )
                model = body.get("model") or "smoke-model"
                usage = {
                    "prompt_tokens": 100,
                    "completion_tokens": 20,
                    "total_tokens": 120,
                }
                if request_class in {"auxiliary", "compaction"}:
                    message: dict[str, Any] = {
                        "role": "assistant",
                        "content": "smoke",
                    }
                    finish = "stop"
                elif has_tool_result or not body.get("tools"):
                    message = {"role": "assistant", "content": FINAL_TEXT}
                    finish = "stop"
                else:
                    message = {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_smoke_1",
                                "type": "function",
                                "function": {
                                    "name": upstream.shell_tool,
                                    "arguments": json.dumps(
                                        {
                                            "command": upstream.tool_command,
                                            "description": "create the smoke file",
                                        }
                                    ),
                                },
                            }
                        ],
                    }
                    finish = "tool_calls"
                self._json(
                    {
                        "id": f"chatcmpl-{len(upstream.requests)}",
                        "object": "chat.completion",
                        "created": 0,
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "message": message,
                                "finish_reason": finish,
                            }
                        ],
                        "usage": usage,
                    }
                )

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        thread.start()
        return int(self.server.server_address[1])

    def stop(self) -> None:
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()


def seed_distribution(cache_dir: str, version: str, binary: Path) -> None:
    directory = distribution_dir(cache_dir, "claude-code", version, "linux-x64")
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / BINARY_NAME
    shutil.copy2(binary, destination)
    destination.chmod(0o755)
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "harness": "claude-code",
                "version": version,
                "platform": "linux-x64",
                "binary": BINARY_NAME,
                "sha256": digest,
                "size_bytes": destination.stat().st_size,
            },
            indent=2,
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="Task image with git+bash")
    parser.add_argument(
        "--harness",
        choices=["claude-code", "deepseek-harness"],
        default="claude-code",
        help="Harness to smoke-test (default: claude-code)",
    )
    parser.add_argument(
        "--tool-command",
        default=None,
        help="Override the scripted Bash command (e.g. one auto mode classifies)",
    )
    parser.add_argument("--claude-binary", type=Path, default=None)
    parser.add_argument(
        "--dsh-runtime",
        type=Path,
        default=None,
        help=(
            "Seed DeepSeek Harness from an extracted runtime executable or a "
            "deepseek_harness_runtime_bin wheel instead of downloading it"
        ),
    )
    parser.add_argument("--dist-mode", choices=["copy", "mount"], default="mount")
    parser.add_argument(
        "--log-upstream-requests",
        choices=["off", "summary", "full"],
        default="off",
        help="Ask the rollout manager to log requests sent to the backend",
    )
    parser.add_argument(
        "--permission-mode",
        choices=[
            "auto",
            "bypassPermissions",
            "acceptEdits",
            "default",
            "plan",
            "danger-full-access",
            "workspace-write",
            "read-only",
        ],
        default=None,
        help="Harness permission mode (e.g. auto exercises the Claude Code classifier)",
    )
    parser.add_argument(
        "--tmp-dir",
        type=Path,
        default=None,
        help=(
            "Host directory for the cache/output (must be visible to the Docker "
            "daemon; defaults to a temp dir under the current directory)"
        ),
    )
    parser.add_argument("--keep", action="store_true", help="Keep temp dirs")
    args = parser.parse_args()

    version = (
        DEFAULT_CLAUDE_CODE_VERSION
        if args.harness == "claude-code"
        else DEFAULT_DEEPSEEK_HARNESS_VERSION
    )
    if args.tmp_dir is not None:
        tmp_dir = str(args.tmp_dir)
        os.makedirs(tmp_dir, exist_ok=True)
    else:
        tmp_dir = tempfile.mkdtemp(prefix=".harness-smoke-", dir=os.getcwd())
    cache_dir = os.path.join(tmp_dir, "cache")
    output_dir = os.path.join(tmp_dir, "out")
    config = HarnessConfig(
        name=args.harness,
        version=version,
        platform="linux-x64",
        cache_dir=cache_dir,
        install_dir="/tmp/harness-dist",
        dist_mode=args.dist_mode,
        context_window=200_000,
        compaction_threshold=80,
        max_turns=6,
        request_params=FORCED_REQUEST_PARAMS,
        permission_mode=args.permission_mode,
        rollout_log_upstream=args.log_upstream_requests,
    )

    upstream = FakeUpstream(
        args.tool_command or TOOL_COMMAND,
        shell_tool="Bash" if args.harness == "claude-code" else "bash",
    )
    port = upstream.start()
    workspace: DockerContainerWorkspace | None = None
    try:
        if args.harness == "deepseek-harness":
            if args.dsh_runtime is not None:
                seed_deepseek_distribution(config, args.dsh_runtime)
        elif args.claude_binary is not None:
            seed_distribution(cache_dir, version, args.claude_binary)
        distribution = ensure_distribution(config)
        print(f"harness: {config.name} {config.version}")
        print(f"distribution: {distribution.binary_path}")

        bind_volumes = build_distribution_bind_volumes(config, distribution)
        workspace = DockerContainerWorkspace.start(
            image=args.image,
            working_dir="/",
            bind_volumes=bind_volumes,
        )
        setup = workspace.execute_command(
            "set -e; rm -rf /testbed; mkdir -p /testbed; cd /testbed; "
            "git init -q; git config user.email smoke@example.com; "
            "git config user.name smoke; echo original > file.txt; "
            "git add -A; git commit -qm base; git rev-parse HEAD",
            timeout=120,
        )
        if setup.exit_code != 0:
            raise RuntimeError(f"repo setup failed: {setup.stderr}")
        base_commit = setup.stdout.strip().splitlines()[-1]
        print(f"base commit: {base_commit}")

        llm = LLM(
            model="openai/smoke-model",
            base_url=f"http://127.0.0.1:{port}/v1",
            api_key="smoke",
        )
        result = run_harness_instance(
            HarnessRunRequest(
                instance=EvalInstance(id="smoke-1", data={"repo": "smoke/repo"}),
                attempt=1,
                repo_path="/testbed",
                base_commit=base_commit,
                instruction=(
                    "Create the file /testbed/smoke.txt containing smoke-ok, then stop."
                ),
                output_dir=output_dir,
                max_iterations=6,
                conversation_timeout=600.0,
            ),
            workspace=workspace,
            config=config,
            llm=llm,
            judge=None,
        )
        print(f"warnings: {result.warnings}")
        print(f"patch:\n{result.git_patch}")
        print(f"history: {result.history_path}")

        assert "smoke.txt" in result.git_patch, "harness change missing from patch"
        assert "smoke-ok" in result.git_patch, "file content missing from patch"
        assert Path(result.history_path).is_file(), "history file missing"
        history = json.loads(Path(result.history_path).read_text())
        assert history["stats"]["requests"] >= 1, "no captured requests"
        assert history["trajectories"], "no captured trajectories"
        assert upstream.requests, "upstream saw no traffic"
        first_request = upstream.requests[0]
        for key, value in FORCED_REQUEST_PARAMS.items():
            assert first_request.get(key) == value, (
                f"forced request param {key!r} missing from upstream request: "
                f"{first_request.get(key)!r}"
            )
        assert result.rollout_export_path is not None
        manager_logs = sorted(Path(tmp_dir).glob("out/logs/*.rollout-manager.log"))
        if args.log_upstream_requests != "off" and manager_logs:
            log_text = manager_logs[0].read_text(errors="replace")
            assert "upstream request:" in log_text, (
                "rollout manager did not log upstream requests"
            )
            print(
                "manager logged upstream requests: "
                f"{log_text.count('upstream request:')} line(s)"
            )
        request_classes = sorted(
            {
                str(entry.get("request_class"))
                for entry in history.get("request_log") or []
            }
        )
        print(f"captured request classes: {request_classes}")
        if config.name == "deepseek-harness":
            assert history["run"]["result"], "missing dsh final result event"
            assert history["run"]["result"].get("text"), "dsh run produced no answer"
        print(
            "SMOKE OK: "
            f"{history['stats']['requests']} request(s), "
            f"{history['stats']['trajectories']} trajectory(ies)"
        )
        return 0
    finally:
        upstream.stop()
        if workspace is not None:
            workspace.cleanup()
        if args.keep:
            print(f"kept temp dir: {tmp_dir}")
        else:
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
