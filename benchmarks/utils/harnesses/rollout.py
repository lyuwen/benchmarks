"""Launch, drive and read the rollout-manager capture proxy.

One ``rollout-manager`` process captures exactly one agent session. The
evaluation worker starts it next to the Docker daemon, points the in-container
harness at it, and finalizes it once the harness exits. The proxy writes the
session's trajectory series as one JSON document, which this module also knows
how to read back into metrics/history payloads.

The ``rollout-manager`` package is not imported: it runs as a subprocess in the
worker's own interpreter, so the harness pipeline does not depend on its
internal API.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import logging
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeGuard

from openhands.sdk.llm import Metrics
from openhands.sdk.llm.utils.metrics import TokenUsage


logger = logging.getLogger(__name__)

ROLLOUT_MODULE = "rollout_manager"
ROLLOUT_SESSION_TYPE = "rollout_session"
ROLLOUT_SCHEMA_VERSION = 2
DEFAULT_ROLLOUT_PORT = 30001
HEALTH_POLL_INTERVAL_SEC = 0.2

# Client-side LiteLLM provider prefixes that are not part of the model name the
# OpenAI-compatible backend expects.
_LITELLM_PROXY_PREFIX = "litellm_proxy/"
_KNOWN_PROVIDER_PREFIXES = frozenset(
    {
        "openai",
        "anthropic",
        "azure",
        "bedrock",
        "deepseek",
        "gemini",
        "groq",
        "mistral",
        "ollama",
        "openrouter",
        "together_ai",
        "vertex_ai",
        "xai",
    }
)


class RolloutManagerError(RuntimeError):
    """The capture proxy could not be started, driven, or read back."""


@dataclass
class RolloutManagerConfig:
    """Everything needed to launch one session capture."""

    upstream_base_url: str
    session_id: str
    output_path: str
    log_path: str
    upstream_api_key: str | None = None
    model: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    exclude_request_classes: str = "auxiliary"
    request_params: dict[str, Any] = field(default_factory=dict)
    log_upstream_requests: str = "off"
    log_upstream_max_chars: int = 20000
    request_timeout_sec: float = 900.0
    finish_wait_sec: float = 60.0
    start_timeout_sec: float = 90.0
    host: str = "127.0.0.1"
    port: int = 0
    python_executable: str = sys.executable


@dataclass
class RolloutManagerStatus:
    session_id: str
    output: str
    finished: bool
    requests: int
    turns: int
    trajectories: int
    compaction_requests: int
    routing_key: str | None = None


def rollout_manager_command(config: RolloutManagerConfig) -> list[str]:
    """Build the ``python -m rollout_manager`` argv for one session."""
    argv = [
        config.python_executable,
        "-m",
        ROLLOUT_MODULE,
        "--upstream",
        config.upstream_base_url,
        "--output",
        config.output_path,
        "--session-id",
        config.session_id,
        "--host",
        config.host,
        "--port",
        str(config.port),
        "--request-timeout-sec",
        str(config.request_timeout_sec),
        "--finish-wait-sec",
        str(config.finish_wait_sec),
        "--log-level",
        "INFO",
    ]
    if config.upstream_api_key:
        argv.extend(["--upstream-api-key", config.upstream_api_key])
    if config.model:
        argv.extend(["--model", config.model])
    if config.metadata:
        argv.extend(["--metadata", json.dumps(config.metadata, sort_keys=True)])
    if config.exclude_request_classes:
        argv.extend(["--exclude-request-classes", config.exclude_request_classes])
    if config.request_params:
        argv.extend(
            [
                "--request-params",
                json.dumps(config.request_params, sort_keys=True),
            ]
        )
    if config.log_upstream_requests != "off":
        argv.extend(
            [
                "--log-upstream-requests",
                config.log_upstream_requests,
                "--log-upstream-max-chars",
                str(config.log_upstream_max_chars),
            ]
        )
    return argv


def rollout_manager_available() -> bool:
    """Whether the rollout-manager package is installed in this interpreter."""
    return importlib.util.find_spec(ROLLOUT_MODULE) is not None


def require_rollout_manager() -> None:
    if rollout_manager_available():
        return
    raise RolloutManagerError(
        "The 'rollout-manager' package is not installed. Install it with "
        "`uv sync` (it is a project dependency) or "
        "`pip install "
        "git+https://github.com/lyuwen/rollout-manager.git`."
    )


def find_free_port(host: str) -> int:
    """Reserve an ephemeral TCP port on ``host`` and return it."""
    bind_host = host or "0.0.0.0"
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((bind_host, 0))
        return int(sock.getsockname()[1])


_CONTROL_OPENER: urllib.request.OpenerDirector | None = None

_PROXY_ENV_NAMES = (
    "http_proxy",
    "HTTP_PROXY",
    "https_proxy",
    "HTTPS_PROXY",
    "all_proxy",
    "ALL_PROXY",
    "no_proxy",
    "NO_PROXY",
)


def _control_opener() -> urllib.request.OpenerDirector:
    """Opener for the local control endpoint that ignores proxy settings.

    Evaluation hosts commonly export ``http_proxy`` / ``all_proxy`` for model
    backends or package mirrors. Those proxies must never be used for the
    rollout-manager control endpoint (usually the Docker bridge gateway): a
    blackholed proxy otherwise looks exactly like "rollout-manager did not
    become healthy (timed out)".
    """
    global _CONTROL_OPENER
    if _CONTROL_OPENER is None:
        _CONTROL_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return _CONTROL_OPENER


def _proxy_env_summary() -> str:
    values = {name: os.environ[name] for name in _PROXY_ENV_NAMES if name in os.environ}
    return json.dumps(values, sort_keys=True) if values else "none"


def http_json(
    url: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    timeout: float = 30.0,
) -> tuple[int, dict[str, Any]]:
    """Small JSON-over-HTTP helper used for the proxy control endpoints.

    Always connects directly: control traffic is host-local and must not be
    affected by ``http_proxy``/``all_proxy`` environment variables.
    """
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(  # noqa: S310 - loopback control endpoint
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with _control_opener().open(request, timeout=timeout) as response:  # noqa: S310
            body = response.read()
            return response.status, _decode_json(body)
    except urllib.error.HTTPError as exc:
        return exc.code, _decode_json(exc.read())
    except (urllib.error.URLError, TimeoutError) as exc:
        reason = getattr(exc, "reason", exc)
        raise ConnectionError(str(reason or exc)) from exc


def _decode_json(raw: bytes | str) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def spawn_rollout_manager(
    argv: list[str],
    *,
    log_path: str,
    cwd: str | None = None,
) -> tuple[subprocess.Popen[str], Any]:
    """Start the proxy process with its log redirected to ``log_path``.

    Returns the process and the (still open) log file handle; the caller closes
    the handle when it stops the process.
    """
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "a", encoding="utf-8")
    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    # The proxy only talks to host-local services (task containers, the
    # upstream model backend); make sure inherited proxy settings never apply
    # to the loopback/bridge part of that traffic.
    local_hosts = "127.0.0.1,localhost,host.docker.internal"
    inherited_bypass = env.get("NO_PROXY") or env.get("no_proxy") or ""
    env["NO_PROXY"] = ",".join(
        part for part in (inherited_bypass.strip(","), local_hosts) if part
    )
    env["no_proxy"] = env["NO_PROXY"]
    try:
        process = subprocess.Popen(  # noqa: S603 - argv list, no shell
            argv,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            cwd=cwd,
            env=env,
            text=True,
        )
    except FileNotFoundError as exc:
        log_file.close()
        raise RolloutManagerError(f"Cannot start rollout-manager: {exc}") from exc
    return process, log_file


class RolloutManager:
    """Lifecycle wrapper around one ``rollout-manager`` capture session."""

    def __init__(
        self,
        config: RolloutManagerConfig,
        *,
        spawn: Callable[..., tuple[subprocess.Popen[str], Any]] = spawn_rollout_manager,
        request: Callable[..., tuple[int, dict[str, Any]]] = http_json,
    ) -> None:
        self.config = config
        self._spawn = spawn
        self._request = request
        self.process: subprocess.Popen[str] | None = None
        self._log_file: Any = None
        self.port = config.port or 0
        self._finished_summary: dict[str, Any] | None = None

    # -------------------- lifecycle ---------------------------------------

    def command(self) -> list[str]:
        return rollout_manager_command(dataclasses.replace(self.config, port=self.port))

    @property
    def base_url(self) -> str:
        """Control URL, reachable from this host process."""
        host = self.config.host
        if host in {"0.0.0.0", "::", ""}:
            host = "127.0.0.1"
        return f"http://{host}:{self.port}"

    @property
    def export_path(self) -> str:
        return self.config.output_path

    def start(self) -> None:
        require_rollout_manager()
        if self.port == 0:
            self.port = find_free_port(self.config.host)
        logger.info(
            "Starting rollout manager for %s on %s (upstream=%s, output=%s)",
            self.config.session_id,
            self.base_url,
            self.config.upstream_base_url,
            self.config.output_path,
        )
        self.process, self._log_file = self._spawn(
            self.command(), log_path=self.config.log_path
        )
        try:
            self.wait_ready(self.config.start_timeout_sec)
        except BaseException:
            # Never leak a capture proxy that failed its readiness probe: the
            # evaluation will retry with a fresh one.
            self.stop()
            raise

    def wait_ready(self, timeout: float | None = None) -> None:
        deadline = time.monotonic() + (timeout if timeout is not None else 90.0)
        last_error = ""
        while time.monotonic() < deadline:
            if self.process is not None and self.process.poll() is not None:
                raise RolloutManagerError(
                    "rollout-manager exited during startup "
                    f"(code {self.process.returncode}). Log:\n"
                    f"{self._log_tail()}"
                )
            try:
                code, _ = self._request(f"{self.base_url}/healthz", timeout=5.0)
            except (ConnectionError, OSError) as exc:
                last_error = str(exc)
            else:
                if code == 200:
                    logger.info("Rollout manager healthy on %s", self.base_url)
                    return
                last_error = f"healthz returned HTTP {code}"
            time.sleep(HEALTH_POLL_INTERVAL_SEC)
        raise RolloutManagerError(
            "rollout-manager did not become healthy within the timeout "
            f"({last_error or 'no response'}). "
            f"Control endpoint {self.base_url}: "
            f"{_probe_endpoint(self.config.host, self.port)}. "
            f"Proxy env: {_proxy_env_summary()}. Log:\n{self._log_tail()}"
        )

    def status(self) -> RolloutManagerStatus:
        code, payload = self._request(f"{self.base_url}/rollout/status", timeout=30.0)
        if code != 200:
            raise RolloutManagerError(
                f"rollout-manager status request failed with HTTP {code}: {payload}"
            )
        return _status_from_payload(payload)

    def finish(self, timeout: float | None = None) -> dict[str, Any]:
        """Drain in-flight turns and write the final export (idempotent)."""
        if self._finished_summary is not None:
            return self._finished_summary
        code, payload = self._request(
            f"{self.base_url}/rollout/finish",
            method="POST",
            payload={},
            timeout=timeout if timeout is not None else 120.0,
        )
        if code != 200:
            raise RolloutManagerError(
                f"rollout-manager finish request failed with HTTP {code}: {payload}"
            )
        self._finished_summary = payload
        logger.info(
            "Rollout session %s finalized: %s turn(s), %s trajectory(ies) -> %s",
            payload.get("session_id"),
            payload.get("turns"),
            payload.get("trajectories"),
            payload.get("output"),
        )
        return payload

    def stop(self) -> None:
        process = self.process
        self.process = None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=15)
        log_file, self._log_file = self._log_file, None
        if log_file is not None:
            try:
                log_file.close()
            except OSError:  # pragma: no cover - defensive
                pass

    def __enter__(self) -> "RolloutManager":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        try:
            if exc_type is None:
                self.finish(self.config.finish_wait_sec + 60.0)
        except RolloutManagerError:
            logger.warning("Failed to finalize rollout session cleanly")
        finally:
            self.stop()

    def _log_tail(self, lines: int = 40) -> str:
        path = Path(self.config.log_path)
        if not path.is_file():
            return ""
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError:  # pragma: no cover - defensive
            return ""
        return "\n".join(content.splitlines()[-lines:])


def _probe_endpoint(host: str, port: int, timeout: float = 3.0) -> str:
    """Best-effort TCP diagnostic for the control endpoint (used on failure)."""
    probe_host = "127.0.0.1" if host in {"0.0.0.0", "::", ""} else host
    try:
        with socket.create_connection((probe_host, port), timeout=timeout):
            return "tcp connect ok"
    except OSError as exc:
        return f"tcp connect failed: {type(exc).__name__}: {exc}"


def _status_from_payload(payload: dict[str, Any]) -> RolloutManagerStatus:
    return RolloutManagerStatus(
        session_id=str(payload.get("session_id") or ""),
        output=str(payload.get("output") or ""),
        finished=bool(payload.get("finished")),
        requests=int(payload.get("requests") or 0),
        turns=int(payload.get("turns") or 0),
        trajectories=int(payload.get("trajectories") or 0),
        compaction_requests=int(payload.get("compaction_requests") or 0),
        routing_key=payload.get("routing_key"),
    )


# ---------------------------------------------------------------------------
# upstream resolution
# ---------------------------------------------------------------------------


def normalize_upstream_base_url(base_url: str) -> str:
    """Return an OpenAI-compatible base URL (``.../v1``) from an LLM config URL."""
    url = (base_url or "").strip().rstrip("/")
    if not url:
        raise RolloutManagerError(
            "The LLM config has no base_url; pass --upstream-base-url pointing "
            "at an OpenAI-compatible endpoint"
        )
    if url.endswith("/v1") or "/v1/" in f"{url}/":
        return url
    return f"{url}/v1"


def resolve_upstream_model(model: str) -> str:
    """Strip client-side LiteLLM provider prefixes from an LLM config model."""
    resolved = (model or "").strip()
    if not resolved:
        raise RolloutManagerError("The LLM config has no model; pass --upstream-model")
    if resolved.startswith(_LITELLM_PROXY_PREFIX):
        # A LiteLLM proxy names its models "<provider>/<model>"; only the
        # proxy-internal prefix is client-side.
        return resolved[len(_LITELLM_PROXY_PREFIX) :]
    first, _, rest = resolved.partition("/")
    if rest and first in _KNOWN_PROVIDER_PREFIXES:
        return rest
    return resolved


# ---------------------------------------------------------------------------
# export inspection
# ---------------------------------------------------------------------------


def load_rollout_export(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load and validate one rollout-manager session export."""
    export_path = Path(path)
    if not export_path.is_file():
        raise RolloutManagerError(f"Rollout export not found: {export_path}")
    try:
        data = json.loads(export_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RolloutManagerError(
            f"Rollout export {export_path} is not valid JSON: {exc}"
        ) from exc
    if not isinstance(data, dict):
        raise RolloutManagerError(f"Rollout export {export_path} is not an object")
    if data.get("type") != ROLLOUT_SESSION_TYPE:
        raise RolloutManagerError(
            f"Rollout export {export_path} has unexpected type {data.get('type')!r}"
        )
    version = data.get("version")
    if version != ROLLOUT_SCHEMA_VERSION:
        raise RolloutManagerError(
            f"Rollout export {export_path} has unsupported schema version "
            f"{version!r} (expected {ROLLOUT_SCHEMA_VERSION})"
        )
    return data


def export_summary(export: dict[str, Any]) -> dict[str, Any]:
    """Compact, log-friendly summary of a rollout export."""
    stats = export.get("stats") or {}
    return {
        "session_id": export.get("session_id"),
        "finished": bool(export.get("finished")),
        "finish_reason": export.get("finish_reason"),
        "model": export.get("model"),
        "requests": int(stats.get("requests") or 0),
        "turns": int(stats.get("turns") or 0),
        "trajectories": int(stats.get("trajectories") or 0),
        "compaction_requests": int(stats.get("compaction_requests") or 0),
        "excluded_turns": int(stats.get("excluded_turns") or 0),
        "request_classes": stats.get("request_classes") or {},
    }


def iter_turns(export: dict[str, Any]) -> Iterator[dict[str, Any]]:
    for trajectory in export.get("trajectories") or []:
        if not isinstance(trajectory, dict):
            continue
        for turn in trajectory.get("turns") or []:
            if isinstance(turn, dict):
                yield turn


def build_metrics(export: dict[str, Any], model_name: str) -> Metrics | None:
    """Aggregate per-turn token usage into the SDK ``Metrics`` model."""
    usages: list[TokenUsage] = []
    for turn in iter_turns(export):
        usage = turn.get("usage") or {}
        if not isinstance(usage, dict):
            continue
        prompt = int(usage.get("prompt_tokens") or 0)
        completion = int(usage.get("completion_tokens") or 0)
        if prompt == 0 and completion == 0:
            continue
        usages.append(
            TokenUsage(
                model=str(turn.get("model") or model_name),
                prompt_tokens=prompt,
                completion_tokens=completion,
                cache_read_tokens=int(usage.get("cache_read_input_tokens") or 0),
                cache_write_tokens=int(usage.get("cache_creation_input_tokens") or 0),
                reasoning_tokens=int(
                    (usage.get("completion_tokens_details") or {}).get(
                        "reasoning_tokens"
                    )
                    or 0
                ),
                per_turn_token=prompt + completion,
                response_id=str(turn.get("response_id") or ""),
            )
        )
    if not usages:
        return None
    total = usages[0]
    for usage in usages[1:]:
        total = total + usage
    return Metrics(
        model_name=model_name,
        accumulated_cost=0.0,
        token_usages=usages,
        accumulated_token_usage=total,
    )


def main_trajectory(export: dict[str, Any]) -> dict[str, Any] | None:
    """Return the main conversation trajectory (longest non-excluded branch)."""
    candidates = [
        trajectory
        for trajectory in export.get("trajectories") or []
        if isinstance(trajectory, dict)
        and trajectory.get("kind", "conversation") == "conversation"
        and not trajectory.get("exclude_from_training")
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda trajectory: (
            int(trajectory.get("leaf_turn_index") or 0),
            len(trajectory.get("messages") or []),
        ),
    )


def is_openai_tool_list(value: Any) -> TypeGuard[list[dict[str, Any]]]:
    """Whether ``value`` is a non-empty OpenAI function-tool list."""
    return (
        isinstance(value, list)
        and bool(value)
        and all(
            isinstance(tool, dict) and str(tool.get("type")) == "function"
            for tool in value
        )
    )


def resolve_trajectory_tools(
    trajectory: dict[str, Any], history: dict[str, Any]
) -> list[dict[str, Any]]:
    """Resolve the OpenAI function-tool list a captured trajectory used.

    The export keeps deduplicated variants in ``tool_sets`` and a per-turn
    ``tool_set`` index; this picks the variant of the last turn (tools can change
    mid-session), ``[]`` for explicitly tool-less requests (auto-mode
    classifier/security monitor), and falls back to the root ``tools`` view only
    for older dumps without per-trajectory sets.
    """
    tool_sets = trajectory.get("tool_sets")
    turns = [turn for turn in trajectory.get("turns") or [] if isinstance(turn, dict)]
    if isinstance(tool_sets, list):
        if not tool_sets:
            # Requests carried no tools at all; never inherit another
            # trajectory's tool list.
            return []
        for turn in reversed(turns):
            index = turn.get("tool_set")
            if (
                isinstance(index, int)
                and 0 <= index < len(tool_sets)
                and is_openai_tool_list(tool_sets[index])
            ):
                return tool_sets[index]
        for candidate in tool_sets:
            if is_openai_tool_list(candidate):
                return candidate
        return []

    # Fallbacks for older/foreign history dumps without per-trajectory sets.
    top_tools = history.get("tools")
    if is_openai_tool_list(top_tools):
        return top_tools
    if isinstance(top_tools, list):
        for candidate in top_tools:
            if is_openai_tool_list(candidate):
                return candidate
    return []


def collect_tool_sets(export: dict[str, Any]) -> list[list[dict[str, Any]]]:
    """Deduplicated union of the function-tool lists seen in the session."""
    seen: set[str] = set()
    tools: list[list[dict[str, Any]]] = []
    for trajectory in export.get("trajectories") or []:
        if not isinstance(trajectory, dict):
            continue
        for tool_set in trajectory.get("tool_sets") or []:
            if not isinstance(tool_set, list):
                continue
            key = json.dumps(tool_set, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            tools.append(tool_set)
    return tools


def parse_stream_json_result(log_path: str | os.PathLike[str]) -> dict[str, Any] | None:
    """Read the final ``result`` event from a ``claude -p --output-format stream-json`` log."""
    path = Path(log_path)
    if not path.is_file():
        return None
    result: dict[str, Any] | None = None
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if not line.startswith("{"):
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(event, dict) and event.get("type") == "result":
                    result = event
    except OSError:  # pragma: no cover - defensive
        return None
    return result


def unwrap_secret(value: Any) -> Any:
    """Unwrap pydantic SecretStr-like values into plain strings."""
    if value is None:
        return None
    unwrap = getattr(value, "get_secret_value", None)
    if callable(unwrap):
        return unwrap()
    return value


def upstream_credentials(llm: Any, config: Any) -> tuple[str, str | None, str, str]:
    """Resolve (upstream_url, api_key, upstream_model, harness_model)."""
    base_url = config.upstream_base_url or getattr(llm, "base_url", None)
    upstream_url = normalize_upstream_base_url(str(base_url or ""))
    api_key = config.upstream_api_key or unwrap_secret(getattr(llm, "api_key", None))
    raw_model = config.upstream_model or getattr(llm, "model", None)
    upstream_model = resolve_upstream_model(str(raw_model or ""))
    harness_model = config.harness_model or upstream_model
    return upstream_url, api_key, upstream_model, harness_model
