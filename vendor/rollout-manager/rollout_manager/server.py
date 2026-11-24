"""Rollout-capture request adapter service for one agent session.

Launch one instance in front of the model backend just before the harness starts,
and stop it when the harness finishes::

    python -m rollout_manager \
        --upstream http://127.0.0.1:30000/v1 \
        --output ./rollout-captures/run-42.json \
        --session-id run-42 \
        --port 30001

The package is self-contained (``pip install -e .`` gives you the
``rollout-manager`` console script); it is not part of the installed ``slime``
package.

The service speaks OpenAI Chat-Completions, Anthropic Messages and OpenAI
Responses to the agent, converts everything to Chat-Completions for the upstream
(SGLang, vLLM, an external API -- anything with ``/v1/chat/completions``), and
exports the session's trajectories to ``--output``. It never renders a chat
template; the backend tokenizes.

The export is a *series* of trajectories: a normal session produces one, while a
context compaction or a sub-agent branch adds more. Messages are strict
OpenAI Chat-Completions messages; training masks and provider reasoning are
separate aligned arrays; per-turn effective requests, original responses and an
ordered request log are retained.

Compaction and request purpose are taken from the harness's explicit
``x-claude-code-*`` gateway hint headers, never inferred from request shape. A
signalled compaction without tools is split into its own trajectory; one with
tools is labelled but stays a continuation. Auxiliary requests can be excluded
from training via ``--exclude-request-classes``.

Client request headers are forwarded to the upstream transparently (only
hop-by-hop/transport headers the proxy must own are replaced); an explicit
``--upstream-api-key`` overrides a forwarded ``Authorization``.

Upstream calls are always non-streaming so the full reply can be captured; when
the client asked for streaming, the reply is re-rendered as the protocol's SSE
stream after it arrives. That mirrors how ``slime.agent.adapters`` serves a fully
realized turn.

Control endpoints::

    POST /rollout/finish   # drain in-flight turns, write the final export
    GET  /rollout/status   # current session state
    GET  /healthz          # readiness probe
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import os
import re
import secrets
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import aiohttp
from aiohttp import web

from .capture import RolloutCapture, SessionClosed
from .convert import (
    ChatReply,
    anthropic_count_tokens,
    anthropic_to_chat_request,
    canonical_reply_message,
    canonicalize_messages,
    canonicalize_tools,
    chat_to_anthropic_response,
    chat_to_responses_response,
    parse_chat_reply,
    responses_to_chat_request,
    wire_history_message,
)
from .trajectory import TurnRecord

logger = logging.getLogger(__name__)

__all__ = ["Config", "RolloutProxy", "build_app", "main"]

_UPSTREAM_PASSTHROUGH_STATUS = {400, 401, 403, 404, 413, 422, 429}
_SAFE_FILENAME = re.compile(r"[^A-Za-z0-9._-]+")

# Routing headers the proxy sets itself so a backend (e.g. an SGLang gateway)
# pins every forwarded request to the same worker/cache. Their value is a fresh
# per-launch routing key; client-supplied values for these names are replaced.
_ROUTING_HEADER_NAMES = ("X-SMG-Routing-Key", "X-Session-ID")

# Request fields a --request-params overlay must never replace: they define the
# captured conversation/protocol rather than sampling behaviour.
_FORBIDDEN_REQUEST_PARAM_KEYS = frozenset({"messages", "stream", "stream_options", "tools"})

# Headers aiohttp must own when talking to the upstream. Everything else is
# forwarded transparently, including Authorization, X-Api-Key, cookies and
# vendor/tracing headers.
_NOT_FORWARDED_HEADERS = frozenset(
    {
        # hop-by-hop / transport level (RFC 7230) and values aiohttp regenerates
        "host",
        "content-length",
        "transfer-encoding",
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "upgrade",
        "expect",
        "accept-encoding",
        # the proxy always calls the upstream non-streaming with a JSON body
        "accept",
        "content-type",
    }
)


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Config:
    """Service configuration; see ``main`` for the CLI flags that fill it."""

    upstream: str
    output: str
    session_id: str
    upstream_api_key: str | None = None
    model: str | None = None
    host: str = "0.0.0.0"
    port: int = 30001
    request_timeout_sec: float = 900.0
    finish_wait_sec: float = 30.0
    max_turns: int | None = None
    max_connections: int = 512
    chars_per_token: float = 4.0
    response_cache_size: int = 256
    snapshot_every_turn: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)
    # Request classes (x-claude-code-request-class) whose turns are kept but
    # excluded from the training mask, e.g. auxiliary classifier/title calls.
    exclude_request_classes: tuple[str, ...] = ()
    # Forced parameters merged (overriding) into every upstream
    # Chat-Completions payload after protocol conversion. Used by benchmark
    # launchers to apply the configured sampling parameters (temperature,
    # top_p, reasoning_effort, ...) to harness traffic that does not carry them.
    request_params: dict[str, Any] = field(default_factory=dict)

    @property
    def upstream_base(self) -> str:
        base = self.upstream.rstrip("/")
        return base[: -len("/chat/completions")] if base.endswith("/chat/completions") else base

    @property
    def chat_completions_url(self) -> str:
        base = self.upstream.rstrip("/")
        return base if base.endswith("/chat/completions") else f"{base}/chat/completions"

    @property
    def models_url(self) -> str:
        return f"{self.upstream_base}/models"


def parse_request_params(raw: str | None) -> dict[str, Any]:
    """Parse and validate the ``--request-params`` JSON overlay."""
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"--request-params is not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("--request-params must be a JSON object")
    forbidden = set(parsed) & _FORBIDDEN_REQUEST_PARAM_KEYS
    if forbidden:
        raise ValueError("--request-params may not override structural fields: " + ", ".join(sorted(forbidden)))
    return parsed


# ---------------------------------------------------------------------------
# errors
# ---------------------------------------------------------------------------


class WebError(Exception):
    """Client-facing 4xx/5xx error carrying a protocol-agnostic message."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


class UpstreamError(Exception):
    """Non-2xx or transport failure from the model backend."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


def _error_response(protocol: str, status: int, message: str) -> web.Response:
    """Shape an error the way the client's protocol expects it."""
    if protocol == "anthropic":
        payload = {"type": "error", "error": {"type": "api_error", "message": message}}
    else:
        payload = {"error": {"message": message, "type": "rollout_manager_error", "code": status}}
    return web.json_response(payload, status=status)


def _upstream_error_status(exc: UpstreamError) -> int:
    return exc.status if exc.status in _UPSTREAM_PASSTHROUGH_STATUS else 502


# ---------------------------------------------------------------------------
# Responses API chaining state
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResponseState:
    """What ``previous_response_id`` chaining and retrieval need from a prior call."""

    wire_messages: list[dict[str, Any]]
    response: dict[str, Any]


class ResponseRegistry:
    """Bounded LRU of Responses objects, keyed by ``response.id``.

    Needed because a Responses client may send only the new input items and
    reference the prior turn by id; the stateless chat backend then has to
    receive the full history we rebuild from this registry.
    """

    def __init__(self, max_entries: int = 256) -> None:
        self.max_entries = max(1, int(max_entries))
        self._entries: OrderedDict[str, ResponseState] = OrderedDict()

    def put(self, response_id: str, state: ResponseState) -> None:
        self._entries[response_id] = state
        self._entries.move_to_end(response_id)
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)

    def get(self, response_id: str) -> ResponseState | None:
        state = self._entries.get(response_id)
        if state is not None:
            self._entries.move_to_end(response_id)
        return state


# ---------------------------------------------------------------------------
# proxy
# ---------------------------------------------------------------------------


class RolloutProxy:
    """aiohttp handlers implementing the single-session capture proxy."""

    def __init__(self, config: Config) -> None:
        self.config = config
        # Every service launch gets its own routing key, shared by all forwarded
        # requests so the backend routes this captured session consistently
        # without colliding with other launches.
        self.routing_key = f"rollout-{secrets.token_hex(16)}"
        self.capture = RolloutCapture(
            config.output,
            session_id=config.session_id,
            metadata=config.metadata,
            snapshot_every_turn=config.snapshot_every_turn,
            exclude_request_classes=config.exclude_request_classes,
        )
        self.responses = ResponseRegistry(config.response_cache_size)
        self._client: aiohttp.ClientSession | None = None

    # -------------------- lifecycle ---------------------------------------

    async def start(self) -> None:
        timeout = aiohttp.ClientTimeout(total=None, sock_connect=10, sock_read=self.config.request_timeout_sec)
        connector = aiohttp.TCPConnector(limit=self.config.max_connections)
        self._client = aiohttp.ClientSession(timeout=timeout, connector=connector)
        if self.capture.output_path.exists():
            logger.warning("capture file %s already exists; it will be overwritten", self.capture.output_path)
        logger.info(
            "rollout manager ready: session=%s routing_key=%s upstream=%s output=%s request_params=%s",
            self.config.session_id,
            self.routing_key,
            self.config.chat_completions_url,
            self.capture.output_path,
            json.dumps(self.config.request_params, sort_keys=True) if self.config.request_params else "none",
        )

    async def cleanup(self) -> None:
        was_finished = self.capture.finished
        summary = await self.capture.finish(reason="shutdown", wait_timeout=self.config.finish_wait_sec)
        if not was_finished:
            logger.info(
                "exported session %s on shutdown: %d turn(s), %d trajectory(ies) -> %s",
                summary["session_id"],
                summary["turns"],
                summary["trajectories"],
                summary["output"],
            )
        if self._client is not None:
            await self._client.close()
            self._client = None

    # -------------------- shared request machinery ------------------------

    @contextlib.asynccontextmanager
    async def _turn(self):
        """Track one in-flight turn, rejecting requests after the session is finalized."""
        session = self.capture.acquire()
        cap = self.config.max_turns
        if cap is not None and session.turn_count >= cap:
            raise WebError(429, f"session {session.session_id!r} exceeded max_turns={cap}")
        task = asyncio.current_task()
        session.inflight.add(task)
        try:
            yield session
        finally:
            session.inflight.discard(task)

    async def _dispatch(
        self, protocol: str, request: web.Request, handler, *, allow_empty_body: bool = False
    ) -> web.StreamResponse:
        try:
            body = await (_read_optional_json(request) if allow_empty_body else _read_json(request))
            return await handler(request, body)
        except WebError as exc:
            return _error_response(protocol, exc.status, exc.message)
        except SessionClosed:
            return _error_response(protocol, 503, "rollout session already finished")
        except UpstreamError as exc:
            logger.warning("upstream error (%s): %s", exc.status, exc.message[:400])
            return _error_response(protocol, _upstream_error_status(exc), exc.message[:2000])
        except Exception:
            logger.exception("unhandled error while serving %s", request.path)
            return _error_response(protocol, 500, "internal error in rollout manager")

    def _upstream_headers(self, request: web.Request, *, content_type: bool) -> list[tuple[str, str]]:
        """Forward the client's request headers to the upstream transparently.

        Every end-to-end header is copied as-is. Transport/hop-by-hop headers and
        the ones the proxy owns are regenerated (see ``_NOT_FORWARDED_HEADERS``).
        An explicit ``--upstream-api-key`` overrides a forwarded ``Authorization``.
        Backend routing headers (``X-SMG-Routing-Key``, ``X-Session-ID``) are set
        to this launch's routing key, replacing any client-supplied value.
        """
        routing_names = {name.lower() for name in _ROUTING_HEADER_NAMES}
        headers = [
            (name, value)
            for name, value in request.headers.items()
            if name.lower() not in _NOT_FORWARDED_HEADERS
            and name.lower() not in routing_names
            and not (name.lower() == "authorization" and self.config.upstream_api_key)
        ]
        headers.extend((name, self.routing_key) for name in _ROUTING_HEADER_NAMES)
        if content_type:
            headers.append(("Content-Type", "application/json"))
        if self.config.upstream_api_key:
            headers.append(("Authorization", f"Bearer {self.config.upstream_api_key}"))
        return headers

    async def _post_chat(self, request: web.Request, body: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        """Forward one request to the upstream, always non-streaming.

        Returns the raw upstream reply and the effective request body actually
        sent (including forced sampling/routing settings), for faithful capture.
        """
        if self._client is None:  # pragma: no cover - handlers run after startup
            raise UpstreamError(502, "service is not started")
        payload = dict(body)
        # The OpenAI SDK expands extra_body client-side; accept a literal object
        # too so raw clients/gateways can use the same escape hatch.
        extra_body = payload.pop("extra_body", None)
        if isinstance(extra_body, dict):
            for key, value in extra_body.items():
                payload.setdefault(key, value)
        payload["stream"] = False
        payload.pop("stream_options", None)
        if self.config.request_params:
            payload.update(self.config.request_params)
        if self.config.model:
            payload["model"] = self.config.model
        headers = self._upstream_headers(request, content_type=True)
        try:
            async with self._client.post(self.config.chat_completions_url, json=payload, headers=headers) as response:
                text = await response.text()
                if response.status >= 400:
                    raise UpstreamError(response.status, text)
        except asyncio.TimeoutError as exc:  # noqa: UP041 - alias of TimeoutError before 3.11
            raise UpstreamError(504, f"upstream request timed out: {exc}") from exc
        except aiohttp.ClientError as exc:
            raise UpstreamError(502, f"upstream request failed: {exc}") from exc
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise UpstreamError(502, f"upstream returned invalid JSON: {text[:300]}") from exc
        if not isinstance(data, dict):
            raise UpstreamError(502, "upstream returned a non-object JSON payload")
        return data, payload

    @staticmethod
    def _parse_reply(data: dict[str, Any]) -> ChatReply:
        try:
            return parse_chat_reply(data)
        except ValueError as exc:
            raise UpstreamError(502, str(exc)) from exc

    @staticmethod
    def _turn_record(
        request: web.Request,
        reply: ChatReply,
        *,
        protocol: str,
        tools,
        latency_ms: float,
        effective_request: dict[str, Any],
    ) -> TurnRecord:
        return TurnRecord(
            response_message=canonical_reply_message(reply),
            finish_reason=reply.finish_reason,
            usage=dict(reply.usage),
            response_id=reply.response_id,
            model=reply.model,
            protocol=protocol,
            tools=tools,
            latency_ms=round(latency_ms, 3),
            request=effective_request,
            response=reply.message,
            client=_client_metadata(request),
        )

    # -------------------- chat completions --------------------------------

    async def handle_chat_completions(self, request: web.Request) -> web.StreamResponse:
        return await self._dispatch("chat", request, self._chat_turn)

    async def _chat_turn(self, request: web.Request, body: dict[str, Any]) -> web.StreamResponse:
        stream = _wants_stream(request, body)
        prompt_messages = canonicalize_messages(body.get("messages"))
        tools = canonicalize_tools(body.get("tools"))

        async with self._turn():
            started = time.monotonic()
            data, effective_request = await self._post_chat(request, body)
            reply = self._parse_reply(data)
            self.capture.record(
                protocol="chat",
                prompt_messages=prompt_messages,
                turn=self._turn_record(
                    request,
                    reply,
                    protocol="chat",
                    tools=tools,
                    latency_ms=(time.monotonic() - started) * 1000,
                    effective_request=effective_request,
                ),
            )
            if stream:
                return await _stream_chat(request, reply)
            # Chat-Completions is the capture wire format, so the upstream reply
            # is forwarded untouched (including ids/usage the client expects).
            return web.json_response(data)

    # -------------------- Anthropic Messages -------------------------------

    async def handle_anthropic_messages(self, request: web.Request) -> web.StreamResponse:
        return await self._dispatch("anthropic", request, self._anthropic_turn)

    async def _anthropic_turn(self, request: web.Request, body: dict[str, Any]) -> web.StreamResponse:
        stream = _wants_stream(request, body)
        upstream_body = anthropic_to_chat_request(body)
        prompt_messages = canonicalize_messages(upstream_body.get("messages"))
        tools = canonicalize_tools(upstream_body.get("tools"))

        async with self._turn():
            started = time.monotonic()
            data, effective_request = await self._post_chat(request, upstream_body)
            reply = self._parse_reply(data)
            self.capture.record(
                protocol="anthropic",
                prompt_messages=prompt_messages,
                turn=self._turn_record(
                    request,
                    reply,
                    protocol="anthropic",
                    tools=tools,
                    latency_ms=(time.monotonic() - started) * 1000,
                    effective_request=effective_request,
                ),
            )
            response = chat_to_anthropic_response(reply, model=body.get("model") or self.config.model)
            if stream:
                return await _stream_anthropic(request, response)
            return web.json_response(response)

    async def handle_anthropic_count_tokens(self, request: web.Request) -> web.StreamResponse:
        try:
            body = await _read_json(request)
        except WebError as exc:
            return _error_response("anthropic", exc.status, exc.message)
        count = anthropic_count_tokens(body, chars_per_token=self.config.chars_per_token)
        return web.json_response({"input_tokens": count})

    # -------------------- OpenAI Responses ---------------------------------

    async def handle_responses(self, request: web.Request) -> web.StreamResponse:
        return await self._dispatch("responses", request, self._responses_turn)

    async def _responses_turn(self, request: web.Request, body: dict[str, Any]) -> web.StreamResponse:
        stream = body.get("stream") is True
        previous_id = body.get("previous_response_id")
        previous = None
        if previous_id:
            previous = self.responses.get(str(previous_id))
            if previous is None:
                raise WebError(400, f"unknown previous_response_id {previous_id!r}")
        upstream_body = responses_to_chat_request(
            body, previous_messages=previous.wire_messages if previous is not None else None
        )
        prompt_messages = canonicalize_messages(upstream_body.get("messages"))
        tools = canonicalize_tools(upstream_body.get("tools"))

        async with self._turn():
            started = time.monotonic()
            data, effective_request = await self._post_chat(request, upstream_body)
            reply = self._parse_reply(data)
            self.capture.record(
                protocol="responses",
                prompt_messages=prompt_messages,
                turn=self._turn_record(
                    request,
                    reply,
                    protocol="responses",
                    tools=tools,
                    latency_ms=(time.monotonic() - started) * 1000,
                    effective_request=effective_request,
                ),
            )
            response = chat_to_responses_response(reply, body, previous_response_id=previous_id)
            self.responses.put(
                response["id"],
                ResponseState(
                    wire_messages=[*upstream_body["messages"], wire_history_message(reply)],
                    response=response,
                ),
            )
            if stream:
                return await _stream_responses(request, response)
            return web.json_response(response)

    async def handle_get_response(self, request: web.Request) -> web.StreamResponse:
        state = self.responses.get(request.match_info["response_id"])
        if state is None:
            return _error_response("responses", 404, "unknown response id")
        return web.json_response(state.response)

    # -------------------- capture lifecycle --------------------------------

    async def handle_finish(self, request: web.Request) -> web.StreamResponse:
        return await self._dispatch("chat", request, self._finish_turn, allow_empty_body=True)

    async def _finish_turn(self, request: web.Request, body: dict[str, Any]) -> web.StreamResponse:
        metadata = body.get("metadata") if isinstance(body.get("metadata"), dict) else None
        summary = await self.capture.finish(
            reason="explicit",
            metadata=metadata,
            wait_timeout=self.config.finish_wait_sec,
        )
        return web.json_response({**summary, "routing_key": self.routing_key})

    async def handle_status(self, request: web.Request) -> web.StreamResponse:
        return web.json_response({**self.capture.status(), "routing_key": self.routing_key})

    async def handle_health(self, request: web.Request) -> web.StreamResponse:
        return web.json_response({"ok": True})

    async def handle_models(self, request: web.Request) -> web.StreamResponse:
        if self._client is None:  # pragma: no cover - handlers run after startup
            return _error_response("chat", 502, "service is not started")
        headers = self._upstream_headers(request, content_type=False)
        try:
            async with self._client.get(self.config.models_url, headers=headers) as response:
                text = await response.text()
                return web.Response(status=response.status, text=text, content_type="application/json")
        except aiohttp.ClientError as exc:
            return _error_response("chat", 502, f"upstream models request failed: {exc}")


# ---------------------------------------------------------------------------
# app wiring
# ---------------------------------------------------------------------------


def build_app(config: Config) -> web.Application:
    """Build the aiohttp application (startup opens the upstream client)."""
    proxy = RolloutProxy(config)
    app = web.Application(client_max_size=64 * 1024 * 1024)
    app.router.add_post("/v1/chat/completions", proxy.handle_chat_completions)
    app.router.add_post("/v1/messages", proxy.handle_anthropic_messages)
    app.router.add_post("/v1/messages/count_tokens", proxy.handle_anthropic_count_tokens)
    app.router.add_post("/v1/responses", proxy.handle_responses)
    app.router.add_get("/v1/responses/{response_id}", proxy.handle_get_response)
    app.router.add_get("/v1/models", proxy.handle_models)
    app.router.add_post("/rollout/finish", proxy.handle_finish)
    app.router.add_get("/rollout/status", proxy.handle_status)
    app.router.add_get("/healthz", proxy.handle_health)

    async def _on_startup(_: web.Application) -> None:
        await proxy.start()

    async def _on_cleanup(_: web.Application) -> None:
        await proxy.cleanup()

    app.on_startup.append(_on_startup)
    app.on_cleanup.append(_on_cleanup)
    return app


# ---------------------------------------------------------------------------
# request helpers
# ---------------------------------------------------------------------------


async def _read_json(request: web.Request) -> dict[str, Any]:
    try:
        body = await request.json()
    except (ValueError, UnicodeDecodeError) as exc:
        raise WebError(400, f"invalid JSON body: {exc}") from exc
    if not isinstance(body, dict):
        raise WebError(400, "request body must be a JSON object")
    return body


async def _read_optional_json(request: web.Request) -> dict[str, Any]:
    """Like :func:`_read_json`, but an absent/empty body means ``{}``."""
    raw = await request.read()
    if not raw.strip():
        return {}
    try:
        body = json.loads(raw)
    except (ValueError, UnicodeDecodeError) as exc:
        raise WebError(400, f"invalid JSON body: {exc}") from exc
    if not isinstance(body, dict):
        raise WebError(400, "request body must be a JSON object")
    return body


def _wants_stream(request: web.Request, body: dict[str, Any]) -> bool:
    return body.get("stream") is True or "text/event-stream" in request.headers.get("Accept", "")


# Claude Code gateway hint headers (x-claude-code-*): request purpose/lineage.
_CLIENT_HEADER_FIELDS = {
    "x-claude-code-request-class": "request_class",
    "x-claude-code-compaction": "compaction_trigger",
    "x-claude-code-context-compacted": "context_compacted",
    "x-claude-code-agent-id": "agent_id",
    "x-claude-code-agent-type": "agent_type",
    "x-claude-code-parent-agent-id": "parent_agent_id",
    "x-claude-code-prompt-id": "prompt_id",
    "x-claude-code-session-id": "client_session_id",
}


def _client_metadata(request: web.Request) -> dict[str, Any]:
    metadata = {}
    for header, field_name in _CLIENT_HEADER_FIELDS.items():
        value = request.headers.get(header)
        if value:
            metadata[field_name] = value
    return metadata


def _safe_stem(session_id: str) -> str:
    stem = _SAFE_FILENAME.sub("_", session_id).strip("._") or "session"
    return stem[:120]


# ---------------------------------------------------------------------------
# SSE rendering (the whole turn is realized upstream before any bytes are sent)
# ---------------------------------------------------------------------------


def _sse_response() -> web.StreamResponse:
    return web.StreamResponse(
        status=200,
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


async def _write_sse(out: web.StreamResponse, payload: dict[str, Any], *, event: str | None = None) -> None:
    prefix = f"event: {event}\n" if event else ""
    await out.write(f"{prefix}data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode())


async def _stream_chat(request: web.Request, reply: ChatReply) -> web.StreamResponse:
    out = _sse_response()
    await out.prepare(request)
    completion_id = reply.response_id or f"chatcmpl_{secrets.token_hex(12)}"
    created = int(time.time())
    model = reply.model or "rollout-manager"

    async def emit(delta: dict[str, Any], *, finish_reason: str | None = None, usage: dict | None = None) -> None:
        chunk: dict[str, Any] = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
        if usage is not None:
            chunk["usage"] = usage
        await out.write(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode())

    await emit({"role": "assistant"})
    if reply.reasoning:
        await emit({"reasoning_content": reply.reasoning})
    if reply.content:
        await emit({"content": reply.content})
    if reply.wire_tool_calls:
        await emit({"tool_calls": [{**call, "index": index} for index, call in enumerate(reply.wire_tool_calls)]})
    await emit({}, finish_reason=reply.finish_reason, usage=reply.usage or None)
    await out.write(b"data: [DONE]\n\n")
    return out


async def _stream_anthropic(request: web.Request, response: dict[str, Any]) -> web.StreamResponse:
    out = _sse_response()
    await out.prepare(request)
    start_message = dict(response)
    start_message["content"] = []
    start_message["stop_reason"] = None
    start_message["stop_sequence"] = None
    start_message["usage"] = {"input_tokens": response.get("usage", {}).get("input_tokens", 0), "output_tokens": 0}
    await _write_sse(out, {"type": "message_start", "message": start_message}, event="message_start")

    for index, block in enumerate(response.get("content") or []):
        block_type = block.get("type")
        if block_type == "thinking":
            start = {"type": "thinking", "thinking": ""}
            delta = {"type": "thinking_delta", "thinking": block.get("thinking", "")}
        elif block_type == "text":
            start = {"type": "text", "text": ""}
            delta = {"type": "text_delta", "text": block.get("text", "")}
        else:  # tool_use
            start = {"type": "tool_use", "id": block.get("id"), "name": block.get("name"), "input": {}}
            delta = {
                "type": "input_json_delta",
                "partial_json": json.dumps(block.get("input") or {}, ensure_ascii=False),
            }
        await _write_sse(
            out,
            {"type": "content_block_start", "index": index, "content_block": start},
            event="content_block_start",
        )
        await _write_sse(
            out,
            {"type": "content_block_delta", "index": index, "delta": delta},
            event="content_block_delta",
        )
        await _write_sse(out, {"type": "content_block_stop", "index": index}, event="content_block_stop")

    await _write_sse(
        out,
        {
            "type": "message_delta",
            "delta": {"stop_reason": response.get("stop_reason"), "stop_sequence": None},
            "usage": {
                "input_tokens": response.get("usage", {}).get("input_tokens", 0),
                "output_tokens": response.get("usage", {}).get("output_tokens", 0),
            },
        },
        event="message_delta",
    )
    await _write_sse(out, {"type": "message_stop"}, event="message_stop")
    return out


async def _stream_responses(request: web.Request, response: dict[str, Any]) -> web.StreamResponse:
    out = _sse_response()
    await out.prepare(request)
    sequence = 0

    async def emit(event: str, payload: dict[str, Any]) -> None:
        nonlocal sequence
        await _write_sse(out, {"type": event, "sequence_number": sequence, **payload}, event=event)
        sequence += 1

    in_progress = {**response, "status": "in_progress", "output": [], "usage": None}
    await emit("response.created", {"response": in_progress})
    await emit("response.in_progress", {"response": in_progress})

    for output_index, item in enumerate(response.get("output") or []):
        await emit(
            "response.output_item.added",
            {"output_index": output_index, "item": {**item, "status": "in_progress"}},
        )
        if item.get("type") == "message":
            item_id = item.get("id")
            content = (item.get("content") or [{}])[0]
            text = content.get("text", "")
            await emit(
                "response.content_part.added",
                {
                    "item_id": item_id,
                    "output_index": output_index,
                    "content_index": 0,
                    "part": {"type": "output_text", "text": "", "annotations": []},
                },
            )
            await emit(
                "response.output_text.delta",
                {
                    "item_id": item_id,
                    "output_index": output_index,
                    "content_index": 0,
                    "delta": text,
                    "logprobs": [],
                },
            )
            await emit(
                "response.output_text.done",
                {
                    "item_id": item_id,
                    "output_index": output_index,
                    "content_index": 0,
                    "text": text,
                    "logprobs": [],
                },
            )
            await emit(
                "response.content_part.done",
                {"item_id": item_id, "output_index": output_index, "content_index": 0, "part": content},
            )
        await emit("response.output_item.done", {"output_index": output_index, "item": item})

    await emit("response.completed", {"response": response})
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rollout-manager",
        description="Capture one agent session's trajectories by proxying an OpenAI-compatible model backend.",
    )
    parser.add_argument(
        "--upstream",
        required=True,
        help="OpenAI-compatible base URL, e.g. http://127.0.0.1:30000/v1 (SGLang, vLLM, external API).",
    )
    parser.add_argument("--upstream-api-key", default=None, help="Bearer token for the upstream, if it needs one.")
    parser.add_argument("--model", default=None, help="Force this model name on upstream requests.")
    parser.add_argument(
        "--output",
        default=None,
        help="Session export path (default: <output-dir>/<session-id>.json).",
    )
    parser.add_argument("--output-dir", default="./rollout-captures", help="Directory for the default export path.")
    parser.add_argument(
        "--session-id",
        default=None,
        help=(
            "Session id recorded in the export (default: $ROLLOUT_SESSION_ID, else a generated run id). "
            "Backend routing uses a separate per-launch key."
        ),
    )
    parser.add_argument(
        "--metadata",
        default=None,
        help="JSON object stored as session metadata (e.g. rollout/task ids).",
    )
    parser.add_argument(
        "--exclude-request-classes",
        default="",
        help=(
            "Comma-separated x-claude-code-request-class values kept in the export but masked out of "
            "training (e.g. auxiliary for session titles/classifiers/security monitors)."
        ),
    )
    parser.add_argument(
        "--request-params",
        default=None,
        help=(
            "JSON object merged (overriding) into every upstream "
            "Chat-Completions payload after protocol conversion, e.g. "
            '{"temperature": 0.7, "top_p": 0.95, "reasoning_effort": "high"}. '
            "Defaults to $ROLLOUT_REQUEST_PARAMS."
        ),
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind host.")
    parser.add_argument("--port", type=int, default=30001, help="Bind port.")
    parser.add_argument("--request-timeout-sec", type=float, default=900.0, help="Upstream read timeout.")
    parser.add_argument(
        "--finish-wait-sec", type=float, default=30.0, help="How long finish waits for in-flight turns."
    )
    parser.add_argument("--max-turns", type=int, default=None, help="Optional session turn cap.")
    parser.add_argument("--max-connections", type=int, default=512, help="Upstream connection pool size.")
    parser.add_argument(
        "--chars-per-token",
        type=float,
        default=4.0,
        help="Heuristic for /v1/messages/count_tokens (there is no tokenizer on the capture path).",
    )
    parser.add_argument(
        "--response-cache-size", type=int, default=256, help="Responses kept for previous_response_id."
    )
    parser.add_argument(
        "--no-snapshot-every-turn",
        dest="snapshot_every_turn",
        action="store_false",
        help="Only write the export on finish instead of after every turn.",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level (default INFO).")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    metadata: dict[str, Any] = {}
    if args.metadata:
        try:
            metadata = json.loads(args.metadata)
        except json.JSONDecodeError as exc:
            parser.error(f"--metadata is not valid JSON: {exc}")
        if not isinstance(metadata, dict):
            parser.error("--metadata must be a JSON object")

    exclude_request_classes = tuple(
        value.strip().lower() for value in str(args.exclude_request_classes or "").split(",") if value.strip()
    )
    try:
        request_params = parse_request_params(args.request_params or os.environ.get("ROLLOUT_REQUEST_PARAMS"))
    except ValueError as exc:
        parser.error(str(exc))

    session_id = args.session_id or os.environ.get("ROLLOUT_SESSION_ID") or f"run-{secrets.token_hex(4)}"
    output = args.output or str(Path(args.output_dir) / f"{_safe_stem(session_id)}.json")
    config = Config(
        upstream=args.upstream,
        output=output,
        session_id=session_id,
        upstream_api_key=args.upstream_api_key,
        model=args.model,
        host=args.host,
        port=args.port,
        request_timeout_sec=args.request_timeout_sec,
        finish_wait_sec=args.finish_wait_sec,
        max_turns=args.max_turns,
        max_connections=args.max_connections,
        chars_per_token=args.chars_per_token,
        response_cache_size=args.response_cache_size,
        snapshot_every_turn=args.snapshot_every_turn,
        metadata=metadata,
        exclude_request_classes=exclude_request_classes,
        request_params=request_params,
    )
    app = build_app(config)
    logger.info("starting rollout manager for session %s on %s:%d", session_id, config.host, config.port)
    web.run_app(app, host=config.host, port=config.port, access_log=None)


if __name__ == "__main__":
    main()
