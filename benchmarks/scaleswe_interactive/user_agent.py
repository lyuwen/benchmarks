"""Stateless-per-turn user agent that drives the coding agent (read-only)."""
from __future__ import annotations

import json
import os

from jinja2 import Environment, FileSystemLoader
from pydantic import BaseModel, Field

from openhands.sdk.llm import Message, TextContent

from benchmarks.scaleswe_interactive.file_reference import (
    extract_file_paths,
    inject_files,
)
from benchmarks.scaleswe_interactive.user_tools import (
    FINISH_TOOL,
    FINISH_TOOL_NAME,
    USER_READONLY_TOOLS,
    execute_readonly_tool,
)


class UserTurn(BaseModel):
    message: str | None = None
    finished: bool = False
    finish_reason: str | None = None
    injected_files: list[dict] = Field(default_factory=list)
    readonly_tool_calls: list[dict] = Field(default_factory=list)
    error: str | None = None


def _content_text(message) -> str:
    parts = []
    for c in getattr(message, "content", []) or []:
        text = getattr(c, "text", None)
        if text:
            parts.append(text)
    return "\n".join(parts).strip()


class UserAgent:
    def __init__(self, user_llm, workspace, repo_path: str, mode: str,
                 user_tools: str, prompts_dir: str, problem_statement: str,
                 max_user_turns: int, max_tool_iters: int = 5,
                 max_retries: int = 2, inject_max_files: int = 5,
                 inject_max_bytes: int = 20000):
        self.llm = user_llm
        self.workspace = workspace
        self.repo_path = repo_path
        self.mode = mode
        self.user_tools = user_tools
        self.problem_statement = problem_statement
        self.max_user_turns = max_user_turns
        self.max_tool_iters = max_tool_iters
        self.max_retries = max_retries
        self.inject_max_files = inject_max_files
        self.inject_max_bytes = inject_max_bytes
        self._env = Environment(loader=FileSystemLoader(prompts_dir))

    def _system_message(self, user_turns: int) -> Message:
        txt = self._env.get_template("user_system.j2").render(
            problem_statement=self.problem_statement, mode=self.mode,
            user_turns=user_turns, max_user_turns=self.max_user_turns)
        return Message(role="system", content=[TextContent(text=txt)])

    def _dialogue_messages(self, dialogue: list[dict]) -> list[Message]:
        msgs: list[Message] = []
        for turn in dialogue:
            # From the user agent's POV, the coding agent is the "assistant"
            # counterpart and the user's own prior lines are "user".
            role = "user" if turn["speaker"] == "user" else "assistant"
            msgs.append(Message(role=role,
                                content=[TextContent(text=turn["text"])]))
        return msgs

    def _injection_message(self, dialogue: list[dict]) -> tuple[Message | None, list[dict]]:
        text_sources = [self.problem_statement]
        for turn in dialogue:
            if turn["speaker"] == "coding":
                text_sources.append(turn["text"])
        paths: list[str] = []
        for src in text_sources:
            for p in extract_file_paths(src):
                if p not in paths:
                    paths.append(p)
        if not paths:
            return None, []
        injected = inject_files(self.workspace, self.repo_path, paths,
                                max_files=self.inject_max_files,
                                max_bytes=self.inject_max_bytes)
        shown = [f for f in injected if f["skipped"] is None]
        if not shown:
            return None, injected
        blocks = [f"--- {f['path']} ---\n{f['content']}" for f in shown]
        msg = Message(role="user", content=[TextContent(
            text="Referenced file contents (read-only):\n\n"
                 + "\n\n".join(blocks))])
        return msg, injected

    def take_turn(self, dialogue: list[dict], user_turns: int) -> UserTurn:
        try:
            return self._take_turn_inner(dialogue, user_turns)
        except Exception as e:  # noqa: BLE001 - surface as error turn
            return UserTurn(error=f"{type(e).__name__}: {e}")

    def _take_turn_inner(self, dialogue: list[dict], user_turns: int) -> UserTurn:
        messages = [self._system_message(user_turns)]
        messages += self._dialogue_messages(dialogue)
        inject_msg, injected = self._injection_message(dialogue)
        if inject_msg is not None:
            messages.append(inject_msg)

        tools = USER_READONLY_TOOLS if self.user_tools == "readonly" else \
            [FINISH_TOOL]

        tool_traces: list[dict] = []
        for _ in range(self.max_tool_iters + 1):
            resp = self._complete_with_retry(messages, tools)
            msg = resp.message
            tool_calls = getattr(msg, "tool_calls", None) or []

            if not tool_calls:
                return UserTurn(message=_content_text(msg) or "(no message)",
                                injected_files=injected,
                                readonly_tool_calls=tool_traces)

            # Handle first tool call (sequential handling keeps it simple).
            call = tool_calls[0]
            name = call.name
            try:
                args = json.loads(call.arguments) if call.arguments else {}
            except (json.JSONDecodeError, TypeError):
                args = {}

            if name == FINISH_TOOL_NAME:
                return UserTurn(finished=True,
                                finish_reason=args.get("reason", ""),
                                injected_files=injected,
                                readonly_tool_calls=tool_traces)

            if self.user_tools != "readonly":
                # Tool offered but not enabled: coerce to a message.
                return UserTurn(message=_content_text(msg) or "(no message)",
                                injected_files=injected,
                                readonly_tool_calls=tool_traces)

            result = execute_readonly_tool(self.workspace, self.repo_path,
                                           name, args)
            tool_traces.append({"name": name, "arguments": args,
                                "result": result[:2000]})
            # Feed the tool result back and let the model continue.
            messages.append(Message(role="assistant",
                                    content=[TextContent(
                                        text=f"[called {name}({args})]")]))
            messages.append(Message(role="user",
                                    content=[TextContent(
                                        text=f"Tool result:\n{result}")]))

        # Ran out of tool iterations: ask the model for a final message.
        messages.append(Message(role="user", content=[TextContent(
            text="Please respond to the coding agent now with a message.")]))
        resp = self._complete_with_retry(messages, tools=[])
        return UserTurn(message=_content_text(resp.message) or "(no message)",
                        injected_files=injected, readonly_tool_calls=tool_traces)

    def _complete_with_retry(self, messages, tools):
        last_exc = None
        for attempt in range(self.max_retries + 1):
            try:
                return self.llm.completion(messages=messages, tools=tools or None)
            except Exception as e:  # noqa: BLE001
                last_exc = e
        raise last_exc
