from types import SimpleNamespace

from benchmarks.scaleswe_interactive.user_agent import UserAgent, UserTurn


class _Msg:
    def __init__(self, text=None, tool_calls=None):
        from openhands.sdk.llm import TextContent
        self.content = [TextContent(text=text)] if text is not None else []
        self.tool_calls = tool_calls


class _Resp:
    def __init__(self, message):
        self.message = message


class _ToolCall:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments  # JSON string


class _StubLLM:
    """Returns queued responses in order."""
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def completion(self, messages, tools=None, **kwargs):
        self.calls.append((messages, tools))
        return _Resp(self._responses.pop(0))


class _WS:
    def execute_command(self, cmd):
        return SimpleNamespace(exit_code=0, stdout="file-body", stderr="")


def _agent(llm, mode="plan", user_tools="none"):
    import os
    prompts = os.path.join(os.path.dirname(__file__), "..", "prompts")
    return UserAgent(user_llm=llm, workspace=_WS(), repo_path="/repo",
                     mode=mode, user_tools=user_tools, prompts_dir=prompts,
                     problem_statement="Fix bug in a/b.py", max_user_turns=20)


def test_plain_message_turn():
    llm = _StubLLM([_Msg(text="Please show me your plan.")])
    turn = _agent(llm).take_turn(dialogue=[{"speaker": "coding", "text": "hi"}],
                                 user_turns=0)
    assert isinstance(turn, UserTurn)
    assert turn.finished is False
    assert turn.message == "Please show me your plan."


def test_finish_tool_ends_session():
    llm = _StubLLM([_Msg(tool_calls=[_ToolCall("finish", '{"reason": "done"}')])])
    turn = _agent(llm).take_turn(dialogue=[{"speaker": "coding", "text": "done?"}],
                                 user_turns=3)
    assert turn.finished is True
    assert turn.finish_reason == "done"


def test_readonly_tool_loop_then_message():
    # First response calls read_file; second returns a plain message.
    llm = _StubLLM([
        _Msg(tool_calls=[_ToolCall("read_file", '{"path": "a/b.py"}')]),
        _Msg(text="Looks right, proceed."),
    ])
    turn = _agent(llm, user_tools="readonly").take_turn(
        dialogue=[{"speaker": "coding", "text": "see a/b.py:10"}], user_turns=0)
    assert turn.message == "Looks right, proceed."
    assert any(c["name"] == "read_file" for c in turn.readonly_tool_calls)


def test_file_injection_happens_in_none_mode():
    llm = _StubLLM([_Msg(text="ok")])
    turn = _agent(llm, user_tools="none").take_turn(
        dialogue=[{"speaker": "coding", "text": "the file a/b.py is broken"}],
        user_turns=0)
    assert any(f["path"] == "a/b.py" for f in turn.injected_files)


def test_agent_passes_real_tool_definitions_to_llm():
    """The tools handed to LLM.completion must be real ToolDefinition objects
    whose to_openai_tool() works (the stub LLM would otherwise hide dict tools
    that crash real LLM.completion)."""
    from openhands.sdk.tool import ToolDefinition
    llm = _StubLLM([_Msg(text="ok")])
    _agent(llm, user_tools="readonly").take_turn(
        dialogue=[{"speaker": "coding", "text": "hi"}], user_turns=0)
    _messages, tools = llm.calls[0]
    assert tools, "readonly mode must pass tools"
    for t in tools:
        assert isinstance(t, ToolDefinition)
        assert t.to_openai_tool()["function"]["name"] == t.name


def test_agent_finish_only_mode_passes_real_finish_tool():
    from openhands.sdk.tool import ToolDefinition
    from benchmarks.scaleswe_interactive.user_tools import FINISH_TOOL_NAME
    llm = _StubLLM([_Msg(text="ok")])
    _agent(llm, user_tools="none").take_turn(
        dialogue=[{"speaker": "coding", "text": "hi"}], user_turns=0)
    _messages, tools = llm.calls[0]
    assert tools and all(isinstance(t, ToolDefinition) for t in tools)
    assert FINISH_TOOL_NAME in {t.to_openai_tool()["function"]["name"]
                                for t in tools}


def test_llm_error_returns_error_turn():
    class _BoomLLM:
        def completion(self, messages, tools=None, **kwargs):
            raise RuntimeError("boom")
    turn = _agent(_BoomLLM()).take_turn(dialogue=[{"speaker": "coding", "text": "x"}],
                                        user_turns=0)
    assert turn.error is not None
    assert turn.finished is False
