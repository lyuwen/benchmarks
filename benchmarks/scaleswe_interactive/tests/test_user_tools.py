from benchmarks.scaleswe_interactive.user_tools import (
    USER_READONLY_TOOLS,
    FINISH_TOOL_NAME,
    execute_readonly_tool,
)


class _WS:
    def __init__(self):
        self.ran = []

    def execute_command(self, cmd):
        self.ran.append(cmd)

        class R:
            exit_code, stdout, stderr = 0, "ok-output", ""
        return R()


def test_finish_tool_present():
    names = {t.name for t in USER_READONLY_TOOLS}
    assert FINISH_TOOL_NAME in names
    assert {"read_file", "grep", "glob", "run_readonly_bash"} <= names


def test_tools_serialize_via_to_openai_tool():
    """Exercises the REAL path LLM.completion uses (a stub LLM hides it)."""
    for tool in USER_READONLY_TOOLS:
        oai = tool.to_openai_tool()
        assert oai["type"] == "function"
        assert oai["function"]["name"] == tool.name
        assert "parameters" in oai["function"]
    names = {t.to_openai_tool()["function"]["name"] for t in USER_READONLY_TOOLS}
    assert {"read_file", "grep", "glob", "run_readonly_bash",
            FINISH_TOOL_NAME} == names


def test_run_readonly_bash_rejects_writer_without_executing():
    ws = _WS()
    out = execute_readonly_tool(ws, "/repo", "run_readonly_bash",
                                {"command": "rm -rf /"})
    assert "read-only" in out.lower()
    assert ws.ran == []  # never executed


def test_run_readonly_bash_allows_reader():
    ws = _WS()
    out = execute_readonly_tool(ws, "/repo", "run_readonly_bash",
                                {"command": "grep -n x a.py"})
    assert out == "ok-output"
    assert ws.ran and "grep -n x a.py" in ws.ran[0]


def test_read_file_uses_cat_within_repo():
    ws = _WS()
    execute_readonly_tool(ws, "/repo", "read_file", {"path": "pkg/m.py"})
    assert ws.ran and "/repo/pkg/m.py" in ws.ran[0] and ws.ran[0].startswith("cat")


def test_read_file_missing_path_returns_error():
    ws = _WS()
    out = execute_readonly_tool(ws, "/repo", "read_file", {})
    assert "missing required argument" in out
    assert "path" in out
    assert ws.ran == []


def test_grep_missing_pattern_returns_error():
    ws = _WS()
    out = execute_readonly_tool(ws, "/repo", "grep", {})
    assert "missing required argument" in out
    assert "pattern" in out
    assert ws.ran == []


def test_glob_missing_pattern_returns_error():
    ws = _WS()
    out = execute_readonly_tool(ws, "/repo", "glob", {})
    assert "missing required argument" in out
    assert "pattern" in out
    assert ws.ran == []


def test_run_readonly_bash_missing_command_returns_error():
    ws = _WS()
    out = execute_readonly_tool(ws, "/repo", "run_readonly_bash", {})
    assert "missing required argument" in out
    assert "command" in out
    assert ws.ran == []
