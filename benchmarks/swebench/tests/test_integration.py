"""
Integration-level tests that verify the wiring between run_infer_lsp.py,
the LSP scripts (now in the SDK), the prompt template, and the native tool
definition — without starting an agent or a Docker container.
"""
import sys
from pathlib import Path

import pytest

SWEBENCH_DIR = Path(__file__).parent.parent
SDK_LSP_DIR = SWEBENCH_DIR.parent.parent / "vendor" / "software-agent-sdk" / "openhands-tools" / "openhands" / "tools" / "lsp"

sys.path.insert(0, str(SWEBENCH_DIR))


# ---------------------------------------------------------------------------
# Script presence (scripts now live in SDK)
# ---------------------------------------------------------------------------
class TestScriptFiles:
    """The LSP scripts must exist in the SDK lsp/ directory."""

    def test_lsp_daemon_exists(self):
        assert (SDK_LSP_DIR / "lsp_daemon.py").is_file()

    def test_lsp_tool_exists(self):
        assert (SDK_LSP_DIR / "lsp_tool.py").is_file()

    def test_run_infer_lsp_exists(self):
        assert (SWEBENCH_DIR / "run_infer_lsp.py").is_file()

    def test_workspace_setup_exists(self):
        assert (SDK_LSP_DIR / "workspace_setup.py").is_file()

    def test_prompt_helpers_exists(self):
        assert (SDK_LSP_DIR / "prompt_helpers.py").is_file()


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------
class TestPromptTemplate:
    """The LSP prompt must exist and contain the right placeholders."""

    @pytest.fixture()
    def template_text(self):
        path = SWEBENCH_DIR / "prompts" / "default_lsp.j2"
        assert path.is_file(), "prompts/default_lsp.j2 is missing"
        return path.read_text()

    def test_contains_lsp_section(self, template_text):
        assert "<lsp_tool>" in template_text
        assert "</lsp_tool>" in template_text

    def test_documents_native_calling(self, template_text):
        """Prompt should show native tool-calling syntax, not CLI syntax."""
        assert "lsp(command=" in template_text

    def test_documents_all_commands_via_jinja(self, template_text):
        """Prompt should reference all 12 commands via {{ cmd.xxx }} Jinja vars."""
        for cmd in [
            "get_definition", "get_type_definition", "find_references",
            "hover", "get_implementation", "get_call_hierarchy",
            "prepare_call_hierarchy", "incoming_calls", "outgoing_calls",
            "get_document_symbols", "get_workspace_symbols", "get_document_highlights",
        ]:
            assert f"cmd.{cmd}" in template_text, f"Prompt missing cmd.{cmd} Jinja var"

    def test_does_not_document_cli_syntax(self, template_text):
        """With native tool calling, CLI examples should not be in prompt."""
        assert "python /tmp/lsp_tool.py" not in template_text

    def test_has_jinja_placeholders(self, template_text):
        assert "{{ instance.repo_path }}" in template_text
        assert "{{ instance.problem_statement }}" in template_text
        assert "{{ instance.base_commit }}" in template_text


# ---------------------------------------------------------------------------
# run_infer_lsp.py references
# ---------------------------------------------------------------------------
class TestRunInferLSPReferences:
    """Verify run_infer_lsp.py correctly uses SDK imports."""

    @pytest.fixture()
    def source(self):
        return (SWEBENCH_DIR / "run_infer_lsp.py").read_text()

    def test_imports_sdk_lsp(self, source):
        """Must import from openhands.tools.lsp SDK module."""
        assert "from openhands.tools.lsp import" in source

    def test_imports_setup_lsp_in_workspace(self, source):
        assert "setup_lsp_in_workspace" in source

    def test_imports_start_lsp_daemon(self, source):
        assert "start_lsp_daemon" in source

    def test_imports_add_lsp_args(self, source):
        assert "add_lsp_args" in source

    def test_imports_apply_lsp_naming(self, source):
        assert "apply_lsp_naming" in source

    def test_imports_get_lsp_command_names(self, source):
        assert "get_lsp_command_names" in source

    def test_imports_lsp_tool_definition(self, source):
        """Must import LSPTool for native tool registration."""
        assert "LSPTool" in source

    def test_adds_lsp_tool_to_tools_list(self, source):
        """Must add the LSP tool to the agent's tools list with params."""
        assert "Tool(name=LSPTool.name" in source

    def test_passes_lsp_naming_param(self, source):
        """Must pass lsp_naming via Tool params."""
        assert 'params={"lsp_naming"' in source or "params={\"lsp_naming\"" in source

    def test_default_prompt_is_lsp(self, source):
        assert "default_lsp.j2" in source

    def test_bind_volume_for_lsp(self, source):
        """Must bind-mount the LSP tool module into the container."""
        assert "openhands/tools/lsp" in source

    def test_no_local_lsp_scripts(self, source):
        """run_infer_lsp.py should NOT define setup_lsp_in_workspace locally."""
        assert "LSP_DAEMON_SCRIPT" not in source
        assert "LSP_TOOL_SCRIPT" not in source


# ---------------------------------------------------------------------------
# lsp_tool.py is a valid CLI (now in SDK)
# ---------------------------------------------------------------------------
class TestLSPToolCLI:
    """lsp_tool.py must be runnable as a CLI script."""

    def test_has_main_guard(self):
        src = (SDK_LSP_DIR / "lsp_tool.py").read_text()
        assert 'if __name__ == "__main__"' in src

    def test_has_argparse(self):
        src = (SDK_LSP_DIR / "lsp_tool.py").read_text()
        assert "argparse" in src

    def test_has_all_cli_arguments(self):
        src = (SDK_LSP_DIR / "lsp_tool.py").read_text()
        for arg in ["--file_path", "--symbol", "--line", "--query", "--item"]:
            assert arg in src, f"CLI argument {arg} not found"


# ---------------------------------------------------------------------------
# lsp_daemon.py basics (now in SDK)
# ---------------------------------------------------------------------------
class TestLSPDaemonBasics:
    """Quick sanity checks on the daemon script."""

    def test_has_main_guard(self):
        src = (SDK_LSP_DIR / "lsp_daemon.py").read_text()
        assert 'if __name__ == "__main__"' in src or "def main" in src

    def test_uses_pyright(self):
        src = (SDK_LSP_DIR / "lsp_daemon.py").read_text()
        assert "pyright" in src.lower()

    def test_writes_port_file(self):
        src = (SDK_LSP_DIR / "lsp_daemon.py").read_text()
        assert "lsp_port_session" in src


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
