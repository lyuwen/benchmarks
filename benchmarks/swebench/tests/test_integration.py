"""
Integration-level tests that verify the wiring between run_infer_lsp.py,
the LSP scripts, the prompt template, and the native tool definition —
without starting an agent or a Docker container.
"""
import sys
from pathlib import Path

import pytest

SWEBENCH_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SWEBENCH_DIR))


# ---------------------------------------------------------------------------
# Script presence
# ---------------------------------------------------------------------------
class TestScriptFiles:
    """The LSP scripts must exist next to run_infer_lsp.py."""

    def test_lsp_daemon_exists(self):
        assert (SWEBENCH_DIR / "lsp_daemon.py").is_file()

    def test_lsp_tool_exists(self):
        assert (SWEBENCH_DIR / "lsp_tool.py").is_file()

    def test_run_infer_lsp_exists(self):
        assert (SWEBENCH_DIR / "run_infer_lsp.py").is_file()


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
        assert 'lsp(command="get_definition"' in template_text or \
               "lsp(command=" in template_text

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
    """Verify run_infer_lsp.py correctly references the LSP setup pieces."""

    @pytest.fixture()
    def source(self):
        return (SWEBENCH_DIR / "run_infer_lsp.py").read_text()

    def test_uses_file_upload(self, source):
        """Must use workspace.file_upload (not upload_file)."""
        assert "file_upload" in source
        assert "upload_file" not in source

    def test_uploads_daemon(self, source):
        assert "lsp_daemon.py" in source

    def test_uploads_tool(self, source):
        assert "lsp_tool.py" in source

    def test_starts_daemon(self, source):
        assert "lsp_daemon" in source
        assert "nohup" in source

    def test_passes_lsp_command_env(self, source):
        """Must pass LSP_COMMAND env var so daemon finds pyright-langserver."""
        assert "LSP_COMMAND" in source

    def test_passes_lsp_project_root_env(self, source):
        """Must pass LSP_PROJECT_ROOT so daemon analyzes the right directory."""
        assert "LSP_PROJECT_ROOT" in source

    def test_verifies_pyright_installed(self, source):
        """Must check that pyright-langserver is accessible."""
        assert "pyright-langserver" in source

    def test_polls_for_port_file(self, source):
        """Must poll for daemon port file, not just sleep once."""
        assert "attempt" in source or "retry" in source.lower() or "range" in source

    def test_imports_lsp_tool_definition(self, source):
        """Must import LSPTool for native tool registration."""
        assert "LSPTool" in source

    def test_adds_lsp_tool_to_tools_list(self, source):
        """Must add the LSP tool to the agent's tools list."""
        assert "Tool(name=LSPTool.name)" in source

    def test_default_prompt_is_lsp(self, source):
        assert "default_lsp.j2" in source

    def test_bind_volume_for_lsp(self, source):
        """Must bind-mount the LSP tool module into the container."""
        assert "openhands/tools/lsp" in source


# ---------------------------------------------------------------------------
# lsp_tool.py is a valid CLI
# ---------------------------------------------------------------------------
class TestLSPToolCLI:
    """lsp_tool.py must be runnable as a CLI script."""

    def test_has_main_guard(self):
        src = (SWEBENCH_DIR / "lsp_tool.py").read_text()
        assert 'if __name__ == "__main__"' in src

    def test_has_argparse(self):
        src = (SWEBENCH_DIR / "lsp_tool.py").read_text()
        assert "argparse" in src

    def test_has_all_cli_arguments(self):
        src = (SWEBENCH_DIR / "lsp_tool.py").read_text()
        for arg in ["--file_path", "--symbol", "--line", "--query"]:
            assert arg in src, f"CLI argument {arg} not found"


# ---------------------------------------------------------------------------
# lsp_daemon.py basics
# ---------------------------------------------------------------------------
class TestLSPDaemonBasics:
    """Quick sanity checks on the daemon script."""

    def test_has_main_guard(self):
        src = (SWEBENCH_DIR / "lsp_daemon.py").read_text()
        assert 'if __name__ == "__main__"' in src or "def main" in src

    def test_uses_pyright(self):
        src = (SWEBENCH_DIR / "lsp_daemon.py").read_text()
        assert "pyright" in src.lower()

    def test_writes_port_file(self):
        src = (SWEBENCH_DIR / "lsp_daemon.py").read_text()
        assert "lsp_port_session" in src


# ---------------------------------------------------------------------------
# Native tool definition module
# ---------------------------------------------------------------------------
class TestNativeToolDefinition:
    """Verify the native SDK tool definition exists and has correct structure."""

    def test_definition_module_exists(self):
        vendor_base = SWEBENCH_DIR.parent.parent / "vendor" / "software-agent-sdk"
        defn = vendor_base / "openhands-tools/openhands/tools/lsp/definition.py"
        assert defn.is_file(), "LSP tool definition module missing"

    def test_definition_has_action_class(self):
        vendor_base = SWEBENCH_DIR.parent.parent / "vendor" / "software-agent-sdk"
        src = (vendor_base / "openhands-tools/openhands/tools/lsp/definition.py").read_text()
        assert "class LSPAction" in src

    def test_definition_has_observation_class(self):
        vendor_base = SWEBENCH_DIR.parent.parent / "vendor" / "software-agent-sdk"
        src = (vendor_base / "openhands-tools/openhands/tools/lsp/definition.py").read_text()
        assert "class LSPObservation" in src

    def test_definition_has_tool_class(self):
        vendor_base = SWEBENCH_DIR.parent.parent / "vendor" / "software-agent-sdk"
        src = (vendor_base / "openhands-tools/openhands/tools/lsp/definition.py").read_text()
        assert "class LSPTool" in src

    def test_definition_registers_tool(self):
        vendor_base = SWEBENCH_DIR.parent.parent / "vendor" / "software-agent-sdk"
        src = (vendor_base / "openhands-tools/openhands/tools/lsp/definition.py").read_text()
        assert "register_tool" in src

    def test_definition_has_all_commands(self):
        vendor_base = SWEBENCH_DIR.parent.parent / "vendor" / "software-agent-sdk"
        src = (vendor_base / "openhands-tools/openhands/tools/lsp/definition.py").read_text()
        for cmd in [
            "get_definition", "get_type_definition", "get_references",
            "get_hover", "get_call_hierarchy", "get_document_symbols",
            "get_workspace_symbols", "get_document_highlights",
        ]:
            assert cmd in src, f"Command {cmd} not in definition"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
