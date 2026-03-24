"""
Integration tests for LSP tool with OpenHands SDK
Tests tool registration, execution flow, and agent integration
"""
import pytest
import json
from pathlib import Path
import sys
from unittest.mock import Mock, AsyncMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from lsp_tool_wrapper import get_lsp_tool
from lsp_tool import EnhancedLSPTool, LSPResult


class TestToolIntegration:
    """Test LSP tool integration with OpenHands"""

    def test_tool_can_be_added_to_tools_list(self):
        """Verify tool can be added to agent's tools list"""
        lsp_tool = get_lsp_tool()
        tools = [lsp_tool]

        assert len(tools) == 1
        assert callable(tools[0])
        assert tools[0].__name__ == "lsp_tool"

    def test_tool_metadata_accessible(self):
        """Verify tool metadata is accessible for registration"""
        lsp_tool = get_lsp_tool()

        assert hasattr(lsp_tool, "__name__")
        assert hasattr(lsp_tool, "__doc__")
        assert lsp_tool.__name__ == "lsp_tool"
        assert "LSP" in lsp_tool.__doc__


class TestToolExecution:
    """Test tool execution flow (mocked)"""

    @pytest.mark.asyncio
    async def test_tool_execution_with_valid_args(self):
        """Test tool execution with valid arguments"""
        with patch('lsp_tool.EnhancedLSPTool') as MockTool:
            # Mock the tool to return a success result
            mock_instance = MockTool.return_value
            mock_result = LSPResult(result={"summary": "Test successful"})
            mock_instance.run_command = AsyncMock(return_value=mock_result)

            lsp_tool = get_lsp_tool()
            result = await lsp_tool(
                command="get_workspace_symbols",
                query="MyClass"
            )

            assert "Test successful" in result

    @pytest.mark.asyncio
    async def test_tool_execution_with_error(self):
        """Test tool execution with error"""
        with patch('lsp_tool.EnhancedLSPTool') as MockTool:
            mock_instance = MockTool.return_value
            mock_result = LSPResult(error="Connection failed")
            mock_instance.run_command = AsyncMock(return_value=mock_result)

            lsp_tool = get_lsp_tool()
            result = await lsp_tool(
                command="get_definition",
                file_path="/test.py",
                symbol="test",
                line=10
            )

            assert "Connection failed" in result


class TestAgentToolFormat:
    """Test tool format for agent registration"""

    def test_tool_signature_matches_openai_format(self):
        """Verify tool signature can be converted to OpenAI format"""
        import inspect
        from lsp_tool import lsp_tool as tool_def

        lsp_tool = get_lsp_tool()
        sig = inspect.signature(lsp_tool)

        # Tool definition should have matching parameters
        tool_params = tool_def["function"]["parameters"]["properties"]
        sig_params = sig.parameters

        for param_name in tool_params.keys():
            assert param_name in sig_params, f"Parameter {param_name} not in signature"

    def test_tool_can_be_serialized_with_other_tools(self):
        """Test tool can be combined with other tools in request"""
        from lsp_tool import lsp_tool as lsp_def

        # Simulate multiple tools in completion request
        tools = [
            lsp_def,
            {
                "type": "function",
                "function": {
                    "name": "bash",
                    "description": "Execute bash command",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {"type": "string"}
                        },
                        "required": ["command"]
                    }
                }
            }
        ]

        # Should serialize without errors
        json_str = json.dumps({"tools": tools})
        parsed = json.loads(json_str)

        assert len(parsed["tools"]) == 2
        tool_names = [t["function"]["name"] for t in parsed["tools"]]
        assert "lsp_tool" in tool_names
        assert "bash" in tool_names


class TestCommandValidation:
    """Test command parameter validation"""

    def test_valid_commands(self):
        """Test all valid commands are accepted"""
        from lsp_tool import ALLOWED_LSP_COMMANDS

        valid_commands = [
            "get_definition",
            "get_references",
            "get_hover",
            "get_workspace_symbols",
        ]

        for cmd in valid_commands:
            assert cmd in ALLOWED_LSP_COMMANDS

    def test_tool_definition_enum_matches_allowed(self):
        """Verify tool definition enum matches allowed commands"""
        from lsp_tool import lsp_tool as tool_def, ALLOWED_LSP_COMMANDS

        enum_commands = tool_def["function"]["parameters"]["properties"]["command"]["enum"]

        assert set(enum_commands) == set(ALLOWED_LSP_COMMANDS)


class TestParameterHandling:
    """Test parameter handling and conversion"""

    def test_args_object_creation(self):
        """Test args object is created correctly from kwargs"""
        # Simulate what wrapper does
        kwargs = {
            "command": "get_definition",
            "file_path": "/test.py",
            "symbol": "MyClass",
            "line": 42,
            "query": None
        }

        args = type('Args', (), kwargs)()

        assert args.command == "get_definition"
        assert args.file_path == "/test.py"
        assert args.symbol == "MyClass"
        assert args.line == 42
        assert args.query is None

    def test_optional_parameters_can_be_none(self):
        """Test optional parameters can be None"""
        kwargs = {
            "command": "get_workspace_symbols",
            "file_path": None,
            "symbol": None,
            "line": None,
            "query": "MyClass"
        }

        args = type('Args', (), kwargs)()

        assert args.command == "get_workspace_symbols"
        assert args.query == "MyClass"
        assert args.file_path is None


class TestToolDescription:
    """Test tool description for LLM understanding"""

    def test_description_mentions_key_capabilities(self):
        """Verify description mentions key LSP capabilities"""
        from lsp_tool import lsp_tool as tool_def

        desc = tool_def["function"]["description"]

        # Should mention key capabilities
        assert "definition" in desc.lower()
        assert "reference" in desc.lower() or "usage" in desc.lower()
        assert "symbol" in desc.lower()

    def test_parameter_descriptions_are_clear(self):
        """Verify parameter descriptions are clear"""
        from lsp_tool import lsp_tool as tool_def

        props = tool_def["function"]["parameters"]["properties"]

        # Each parameter should have a description
        for param_name, param_def in props.items():
            assert "description" in param_def, f"No description for {param_name}"
            assert len(param_def["description"]) > 10, f"Description too short for {param_name}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
