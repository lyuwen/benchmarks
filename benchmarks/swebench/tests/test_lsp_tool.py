"""
Test suite for LSP tool integration
Tests tool definitions, wrapper functionality, and OpenHands integration
"""
import pytest
import json
import asyncio
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lsp_tool_wrapper import get_lsp_tool
from lsp_tool import (
    LSPResult,
    LSPResponseParser,
    ALLOWED_LSP_COMMANDS,
    lsp_tool as lsp_tool_definition,
)


class TestLSPToolDefinition:
    """Test the tool definition format for OpenAI completion requests"""

    def test_tool_definition_structure(self):
        """Verify lsp_tool definition has correct OpenAI format"""
        assert lsp_tool_definition["type"] == "function"
        assert "function" in lsp_tool_definition
        func = lsp_tool_definition["function"]

        assert func["name"] == "lsp_tool"
        assert "description" in func
        assert "parameters" in func

    def test_tool_parameters_schema(self):
        """Verify parameters follow JSON Schema format"""
        params = lsp_tool_definition["function"]["parameters"]

        assert params["type"] == "object"
        assert "properties" in params
        assert "required" in params
        assert "command" in params["required"]

    def test_command_parameter(self):
        """Verify command parameter has enum of allowed commands"""
        props = lsp_tool_definition["function"]["parameters"]["properties"]
        cmd = props["command"]

        assert cmd["type"] == "string"
        assert "enum" in cmd
        assert set(cmd["enum"]) == set(ALLOWED_LSP_COMMANDS)

    def test_optional_parameters(self):
        """Verify optional parameters are properly defined"""
        props = lsp_tool_definition["function"]["parameters"]["properties"]

        # file_path
        assert "file_path" in props
        assert props["file_path"]["type"] == "string"

        # symbol
        assert "symbol" in props
        assert props["symbol"]["type"] == "string"

        # line
        assert "line" in props
        assert props["line"]["type"] == "integer"

        # query
        assert "query" in props
        assert props["query"]["type"] == "string"

    def test_serializable_to_json(self):
        """Verify tool definition can be serialized to JSON"""
        json_str = json.dumps(lsp_tool_definition)
        parsed = json.loads(json_str)

        assert parsed["function"]["name"] == "lsp_tool"
        assert len(parsed["function"]["parameters"]["properties"]) >= 5


class TestLSPToolWrapper:
    """Test the OpenHands SDK wrapper"""

    def test_get_lsp_tool_returns_callable(self):
        """Verify get_lsp_tool returns a callable function"""
        tool = get_lsp_tool()
        assert callable(tool)

    def test_tool_has_correct_name(self):
        """Verify tool function has correct name"""
        tool = get_lsp_tool()
        assert tool.__name__ == "lsp_tool"

    def test_tool_has_docstring(self):
        """Verify tool has descriptive docstring"""
        tool = get_lsp_tool()
        assert tool.__doc__ is not None
        assert "LSP" in tool.__doc__
        assert "get_definition" in tool.__doc__

    def test_tool_signature(self):
        """Verify tool accepts correct parameters"""
        tool = get_lsp_tool()
        import inspect
        sig = inspect.signature(tool)

        params = list(sig.parameters.keys())
        assert "command" in params
        assert "file_path" in params
        assert "symbol" in params
        assert "line" in params
        assert "query" in params


class TestLSPResult:
    """Test LSP result formatting"""

    def test_success_result(self):
        """Test successful result formatting"""
        result = LSPResult(result={"summary": "Test result"})

        assert result.status_code == "success"
        assert result.error is None
        text = result.to_text()
        assert "Test result" in text

    def test_error_result(self):
        """Test error result formatting"""
        result = LSPResult(error="Connection failed")

        assert result.status_code == "error"
        assert result.error == "Connection failed"
        text = result.to_text()
        assert "Connection failed" in text

    def test_to_dict(self):
        """Test dictionary conversion"""
        result = LSPResult(result={"summary": "Test"})
        d = result.to_dict()

        assert d["status_code"] == "success"
        assert d["result"]["summary"] == "Test"

    def test_empty_result(self):
        """Test empty result handling"""
        result = LSPResult(result=None)
        text = result.to_text()

        assert "No output content" in text or "Succeeded" in text


class TestLSPResponseParser:
    """Test LSP response parsing"""

    def test_parse_references(self):
        """Test reference parsing"""
        data = [
            {"uri": "file:///test.py", "range": {"start": {"line": 10, "character": 5}}},
            {"uri": "file:///test.py", "range": {"start": {"line": 20, "character": 10}}},
        ]

        result = LSPResponseParser.parse_references(data)
        assert "summary" in result
        assert "2 reference" in result["summary"]

    def test_parse_empty_references(self):
        """Test empty reference list"""
        result = LSPResponseParser.parse_references([])
        assert "No references" in result["summary"]

    def test_parse_hover(self):
        """Test hover info parsing"""
        data = {
            "contents": {
                "kind": "markdown",
                "value": "```python\ndef my_function() -> None\n```\nA test function"
            }
        }

        result = LSPResponseParser.parse_hover(data)
        assert "summary" in result
        assert "my_function" in result["summary"]

    def test_parse_document_symbols(self):
        """Test document symbols parsing"""
        data = [
            {
                "name": "MyClass",
                "kind": 5,  # Class
                "range": {"start": {"line": 0, "character": 0}},
                "children": [
                    {
                        "name": "my_method",
                        "kind": 6,  # Method
                        "range": {"start": {"line": 5, "character": 4}}
                    }
                ]
            }
        ]

        result = LSPResponseParser.parse_document_symbols(data)
        assert "summary" in result
        assert "MyClass" in result["summary"]
        assert "my_method" in result["summary"]


class TestAllowedCommands:
    """Test command validation"""

    def test_all_commands_defined(self):
        """Verify all expected commands are defined"""
        expected = {
            "get_definition",
            "get_type_definition",
            "get_references",
            "get_hover",
            "get_document_highlights",
            "get_document_symbols",
            "get_workspace_symbols",
            "get_call_hierarchy",
        }

        assert expected.issubset(set(ALLOWED_LSP_COMMANDS))

    def test_commands_match_tool_definition(self):
        """Verify commands in tool definition match ALLOWED_LSP_COMMANDS"""
        tool_commands = set(
            lsp_tool_definition["function"]["parameters"]["properties"]["command"]["enum"]
        )

        assert tool_commands == set(ALLOWED_LSP_COMMANDS)


class TestOpenAIToolFormat:
    """Test that tool definition matches OpenAI's expected format"""

    def test_openai_function_calling_format(self):
        """Verify format matches OpenAI function calling spec"""
        tool = lsp_tool_definition

        # Top level structure
        assert tool["type"] == "function"
        assert "function" in tool

        # Function structure
        func = tool["function"]
        assert "name" in func
        assert "description" in func
        assert "parameters" in func

        # Parameters structure (JSON Schema)
        params = func["parameters"]
        assert params["type"] == "object"
        assert "properties" in params
        assert "required" in params

        # All required fields are in properties
        for req in params["required"]:
            assert req in params["properties"]

    def test_can_be_used_in_completion_request(self):
        """Verify tool can be serialized for OpenAI API request"""
        # Simulate OpenAI completion request format
        request = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Test"}],
            "tools": [lsp_tool_definition],
            "tool_choice": "auto"
        }

        # Should serialize without errors
        json_str = json.dumps(request)
        parsed = json.loads(json_str)

        assert len(parsed["tools"]) == 1
        assert parsed["tools"][0]["function"]["name"] == "lsp_tool"

    def test_parameter_types_are_valid(self):
        """Verify all parameter types are valid JSON Schema types"""
        valid_types = {"string", "integer", "number", "boolean", "object", "array"}
        props = lsp_tool_definition["function"]["parameters"]["properties"]

        for param_name, param_def in props.items():
            assert "type" in param_def, f"Parameter {param_name} missing type"
            assert param_def["type"] in valid_types, f"Invalid type for {param_name}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
