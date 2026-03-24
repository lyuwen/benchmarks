"""
Tests for OpenAI completion request format
Validates that LSP tool definition works with actual OpenAI API format
"""
import pytest
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from lsp_tool import lsp_tool as lsp_tool_definition


class TestOpenAICompletionFormat:
    """Test LSP tool in OpenAI completion request format"""

    def test_minimal_completion_request(self):
        """Test tool in minimal completion request"""
        request = {
            "model": "gpt-4",
            "messages": [
                {"role": "user", "content": "Find the definition of MyClass"}
            ],
            "tools": [lsp_tool_definition]
        }

        # Should serialize cleanly
        json_str = json.dumps(request, indent=2)
        parsed = json.loads(json_str)

        assert parsed["tools"][0]["type"] == "function"
        assert parsed["tools"][0]["function"]["name"] == "lsp_tool"

    def test_completion_request_with_tool_choice(self):
        """Test tool with tool_choice parameter"""
        request = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Test"}],
            "tools": [lsp_tool_definition],
            "tool_choice": {"type": "function", "function": {"name": "lsp_tool"}}
        }

        json_str = json.dumps(request)
        parsed = json.loads(json_str)

        assert parsed["tool_choice"]["function"]["name"] == "lsp_tool"

    def test_completion_request_with_multiple_tools(self):
        """Test LSP tool alongside other tools"""
        bash_tool = {
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Execute bash command",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "Command to execute"}
                    },
                    "required": ["command"]
                }
            }
        }

        request = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Test"}],
            "tools": [lsp_tool_definition, bash_tool],
            "tool_choice": "auto"
        }

        json_str = json.dumps(request)
        parsed = json.loads(json_str)

        assert len(parsed["tools"]) == 2
        names = [t["function"]["name"] for t in parsed["tools"]]
        assert "lsp_tool" in names
        assert "bash" in names

    def test_tool_call_response_format(self):
        """Test simulated tool call in response"""
        # Simulate OpenAI response with tool call
        response = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "model": "gpt-4",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_abc123",
                        "type": "function",
                        "function": {
                            "name": "lsp_tool",
                            "arguments": json.dumps({
                                "command": "get_definition",
                                "file_path": "/workspace/test.py",
                                "symbol": "MyClass",
                                "line": 42
                            })
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }]
        }

        # Should parse cleanly
        json_str = json.dumps(response)
        parsed = json.loads(json_str)

        tool_call = parsed["choices"][0]["message"]["tool_calls"][0]
        assert tool_call["function"]["name"] == "lsp_tool"

        args = json.loads(tool_call["function"]["arguments"])
        assert args["command"] == "get_definition"
        assert args["symbol"] == "MyClass"

    def test_tool_result_in_messages(self):
        """Test tool result can be added to messages"""
        messages = [
            {"role": "user", "content": "Find MyClass definition"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_123",
                    "type": "function",
                    "function": {
                        "name": "lsp_tool",
                        "arguments": json.dumps({
                            "command": "get_definition",
                            "file_path": "/test.py",
                            "symbol": "MyClass",
                            "line": 10
                        })
                    }
                }]
            },
            {
                "role": "tool",
                "tool_call_id": "call_123",
                "content": "Found definition at /test.py line 5:\nclass MyClass:\n    pass"
            }
        ]

        # Should serialize cleanly
        json_str = json.dumps(messages)
        parsed = json.loads(json_str)

        assert len(parsed) == 3
        assert parsed[2]["role"] == "tool"
        assert "MyClass" in parsed[2]["content"]


class TestParameterValidation:
    """Test parameter validation in completion requests"""

    def test_required_parameter_command(self):
        """Test command is marked as required"""
        params = lsp_tool_definition["function"]["parameters"]

        assert "required" in params
        assert "command" in params["required"]

    def test_optional_parameters_not_required(self):
        """Test optional parameters are not in required list"""
        params = lsp_tool_definition["function"]["parameters"]
        required = params["required"]

        assert "file_path" not in required
        assert "symbol" not in required
        assert "line" not in required
        assert "query" not in required

    def test_enum_constraint_on_command(self):
        """Test command has enum constraint"""
        props = lsp_tool_definition["function"]["parameters"]["properties"]
        cmd = props["command"]

        assert "enum" in cmd
        assert isinstance(cmd["enum"], list)
        assert len(cmd["enum"]) > 0
        assert "get_definition" in cmd["enum"]


class TestJSONSchemaCompliance:
    """Test JSON Schema compliance"""

    def test_parameters_are_valid_json_schema(self):
        """Test parameters follow JSON Schema spec"""
        params = lsp_tool_definition["function"]["parameters"]

        # Must have type
        assert "type" in params
        assert params["type"] == "object"

        # Must have properties
        assert "properties" in params
        assert isinstance(params["properties"], dict)

    def test_property_definitions_are_valid(self):
        """Test each property has valid schema"""
        props = lsp_tool_definition["function"]["parameters"]["properties"]

        for prop_name, prop_def in props.items():
            # Each property must have type
            assert "type" in prop_def, f"Property {prop_name} missing type"

            # Type must be valid
            valid_types = ["string", "integer", "number", "boolean", "object", "array", "null"]
            assert prop_def["type"] in valid_types, f"Invalid type for {prop_name}"

            # Should have description
            assert "description" in prop_def, f"Property {prop_name} missing description"

    def test_no_additional_properties_restriction(self):
        """Test additionalProperties is not overly restrictive"""
        params = lsp_tool_definition["function"]["parameters"]

        # Should either not have additionalProperties or it should be flexible
        if "additionalProperties" in params:
            # If present, should not be false (too restrictive)
            assert params["additionalProperties"] != False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
