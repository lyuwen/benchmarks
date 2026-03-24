# LSP Tool Tests

Comprehensive test suite for LSP tool integration with OpenHands.

## Test Files

### `test_lsp_tool.py` (Main Tool Tests)
- **TestLSPToolDefinition**: Validates OpenAI function calling format
- **TestLSPToolWrapper**: Tests OpenHands SDK wrapper
- **TestLSPResult**: Tests result formatting for LLM consumption
- **TestLSPResponseParser**: Tests LSP response parsing
- **TestAllowedCommands**: Validates command definitions
- **TestOpenAIToolFormat**: Ensures OpenAI API compatibility

### `test_openai_format.py` (Completion Request Tests)
- **TestOpenAICompletionFormat**: Tests tool in actual completion requests
- **TestParameterValidation**: Validates required/optional parameters
- **TestJSONSchemaCompliance**: Ensures JSON Schema compliance

### `test_integration.py` (Integration Tests)
- **TestToolIntegration**: Tests tool registration with OpenHands
- **TestToolExecution**: Tests execution flow (mocked)
- **TestAgentToolFormat**: Validates agent tool format
- **TestCommandValidation**: Tests command validation
- **TestParameterHandling**: Tests parameter conversion

## Running Tests

### Install Dependencies
```bash
pip install pytest pytest-asyncio
```

### Run All Tests
```bash
cd benchmarks/benchmarks/swebench
pytest tests/ -v
```

### Run Specific Test File
```bash
pytest tests/test_lsp_tool.py -v
pytest tests/test_openai_format.py -v
pytest tests/test_integration.py -v
```

### Run Specific Test Class
```bash
pytest tests/test_lsp_tool.py::TestLSPToolDefinition -v
pytest tests/test_openai_format.py::TestOpenAICompletionFormat -v
```

### Run Specific Test
```bash
pytest tests/test_lsp_tool.py::TestLSPToolDefinition::test_tool_definition_structure -v
```

### Run with Coverage
```bash
pytest tests/ --cov=. --cov-report=html
```

## What Tests Cover

### 1. Tool Definition Format ✓
- OpenAI function calling structure
- JSON Schema compliance
- Parameter types and descriptions
- Required vs optional parameters
- Enum constraints on commands

### 2. Tool Registration ✓
- Tool can be added to agent's tools list
- Tool metadata is accessible
- Tool signature matches definition
- Tool can be serialized with other tools

### 3. Completion Request Format ✓
- Tool works in minimal completion request
- Tool works with `tool_choice` parameter
- Tool works alongside other tools (bash, etc.)
- Tool call responses parse correctly
- Tool results can be added to messages

### 4. Parameter Handling ✓
- Args object creation from kwargs
- Optional parameters can be None
- Command enum validation
- Parameter type validation

### 5. Response Formatting ✓
- Success results format correctly
- Error results format correctly
- Empty results handled gracefully
- LSP responses parsed to human-readable text

### 6. Integration ✓
- Wrapper returns callable function
- Tool has correct name and docstring
- Tool execution flow (mocked)
- Error handling

## Test Output Example

```
tests/test_lsp_tool.py::TestLSPToolDefinition::test_tool_definition_structure PASSED
tests/test_lsp_tool.py::TestLSPToolDefinition::test_tool_parameters_schema PASSED
tests/test_lsp_tool.py::TestLSPToolDefinition::test_command_parameter PASSED
tests/test_lsp_tool.py::TestLSPToolDefinition::test_optional_parameters PASSED
tests/test_lsp_tool.py::TestLSPToolDefinition::test_serializable_to_json PASSED
...
======================== 45 passed in 0.23s ========================
```

## Key Validations

### OpenAI Tool Format
```python
{
  "type": "function",
  "function": {
    "name": "lsp_tool",
    "description": "...",
    "parameters": {
      "type": "object",
      "properties": {
        "command": {
          "type": "string",
          "enum": ["get_definition", "get_references", ...]
        },
        "file_path": {"type": "string"},
        "symbol": {"type": "string"},
        "line": {"type": "integer"},
        "query": {"type": "string"}
      },
      "required": ["command"]
    }
  }
}
```

### Completion Request
```python
{
  "model": "gpt-4",
  "messages": [...],
  "tools": [lsp_tool_definition, bash_tool, ...],
  "tool_choice": "auto"
}
```

### Tool Call Response
```python
{
  "tool_calls": [{
    "id": "call_123",
    "type": "function",
    "function": {
      "name": "lsp_tool",
      "arguments": '{"command": "get_definition", "file_path": "/test.py", ...}'
    }
  }]
}
```

## Notes

- Tests do **not** require spinning up the agent
- Tests do **not** require LSP daemon to be running
- Tests use mocking for execution flow validation
- Tests focus on tool definition and format validation
- All tests are fast and can run in CI/CD

## Troubleshooting

### Import Errors
If you get import errors, ensure you're running from the correct directory:
```bash
cd benchmarks/benchmarks/swebench
python -m pytest tests/ -v
```

### Missing pytest
```bash
pip install pytest pytest-asyncio
```

### Path Issues
Tests add parent directory to sys.path automatically:
```python
sys.path.insert(0, str(Path(__file__).parent.parent))
```
