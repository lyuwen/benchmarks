# LSP Tool Integration for SWE-bench

This directory contains the LSP (Language Server Protocol) tool integration for OpenHands SWE-bench inference, providing semantic code intelligence for improved bug localization and understanding.

## Files

### Core LSP Components

- **`lsp_daemon.py`** (1136 lines) - Background daemon managing Pyright LSP server
  - Runs as persistent process in Docker workspace
  - Handles LSP protocol communication via stdio
  - Provides TCP socket API for tool requests
  - Manages document synchronization and caching

- **`lsp_tool.py`** (1044 lines) - LSP client tool implementation
  - Connects to daemon via TCP socket
  - Provides 9 semantic code analysis commands
  - Formats responses for LLM consumption
  - Handles symbol-to-position conversion

- **`lsp_tool_wrapper.py`** - OpenHands SDK integration wrapper
  - Adapts LSP tool for OpenHands tool format
  - Provides async function-based tool interface
  - Handles tool registration and execution

### Inference Scripts

- **`run_infer_lsp.py`** - LSP-enabled SWE-bench inference
  - Based on `run_infer.py` with LSP integration
  - Automatically starts LSP daemon in workspace
  - Adds LSP tool to agent's tool list
  - Enables semantic code navigation during inference

## LSP Tool Capabilities

The LSP tool provides semantic code understanding via Pyright:

1. **`get_definition`** - Find symbol definition with full source code
2. **`get_type_definition`** - Find type definition with source code
3. **`get_references`** - Find all symbol usages across project
4. **`get_hover`** - Get docstring, type info, and signature
5. **`get_call_hierarchy`** - Complete incoming/outgoing call analysis
6. **`get_document_symbols`** - List all symbols in file (outline)
7. **`get_workspace_symbols`** - Search symbols across workspace
8. **`get_document_highlights`** - Highlight symbol usages in file

## Usage

### Running LSP-enabled Inference

```bash
python benchmarks/swebench/run_infer_lsp.py \
  --dataset princeton-nlp/SWE-bench_Lite \
  --split test \
  --llm-config .llm_config/gpt-4o.json \
  --output-dir outputs/swebench_lsp \
  --num-workers 4
```

### How It Works

1. **Workspace Setup**: When workspace is created, LSP daemon is started:
   ```python
   workspace.upload_file("lsp_daemon.py", "/tmp/lsp_daemon.py")
   workspace.execute_command("nohup python /tmp/lsp_daemon.py &")
   ```

2. **Tool Registration**: LSP tool is added to agent's tools:
   ```python
   from lsp_tool_wrapper import get_lsp_tool
   lsp_tool = get_lsp_tool()
   tools.append(lsp_tool)
   ```

3. **Agent Usage**: Agent can now use LSP for code intelligence:
   ```
   lsp_tool(command="get_workspace_symbols", query="MyClass")
   lsp_tool(command="get_definition", file_path="/workspace/repo/file.py",
            symbol="my_function", line=42)
   ```

## Architecture

```
Agent → lsp_tool_wrapper → lsp_tool.py → TCP Socket → lsp_daemon.py → Pyright
                                                                          ↓
Agent ← Formatted Results ← Response Parser ← LSP Response ← JSON-RPC ←
```

## Configuration

### Environment Variables

- `LSP_PORT_FILE` - Port file location (default: `/var/tmp/lsp_port_session_abc.pid`)
- `LSP_COMMAND` - LSP server command (default: `pyright-langserver --stdio`)

### Dependencies

The LSP tool requires:
- `pyright` - Language server
- `orjson` - Fast JSON serialization

These are automatically installed during workspace setup.

## Benefits for SWE-bench

1. **Semantic Understanding** - Goes beyond text search to understand code structure
2. **Cross-file Analysis** - Find references and definitions across entire project
3. **Type Information** - Get type hints and signatures for better understanding
4. **Call Hierarchy** - Understand function call relationships
5. **Fast Navigation** - Quickly locate relevant code for bug fixing

## Differences from run_infer.py

The LSP-enabled version adds:

1. LSP daemon startup in `prepare_workspace()`
2. LSP tool registration in `evaluate_instance()`
3. Import of `lsp_tool_wrapper` module

All other functionality remains identical to the base `run_infer.py`.

## Troubleshooting

### Daemon Not Starting

Check daemon logs:
```bash
workspace.execute_command("cat /tmp/lsp_daemon.log")
```

### Connection Issues

Verify daemon is running:
```bash
workspace.execute_command("ps aux | grep lsp_daemon")
workspace.execute_command("cat /var/tmp/lsp_port_session_abc.pid")
```

### LSP Errors

Check Pyright installation:
```bash
workspace.execute_command("which pyright-langserver")
workspace.execute_command("pyright --version")
```

## Performance

- **Daemon Persistence**: Single daemon serves all requests (no startup overhead)
- **Async I/O**: Non-blocking communication
- **Caching**: LSP server caches parsed ASTs
- **Incremental Updates**: Only changed portions sent to LSP

## Future Enhancements

- Support for additional language servers (TypeScript, Rust, Go)
- Enhanced error handling and recovery
- Metrics collection for LSP usage
- Integration with agent prompts for LSP guidance
