# LSP Tool Integration Summary

## Completed Tasks

### 1. Created New Branch
- Branch: `lsp-tool-integration`
- Base: `dev` branch of benchmarks repository

### 2. Ported LSP Tool Components

#### Core Files Copied and Adapted:
1. **lsp_daemon.py** (1136 lines)
   - Source: `R2E-Gym/src/r2egym/agenthub/tools/lsp_daemon.py`
   - Destination: `benchmarks/benchmarks/swebench/lsp_daemon.py`
   - Changes: Minimal - kept original implementation

2. **lsp_tool.py** (1044 lines)
   - Source: `R2E-Gym/src/r2egym/agenthub/tools/lsp_tool.py`
   - Destination: `benchmarks/benchmarks/swebench/lsp_tool.py`
   - Changes: Updated shebang from `/root/.venv/bin/python` to `/usr/bin/env python3`

#### New Integration Files:
3. **lsp_tool_wrapper.py** (60 lines)
   - Purpose: Adapts LSP tool for OpenHands SDK
   - Provides `get_lsp_tool()` function returning async tool
   - Wraps `EnhancedLSPTool` for agent use

4. **run_infer_lsp.py** (483 lines)
   - Based on: `run_infer.py`
   - Key additions:
     - LSP daemon startup in `prepare_workspace()`
     - LSP tool registration in `evaluate_instance()`
     - Automatic installation of dependencies (pyright, orjson)

### 3. Integration Points

#### In `prepare_workspace()`:
```python
# Start LSP daemon
daemon_script = Path(__file__).parent / "lsp_daemon.py"
workspace.upload_file(str(daemon_script), "/tmp/lsp_daemon.py")
workspace.execute_command("chmod +x /tmp/lsp_daemon.py")
workspace.execute_command("pip install orjson pyright 2>/dev/null || true")
workspace.execute_command("nohup python /tmp/lsp_daemon.py > /tmp/lsp_daemon.log 2>&1 &")
```

#### In `evaluate_instance()`:
```python
# Add LSP tool
from lsp_tool_wrapper import get_lsp_tool
lsp_tool = get_lsp_tool()
tools.append(lsp_tool)
```

### 4. Documentation Created

- **LSP_README.md** - Complete usage guide for LSP integration
- **LSP_TOOL_DOCUMENTATION.md** (in main repo) - Comprehensive technical documentation

## File Structure

```
benchmarks/benchmarks/swebench/
├── lsp_daemon.py          # LSP daemon server (1136 lines)
├── lsp_tool.py            # LSP client tool (1044 lines)
├── lsp_tool_wrapper.py    # OpenHands integration (60 lines)
├── run_infer_lsp.py       # LSP-enabled inference (483 lines)
├── LSP_README.md          # Usage documentation
└── run_infer.py           # Original (unchanged)
```

## Commits

1. **532bd0a** - "Add LSP tool integration for SWE-bench inference"
   - Added all 4 core files
   - 2726 lines added

2. **d63a934** - "Add LSP integration documentation"
   - Added LSP_README.md
   - 160 lines added

## Usage

### Running LSP-enabled Inference:
```bash
python benchmarks/swebench/run_infer_lsp.py \
  --dataset princeton-nlp/SWE-bench_Lite \
  --split test \
  --llm-config .llm_config/gpt-4o.json \
  --output-dir outputs/swebench_lsp \
  --num-workers 4
```

### LSP Tool Commands Available to Agent:
- `get_definition` - Find symbol definition with source code
- `get_type_definition` - Find type definition
- `get_references` - Find all usages across project
- `get_hover` - Get docstring and type info
- `get_call_hierarchy` - Incoming/outgoing calls
- `get_document_symbols` - File outline
- `get_workspace_symbols` - Search symbols
- `get_document_highlights` - Highlight usages

## Key Features

1. **Automatic Setup** - LSP daemon starts automatically in workspace
2. **Semantic Intelligence** - Goes beyond grep/find with AST-based analysis
3. **Cross-file Analysis** - Understands imports and references
4. **Type Information** - Provides type hints and signatures
5. **Minimal Changes** - Only 2 methods modified from original run_infer.py

## Architecture

```
Agent (OpenHands)
  ↓
lsp_tool_wrapper.get_lsp_tool()
  ↓
lsp_tool.EnhancedLSPTool
  ↓
TCP Socket (localhost)
  ↓
lsp_daemon.LSPDaemonServer
  ↓
Pyright LSP Server (stdio)
```

## Dependencies

- **pyright** - Language server for Python
- **orjson** - Fast JSON serialization
- Both installed automatically during workspace setup

## Testing Recommendations

1. Test with single instance first:
   ```bash
   python benchmarks/swebench/run_infer_lsp.py \
     --dataset princeton-nlp/SWE-bench_Lite \
     --split test \
     --n-limit 1 \
     --llm-config .llm_config/gpt-4o.json
   ```

2. Check daemon logs:
   ```python
   workspace.execute_command("cat /tmp/lsp_daemon.log")
   ```

3. Verify LSP tool in agent tools list

## Next Steps

1. Test inference with LSP tool on sample instances
2. Compare results with baseline run_infer.py
3. Analyze LSP tool usage patterns in agent trajectories
4. Optimize prompts to encourage LSP tool usage for localization
5. Consider adding LSP-specific guidance in system prompts

## Notes

- All code resides in `benchmarks/benchmarks/swebench/` directory
- No changes to other parts of the codebase
- Original `run_infer.py` remains unchanged
- LSP integration is opt-in via `run_infer_lsp.py`
