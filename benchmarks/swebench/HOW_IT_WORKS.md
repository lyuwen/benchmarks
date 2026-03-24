# How the LSP Tool Works - Technical Deep Dive

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         Agent (OpenHands)                        │
│  Receives: "Find the definition of MyClass in file.py line 42"  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              lsp_tool_wrapper.get_lsp_tool()                     │
│  - Async function wrapper for OpenHands SDK                      │
│  - Converts kwargs to args object                                │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│           lsp_tool.EnhancedLSPTool.run_command()                 │
│  - Converts (file_path, line, symbol) → (file_path, line, char) │
│  - Reads file to find character position of symbol              │
│  - Enhances responses with source code extraction                │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│            lsp_tool.LSPToolClient.send_request()                 │
│  - TCP socket client (localhost:PORT)                            │
│  - Protocol: [4-byte length][JSON payload]                       │
│  - Reads port from /var/tmp/lsp_port_session_abc.pid            │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼ TCP Socket
┌─────────────────────────────────────────────────────────────────┐
│          lsp_daemon.LSPDaemonServer (Background Process)         │
│  - Listens on localhost with dynamic port                        │
│  - Handles concurrent client connections                         │
│  - Routes requests to LSPClient                                  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              lsp_daemon.LSPClient (Async Manager)                │
│  - Manages Pyright subprocess (stdin/stdout)                     │
│  - Handles document synchronization                              │
│  - Tracks file versions and diagnostics                          │
│  - Implements debouncing for file updates                        │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼ JSON-RPC over stdio
┌─────────────────────────────────────────────────────────────────┐
│                  Pyright LSP Server Process                      │
│  - Language server for Python                                    │
│  - Parses AST, builds symbol table                               │
│  - Understands types, imports, references                        │
│  - Caches parsed files for performance                           │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼ Response flows back
┌─────────────────────────────────────────────────────────────────┐
│                    LSPResponseParser                             │
│  - Converts LSP JSON to human-readable text                      │
│  - Groups references by file                                     │
│  - Formats locations with line numbers                           │
│  - Extracts relevant information for LLM                         │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Agent receives result                       │
│  "Found definition at /workspace/file.py line 10:                │
│   class MyClass:                                                 │
│       def __init__(self): ..."                                   │
└─────────────────────────────────────────────────────────────────┘
```

## Step-by-Step Execution Flow

### 1. Agent Calls Tool

Agent decides to use LSP tool:
```python
lsp_tool(
    command="get_definition",
    file_path="/workspace/repo/module.py",
    symbol="MyClass",
    line=42
)
```

### 2. Wrapper Converts to Args

`lsp_tool_wrapper.py` converts kwargs to args object:
```python
args = type('Args', (), {
    'command': 'get_definition',
    'file_path': '/workspace/repo/module.py',
    'symbol': 'MyClass',
    'line': 42,
    'query': None
})()
```

### 3. Symbol-to-Position Conversion

`EnhancedLSPTool` finds character position:
```python
# Read line 42 from file
line_text = "    my_obj = MyClass(arg1, arg2)"
#                     ^
#                     character position = 13

# Convert to 0-indexed
args.line = 41  # 42 - 1
args.character = 13  # position of 'M' in 'MyClass'
```

### 4. Socket Request

`LSPToolClient` sends request via TCP:
```python
# Request format
request = {
    "command": "get_definition",
    "file_path": "/workspace/repo/module.py",
    "line": 41,
    "character": 13
}

# Socket protocol
message = struct.pack('!I', len(json_bytes)) + json_bytes
# [4 bytes: 156][{"command": "get_definition", ...}]
```

### 5. Daemon Receives Request

`LSPDaemonServer` handles connection:
```python
# Read length prefix
length = struct.unpack('!I', await reader.read(4))[0]

# Read JSON payload
json_data = await reader.read(length)
request = json.loads(json_data)

# Route to LSPClient
response = await lsp_client.handle_request(request)
```

### 6. LSP Protocol Communication

`LSPClient` sends JSON-RPC to Pyright:
```python
# LSP request format
lsp_request = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "textDocument/definition",
    "params": {
        "textDocument": {
            "uri": "file:///workspace/repo/module.py"
        },
        "position": {
            "line": 41,
            "character": 13
        }
    }
}

# Send via stdin
message = f"Content-Length: {len(json_bytes)}\r\n\r\n{json_bytes}"
process.stdin.write(message)
```

### 7. Pyright Processes Request

Pyright LSP server:
1. Parses AST of module.py
2. Finds symbol at position (41, 13)
3. Resolves symbol to definition location
4. Returns location + optional source range

```python
# Pyright response
{
    "jsonrpc": "2.0",
    "id": 1,
    "result": [{
        "uri": "file:///workspace/repo/classes.py",
        "range": {
            "start": {"line": 10, "character": 0},
            "end": {"line": 25, "character": 0}
        }
    }]
}
```

### 8. Enhanced Response Processing

`EnhancedLSPTool` adds source code:
```python
# Get definition location
location = result[0]
file_path = "/workspace/repo/classes.py"
start_line = 10
end_line = 25

# Read source code
with open(file_path) as f:
    lines = f.readlines()[start_line:end_line]
    source_code = ''.join(lines)

# Enhanced result
enhanced_result = {
    "summary": "Found definition at /workspace/repo/classes.py line 11",
    "locations": [location],
    "source_code": source_code
}
```

### 9. Response Formatting

`LSPResponseParser` formats for LLM:
```python
output = """
Found definition at /workspace/repo/classes.py line 11

--- SOURCE CODE START ---
class MyClass:
    def __init__(self, arg1, arg2):
        self.arg1 = arg1
        self.arg2 = arg2

    def process(self):
        return self.arg1 + self.arg2
--- SOURCE CODE END ---
"""
```

### 10. Agent Receives Result

Agent gets human-readable text:
```
[status_code]:
 success
[Result]:
Found definition at /workspace/repo/classes.py line 11

--- SOURCE CODE START ---
class MyClass:
    def __init__(self, arg1, arg2):
        self.arg1 = arg1
        self.arg2 = arg2

    def process(self):
        return self.arg1 + self.arg2
--- SOURCE CODE END ---
```

## Key Design Decisions

### 1. Why TCP Socket Instead of Stdio?

**Problem**: Agent already uses stdio for communication
**Solution**: Daemon uses TCP socket to avoid conflicts
**Benefit**: Multiple clients can connect simultaneously

### 2. Why Length-Prefixed Protocol?

**Problem**: Need to know message boundaries
**Solution**: 4-byte big-endian length prefix
**Benefit**: Simple, efficient, no parsing ambiguity

### 3. Why Symbol + Line Instead of Character Position?

**Problem**: LLMs don't know character positions
**Solution**: Tool converts symbol name + line number
**Benefit**: More intuitive for LLM to use

### 4. Why Persistent Daemon?

**Problem**: Starting Pyright for each request is slow (~2s)
**Solution**: Single daemon serves all requests
**Benefit**: Sub-100ms response time after warmup

### 5. Why Enhanced Responses?

**Problem**: LSP returns only locations, not source code
**Solution**: Tool fetches and includes source code
**Benefit**: LLM sees definition without extra file read

## Communication Protocols

### Socket Protocol (Tool ↔ Daemon)

```
Request:
┌────────────┬──────────────────────────────────┐
│ 4 bytes    │ N bytes                          │
│ Length (N) │ JSON payload                     │
└────────────┴──────────────────────────────────┘

Response:
┌────────────┬──────────────────────────────────┐
│ 4 bytes    │ N bytes                          │
│ Length (N) │ JSON payload                     │
└────────────┴──────────────────────────────────┘
```

### LSP Protocol (Daemon ↔ Pyright)

```
Request:
Content-Length: 156\r\n
\r\n
{"jsonrpc":"2.0","id":1,"method":"textDocument/definition",...}

Response:
Content-Length: 234\r\n
\r\n
{"jsonrpc":"2.0","id":1,"result":[...]}
```

## Performance Characteristics

- **Daemon startup**: ~500ms (one-time)
- **First request**: ~1-2s (Pyright warmup)
- **Subsequent requests**: ~50-200ms
- **Memory usage**: ~100-200MB (Pyright + daemon)
- **Concurrent requests**: Supported via async I/O

## Error Handling

1. **Daemon not running**: Connection refused → Clear error message
2. **File not found**: LSP returns empty → "No definition found"
3. **Symbol not found**: LSP returns empty → "No definition found"
4. **Timeout**: 600s timeout → "Request timed out"
5. **Invalid command**: Validation error → "Invalid command"

## Testing Strategy

Tests validate:
1. ✓ Tool definition matches OpenAI format
2. ✓ Parameters follow JSON Schema
3. ✓ Tool can be serialized in completion requests
4. ✓ Tool calls parse correctly in responses
5. ✓ Wrapper converts kwargs to args
6. ✓ Results format correctly for LLM

Tests do NOT require:
- ✗ Running daemon
- ✗ Spinning up agent
- ✗ Network connections
- ✗ File system access (except imports)
