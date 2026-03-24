"""
LSP Tool Wrapper for OpenHands SDK Integration
"""
import asyncio
from pathlib import Path
import sys

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from lsp_tool import EnhancedLSPTool


def get_lsp_tool():
    """
    Returns LSP tool as a simple function-based tool for OpenHands.
    This avoids complex ToolDefinition instantiation issues.
    """
    tool = EnhancedLSPTool()

    async def lsp_tool_func(command: str, file_path: str = None, symbol: str = None,
                           line: int = None, query: str = None) -> str:
        """
        LSP code intelligence tool for semantic code understanding.

        Args:
            command: LSP command (get_definition, get_references, get_hover, etc.)
            file_path: Absolute path to file
            symbol: Symbol name (e.g., 'MyClass', 'my_function')
            line: Line number (1-indexed)
            query: Symbol to search (for get_workspace_symbols)
        """
        args = type('Args', (), {
            'command': command,
            'file_path': file_path,
            'symbol': symbol,
            'line': line,
            'query': query
        })()
        result = await tool.run_command(args)
        return result.to_text()

    # Add metadata for tool registration
    lsp_tool_func.__name__ = "lsp_tool"
    lsp_tool_func.__doc__ = """LSP code intelligence tool providing semantic understanding via Pyright.

Commands:
- get_definition: Find symbol definition with full source code
- get_type_definition: Find type definition with source code
- get_references: Find all symbol usages across project
- get_hover: Get docstring, type info, and signature
- get_call_hierarchy: Complete incoming/outgoing call analysis
- get_document_symbols: List all symbols in file (outline)
- get_workspace_symbols: Search symbols across workspace
- get_document_highlights: Highlight symbol usages in file

Use for code navigation, understanding structure, and bug localization."""

    return lsp_tool_func

