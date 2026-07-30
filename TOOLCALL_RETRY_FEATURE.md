# Malformed Tool Call Retry Guardrail

## Overview
This feature adds a guardrail to detect and retry malformed tool calls from LLM providers that have parsing issues.

## Problem
Some LLM inference providers occasionally return malformed tool call arguments with embedded parsing artifacts, such as:
```json
{
  "command": "view",
  "path": "/workspace/beets/.github/workflows/integration_test.yaml</｜DSML｜parameter<｜DSML｜parameter name=\"security_risk\" string=\"true\">LOW",
  "security_risk": "LOW",
  "summary": "View integration_test.yaml"
}
```

These malformed responses cause tool execution errors and contaminate training data.

## Solution
A pattern-based detection mechanism that:
1. Serializes the LLM response to JSON
2. Checks for user-defined malformed patterns
3. Triggers a retry by raising `LLMNoResponseError` when patterns are detected

## Implementation
Modified `vendor/software-agent-sdk/openhands-sdk/openhands/sdk/llm/llm.py`:

### Added `_check_malformed_response` method
- Reads patterns from `OH_MALFORM_PATTERNS` environment variable
- Patterns are semicolon-delimited (e.g., `"</｜DSML｜parameter;other_pattern"`)
- Converts `ModelResponse` to JSON and searches for patterns
- Returns `True` if any pattern is found

### Modified `_transport_call` method
- After receiving response from `litellm_completion`
- Before returning, checks for malformed patterns
- Raises `LLMNoResponseError` to trigger existing retry logic

## Usage
Set the environment variable with patterns to detect:
```bash
export OH_MALFORM_PATTERNS="</｜DSML｜parameter;<｜DSML｜parameter"
```

Multiple patterns can be specified, separated by semicolons.

## Retry Behavior
- Uses existing `LLM_RETRY_EXCEPTIONS` mechanism
- Respects configured retry parameters:
  - `num_retries` (default: 5)
  - `retry_min_wait` (default: 8s)
  - `retry_max_wait` (default: 64s)
  - `retry_multiplier` (default: 8.0)

## Logging
- Logs warnings when malformed patterns are detected
- Logs errors if pattern checking fails (but continues without blocking)

---

# Static Tool-Call Parameter Validation Guardrail

## Overview
A second, complementary guardrail that validates tool-call *parameters* against
the known tool schemas after each inference and retries when they are invalid.
Where `OH_MALFORM_PATTERNS` does a text scan for known garbage substrings, this
guardrail parses the arguments and checks them structurally.

## Problem
Some providers emit tool calls whose arguments are only broken once parsed:

- **Un-parseable JSON** — raw backslash escapes such as `\boxed` / `\Box`
  (common in LaTeX/math tasks) are not valid JSON escapes, so the arguments
  string fails `json.loads` with `Invalid \escape`. Before this guardrail such a
  call slipped straight through and only exploded later in the agent at
  `json.loads(tool_call.arguments)`, surfacing as a hard `Agent Error` with no
  retry.
- **Schema-illegal parameters** — e.g. a `str_replace` with `old_str == new_str`,
  a `terminal` call combining `reset=True` with `is_input=True`, or a
  `task_tracker` `plan` with no `task_list`.

## Solution
`openhands/sdk/llm/utils/tool_call_validation.py` provides a purely static
checker (no tool execution, no filesystem access) with two layers, invoked from
`LLM._check_invalid_tool_call_params` in `_transport_call`:

1. **Generic JSON gate (all tool calls):** every tool call's `function.arguments`
   must parse to a JSON object, regardless of tool name. This catches
   un-parseable args (e.g. `\Box`) for *any* tool, including `finish` and MCP
   tools that have no dedicated schema checker.
2. **Tool-specific schema checks** for the tools that dominate agent traces:
   - `file_editor` / `str_replace_editor`
   - `terminal` / `execute_bash`
   - `task_tracker`
   - `finish` / `think` (SDK builtins) — e.g. `finish` called with only a
     `summary` and no required `message` field, which otherwise surfaces as a
     pydantic `Field required` error at agent-side validation time.

   Calls for any other tool clear the generic gate and are otherwise left alone.

When a call fails either layer, `find_invalid_tool_call` returns
`(tool_name, raw_arguments)` and `_transport_call` raises `LLMNoResponseError`,
reusing the same retry boundary as the malformed-pattern check.

## Usage
Opt-in via environment variable (truthy = `1`/`true`/`yes`/`on`):
```bash
export OH_VALIDATE_TOOLCALL_PARAMS=1
```
When unset or falsy, the check is a no-op.

## Retry Behavior
Identical to the malformed-pattern check — raises `LLMNoResponseError` inside the
existing `LLM_RETRY_EXCEPTIONS` boundary, respecting `num_retries` /
`retry_min_wait` / `retry_max_wait` / `retry_multiplier`.

## Logging
- Logs a warning with the offending tool name and raw arguments before retrying
- Logs a warning and treats the call as valid (non-blocking) if the check itself
  raises

## Commits
- Submodule: `36f5144` - feat: add guardrail to detect and retry malformed tool calls
- Submodule: `70a524d` - feat: add static tool-call parameter validation guardrail
- Submodule: `b169437` - feat: add generic JSON gate + task_tracker schema check to toolcall guardrail
- Submodule: `ae13fc2` - feat: add finish/think builtin schema checks to toolcall guardrail
- Parent: `6e396ce` - feat: update submodule with malformed tool call retry guardrail
