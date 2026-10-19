# Scale-SWE Interactive (two-agent mode)

Two LLM agents collaborate in one shared workspace to produce a multi-turn
trajectory:

- **coding agent** (positional config argument): full tools, does the work.
- **user agent** (`--user-llm-config-path`, defaults to the coding LLM config
  when omitted): read-only, drives the coding agent, approves plans, and calls
  `finish` to end.

## Usage

    python -m benchmarks.scaleswe_interactive.run_infer \
      <coding_llm_config.json> \
      --user-llm-config-path configs/user.json \
      --mode plan \
      --user-tools readonly \
      --max-user-turns 20 \
      --n-limit 1

The coding LLM config is a positional argument. `--user-llm-config-path`
defaults to that same coding LLM config when omitted.

## Modes
- `--mode plan` (default): coding agent must present a plan and wait for user
  approval before editing (prompt-enforced).
- `--mode auto`: coding agent proceeds autonomously, returning to the user only
  when it wants input.

## User repo access
- `--user-tools none` (default): user sees only injected contents of files
  referenced in the problem statement / coding agent messages.
- `--user-tools readonly`: user additionally gets read-only tools
  (`read_file`, `grep`, `glob`, guarded `run_readonly_bash`).

## Turn cap
- `--max-user-turns` (default 20): caps the number of user turns before the
  session terminates.

## Outputs (in the eval output dir)
- `{id}.history.json` — scaleswe-compatible (git_patch, messages, tools).
- `{id}.interactive.json` — speaker-tagged turns, mode, models, injected files,
  read-only tool traces, termination reason.

Read-only enforcement is a best-effort command allowlist, not an OS sandbox.
