# Scale-SWE Interactive: Two-Agent Collaborative Mode — Design

**Date:** 2026-08-11
**Status:** Approved (design), pending implementation plan
**Author:** collaborative brainstorm (user + Claude)

## Summary

Branch a new evaluation task, `benchmarks/scaleswe-interactive/`, from the
existing `benchmarks/scaleswe/` task. It replaces the single coding agent +
canned `fake_user_response` with **two LLM-backed agents collaborating in one
shared workspace/container**:

- a **coding agent** (the existing agent, full tools), and
- a **user agent** that plays the human role: reads the problem statement,
  drives the coding agent (explore → plan → fix → test), reads code files
  read-only, approves or modifies the plan, and calls a **finish** action to end
  the session.

The two agents may use **different LLM backends**. The **product is the
multi-turn transcript** (trajectory / data generation), not primarily the git
patch.

## Goals

- Produce realistic multi-turn user↔agent dialogue as training/eval data.
- Support two different LLM backends (one per agent).
- Keep the user agent **read-only** with respect to the environment.
- Preserve compatibility with existing scaleswe downstream tooling
  (`history.json` + `output.jsonl`), while adding a richer speaker-tagged side
  file.

## Non-Goals

- No new judge/scoring work; the existing scaleswe judge can still run on the
  final patch but is not the focus.
- No structural sandbox enforcement of read-only at the OS level (guard is a
  command allowlist, see Risks).
- Not modifying the shared `benchmarks/utils/` base classes beyond what is
  strictly necessary; task-specific logic lives in the new task dir.

## Key Decisions (from brainstorm)

1. **Purpose:** trajectory / data generation — the transcript is the deliverable.
2. **Wiring:** one persisted `Conversation` (owned by the coding agent). The user
   agent is a custom `fake_user_response_fn` that calls its own LLM to generate
   each user turn — **not** a second full Conversation.
3. **User-agent implementation:** *Approach A — stateless LLM call per user
   turn.* Each turn is built from user system prompt + problem statement +
   running dialogue + injected file contents + (optional) read-only tool
   results. This keeps the persisted event stream owned solely by the coding
   agent and sidesteps the RemoteConversation event-cache timing gotcha
   documented in `CLAUDE.local.md`.
4. **User read access:** a switch `--user-tools {none|readonly}`.
   - **File-reference injection runs in BOTH modes** as the baseline: when the
     problem statement or the coding agent's latest message references a file
     path, that file is `cat`-ed from the shared workspace and injected into the
     user agent's context.
   - **`readonly`** additionally gives the user agent a small manual tool-call
     loop with read-only tools (read-file / grep / glob, and a bash wrapper
     restricted to read-only commands).
   - **`none`** relies solely on injected file content.
5. **Finish control:** **user-only finish.** The coding agent never ends the
   session; it always yields back to the user, who decides when solved.
6. **Collaboration modes:** `--mode {plan|auto}`.
   - **plan:** coding agent must present a plan and wait for the user agent to
     approve/modify before proceeding with edits. **Prompt-enforced** (coding
     agent yields via message; user agent gates via reply).
   - **auto:** coding agent proceeds and only returns to the user when it wants
     input; the user still reads, may redirect, and calls finish when satisfied.
7. **Turn limit:** cap total user↔agent exchanges via `--max-user-turns`, on top
   of the existing `max_iterations` for the coding agent's inner tool loop.
8. **LLM config:** reuse `--llm-config-path` for the coding agent; add
   `--user-llm-config-path` for the user agent. If omitted, the user agent
   defaults to the coding agent's config.
9. **Output:** reuse scaleswe's `history.json` + `output.jsonl` **plus** a richer
   speaker-tagged side file.

## Architecture

New task directory `benchmarks/scaleswe-interactive/`, structured like
`benchmarks/scaleswe/`. Image resolution, workspace creation, `pre_commands`,
repo location (`workdir`), `parent_commit`→`base_commit` normalization, git
add/commit/diff, and history dumping are **carried over unchanged**. Only the
agent-execution step changes.

### Components

- **`run_infer.py`** — `ScaleSWEInteractiveEvaluation(Evaluation)`. Reuses
  scaleswe's `prepare_instances` and `prepare_workspace`. `evaluate_instance`
  builds the coding agent + Conversation exactly as scaleswe does, then invokes
  the new driver loop (below) instead of
  `run_conversation_with_fake_user_response`.

- **`user_agent.py`** — the user agent as a stateless-per-turn component:
  - `UserAgent(user_llm, workspace, mode, user_tools, limits)`.
  - `respond(conversation) -> UserTurn` — the `fake_user_response_fn`-compatible
    entry point. Returns either a natural-language message or a finish signal.
  - Internals: dialogue extraction from `conversation.state.events`, file-ref
    extraction + injection, optional read-only tool loop, prompt assembly, one
    (or a few, if using tools) `user_llm.completion()` calls.

- **`readonly_tools.py`** — read-only tool definitions for the user agent and a
  **bash-command guard** (allowlist of read-only commands; rejects anything that
  can mutate the workspace). Used only in `--user-tools readonly`.

- **`file_reference.py`** — extract candidate file paths from text (problem
  statement + coding agent messages), resolve against the repo path, `cat` with
  size/count caps, format for injection.

- **`driver.py`** — `run_interactive_session(conversation, user_agent, limits)`:
  the loop that alternates coding-agent runs and user-agent turns, enforces
  `--max-user-turns`, and detects termination.

- **`prompts/`** — Jinja2 templates: `coding_system` (plan vs auto variants),
  `user_system` (plan vs auto variants), and the initial instruction.

- **`transcript.py`** — assembles the richer side file.

### Driver loop

```
send problem statement as first user turn
loop:
  coding agent runs its inner tool loop (respects max_iterations)
  coding agent yields a message (or errors/stucks -> stop)
  user_turns += 1
  user agent turn:
     inject referenced files (both modes)
     if user_tools == readonly: run read-only tool loop
     produce reply OR finish
  if finish: stop (reason = user_finish)
  if user_turns >= max_user_turns: stop (reason = turn_cap)
  else: conversation.send_message(reply); continue
```

Termination reasons: `user_finish`, `turn_cap`, `agent_error`, `agent_stuck`,
`user_error` (user LLM failed after retries — see Error handling).

### Data flow / ownership

- The **coding agent owns the single persisted `Conversation`** and its event
  stream. User messages enter that stream via `conversation.send_message()` —
  identical to how the canned fake user works today — so `messages` in
  `history.json` naturally contains the full user↔agent dialogue.
- The **user agent keeps no persisted Conversation.** Its per-turn LLM calls and
  read-only tool traces are captured only in memory and written to the side
  file.

## CLI surface (new/changed args)

- `--user-llm-config-path PATH` (default: same as `--llm-config-path`).
- `--mode {plan|auto}` (default: `plan`).
- `--user-tools {none|readonly}` (default: `none`).
- `--max-user-turns INT` (default: e.g. 15).
- File-injection caps: `--inject-max-files`, `--inject-max-bytes` (sensible
  defaults; may be constants initially).

## Output format

1. **Reuse scaleswe** (`{instance_id}.history.json`, `output.jsonl`): unchanged
   shape — `git_patch`, `messages` (dialogue as user/assistant turns), `tools`,
   model, temperature. Existing tooling keeps working.

2. **Richer side file `{instance_id}.interactive.json`:**
   - `instance_id`, `mode`, `user_tools`, `coding_model`, `user_model`.
   - `turns`: ordered list, each tagged `speaker` (`coding` | `user`), the
     message text, and for user turns: `injected_files` (paths + sizes),
     `readonly_tool_calls` (name, args, truncated result) when in readonly mode.
   - `termination`: `{reason, user_turns}`.

## Error handling

- **User LLM failure** on a turn: retry a small number of times; on exhaustion,
  terminate the session with reason `user_error` and still dump partial
  transcript + patch.
- **File injection:** missing/oversized files are skipped with a note in
  `injected_files`; never fatal.
- **Read-only guard:** a rejected command returns a tool error to the user agent
  (not a crash), instructing it to use read-only operations.
- **Coding agent error/stuck:** reuse existing `ConversationExecutionStatus`
  checks; stop and dump what exists.

## Testing

- **Unit:** read-only bash guard (accepts `cat`/`grep`/`ls`/`git log`/`git
  diff`; rejects `rm`/`>`/`sed -i`/`git commit`/`apply`/`tee`/`mv`/`cp` into
  repo, and command chaining that hides a writer).
- **Unit:** file-reference extractor (paths in prose, backticked paths, line
  refs like `path:line`), and injector caps.
- **Unit:** driver termination — `user_finish`, `turn_cap`, `agent_error`, using
  a stubbed user LLM and a fake conversation.
- **Unit:** transcript assembly shape.
- **Smoke:** one real instance end-to-end (docker/flex), assert both output
  files are produced and dialogue has ≥2 user turns.

## Risks / open items

- **Read-only guard is best-effort** (command allowlist), not an OS-level
  sandbox. A cleverly crafted command could slip through. Acceptable for
  data-gen; documented. Could later run the user agent as a distinct low-priv
  OS user if stronger isolation is needed.
- **plan mode is prompt-enforced**, so a non-compliant coding agent could edit
  before approval. Acceptable initially; a structural gate can be added later if
  the data shows leakage.
- **File-reference extraction is heuristic** — over-injection (noise) vs
  under-injection (user under-informed). Caps + iteration on the regex mitigate.
- Ensure the `Conversation(...)` → first `send_message` gap from
  `CLAUDE.local.md` is preserved (repo prep already provides it in scaleswe;
  keep that ordering).

## Reference

- Base task: `benchmarks/scaleswe/run_infer.py`.
- Replaced mechanism: `benchmarks/utils/fake_user_response.py`.
- Multi-stage prompt patterns to borrow from: `benchmarks/swebench/CLAUDE.md`.
