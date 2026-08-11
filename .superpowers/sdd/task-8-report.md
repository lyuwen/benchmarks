# Task 8 Report: run_infer.py wiring + CLI

## Status: DONE

## Files created
- `benchmarks/scaleswe_interactive/run_infer.py` — `ScaleSWEInteractiveEvaluation(ScaleSWEEvaluation)` overriding `evaluate_instance`, `load_llm_from_config(path) -> LLM`, `build_arg_parser()`, `main()`.
- `benchmarks/scaleswe_interactive/tests/test_run_infer_wiring.py` — 3 wiring tests.

## Corrections applied (all three verified against live code)

### CORRECTION 1 — `llm_config_path` is positional
Confirmed in `benchmarks/utils/args_parser.py::get_parser()`: `parser.add_argument("llm_config_path", ...)` (positional, no leading dashes). Adjusted both test cases to pass it positionally:
- `test_parser_has_interactive_args`: `parser.parse_args(["x.json"])`
- `test_parser_accepts_overrides`: `parser.parse_args(["x.json", "--user-llm-config-path", "y.json", ...])`
`main()` reads it via `args.llm_config_path`. No `--llm-config-path` optional was added (would collide with the positional dest). All assertions unchanged.

### CORRECTION 2 — `EvalMetadata.critic` is required
Confirmed in `benchmarks/utils/models.py::EvalMetadata`: `critic: CriticBase` has no default (required). Added `from benchmarks.utils.critics import create_critic`, call `critic = create_critic(args)` after parsing, and passed `critic=critic` into `EvalMetadata(...)`. Mirrors `benchmarks/scaleswe/run_infer.py::main()`.

### CORRECTION 3 — only valid EvalMetadata fields passed; interactive settings in `details`
All kwargs passed to `EvalMetadata(...)` are real fields (verified against the model): `llm, dataset, dataset_split, max_iterations, eval_output_dir, details, prompt_path, eval_limit, env_setup_commands, max_attempts, critic, selected_instances_file, max_retries, workspace_type, conversation_timeout`. Interactive settings stored in `details={'mode', 'user_tools', 'max_user_turns', 'user_llm': user_llm}`. `details` is typed `dict[str, Any] | None`, so the `LLM` object is accepted and construction succeeds given a real critic.

## Deviations from the brief
None beyond the three corrections. The large `evaluate_instance` body was reproduced faithfully. All brief-listed imports resolved without path adjustments (verified by the passing full-suite run, which imports the module at collection time). `args.split`, `args.output_dir`, `args.note`, `args.n_limit`, `args.max_attempts`, `args.select`, `args.max_retries`, `args.workspace`, `args.conversation_timeout`, `args.num_workers` all exist in `get_parser()`.

## Interfaces verified before implementation
- `driver.run_interactive_session(conversation, user_agent, initial_instruction, max_user_turns, timeout=None) -> SessionResult`
- `transcript.build_interactive_transcript(instance_id, mode, user_tools, coding_model, user_model, session) -> dict`
- `user_agent.UserAgent(user_llm, workspace, repo_path, mode, user_tools, prompts_dir, problem_statement, max_user_turns, ...)`
- Prompt templates present: `coding_system_plan.j2`, `coding_system_auto.j2`, `user_system.j2`, `initial_instruction.j2`.

## Test commands + output

### Wiring test
`python -m pytest benchmarks/scaleswe_interactive/tests/test_run_infer_wiring.py -q`
```
...                                                                      [100%]
3 passed, 1 warning in 3.39s
```
(1 warning = harmless RequestsDependencyWarning, ignored per instructions.)

Prior to implementation, the same command failed as expected:
`ModuleNotFoundError: No module named 'benchmarks.scaleswe_interactive.run_infer'`.

### Full package suite
`python -m pytest benchmarks/scaleswe_interactive/tests/ -q`
```
........................................................................ [ 78%]
....................                                                     [100%]
92 passed, 1 warning in 3.09s
```

## Commit
`d01392af301f89eba98c2918444dcca44013f5f6` — "feat(scaleswe-interactive): run_infer wiring + CLI for two-agent mode"

## Final-review fixes

Fixed all five verified Critical/Important findings from the whole-branch review.

- **CRITICAL #1 — raw dict tools crash `LLM.completion`.** Rewrote
  `user_tools.py` to define an `Action` subclass per tool
  (`ReadFileAction`, `GrepAction`, `GlobAction`, `RunReadonlyBashAction`,
  `FinishAction`) and build real `ToolDefinition` instances via
  `ToolDefinition.create(...)` with no executor. Tool names unchanged
  (`read_file`, `grep`, `glob`, `run_readonly_bash`, `finish`);
  `FINISH_TOOL_NAME="finish"`. `user_agent.py` already selects the
  `ToolDefinition` list / `[FINISH_TOOL]` and dispatches by name, so the
  manual executor path is untouched. `.to_openai_tool()` now works.
- **IMPORTANT #2 — single `&` not a separator.** `readonly_guard.py` split
  regex now `&&|\|\||;|\||&|\n` (the `&&`/`||` alternatives precede the bare
  `&`/`|`, so `a && b` still splits into exactly two segments).
- **IMPORTANT #3 — allowlisted readers writing via flags.** Added
  `_SORT_OUTPUT_FLAGS` rejection for `sort -o/--output/--output=` and a
  `_SED_PROGRAM_WRITE` regex rejecting sed `w`/`W` commands and `s///w`
  write flags, extending the existing `-i` handling. Read-only
  `sed -n '1,5p'` / `sort a` still pass.
- **IMPORTANT #4 — contradictory finish guidance.** Added a strong
  "OVERRIDE — SESSION CONTROL" block to both
  `prompts/coding_system_plan.j2` and `coding_system_auto.j2` stating the
  earlier finish-phase instructions are superseded, the coding agent must
  NEVER call finish, and only the user ends the session. `default.j2`
  untouched.
- **IMPORTANT #5 — driver never re-syncs event cache.** Added shared
  `_sync_events(conversation)` helper in `fake_user_response.py` (calls
  `_do_full_sync()` + `_reconcile_event_cache()` when present, never raises;
  existing single-shot signatures unchanged). `driver.py` calls it before
  `_latest_agent_text` each turn and now dedups by stable `event.id`
  (`_event_key`), falling back to `id(event)` when absent.

### Test additions
- `tests/test_user_tools.py::test_tools_serialize_via_to_openai_tool` —
  constructs each tool, asserts `.to_openai_tool()` returns a dict with the
  expected function name (exercises the real path the stub hid). Updated
  `test_finish_tool_present` for `ToolDefinition` objects.
- `tests/test_user_agent.py` — asserts the tools the agent passes are real
  `ToolDefinition` objects and their `.to_openai_tool()` conversion succeeds.
- `tests/test_readonly_guard.py::test_rejects_bypasses` — added
  `grep x f & rm y`, `cat a & rm b`, `sort -o out in`,
  `sort --output=out in`, `sed -n 'w /tmp/x' a`, `sed 's/a/b/w f' a`,
  `sed 'W f' a`; plus `test_readonly_sed_sort_still_pass` keeping
  `sed -n '1,5p' a` / `sort a` / `a && b` clean.
- `tests/test_driver.py::test_driver_syncs_events_before_reading_agent_text`
  — fake conversation whose agent text only lands when `_do_full_sync()` is
  called; asserts the driver still sees the text and `sync_calls >= 1`. Plus
  `test_latest_agent_text_dedups_by_stable_id` proving stable-`.id` dedup.

### Final pytest output
```
$ python -m pytest benchmarks/scaleswe_interactive/tests/ -q
107 passed, 1 warning in 3.10s
```
(RequestsDependencyWarning ignored per instructions.)
