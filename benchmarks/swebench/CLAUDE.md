# SWE-bench multi-stage explore → fix → review workflow

This directory hosts an optional **three-stage pipeline** layered on top of the
standard single-shot `run_infer.py`. A *teacher* model first explores the repo
and writes a plan; a *student* model implements the fix guided by that plan; a
*reviewer* model then verifies the resulting patch in a fresh environment and
emits a judgement. Each stage is a separate `run_infer-*.py` variant plus a
Jinja prompt in `prompts/`, and consumes the previous stage's per-instance
`{instance_id}.history.json`.

```
Stage 1 EXPLORE+PLAN  ──plan──▶  Stage 2 FIX  ──patch──▶  Stage 3 REVIEW ──verdict
run_infer-nothink-plan.py        run_infer-with-plan.py   run_infer-review.py
prompts/explore-and-plan.j2      prompts/fix-with-plan.j2 prompts/review.j2
```

Each stage reads the prior stage's output via a `--*-dir` flag pointing at the
directory containing `{instance_id}.history.json`, and fails fast if the
expected artifact is missing (no silent fallback to a fixerless / patchless
run).

## Stage 1 — Explore & plan (`run_infer-nothink-plan.py` + `explore-and-plan.j2`)

- **Goal:** produce a plan without editing code. The agent explores, understands
  the tests, and calls `finish` with the finalized plan as the finish message.
- **Prompt (`explore-and-plan.j2`)** is plan-only and forbids any file edits. Key
  structure:
  - Phase 5 (PLANNING) is the payload. **5.2 FIX IMPLEMENTATION** requires
    **execution-path tracing** (5.2.1) and **per-path coverage** (5.2.2): trace
    the real dispatch path for each input category (same-type vs mixed-type,
    reflected/flipped ops, broadcast/alignment, type-specific overrides) and give
    each distinct path its own edit — a change to a shared helper does not cover
    paths that bypass it.
  - **5.3 TEST STRATEGY** enumerates boundary/special-value cases, maps each case
    to a concrete test *and the code path it exercises*, states expected
    before/after outcomes, and gives an execution plan. A case whose path no edit
    touches is flagged as a coverage gap.
  - **5.5 TIME_ESTIMATE** ends with a machine-parseable block (single decimal
    `total_hours`), mirrored below by the review verdict block:
    ```
    ### TIME_ESTIMATE
    total_hours: <number>
    ```
  - Phase 6 self-review re-checks execution-path coverage (6.4) — mixed-type,
    reflected-op, broadcast, empty-input, override paths are the blind spots.
- **Why a dedicated variant** (not just `--prompt-path`): the default
  `fake_user_response` nudges the agent to keep *fixing*, which is wrong for a
  planner. `run_infer-nothink-plan.py` injects **`fake_user_response_plan`**
  ("continue drafting the plan … submit with the `finish` tool"). It also
  symlinks `/testbed → {repo_path}` so path-based reads resolve, and drops the
  ThinkTool via `include_default_tools=['FinishTool']`.
- Default prompt is `default.j2`; pass `--prompt-path .../explore-and-plan.j2`.
- **Rationale note (pandas-59636):** an earlier version only asked for a thin
  "testing" item and a flat list of test cases; students still missed a second
  code path (a DataFrame+Series broadcast helper) that mixed-type cases
  exercised. Enumerating the case was not enough — the plan never traced how it
  dispatched. Hence the execution-path-tracing requirement. Conversely, removing
  `xfail` marks to verify a fix is *correct*, not a test-editing violation — the
  prompt does not forbid it.

## Stage 2 — Fix with plan (`run_infer-with-plan.py` + `fix-with-plan.j2`)

- **Goal:** implement the fix, guided by the Stage-1 plan.
- **Plan source:** `--plan-dir <stage1 output dir>` (required). `plan_dir` is a
  field on the evaluator; `load_plan_from_history()` reads the **last `finish`
  tool call's `message`** from `{plan_dir}/{instance_id}.history.json` and injects
  it as `{{ instance.plan }}` into the prompt.
- **Prompt (`fix-with-plan.j2`)** treats the plan as an authoritative *starting
  point*, not gospel. The "How to work with the plan" preamble sets the latitude:
  - If the plan's code assumptions are wrong on inspection → prefer the code,
    adapt, note the deviation.
  - For **genuine ambiguity** (plan-flagged, or issue+code don't pin the intended
    behavior) → **defer and flag**, do not guess: take the most conservative /
    minimal / backward-compatible behavior, keep the change scoped, and surface
    each deferred case in the final finish message (what's ambiguous, options,
    interim behavior, what a maintainer must decide). Reinforced at 4.2 (no forced
    assertions on ambiguous boundary cases) and 5.4 (final-report summary).
  - The `<interface>` block is **not** included here (it is already embedded in
    the plan text).
- Default prompt is `fix-with-plan.j2`. Output patch lands at
  `test_result.git_patch` in the history JSON — the Stage-3 input.

## Stage 3 — Review / verify (`run_infer-review.py` + `review.j2`)

- **Goal:** independently verify the fix. Apply the Stage-2 patch to a **fresh**
  instance environment and have a reviewer agent judge whether it resolves the
  issue, ending with a machine-parseable verdict.
- **Patch source:** `--patch-dir <stage2 output dir>` (required).
  `load_patch_from_history()` reads `test_result.git_patch` from
  `{patch_dir}/{instance_id}.history.json` (fails fast if missing/empty) and
  attaches it as `{{ instance.patch }}`.
- **Fresh env + apply:** `evaluate_instance` copies `/testbed → repo`,
  `git reset --hard`, then transfers the patch via base64 and applies with
  `git apply` (falling back to `--3way` / `patch -p1`); a `patch_applied` flag is
  recorded even on failure.
- **Reviewer context:** issue description + the applied patch diff only (not the
  plan — kept unbiased and decoupled from Stage 1).
- **`--review-mode` (two modes):**
  - `test` (default): tools = `StrReplaceEditor(view)` + `ExecuteBash` + `Finish`.
    The reviewer may read code and *run the test suite* to empirically verify;
    the prompt includes a VERIFY-BY-TESTING phase (conda `testbed` activation).
  - `readonly`: tools = `StrReplaceEditor(view)` + `Finish`, **no shell** — a hard
    no-execution guarantee. The prompt drops the testing phase and tells the agent
    to reason from code inspection.
  - Neither mode gets the file-editor's edit commands: the reviewer must not
    modify the patched code.
- **Fake user response:** uses **`fake_user_response_review`** (not the default,
  which would nudge toward fixing) — it nudges the agent to finish with a verdict
  including the `### VERDICT` block.
- **Verdict capture:** `extract_finish_message()` reads the last `FinishAction`'s
  `message`; `parse_verdict()` regex-extracts the block:
  ```
  ### VERDICT
  decision: <pass|fail>
  confidence: <low|medium|high>
  reasoning: <one or two sentences>
  ```
  The parsed result (`decision`, `confidence`, `reasoning`, full `review_message`,
  `patch_applied`, `review_mode`) is stored under
  **`test_result.review`** in both the `{instance_id}.history.json` dump and the
  `EvalOutput` (so it reaches `output.jsonl`). A missing/malformed block yields
  `decision: null` rather than an error.

## Conventions shared across the three stages

- **Handoff via history JSON.** Each stage reads `{instance_id}.history.json` from
  the previous stage's output dir via a required `--*-dir` flag, and fails fast on
  a missing artifact.
- **Machine-parseable trailer blocks.** Stage 1 emits `### TIME_ESTIMATE`, Stage 3
  emits `### VERDICT` — both greppable with `^key:\s*value` regexes from the finish
  message so downstream tooling can extract them.
- **Stage-appropriate `fake_user_response`.** The default (fixer-oriented) message
  is wrong for the plan and review stages; each uses its own
  `fake_user_response_*` that points the agent at the correct terminal action.
- **Each stage is a copy of `run_infer.py`** with only the deltas above, following
  the repo convention of separate `run_infer-*.py` variants rather than flags on
  the base script.

## Typical run

```bash
# Stage 1 — explore & plan
python -m benchmarks.swebench.run_infer-nothink-plan \
    --prompt-path benchmarks/swebench/prompts/explore-and-plan.j2 \
    --output-dir OUT/plan ...

# Stage 2 — fix, guided by the plan
python -m benchmarks.swebench.run_infer-with-plan \
    --plan-dir OUT/plan/<history dir> --output-dir OUT/fix ...

# Stage 3 — review the fix in a fresh env
python -m benchmarks.swebench.run_infer-review \
    --patch-dir OUT/fix/<history dir> --review-mode test \
    --output-dir OUT/review ...
```

Version history of these prompts/scripts lives on the `explore-and-plan` branch;
the swerebenchv2 task carries the same explore/fix pair (not yet the reviewer).
