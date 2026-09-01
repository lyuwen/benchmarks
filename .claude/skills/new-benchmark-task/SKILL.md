---
name: new-benchmark-task
description: >-
  Use this skill whenever creating a NEW OpenHands benchmark task by mirroring an
  existing one (e.g. cloning benchmarks/scaleswe into benchmarks/<newtask>), or
  when adapting a task's execution-based judge to a new dataset. Triggers on
  requests like "mirror the scaleswe task", "create a new benchmark task like X",
  "add a run_infer + judge for <dataset>", "adapt the judge to <validator>", or
  when porting the plan→implement→review or single-shot inference harness to a
  new SWE-bench-style dataset with per-instance Docker images. Covers the file
  set to mirror, dataset-field mapping, judge adaptation to an offline validator,
  judge registration wiring, and the security/registry gotchas that are easy to
  get wrong.
---

# Mirroring an OpenHands benchmark task

This repo's benchmark tasks (`benchmarks/<task>/`) are near-identical in shape;
a new task is created by mirroring an existing one and adapting only the
dataset-specific and judge-specific parts. `benchmarks/scaleswe` is the reference
for **image-URL tasks** — datasets that ship a pre-built per-instance Docker image
(field like `image_url`/`image_name`), as opposed to swebench which derives the
image from `instance_id`. Follow the phases below in order.

## Phase 0 — Inventory the source task

Before writing anything, read the source task end-to-end so the mirror matches its
current structure (it drifts over time). A task directory contains:

- `__init__.py` — **empty** (0 bytes).
- `judge.py` — an `ExecutionBasedJudge` subclass, `@register_judge("<task>")`.
- `run_infer.py` — the main inference entrypoint (an `Evaluation` subclass with
  `prepare_instances` / `prepare_workspace` / `evaluate_instance`).
- `run_infer-no-think-no-task.py` — optional variant, identical to `run_infer.py`
  except the tool block hardcodes `StrReplaceEditor` + `ExecuteBash` +
  `include_default_tools=['FinishTool']`.
- `prompts/*.j2` — Jinja2 instruction templates. Copy verbatim; they reference
  `instance.repo_path`, `instance.problem_statement`, `instance.base_commit`.

Also read `benchmarks/utils/execution_judge.py` (the judge base class, registry,
`add_judge_args`, `create_judge`) and `benchmarks/utils/evaluation.py` (the
`Evaluation` base class and its 3-phase contract).

## Phase 1 — Map the dataset schema

Inspect the dataset (usually a JSONL under `benchmarks/<task>/data/`) and map its
field names to the SWE-bench concepts the harness expects. **Do this before
coding** — the field names differ per dataset and wrong assumptions here cause
silent evaluation errors. See `references/field-mapping.md` for the concrete
z021data map and the questions to answer for any new dataset.

Key questions: which field is the base commit? which identifies the Docker image,
and is it a bare name or already registry-qualified? are `FAIL_TO_PASS` /
`PASS_TO_PASS` present (and are they JSON-encoded strings)? is
`problem_statement` sometimes empty, and what is the fallback?

## Phase 2 — Mirror the scaffolding

Mechanical, low-risk; safe to delegate to a subagent:

1. Empty `__init__.py`.
2. Copy `prompts/*.j2` verbatim.
3. Copy `run_infer.py`, then adapt (Phase 4). Create the no-think variant by
   copying the finished `run_infer.py` and applying only the tool-block edit.

## Phase 3 — Adapt the judge (the hard part)

The judge determines whether an agent patch resolves the instance. There are two
shapes; pick by what the task provides:

- **Delegating judge** (like `scaleswe/judge.py`): hands off to an external
  package's evaluator (`awe_agent`). Only viable if that package exists and is
  importable. Cheap to mirror but opaque.
- **Self-contained judge** (like `benchmarks/swerebenchv2/judge.py`): applies
  patches and runs tests directly in the instance image. **Prefer this when the
  task ships an offline validator** — the judge must reproduce that validator's
  exact procedure so a passing judge means the same thing as passing offline
  validation.

When adapting to an offline validator (e.g. a `validate_*.py`):

1. **Reuse the validator's own helpers** rather than reimplementing them — import
   its patch-separation, `apply_patch`, `build_run_command`, `parse_run_output`,
   and `wait_for_container_ready` functions. This keeps the judge in lockstep;
   the validator can evolve and the judge follows.
2. **Reproduce the step order exactly.** For z021data: checkout base commit →
   apply ground-truth test patch + f2p_patch → write the generated test file
   (`test_script`) → run `setup ; test_command` BEFORE the fix (Result 1) →
   apply the **agent's** patch as the fix (test-file edits stripped) → run again
   AFTER (Result 2) → `F2P = before_failed ∩ after_passed`,
   `P2F = before_passed ∩ after_failed`, `resolved = F2P>0 and P2F==0`.
   The one substitution vs. the offline validator: the agent's `git_patch`
   replaces the gold *source* patch; the gold *test* artifacts stay.
3. **Strip test-file edits from the agent patch** before applying, so a candidate
   cannot smuggle changes into the ground-truth test files.
4. **Import docker-py lazily** and return `None` (skip, don't crash) when it — or
   the validator's core module — is unavailable. Empty/blank `git_patch` →
   return `False` immediately.
5. Match the validator's **image resolution and container run** exactly (CMD
   `sleep infinity`, don't override ENTRYPOINT, wait for the readiness marker,
   run commands via `bash -c "cd {workdir} && timeout ... "`).

## Phase 4 — Adapt run_infer.py

Change only the task-specific bits (leave `prepare_workspace`'s docker/flex/remote
branches alone unless the image differs):

- **Image resolution:** for image-URL tasks you *prepend* a registry prefix to a
  bare image name (see Phase 5 gotchas) — NOT scaleswe's "replace everything
  before the last `/`". Guard against an already-qualified URL.
- **Base commit:** set `instance.data["base_commit"]` from the dataset's base
  field (z021data: `parent_commit`). Check it out before the agent starts so the
  `git diff <base> HEAD` diff base is the pre-fix state.
- **Problem statement fallback:** if the dataset leaves `problem_statement` empty
  for some rows, compose a fallback (z021data: from `pr_description`).
- **Do NOT blindly run the dataset's `pre_commands`** — verify what that field
  actually holds (in z021data it's the *test* command, not setup).
- **Judge wiring:** import the judge for its registration side effect at the top,
  `add_judge_args(parser, default_judge="<task>")`, set the dataset/workspace
  defaults, and pass the judge into the `Evaluation` subclass.
- In `benchmarks/utils/execution_judge.py`, add the task name to the tuple in
  `create_judge()` that forwards `docker_image_prefix`.

## Phase 5 — Security & correctness gotchas

These are the mistakes that are easy to make and expensive to miss:

- **No hardcoded private registry as a default.** `docker_image_prefix` /
  `DEFAULT_DOCKER_IMAGE_PREFIX` must default to `None`/`""`; the registry is
  supplied only at runtime via `--docker-image-prefix`. A committed registry
  hostname is sensitive-info disclosure. `grep` new files for the registry
  string before committing — and scrub it from any offline validator that was
  copied in (its argparse default / function defaults often bake it in).
- **Detect an already-qualified `image_url` before prepending.** Only prepend to
  a bare `namespace/repo:tag`. Docker's rule: the first `/`-separated component
  is a registry host when it contains `.` or `:` or equals `localhost`. If a
  host is present (or no prefix given), return the URL unchanged (lowercased).
  Apply the same helper identically in `judge.py` and `run_infer*.py`.
- **The dataset `.gitignore` may ignore everything** (`*`). `git add` respects
  it so the large dataset stays out of commits — confirm with `git add -n`
  before committing, and commit only the code files.
- **Scrub git history when the image is built AHEAD of the base commit.** If the
  pre-built image sits at the *fix* state (e.g. z021data at `pr_commit`) and
  inference checks out *backwards* to the base commit, the gold fix and
  test-generation commits stay reachable via branches, tags, remote-tracking
  refs, and the reflog — the agent can inspect the solution (`git log --all`,
  `git show <fix>`) and stray refs pollute the final `git diff base HEAD`. After
  checkout, pin one branch at the base commit and delete every other
  branch/tag/remote, then `git reflog expire --expire=now --all` and
  `git gc --prune=now` so the base is the sole reachable tip. Verify with a
  local git simulation that `git show <fix>` returns "bad object" and
  `git diff base HEAD` still works after committing. The judge is unaffected —
  it runs in a fresh container from the image with full history. (Tasks like
  swebench/swerebench that start at HEAD=base don't need this — a plain
  `git reset --hard` suffices.)
- **Empty patch → `False`; missing docker/env → `None`.** A judge must never
  crash the run.

## Phase 6 — Validate

Run in the project venv (`.venv/bin/python`), which has the openhands SDK:

- `python -c "import ast; ast.parse(open(f).read())"` on each new file.
- Import each `run_infer*.py` (hyphenated names via `importlib.util`) and confirm
  the judge registers: `"<task>" in get_registered_judges()`.
- Exercise the image-resolution helper on bare, prefixed, and already-qualified
  URLs; confirm `judge.py` and `run_infer.py` agree.
- Confirm empty patch → `False` and no-docker → `None` without raising.

## Delegation note

This repo's convention is subagent-driven development. Phase 2 (scaffolding) and
Phase 6 (validation) delegate cleanly. Write Phase 3 (judge) and Phase 4
(run_infer) yourself or with tight specs — they carry the task-specific logic
where drift is costly.
