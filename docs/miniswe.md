# mini-swe-agent harness for SWE-bench and Scale-SWE

**Status:** initial port implemented on `feat/miniswe`. Offline adapter, history,
CLI and wiring tests pass; a live smoke against a real agent-server container and
a real model endpoint also passes (environment adapter + one full mini rollout).
SWE-bench/Scale-SWE end-to-end runs with scoring and Flex/remote backends are
still **unverified** (see [Verification status](#verification-status)).

The benchmark module's `Evaluation` class stays in charge of every run: instance
selection, workspace provisioning, retries/resume, runtime accounting, result
callbacks, aggregation and cleanup. Only the per-instance agent loop is replaced
with [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) `v2.4.6`.

## Entry points

| Task | Entry point | Prompt |
| --- | --- | --- |
| SWE-bench | `benchmarks/swebench/run_infer_miniswe.py` | `benchmarks/swebench/prompts/miniswe-default.j2` |
| Scale-SWE | `benchmarks/scaleswe/run_infer_miniswe.py` | `benchmarks/scaleswe/prompts/miniswe-default.j2` |
| z021data | `benchmarks/z021data/run_infer_miniswe.py` | `benchmarks/z021data/prompts/miniswe-default.j2` |
| SWE-rebench-V2 | `benchmarks/swerebenchv2/run_infer_miniswe.py` | `benchmarks/swerebenchv2/prompts/miniswe-default.j2` |

Both subclass the existing evaluator, reuse `prepare_instances()`,
`prepare_workspace()` and the extracted `prepare_repo()` checkout helper, and
override only `evaluate_instance()`.

## Install

`mini-swe-agent` is an optional benchmark-worker dependency (it is imported
lazily, so all other entry points keep working without it):

```bash
uv sync --extra miniswe     # preferred, pins mini-swe-agent==2.4.6
# or, for an already-synced environment:
uv pip install mini-swe-agent==2.4.6
```

The harness reuses the repo's `litellm`; no separate model client is installed.

## Usage

```bash
# SWE-bench: one Verified instance in Docker with the execution-based judge
uv run python -m benchmarks.swebench.run_infer_miniswe model.json \
  --dataset princeton-nlp/SWE-bench_Verified --split test \
  --workspace docker --n-limit 1 --num-workers 1 --max-attempts 1 \
  --note miniswe-smoke --judge swebench

# Scale-SWE: one batch row in Flex
uv run python -m benchmarks.scaleswe.run_infer_miniswe model.json \
  --dataset /path/to/scale-swe-batch1.jsonl \
  --workspace flex --n-limit 1 --num-workers 1 --max-attempts 1 \
  --judge scaleswe
```

Console scripts `swebench-infer-miniswe` and `scaleswe-infer-miniswe` are also
registered in `pyproject.toml`.

Outputs go to the usual structured directory with a harness-specific `miniswe/`
suffix, so runs with identical model/dataset/note resume independently from
OpenHands runs:

```
eval_outputs/<dataset>-<split>/<model>_sdk_<sha>_maxiter_<n>_N_<note>/miniswe/
├── metadata.json
├── output.jsonl                  # candidate patches, same EvalOutput schema
├── output_errors.jsonl
├── <instance>.attempt<N>.history.json   # per-attempt normalized history
├── <instance>.history.json       # canonical history of the selected attempt
├── miniswe_traces/               # native mini transcripts (redacted), per rollout
└── judge_logs/                   # when --judge swebench
```

## CLI mapping

| Existing flag | mini-swe-agent meaning |
| --- | --- |
| `--max-iterations` | `step_limit`: the number of **model calls** (steps), not OpenHands iterations |
| `--conversation-timeout` | wall-clock rollout deadline (`wall_time_limit_seconds`); each workspace command is additionally clamped to the remaining budget, and each model request is bounded by the rollout budget (mini v2 exposes no per-call timeout hook) |
| `--workspace {docker,flex,remote}` | reused unchanged; the agent loop runs on the benchmark worker, bash actions run in the task container |
| `--max-attempts`, `--max-retries` | unchanged; `Evaluation` owns retry/aggregation |
| `--judge [name]` | unchanged; the judge scores the **submitted** patch |
| `--critic {pass,empty_patch_critic}` | supported; `finish_with_patch` depends on OpenHands events and is rejected |
| `--condenser` | rejected (OpenHands-only feature) |
| `--mini-cost-limit` | optional rollout cost cap in USD (default `0` = no limit); only effective when the model has a LiteLLM price entry (unpriced aliases record cost `0`) |
| `--mini-progress`, `--mini-command-timeout`, `--mini-bash-env-file` | shared across all four mini entry points via `benchmarks/utils/miniswe/cli.py` (see [Live progress](#live-progress)) |
| `--mini-command-timeout` | per-bash-command timeout inside the container (default 60 s) |
| `--mini-bash-env-file` | optional `BASH_ENV` file for non-login `bash` (off by default: agent-server images already put the task environment on `PATH`, and `/root/.bashrc` is unreadable to the non-root agent user; example override: `/home/openhands/.bashrc`) |

## Live progress

Unlike OpenHands runs, mini has no event stream, so the adapter emits one compact
line per rollout event while the run is in flight:

```
2026-11-23 13:16:30,215 [miniswe instance=astropy__astropy-12907 attempt=1] rollout_start model=openai/deepseek-flash cwd=/testbed step_limit=8 command_timeout=120s rollout_timeout=300s
2026-11-23 13:16:31,509 [miniswe instance=astropy__astropy-12907 attempt=1] model_call n=1 chars=0 tools=2 dt=1.29s tokens=629+104 cost=$0.0000 preview=
2026-11-23 13:16:31,511 [miniswe instance=astropy__astropy-12907 attempt=1] action n=1 command=echo miniswe-live-ok
2026-11-23 13:16:31,626 [miniswe instance=astropy__astropy-12907 attempt=1] observation n=1 rc=0 chars=16 dt=0.11s preview=miniswe-live-ok\n
2026-11-23 13:16:31,745 [miniswe instance=astropy__astropy-12907 attempt=1] interrupt n=1 type=Submitted dt=0.11s
2026-11-23 13:16:31,752 [miniswe instance=astropy__astropy-12907 attempt=1] rollout_end status=Submitted calls=1 cost=$0.0000 dt=1.54s submission_chars=0
```

Events: `rollout_start`, `model_call` (latency, output/tool-call counts, tokens,
cost), `action` (the bash command), `observation` (return code, output size,
preview), `interrupt` (mini control-flow: `FormatError`, `Submitted`,
`LimitsExceeded`, `TimeExceeded`), `rollout_end`, `model_error`, `rollout_error`.
Previews are newline-escaped and truncated to 320 chars; the full transcript is
in the native trace.

Three sinks (`--mini-progress {console,file,none}`, default `console`):

| Sink | File / stream | Notes |
| --- | --- | --- |
| dedicated progress log | `logs/instance_<id>.mini.log` | always written unless `none`; `tail -f` it for one instance |
| framework run log | `logs/instance_<id>.log` | the progress lines are also INFO records on the normal logging pipeline |
| console | the worker's real `sys.__stdout__` | escapes `Evaluation`'s per-instance stdout redirection, like the OpenHands conversation-event handler; lines are prefixed with `[miniswe instance=… attempt=…]`, so parallel workers stay attributable |

The inherited `logs/instance_<id>.output.log` is the worker process's redirected
stdout/stderr, not a channel for our progress lines: it carries stray prints and
library output (and container command output is fetched over the API, not written
here). The dedicated `.mini.log` + framework `.log` + console stream are the
progress surfaces; nothing is written into a container. For a live run:

```bash
tail -f eval_outputs/.../miniswe/logs/instance_*.mini.log
```

## How the adapter works

`benchmarks/utils/miniswe/` owns the shared integration:

| Module | Responsibility |
| --- | --- |
| `environment.py` | mini `Environment` protocol over `workspace.execute_command()`: task cwd, shell interpreter/`BASH_ENV`, stdout+stderr observation, timeout mapping, submission marker detection, roll-out deadline |
| `model.py` | translates the OpenHands `LLM` config into mini's `LitellmModel`, reusing the SDK's `select_chat_options()` for provider parity; regenerates `X-SMG-Routing-Key` per rollout when router affinity is configured (put the header in the LLM config — unlike `run_infer.py`, the mini entry point does not inject one); never serializes credentials; defaults mini's `cost_tracking` to `ignore_errors` so unpriced aliases/local models do not abort rollouts (cost is then `0`) |
| `rollout.py` | runs `DefaultAgent`, maps limits to mini's `step_limit`/`wall_time_limit_seconds`, collects metrics, and writes a redacted native trace before workspace release |
| `progress.py` | human-readable live progress: decorates mini's model/environment protocols and emits one line per model call, action, observation, interrupt and exit |
| `history.py` | normalizes mini's transcript to the `*.history.json` contract and publishes the aggregated attempt's history atomically |
| `submission.py` | uploads the candidate patch and checks it against the task baseline with `git apply --check`; a failed check is recorded, never rewrites candidate bytes |

Per-task defaults:

- **SWE-bench**: cwd is `/workspace/<repo>/` (existing `/testbed` copy + symlink +
  reset flow), and the checkout is reset to the dataset `base_commit` before the
  agent starts. Agent-server images can sit at an environment-setup commit rather
  than the dataset base commit (e.g. a local agent-server image whose HEAD is an
  env-setup snapshot); resetting guarantees the agent's `git diff` submission and
  the judge's patch application share one baseline, matching the official
  `git reset --hard {base_commit}` image build. The inherited OpenHands runner
  still keeps the image HEAD (legacy behavior). The judge gets the submitted
  patch and `SWEBenchJudge` log dir is unchanged.
- **Scale-SWE**: cwd is the image `workdir`, field normalization and
  `pre_commands` run before the first model request, and the checkout is verified
  against `parent_commit` (falling back to `base_commit`) before inference so a
  bad checkout fails the attempt instead of scoring against the wrong baseline.
- **z021data**: cwd is the image `workdir` (default `/app`), `parent_commit` is
  checked out, and an empty `problem_statement` falls back to the PR/issue text
  (same `resolve_problem_statement` helper as the OpenHands runner). The
  dataset's `pre_commands` hold the test invocation, so they are deliberately not
  run during inference. The checkout is verified before the rollout fails the
  attempt on mismatch.
- **SWE-rebench-V2**: the image repo at `/<repo>` is copied to
  `/workspace/<repo>/` and symlinked back, then reset to the dataset
  `base_commit` (multi-language images are not guaranteed to sit there). Pre-existing
  untracked build artifacts are recorded and filtered from the submitted patch
  exactly like the OpenHands runner (binary blobs would otherwise break the
  judge's `git apply`).

## Patch policy

The candidate is **mini's submitted patch**, used byte-identically for the judge,
`EvalOutput.test_result.git_patch` and the history file. The OpenHands runner's
blanket `git add -A` is deliberately not used: it would capture mini's scratch
patch file. Existing scorer postprocessing (e.g. SWE-bench offline filtering of
`pyproject.toml`/`setup.py`/`tox.ini` in `swebench-eval`) keeps its current
boundary and operates on the same JSONL.

Submission protocol (pinned mini semantics):

- the agent ends a rollout by running
  `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat /tmp/patch.txt`;
- the marker must be the **first line of stdout** and the command must exit `0`;
- everything after the marker is the patch;
- stderr is part of the observation but never part of the patch;
- a nonzero exit with the marker is a command failure, not a submission;
- submissions at/above the agent-server `BashOutput` chunk size (1 MiB) are
  **rejected explicitly** rather than forwarded as a possibly truncated or
  reordered patch (the remote polling path sees at most 100 bash events per
  command).

## History and JSONL compatibility

`<instance_id>.attempt<N>.history.json` uses the same schema as the OpenHands
runners (see `docs/condenser-wiring.md`):

| Field | Behavior |
| --- | --- |
| `instance_id`, `model`, `temperature`, `top_p` | same names/types; `temperature`/`top_p` are the **effective** values sent for this attempt (a reasoning model that drops sampling controls records `null`) |
| `messages` | normalized transcript: assistant tool calls, tool-call ids, matching tool results (`name: bash`), multiple calls per assistant message, and `reasoning_content` when available. Content uses the exporter's list encoding (`[{"type": "text", "text": ...}]`). mini's internal `role="exit"` records stay in the native trace only |
| `tools` | the exact tool schema sent to the endpoint (mini's `bash` tool) |
| `test_result.git_patch` | byte-identical to the JSONL candidate |
| `evaluation` | present only when a judge is configured; `null` means the judge could not produce a verdict |
| `trajectories` | always one `type: trajectory` segment with the model-facing message list (mini has no pruning/compaction) |
| `pruning` | never emitted (mini does not prune the model view) |
| `miniswe` | harness diagnostics: `exit_status`, `n_calls`, `cost`, `duration_seconds`, `trace_path`, `patch_applies_cleanly`, limits |

Attempt selection: each attempt writes its own
`<instance_id>.attempt<N>.history.json`. After aggregation the evaluator reads the
selected attempt back from `output.jsonl`/`output_errors.jsonl` and atomically
publishes it as `<instance_id>.history.json`. Earlier attempts and diagnostic
traces stay on disk. Attempts that raise (model/workspace failures handled by
`Evaluation`'s retry loop) keep only their native trace: they are recorded in
`output_errors.jsonl` and have no per-attempt history, so an error-only instance
has no canonical history file.

`EvalOutput.history` is empty for mini rollouts (mini produces no OpenHands SDK
events). `EvalOutput.metrics` is populated from the responses mini captured
(cost, prompt/completion tokens, cached/reasoning tokens when the provider
reports them).

## Critics and judges

- `--critic pass` (default) and `--critic empty_patch_critic` work (empty event
  list + patch-only logic).
- `--critic finish_with_patch` requires OpenHands events and is rejected with a
  clear error until event synthesis exists.
- Judges are unchanged; they receive the submitted patch. A judge that cannot run
  returns `null`, which is recorded explicitly (distinct from a genuine failed
  evaluation).

## Migration recipe for another task

1. **Describe the task contract**: dataset source, instance id, instruction,
   image resolver, cwd, setup commands, baseline commit, submission payload and
   judge input. Capture a fixture row.
2. **Reuse `Evaluation`**: add a sibling `run_infer_miniswe.py` that subclasses the
   existing evaluator. If the OpenHands `evaluate_instance()` interleaves setup
   with `Conversation` creation, extract a `prepare_repo()` helper first (as done
   for `swebench`/`scaleswe`) so both paths share it.
3. **Prepare before inference**: call `prepare_repo()`, verify the baseline commit,
   render the task prompt (add `prompts/miniswe-default.j2` containing the problem
   statement, real cwd, test-edit policy and the submission command), then build
   `MiniRolloutSettings(cwd=..., step_limit=--max-iterations,
   timeout_seconds=--conversation-timeout, ...)`.
4. **Connect scoring and output**: pass `rollout.submission` to the task judge,
   `build_history_dump(..., include_evaluation=judge is not None)` +
   `write_history()`, and return `EvalOutput(history=[], metrics=rollout.metrics)`.
   Override `run()` to call `publish_canonical_histories()` after aggregation.
5. **Verify**: reuse `benchmarks/utils/tests/` fakes; add a wiring test with a
   scripted model, then run once against a pinned fixture (setup, agent, export,
   scoring) and feed the history to the real reader. If the task has a dataset
   base commit, reset the checkout to it before the rollout (see
   `prepare_repo(checkout_commit=...)` in the SWE-bench path): agent-server images
   are not guaranteed to sit at that commit, while the judge always applies the
   patch to it.

## Troubleshooting: "Container failed to become healthy in time"

The SDK's `DockerWorkspace` waits a **hardcoded 120 s** for the agent server to
answer `/health`. Large task images can need considerably longer on a busy or
slow-disk host (a scikit-learn SWE-bench agent-server image was observed taking
**213 s** from container start to "Application startup complete"). When the wait
expires, the SDK raises and — because its docker-log streaming thread keeps a
reference to the workspace — never cleans up the container. `Evaluation` then
retries, and every retry fails at the same 120 s mark while leaking one more
container, so the cascade gets worse.

`swebench` and `scaleswe` now provision the `docker` backend with
[`benchmarks/utils/docker_workspace.py`](benchmarks/utils/docker_workspace.py)
(`BenchDockerWorkspace`):

- startup budget `OH_WORKSPACE_STARTUP_TIMEOUT` seconds (default **600**;
  e.g. `OH_WORKSPACE_STARTUP_TIMEOUT=900 uv run python -m ...`);
- a failed start stops/removes its container before re-raising, so retries do
  not leak;
- the failure message includes the last 20 container log lines, so a crash loop
  (broken image) is distinguishable from a slow start;
- a failure while running setup commands after the workspace is created also
  releases the workspace.

If a previous run was interrupted or failed this way, clean up its containers
(only when no run is active):

```bash
docker ps -a --filter name=agent-server-        # inspect
docker rm -f $(docker ps -aq --filter name=agent-server-)
```

`--workspace flex`/`remote` are unaffected. Other benchmarks can switch one line
(`DockerWorkspace(...)` -> `BenchDockerWorkspace(...)`) to get the same behavior.

## Verification status

Matrix rows required by the porting plan. "Offline" means pytest with fakes/stubs,
no Docker, no network, no model endpoint. "Live" means a real agent-server
container and/or a real model endpoint.

| Row | Status | Evidence |
| --- | --- | --- |
| Contracts / history schema | passed (offline) | `benchmarks/utils/tests/test_miniswe_history.py` compares normalized messages against `Message.to_chat_dict()` (the OpenHands exporter) for assistant tool-call and tool-result encodings, plus trajectories/evaluation/miniswe fields |
| Environment adapter (cwd, quoting, multiline, stderr, exit codes, deadlines) | passed (offline + live) | `test_miniswe_environment.py`; live `scripts/miniswe_smoke_env.py` against an agent-server image (as user `openhands`, non-root): cwd/`python -V`, multiline heredoc, stderr + nonzero exit, submission marker/payload exact |
| Submission marker (first line, payload, nonzero exit, stderr exclusion, 1 MiB rejection) | passed (offline + live) | `test_miniswe_environment.py`; live smoke transfers a patch exactly, with stderr excluded |
| Model config translation (alias, api base/key, sampling, token limit, headers, routing key, redaction, cost-tracking fallback) | passed (offline + live) | `test_miniswe_model.py`; live `openai/deepseek-flash` via `https://api.deepseek.com/v1` emitted a `bash` tool call using the translated kwargs. Live run exposed mini aborting on LiteLLM's missing price entry for the alias, fixed with non-fatal `cost_tracking` |
| Rollout limits/terminal statuses/metrics/trace redaction | passed (offline + live) | `test_miniswe_rollout.py`; live rollout ended `Submitted` in 1 call (654 prompt / 104 completion tokens, two parallel bash tool calls) with metrics populated and a redacted trace written |
| Docker workspace startup (`BenchDockerWorkspace`) | passed (live) | scikit-learn agent-server image: succeeded under the 600s budget (84s in an idle window; 213s was observed while several containers were starting, which the SDK's hardcoded 120s cannot survive), the workspace ran a command as the `openhands` user, and cleanup removed the container |
| Live progress (per-instance log, console mirror, truncation, interrupts/format errors) | passed (offline + live) | `benchmarks/utils/tests/test_miniswe_progress.py`; a live rollout produced `rollout_start`/`model_call`/`action`/`observation`/`interrupt`/`rollout_end` lines in both the `.mini.log` and the console |
| Evaluator wiring (history publication, judge candidate, JSONL fields, metrics) | passed (offline) | `benchmarks/swebench/tests/test_run_infer_miniswe.py`, `benchmarks/z021data/tests/test_run_infer_miniswe_z021data.py`, `benchmarks/swerebenchv2/tests/test_run_infer_miniswe.py` |
| Legacy runner regression (extracted `prepare_repo`) | passed (offline) | `benchmarks/{swebench,scaleswe}/tests/test_prepare_repo*.py`; existing suites `tests/`, `benchmarks/utils/test_*.py` show only the 6 `tests/test_metrics.py` failures that already exist on `dev` |
| SWE-bench baseline checkout (image HEAD -> dataset `base_commit`) | passed (live) | astropy instance image: HEAD was an env-setup commit; `prepare_repo(..., checkout_commit=...)` reset to `base_commit`, `head_matches_commit` True, simulated `git diff` validated by `check_patch_applies` |
| SWE-bench end-to-end in Docker + judge + `swebench-eval` converter | **unverified** | needs a dataset row + judge image build/pull; the environment, model and baseline halves are now live-verified separately |
| z021data end-to-end (image, checkout, judge) | **unverified** | offline wiring + prepare_repo tests only; needs an image and a dataset row |
| SWE-rebench-V2 end-to-end (multi-language image, judge) | **unverified** | offline wiring + prepare_repo tests only; needs an image and a dataset row |
| Scale-SWE end-to-end in Flex + concrete judge verdict | **unverified** | needs image pull, Flex agent-plugin runtime, Docker/AweAgent on the judge host |
| Parallel batch, interrupt/resume, forced retry, aggregation with earlier attempt winning | **unverified** (aggregation selection unit-tested only) | needs a live batch |
| Real downstream history readers (`scripts/extract_trajectories_jsonl.py`, training normalizers) | **unverified**; schema follows the shared contract and is fixture-tested | run the real reader on a produced history |
| `--workspace remote` | **unverified** | same adapter as docker/flex, but remote runtime not exercised |

### Live smoke record

- Base commit: `dev` @ `bb96d18`; feature branch `feat/miniswe`.
- `mini-swe-agent==2.4.6`, `openhands-sdk/tools/workspace 1.8.1` (submodule
  `b0725d6`), `litellm 1.80.10`.
- Backend: Docker workspace against the pre-built agent-server image
  `.../openhands-eval-agent-server:f011127-sweb.eval.x86_64.astropy_1776_astropy-12907-source-minimal`
  (task checkout at `/testbed`); model alias `openai/deepseek-flash`
  (`https://api.deepseek.com/v1`).
- Commands:
  - `uv run python scripts/miniswe_smoke_env.py --image <agent-server-image> --cwd /testbed`
    → `live workspace-environment smoke PASSED`.
  - one `run_mini_rollout` call against the same workspace with
    `MiniRolloutSettings(cwd="/testbed", step_limit=8, timeout_seconds=300)`
    → `status=Submitted`, `n_calls=1`, metrics populated, exits excluded from the
    history dump, trace written.
- Baseline checkout (SWE-bench): on the astropy agent-server image, image HEAD
  was the env-setup snapshot `e66c5e38 "SWE-bench"` while the dataset base commit
  was `d16bfe05...`. `prepare_repo(..., checkout_commit=base_commit)` reset HEAD
  to `d16bfe05`, `head_matches_commit` returned True, a simulated agent `git diff`
  passed `check_patch_applies` (True), and the validation left the worktree clean.
  This is why the earlier `Prepared checkout does not match base commit` error
  appeared; the mini entry point now resets before the first model request.
- Not run: dataset/judge/converter (SWE-bench) and Flex/pre-commands (Scale-SWE)
  end-to-end.

Known limitations to address in follow-ups: no >1 MiB submissions (explicit
rejection), no OpenHands-only critics/condensers, `EvalOutput.history` empty,
`--bind-dev-sdk`/fake-user flags not offered, mini's global model cost/call
limits (`MSWEA_GLOBAL_*`) are unset, per-model-request deadlines use the rollout
budget rather than the exact remaining time, workspace transport errors abort the
attempt (so `Evaluation` retries it) instead of being surfaced to the agent, and
cost metrics/limits are unavailable (`0`) for models missing from LiteLLM's price
map.
