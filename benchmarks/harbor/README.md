# Harbor task support for OpenHands Benchmarks

`benchmarks/harbor` runs [Harbor-format](https://docs.harborframework.com/tasks/overview)
task directories with the customized OpenHands `software-agent-sdk` harness.
The benchmark runner owns the trial lifecycle (workspace, agent, verifier,
artifacts, result records); Harbor supplies the task specification and the
verifier contract. No part of the `harbor` Python package is imported, so this
adapter cannot conflict with the SDK dependency tree.

Two entry points are registered in `pyproject.toml`:

| Command | Module | Purpose |
|---|---|---|
| `harbor-infer` | `benchmarks.harbor.run_infer` | Run inference + verification for selected tasks. |
| `harbor-eval` | `benchmarks.harbor.eval_infer` | Summarize persisted results (never re-runs inference). |

## Pinned reference

The implementation targets **Harbor `v0.9.0`** (commit
`1b1dbc43729edff705071a797bb0f98a90cf62f5`). The public documentation describes
newer contracts (for example `[environment].network_mode`, `[[verifier.collect]]`,
per-step network policy) that this release does not execute; the plan explicitly
treats live docs as possible contracts, not the supported version. `task.py`
records the pinned version/commit, and resume validation covers it.

Format-version support is declared per task: `schema_version` (or the legacy
`version` key, as used by CalibForge) may be `1.0`–`1.4`. Fields added after
v0.9.0 are parsed only so unsupported execution semantics can be rejected
explicitly instead of silently ignored.

## Supported subset (Phase 1)

* Single-step tasks: `task.toml`, `instruction.md`, `environment/`, `tests/`.
* `[environment].docker_image` (prebuilt) **or** a single-container
  `environment/Dockerfile` build (build context = `environment/`).
* Linux containers only; shared verifier (the verifier runs in the agent's
  container); top-level `artifacts` collection.
* Effective working directory = `[environment].workdir`, else the image
  `WORKDIR`, else `/` (Harbor's `exec -w` semantics).
* `[environment].env` (with `${VAR}` / `${VAR:-default}` resolution),
  `[environment].healthcheck`, `[environment].cpus`, `memory_mb`,
  `[agent].timeout_sec`, `[verifier].timeout_sec`, `[verifier].env`,
  `[verifier].user`.
* `instruction.md` is used **verbatim** (Harbor's canary-line stripping is not
  applied; the plan requires verbatim instructions).

Rejected before any worker starts (clear error, never silently relaxed):

| Feature | Reason |
|---|---|
| `[[steps]]` | Multi-step rollout/rollup is Phase 2. |
| `[verifier].environment_mode = "separate"` / `[verifier.environment]` | Separate verifier sandbox + artifact handoff is Phase 2. |
| `environment/docker-compose.yaml` | Compose/multi-container environments are unsupported. |
| Windows (`environment.os`) | Linux `bash -c` execution is assumed. |
| GPU/TPU requests | Local Docker adapter is CPU-only. |
| `mcp_servers`, `skills_dir`, custom `mounts_json` | Not implemented by this harness. |
| `network_mode != "public"`, `allowed_hosts`, `allow_internet = false` | The injected agent server must stay reachable over a published port. |
| `[agent].user` other than root | The injected agent server cannot switch users for tool execution. |
| `[[verifier.collect]]` | Post-v0.9.0 collection hooks are not part of the pinned contract. |
| schema version outside `1.0`–`1.4` | Not a declared supported schema. |

Unsupported combinations are rejected at load time or CLI preflight, before the
process pool starts.

## Usage

```bash
# CalibForge (schema 1.0, prebuilt images, nested subsets)
uv run harbor-infer .llm_config/model.json \
  --tasks-path thirdparty/CalibForge/multi_solver \
  --docker-image-prefix "$REGISTRY" \
  --agent-plugin-image "$AGENT_PLUGIN_IMAGE" \
  --num-workers 4 --n-limit 10 \
  --max-attempts 1 --max-retries 0 \
  --note calibforge-smoke

# One task directory (outside any Git repo is fine)
uv run harbor-infer .llm_config/model.json \
  --tasks-path /data/harbor/my-task --agent-plugin-image "$AGENT_PLUGIN_IMAGE"

# Explicit inference-only (no verifier): results are marked unscored
uv run harbor-infer .llm_config/model.json --tasks-path /data/harbor/tasks \
  --agent-plugin-image "$AGENT_PLUGIN_IMAGE" --skip-verifier

# Summarize a completed run
uv run harbor-eval eval_outputs/<dataset>/<model-config>
```

Key options:

| Option | Meaning |
|---|---|
| `--tasks-path` | Individual task directory or dataset root. Discovery is recursive so nested subsets work (a deliberate superset of Harbor v0.9.0, which only reads one level). |
| `--docker-image-prefix` | Prepended to *bare* `docker_image` references, e.g. a local mirror. Already-qualified references (registry host or digest) are untouched. Never hard-coded. |
| `--agent-plugin-image` | Image providing the `/agent-server` volume injected into task images (also `AGENT_PLUGIN_IMAGE`). |
| `--select` | Text file of instance ids (one per line, `#` comments) or dataset-relative paths. Applied **after** path sorting and **before** `--n-limit`. |
| `--n-limit` | Number of tasks to keep after selection (`0` = all). |
| `--max-attempts` | Must be `1` in Phase 1; verifier-driven retries are not implemented. |
| `--max-retries` | Whole-rollout infrastructure retries; defaults to `0` (matching Harbor v0.9.0's `RetryConfig`). |
| `--skip-verifier` | Inference only; trials are recorded as `unscored`. |
| `--primary-reward-key` | Reward key used as the primary score (default `reward`). Set it explicitly for datasets with a different primary metric. |
| `--agent-timeout`, `--verifier-timeout` | Job-level overrides of the task budgets (also useful for smoke tests). |
| `--workspace-memory-limit` | Fallback Docker memory limit when a task omits `memory_mb` (default `14g`, `HARBOR_WORKSPACE_MEMORY_LIMIT`). |
| `HARBOR_WORKSPACE_STARTUP_TIMEOUT` | Seconds to wait for the injected agent server health endpoint (default `180`). |

The one supported workspace backend is local Docker (`--workspace docker`);
`flex`, `remote`, and the SDK's 14 GB hardcoded memory path are not used.

## Execution semantics

1. **Image resolution.** A prebuilt `[environment].docker_image` wins over
   `environment/Dockerfile` (Harbor parity). Otherwise the Dockerfile is built
   with `environment/` as context and a tag derived from the task id and
   content fingerprint. The resolved image digest/id is recorded per trial.
2. **Workspace.** The task image is started with the `openhands-agent-plugin`
   volume injected and the agent server as entrypoint (the
   `FlexWorkspace` pattern), while preserving Harbor's container settings:
   task env vars, cpus/memory limits, and the image's `WORKDIR`. The
   agent-server's conversation store is redirected to `/tmp/harbor/...` so it
   does not pollute the task filesystem. For prebuilt-image tasks whose
   `environment/` directory has no build spec but does contain runtime files,
   the directory is copied into the effective working directory after startup
   (Harbor parity).
3. **Healthcheck.** `[environment].healthcheck` runs once after startup and
   before the agent, with Harbor's retry/interval/start-period semantics.
4. **Agent.** The SDK conversation runs the verbatim instruction with the
   default tools and configured condenser. `[agent].timeout_sec` is enforced as
   a **total** budget across fake-user continuations; on expiry the conversation
   is paused and verification still runs (Harbor parity: a timed-out agent can
   still score).
5. **Artifacts.** Collected after the agent and before the verifier, into
   `trials/<id>/attempt-<n>/<execution-id>/artifacts/`. The `/logs/artifacts`
   convention source is always included; directory exclusions use
   `tar --exclude`; failures are recorded, never raised. `solution/` is never
   uploaded or exposed.
6. **Verifier (shared).** `/logs/verifier` is cleared (stale agent-written
   rewards are removed), `tests/` is copied to `/tests` after the agent,
   `chmod +x` runs as root, and the entrypoint is invoked exactly as
   `(/tests/test.sh) > /logs/verifier/test-stdout.txt 2>&1` via `bash -c` in
   the effective working directory, with the resolved `[verifier].env` and
   `[verifier].user` and the `[verifier].timeout_sec` budget.
7. **Rewards.** `/logs/verifier/reward.json` takes precedence over
   `/logs/verifier/reward.txt` (Harbor parity). Rewards may be named and
   fractional; zero is a valid score. The verifier exit code does **not** gate
   scoring. `reward.json` must be an object of numeric values; numeric strings
   and booleans are coerced as Harbor's pydantic model does, while non-finite
   values (NaN/Infinity) are rejected (plan requirement) and an empty reward
   object is a verifier error, not a zero score.
8. **Cleanup.** Partial workspace startup is cleaned up inside the workspace
   adapter (the base worker has no handle when `prepare_workspace` raises), and
   timed-out verifier processes are killed before teardown.

## Output layout

```
<eval_output_dir>/
├── metadata.json                 # EvalMetadata incl. Harbor provenance + hashes
├── harbor_tasks.json             # selected tasks, relative paths, fingerprints
├── output.critic_attempt_1.jsonl # per-attempt records (canonical resume source)
├── output.jsonl                  # aggregated successful records
├── output_errors.jsonl           # callback copy of top-level errors
├── summary.json                  # written by harbor-eval
├── logs/                         # per-instance framework/agent logs
└── trials/<safe-id>/attempt-<n>/<execution-id>/
    ├── trial.json                # instance, attempt, execution id, fingerprint
    ├── history.json              # events/model-view trajectory + test_result
    ├── conversations.tar.gz      # agent-server conversation archive
    ├── verifier/                 # reward files, test-stdout, exec logs, meta.json
    └── artifacts/                # collected artifacts + manifest.json
```

`EvalOutput.test_result` (also embedded verbatim in `history.json`) has this
shape:

```json
{
  "verifier": {
    "status": "scored | verifier_error | verifier_timeout | unscored",
    "reward": 1.0,
    "rewards": {"reward": 1.0},
    "error": null,
    "exit_code": 0,
    "duration_sec": 12.3,
    "reward_file": "reward.txt",
    "log_dir": "trials/<id>/attempt-1/<exec>/verifier",
    "test_stdout_file": "...",
    "stdout_file": "...",
    "stderr_file": "...",
    "timed_out": false
  },
  "agent": {"status": "completed | timeout | error", "error": null, "duration_sec": 42.0},
  "artifacts": {"entries": [], "manifest": "...", "failed_count": 0},
  "trial": {"instance_id": "...", "attempt": 1, "trial_dir": "...",
            "image": "...", "image_digest": "...", "workdir": "/app",
            "effective_settings": {}},
  "task": {"task_id": "...", "relative_path": "...", "schema_version": "1.0",
           "fingerprint": "...", "verifier_enabled": true,
           "primary_reward_key": "reward"}
}
```

Scoring/status contract:

* `scored` — a valid reward was produced (a valid `0` is scored).
* `verifier_error` — missing/malformed/non-finite reward, or verifier execution
  failure. Not a zero score; counted as an infrastructure/verifier error.
* `verifier_timeout` — the verifier exceeded its budget; any reward file already
  written is preserved as a raw diagnostic score.
* `unscored` — explicit inference-only run (`--skip-verifier`).

A completed rollout with a grading error keeps `EvalOutput.error = null` and
stores the grading error under `test_result.verifier.error`, so the submission
and trajectory remain in the main results. Top-level `error` is reserved for
failures to produce a usable rollout.

Resume reads completed ids from `output.critic_attempt_<n>.jsonl` (including
error rows). Before `metadata.json` is rewritten, resume validates the selected
task set/content hash and the run-settings hash; a mismatch is rejected instead
of blending two different runs. This is why the output directory is a *run*
directory: use a new `--output-dir`/`--note` when the task set or settings
change.

Phase 1 keeps the framework's **completion-based** resume policy: an instance
with a row in the attempt file is considered done for that attempt, even when
the row is a top-level error. `output_errors.jsonl` retains those errors for
reporting, but a plain resume does not retry every infrastructure failure.
Explicit grading replay (Phase 2) will mint a new grading id without
overwriting the original records.

## `harbor-eval` methodology

`harbor-eval <run-dir>` reports:

* selected task count, recorded count, attempt records, missing records;
* counts by agent status (completed/timeout/error/missing) and verifier status
  (scored/verifier_error/verifier_timeout/unscored/missing);
* infrastructure/verifier error count, computed from attempt records without
  double-counting the `output_errors.jsonl` callback copies;
* the **primary-score mean over scored tasks**, with its explicit denominator
  (tasks that produced both `scored` status and the primary key);
* scored coverage over the selected task count.

Optional policies are printed explicitly rather than implied:

* `--errors-as-zero` — a second mean that counts unscored/errored/missing tasks
  as `0` over the selected denominator (for datasets that require it).
* `--success-threshold FLOAT` — a success rate for primary reward `>= FLOAT`.

A successful critic decision is not a benchmark score: the `pass` critic is
required in Phase 1 and aggregation ranks records by critic success, so Harbor
scores come exclusively from verifier rewards.

## Parity and rollout gates

The initial milestone is the local fixture + CalibForge gate:

* a task outside any Git repo produces an OpenHands trajectory and a verifier
  reward through one `harbor-infer` invocation;
* known-good and no-op outcomes are compared against the pinned Harbor release.

Later dataset gates (not part of the initial delivery):

| Gate | Status |
|---|---|
| DeepSWE | Requires Phase 2 (separate verifier sandbox, `[[verifier.collect]]` dialect, named rewards, infrastructure-error accounting). |
| SWE-bench Pro V2 | Pin a separate V2 checkout; reproduce offline agent phase + fresh-sandbox patch replay. |
| Converted MiMo Code / Terminal | Loader/workspace support exists (schema 1.4, workdir, healthcheck, verifier user); compare a sample against the reference job before claiming parity. |

Parity comparisons must replay the same known submission (oracle / no-op /
bad patch) so model sampling is not a confound. Dataset revisions and
reference commands are pinned in the comparison notes.

## Deliberate deviations from Harbor v0.9.0

| Area | This adapter | Harbor v0.9.0 |
|---|---|---|
| Dataset discovery | Recursive (nested subsets supported, per the plan). | One level only. |
| Task validation | Malformed/unsupported tasks fail the run with a clear error. | Discovery silently skips invalid tasks. |
| Instruction | Verbatim `instruction.md`. | Strips canary lines. |
| Non-finite rewards | Rejected as verifier errors. | Accepted by pydantic. |
| Empty `reward.json` object | Verifier error (not scoreable). | Accepted as empty rewards. |
| Verifier timeout | Hard `docker exec` deadline + process kill; budget shared across grading steps. | `asyncio.wait_for` around the whole verify; client cancellation may leave the process running. |
| Image digest | Recorded per trial when available. | Not recorded. |
| Trial layout | Per-execution directories (`trials/<id>/attempt-n/<execution-id>/`). | Per-trial directory, overwritten on retry. |

## Testing

```bash
# Unit tests (no Docker, no LLM)
uv run pytest benchmarks/harbor/tests -q

# Docker integration (starts real containers, still no LLM)
HARBOR_DOCKER_TEST=1 \
HARBOR_TEST_AGENT_PLUGIN_IMAGE=<image with /agent-server> \
uv run pytest benchmarks/harbor/tests/test_integration_docker.py -q
```

The integration test builds a tiny fixture task from `debian:bookworm-slim`
(override with `HARBOR_TEST_BASE_IMAGE`), starts the workspace with the real
agent-server injection, runs the healthcheck, and grades a real `tests/test.sh`.

Style/type checks:

```bash
uv run pre-commit run --files benchmarks/harbor/**/*.py benchmarks/utils/fake_user_response.py
```

## Security notes

* No registry hostname is hard-coded; mirrors are supplied at runtime via
  `--docker-image-prefix` / `--agent-plugin-image`.
* The verifier's `tests/` is only uploaded after inference; `solution/` and the
  host task root are never exposed to the agent.
* `/logs/verifier` is cleared before grading so an agent cannot pre-write a
  reward; the verifier's exit code alone never determines the score.
* Secrets in `[environment].env` / `[verifier].env` are recorded by key name in
  `effective_settings`, not by value.

## Phase 2 (not implemented)

Separate verifier sandboxes with artifact handoff, `[[verifier.collect]]`
collection hooks, multi-step tasks, compose environments, non-public network
modes, non-root agent users, replay/regrade grading, and Harbor-aware
critic/retry policies (verifier-driven retries, best-of-k, pass@k). The record
model (`trials/...`, immutable execution ids, persisted artifacts) is designed
so replay grading can be added without overwriting original records.

## Validation status (development checkout)

The following checks were run for this implementation:

* **Unit tests** — `uv run pytest benchmarks/harbor/tests -q`: loader/validation,
  reward adapter, verifier execution (fake workspace), artifact collection,
  lifecycle/manifest/resume, trial layout, workspace flag helpers, and
  `harbor-eval` aggregation. Docker integration tests are skipped by default.
* **Docker integration** — `HARBOR_DOCKER_TEST=1 ... pytest
  benchmarks/harbor/tests/test_integration_docker.py`: a Dockerfile-built
  fixture (healthcheck → agent-server command → real `tests/test.sh`, reward
  0.5) and a prebuilt-image fixture (reward 1.0), including container cleanup.
* **End-to-end `harbor-infer`** — one local fixture task with a live model:
  OpenHands trajectory and model-view trajectory exported, verifier reward
  `1.0`, conversation archive captured. The same fixture copied **outside any
  Git repository** scored `1.0`, confirming the runner needs no Git repo.
* **CalibForge no-op smoke** — a real `multi_solver` task image resolved its
  image `WORKDIR` to `/app`, artifact collection succeeded, and the task's own
  `tests/test.sh` produced `reward.txt = 0` for the unmodified environment
  (`status = scored`, reward `0.0`) inside its 600 s budget.
