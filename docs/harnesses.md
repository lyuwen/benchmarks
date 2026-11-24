# Blackbox harness inference (Claude Code)

This page documents `run_infer_harnesses.py`, the inference entry points that run
an **external agent harness as a black box inside the task container** and
capture its model traffic for trajectory synthesis.

Two harness-family entry points exist today:

| Task | Entry point | Module |
| --- | --- | --- |
| SWE-bench | `swebench-infer-harnesses` | `benchmarks/swebench/run_infer_harnesses.py` |
| SWE-rebench-V2 | `swerebenchv2-infer-harnesses` | `benchmarks/swerebenchv2/run_infer_harnesses.py` |

The first supported harness is **Claude Code**; the architecture is deliberately
harness-agnostic so other CLIs (Codex, Gemini CLI, ...) can be added behind the
same runner.

## Why a rollout manager

Unlike `run_infer.py` (OpenHands agent through an agent server) and
`run_infer_miniswe.py` (mini-swe-agent driving the container from the host), a
blackbox harness is an opaque CLI: we can observe its effects on the container,
but not its messages. To get trajectories anyway, the harness is pointed at a
local **capture proxy**:

```text
task container                          host
┌──────────────────────────────┐    ┌───────────────────────────────────────┐
│  Claude Code (native binary) │    │  rollout-manager (one process/session) │
│    ANTHROPIC_BASE_URL ───────┼───▶│  Anthropic Messages → OpenAI chat      │
│    (tool calls, compaction)  │    │  requests/responses captured + tree    │
│  repo at /testbed            │    │  ↓                                     │
└──────────────────────────────┘    │  OpenAI-compatible model backend       │
                                    │  (LLM config base_url)                 │
                                    └───────────────────────────────────────┘
```

The proxy is [`lyuwen/rollout-manager`](https://github.com/lyuwen/rollout-manager),
pinned as a project dependency. Per instance it:

- forwards the harness's **Anthropic Messages** traffic to the benchmark's
  OpenAI-compatible backend;
- records every request/response pair;
- routes requests into a prefix tree and exports the session's **trajectory
  series** (main conversation, context compactions, sub-agents, forks) as one
  JSON document (`schema version 2`);
- labels compaction/auxiliary requests from Claude Code's explicit
  `x-claude-code-*` gateway hint headers (enabled via
  `CLAUDE_CODE_GATEWAY_HINT_HEADERS=1`, set by the harness adapter).

The harness never sees credentials: the container gets a dummy token, and the
proxy uses the real upstream key from the LLM config.

## Prerequisites

- Docker available locally (the harness runs in a *plain* task container built
  from the official image — no agent server involved).
- An OpenAI-compatible model backend reachable from the host that supports
  **tool calling**. The LLM config is the same JSON used by the other
  benchmarks (`.llm_config/*.json`).
- ~380 MiB of disk and a one-time download for the pinned native Claude Code
  distribution (cached under `~/.cache/openhands-benchmarks/harness-dist`).

### Preparing the harness distribution in advance

The evaluation entry points build/verify the distribution in the main process
before workers start, so it is downloaded exactly once per host. To fetch it
ahead of time (login node, image build, shared cache) use the dedicated command:

```bash
uv run harness-prepare                     # download the pinned release
uv run harness-prepare --harness-cache-dir /shared/harness-dist
uv run harness-prepare --check             # cached? (no network)
```

Resulting cache layout:

```text
<cache>/claude-code/<version>/<platform>/{claude, manifest.json}
# e.g. ~/.cache/openhands-benchmarks/harness-dist/claude-code/2.1.331/linux-x64/
```

For **air-gapped hosts**, download on a machine with network access and seed the
cache from the binary (checksum is optional but recommended; `harness-prepare`
prints it, and the release manifest publishes it):

```bash
# with network:
curl -fsSL -o claude \
  https://downloads.claude.ai/claude-code-releases/2.1.331/linux-x64/claude
sha256sum claude

# on the eval host:
uv run harness-prepare --binary /path/to/claude --sha256 <checksum>
```

Seed mode verifies size/checksum (and runs `claude --version` when the binary
matches the host platform) before writing the cache, so a truncated or wrong
platform download fails fast.

Set the cache location once for a whole cluster instead of passing a flag to
every command:

```bash
export HARNESS_DIST_CACHE_DIR=/mnt/shared/harness-dist   # NFS or other shared FS
uv run harness-prepare                                   # fill it once
uv run swebench-infer-harnesses ...                      # all nodes/workers use it
```

### Distributed / shared cache (NFS)

The cache directory can live on NFS (or any shared filesystem) and be shared by
every evaluation node and worker. It is safe against concurrent first-use:

- the cached entry is checked lock-free before any work;
- an advisory `flock` on `<entry>/.distribution.lock` serializes a first-time
  build so a shared cache is downloaded **once**, not once per node (other
  processes re-check the cache after acquiring the lock);
- downloads go to a per-process unique temp file and are published with an
  atomic `os.replace` (same filesystem as the final path), and `manifest.json`
  is written atomically — so a reader never sees a partial binary/manifest;
- the layout is per version *and* platform
  (`<cache>/claude-code/<version>/<platform>/`), so nodes preparing different
  container platforms never collide.

Filesystem caveats:

- `flock` is advisory and not guaranteed on every NFS setup. The code detects
  “lock not supported” and continues lock-free; correctness is preserved by the
  unique temp files + atomic renames, but simultaneous first downloads may be
  duplicated. Warm the cache once (`harness-prepare`) before a large run to
  avoid that.
- The cache directory must be writable by the evaluation users (and `os.replace`
  requires the temp file to be on the same filesystem, which is why temp files
  are created as siblings).
- For **performance**, prefer a shared cache as a *cold store* and a per-node
  local cache for the actual runs: 380 MiB is read per instance in `copy` mode
  (`docker cp`) and per execution in `mount` mode, which is expensive over NFS.
  Seed each node once from the shared copy:

  ```bash
  # on each evaluation node
  uv run harness-prepare \
    --harness-cache-dir ~/.cache/openhands-benchmarks/harness-dist \
    --binary /mnt/shared/harness-dist/claude-code/2.1.331/linux-x64/claude \
    --sha256 4c00839cf46beb2b38ec310105f2d538f7f9321fe46c636299b74bcab94626b1
  ```

  (`--check` confirms the local copy is present.)

## Usage

SWE-bench:

```bash
uv run swebench-infer-harnesses .llm_config/model.json \
  --dataset princeton-nlp/SWE-bench_Verified --split test \
  --harness claude-code \
  --n-limit 10 --num-workers 2 --max-attempts 1 \
  --judge swebench
```

SWE-rebench-V2:

```bash
uv run swerebenchv2-infer-harnesses .llm_config/model.json \
  --dataset thirdparty/SWE-rebench-V2/SWE-rebench-V2-py.jsonl \
  --harness claude-code \
  --n-limit 10 --num-workers 2 --max-attempts 1 \
  --judge swerebenchv2
```

The full argument surface is shared with the regular inference entry points
(`--dataset`, `--split`, `--n-limit`, `--select`, `--num-workers`,
`--max-attempts`, `--max-retries`, `--conversation-timeout`, `--output-dir`,
`--note`, `--critic`, judge flags, ...), plus the harness flags below.

Only `--workspace docker` is supported: the harness must execute inside the task
container and the proxy runs next to the local Docker daemon.

## Harness configuration

| Flag | Default | Meaning |
| --- | --- | --- |
| `--harness` | `claude-code` | Harness implementation |
| `--harness-version` | `2.1.331` | Pinned Claude Code release (native package version) |
| `--harness-platform` | host auto-detect | `linux-x64`, `linux-arm64`, `linux-x64-musl`, `linux-arm64-musl` |
| `--harness-cache-dir` | `~/.cache/openhands-benchmarks/harness-dist` | Host cache for the portable distribution |
| `--harness-install-dir` | `/opt/harness` | Where the binary is exposed inside the task container |
| `--harness-dist-mode` | `copy` | `copy` (`docker cp` the binary) or `mount` (bind-mount the cache; faster for large local runs) |
| `--harness-context-window` | `200000` | Real context window of the served model, reported to the harness (`0`: harness default) |
| `--harness-compaction-threshold` | `80` | Compact at this percentage of the auto-compact window (`0`: harness default) |
| `--harness-max-output-tokens` | unset | Per-response output-token cap |
| `--harness-max-turns` | `--max-iterations` | Agentic-turn cap |
| `--harness-bash-timeout-ms` | `600000` | Default/max harness shell-command timeout |
| `--harness-api-timeout-ms` | `1200000` | Per-model-request timeout |
| `--harness-bare` | off | Run Claude Code with `--bare` (no hooks/CLAUDE.md/skills/MCP discovery) |
| `--harness-no-skip-permissions` | off | Do not use bypass-permissions mode |
| `--harness-env KEY=VALUE` | none | Extra environment variable (repeatable) |
| `--harness-arg ARG` | none | Extra harness CLI argument (repeatable) |
| `--harness-header 'Name: Value'` | from LLM config | Extra HTTP header sent by the harness (forwarded by the proxy to the upstream) |

### Compaction is configured explicitly

Claude Code assumes a **1M-token** context window for model IDs it does not
recognize (which is the usual case for a served backend behind a proxy), so its
default compaction point would be far beyond the model's real limit. The
pipeline therefore sets, inside the harness process:

| Variable | Value |
| --- | --- |
| `CLAUDE_CODE_MAX_CONTEXT_TOKENS` | `--harness-context-window` |
| `CLAUDE_CODE_AUTO_COMPACT_WINDOW` | `--harness-context-window` (only when ≥ 100k) |
| `CLAUDE_AUTOCOMPACT_PCT_OVERRIDE` | `--harness-compaction-threshold` |

With the defaults (200k window, 80%), the harness compacts when the conversation
reaches ~160k tokens. Compaction requests are captured as their own trajectory
(kind `compaction`) and labelled in the export, so the trajectory series shows
the pre-compaction conversation, the summary call, and the continuation.

### Upstream model resolution

The rollout manager needs the OpenAI-compatible URL, key and model name:

- URL: `--upstream-base-url`, else the LLM config `base_url` with `/v1`
  appended when missing (`https://llm-proxy...` → `https://llm-proxy.../v1`).
- key: `--upstream-api-key`, else the LLM config `api_key`.
- model: `--upstream-model`, else the LLM config `model` with client-side
  LiteLLM prefixes stripped (`litellm_proxy/anthropic/claude-sonnet-4` →
  `anthropic/claude-sonnet-4`, `openai/foo` → `foo`).
- the harness is told the resolved model name (`--harness-model` overrides it).

Headers configured under `litellm_params.extra_headers` in the LLM config are
injected into the harness requests via `ANTHROPIC_CUSTOM_HEADERS`, so provider
beta flags keep working through the proxy. Transport/auth/routing headers are
stripped.

## What happens per instance

1. The official task image is pulled (if missing) and a long-lived container is
   started from it (`--entrypoint sleep infinity`); no agent server is involved.
2. The portable Claude Code binary is installed (`docker cp`) or mounted
   (`-v ...:ro`), then smoke-checked with `claude --version` inside the
   container.
3. The repository is reset to the dataset `base_commit`
   (`/testbed` for SWE-bench, `/<repo>` for SWE-rebench-V2) and pre-existing
   untracked files are recorded so image build artifacts never leak into the
   patch.
4. One `rollout-manager` process is started on the host, bound to the Docker
   bridge gateway, and the harness runs headless inside the container:

   ```text
   claude -p "<instruction>" --output-format stream-json --verbose \
     --settings /tmp/harness/settings.json --model <model> \
     --dangerously-skip-permissions --max-turns <n> --no-session-persistence
   ```

   with `IS_SANDBOX=1` (required for bypass-permissions as root), telemetry and
   auto-update disabled, and `ANTHROPIC_BASE_URL` pointing at the proxy.
5. The harness stream-json log is streamed to the output directory; the proxy is
   finalized (`POST /rollout/finish`) and the export is validated.
6. `git add -A` + a `harness patch` commit + `git diff <base_commit> HEAD`
   yields the candidate patch (pre-existing untracked files are removed from it).
7. The optional `--judge` runs exactly like in the regular inference entry
   points, and a per-attempt history dump is written.

## Outputs

Everything lands in the evaluator's structured output directory, under
`.../harnesses/claude-code/` (separate from OpenHands runs, so resume and
attempt bookkeeping never mix):

```text
<eval_output_dir>/harnesses/claude-code/
├── metadata.json
├── output.jsonl                          # EvalOutput per instance (test_result.git_patch)
├── output_errors.jsonl
├── output.critic_attempt_<N>.jsonl
├── <instance>.attempt<N>.history.json    # trajectory + patch + run metadata
├── <instance>.history.json               # winning attempt (published after the run)
├── rollout/
│   └── <instance>.attempt<N>.json        # rollout-manager trajectory export (schema v2)
├── logs/
│   ├── instance_<id>.log                 # framework log
│   ├── instance_<id>.output.log          # worker stdout/stderr
│   ├── <id>.attempt<N>.harness.jsonl     # Claude Code stream-json transcript
│   └── <id>.attempt<N>.rollout-manager.log
└── judge_logs/                           # swebench judge logs, when enabled
```

The **trajectory export** is the primary artifact: `trajectories` (prefix-tree
order) with strict OpenAI Chat-Completions `messages`, `loss_mask`,
`reasoning_content`, per-turn `request`/`response`, plus `request_log` (capture
order) and `stats`. See the upstream
[rollout-manager README](https://github.com/lyuwen/rollout-manager#export-format)
for the full schema.

The per-instance **history JSON** mirrors the other benchmarks' dumps:

| Key | Content |
| --- | --- |
| `messages` | messages of the main conversation trajectory |
| `tools` | deduplicated tool lists captured for the session |
| `trajectories`, `request_log`, `stats` | the rollout export's trajectory data |
| `test_result.git_patch` | candidate patch (also in `output.jsonl`) |
| `evaluation` | judge verdict (`true`/`false`/`null`), when a judge ran |
| `run` | harness exit code, duration, timeout flag, stream-json `result` event |
| `rollout_export` | relative path to the trajectory export |
| `repository` | repo path, base commit, ignored pre-existing untracked files |
| `warnings` | non-fatal harness/proxy problems (e.g. no captures) |

## Reliability notes

- Harness-level failures (non-zero exit, timeout, empty capture) are **outcomes,
  not exceptions**: the patch (possibly empty) is still scored, and the attempt
  is not re-run. Infrastructure failures (image pull, container start, proxy
  start, repo reset, patch extraction) raise and go through the normal retry
  loop with a fresh container.
- `--conversation-timeout` is a wall-clock budget for the harness run; the
  process is wrapped in the container's `timeout` and a host-side backstop kills
  the exec client if Docker hangs.
- A `rollout-manager` snapshot is written after every turn, so a crashed
  harness/proxy still leaves a partial (but valid) export.
- Requests are always made non-streaming to the upstream and re-rendered as SSE
  by the proxy; the proxy also serves `/v1/messages/count_tokens` with a
  character-based estimate (no tokenizer on the capture path).

## Manual smoke test

`scripts/harness_smoke.py` runs the full pipeline against a scripted fake
upstream and a local task image (no model backend needed):

```bash
# Reuse a downloaded native binary (skips the ~380 MiB download)
uv run python scripts/harness_smoke.py \
  --image <task-image-with-git-and-bash> \
  --claude-binary /path/to/claude \
  --dist-mode mount
```

It creates a git repo in the container, scripts one Bash tool call through the
capture proxy, and asserts that the patch and the trajectory export are
produced.

## Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| `Cannot use --dangerously-skip-permissions as root` | The adapter sets `IS_SANDBOX=1`; if you removed it, restore it or run as non-root |
| `rollout-manager` not installed | Run `uv sync` (it is a core dependency) |
| `harness binary ... does not run in the task container` | Wrong `--harness-platform`, musl vs glibc image, or a non-Linux container |
| `Cannot create /opt/harness` | The image user cannot write `/opt`; pass `--harness-install-dir /tmp/harness-dist` |
| Upstream 400 / unsupported params | The backend must accept Chat-Completions with function tools; try `--harness-model` pinning, `--harness-bare`, or disable provider betas (`--harness-header` / LLM-config headers) |
| Proxy unreachable from container | Use the default Docker bridge; `--rollout-bind-host` can pin the host interface |
| Empty patch / zero captures | Harness never reached the model (check `logs/*.harness.jsonl` and `logs/*.rollout-manager.log`); the instance is scored unresolved |

## Adding another harness

1. Implement an `install`/`run` adapter (see
   `benchmarks/utils/harnesses/claude_code.py`) that installs a portable
   distribution into the container and runs the CLI headless with its model
   endpoint pointed at `proxy_url`.
2. Register it in `benchmarks/utils/harnesses/base.py` (`_FACTORIES`) and add the
   name to `SUPPORTED_HARNESSES`.
3. Add a distribution source in `benchmarks/utils/harnesses/distribution.py` if
   the harness does not use the Claude Code release feed.

The per-instance pipeline (`runner.py`), capture proxy lifecycle (`rollout.py`),
Docker plumbing (`workspace.py`), patch extraction, judging and history dumping
are harness-independent.
