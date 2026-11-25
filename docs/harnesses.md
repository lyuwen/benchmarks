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
| `--harness-permission-mode` | unset | Claude Code permission mode: `auto`, `bypassPermissions`, `acceptEdits`, `default`, `plan` |
| `--harness-include-auxiliary` | off | Train auxiliary requests (auto-mode classifier, titles) instead of masking them (`--exclude-request-classes ''`) |
| `--harness-auto-mode-model` | main agent model | Optional model ID for auto-mode classifier requests |
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

### Sampling / request parameters

Harness CLIs do not necessarily send the sampling parameters a benchmark wants
(and Claude Code may use its own defaults), so the pipeline forwards the LLM
config's explicit generation settings to the rollout manager. By default these
come from the LLM config:

- top-level `temperature`, `top_p`, `top_k`, `seed`, `reasoning_effort`,
  `max_tokens` / `max_output_tokens` (mapped to `max_tokens`), and
  `presence_penalty` / `frequency_penalty` / `repetition_penalty`;
- the same keys under `litellm_params` as fallbacks;
- provider `extra_body` entries (top-level or under `litellm_params`), e.g.
  `{"thinking": {"type": "enabled"}}`;
- structural fields (`messages`, `stream`, `stream_options`, `tools`) are never
  forwarded as parameters.

The rollout manager overlays them onto **every** upstream Chat-Completions
payload after protocol conversion (overriding values the harness sent), and the
effective per-turn `request` in the trajectory export shows them. `--model`
still wins over a `model` request-param.

Override or extend on the command line:

```bash
# single key/value pairs, JSON-parsed when possible (repeatable)
--request-param temperature=0.2 --request-param reasoning_effort=high
# or a JSON object
--request-params '{"temperature": 0.2, "top_p": 0.95, "thinking": {"type": "enabled"}}'
# ignore the LLM config's sampling values (CLI values still apply)
--no-llm-request-params
```

The proxy side lives in the vendored `vendor/rollout-manager` package (pinned
upstream snapshot + the `--request-params` overlay; see its `VENDORED.md`).

### Permission modes, auto mode and auxiliary traffic

By default the harness runs with bypass permissions (`--dangerously-skip-permissions`
plus `IS_SANDBOX=1`). To run Claude Code's **auto mode** — where a second model,
the classifier, reviews each action — use:

```bash
uv run swebench-infer-harnesses .llm_config/model.json \
  --harness-permission-mode auto --judge swebench
```

How this is wired:

- The classifier uses the **same model as the main agent**: no extra model is
  configured by default, and the rollout manager forces the served backend model
  on every upstream request anyway. `--harness-auto-mode-model <id>` pins a
  different classifier model ID only if you need one.
- `CLAUDE_CODE_AUTO_MODE_SERVER=0` forces client-side classifier requests so
  they go through the capture proxy (server-side review is unavailable behind a
  custom gateway and would not be visible to the capture).
- Classifier calls are Claude Code **`auxiliary`** requests: the rollout manager
  records them as their own trajectories/turns with `request_class: auxiliary`.
  They are masked out of training by default (`--exclude-request-classes
  auxiliary`): they stay in the export (`excluded_turns`,
  `exclude_from_training=true`, `loss_mask=0`) but `harness-flatten-histories`
  drops them unless you pass `--include-excluded`.

How to read the masks in an auto-mode history dump:

- The **main conversation** is split into fragment trajectories by the capture
  tree; each fragment trains only the turns it owns (`loss_mask=1` on those
  assistant messages) while the shared prefix is context (`loss_mask=0`). A
  29-request auto-mode run, for example, had 6 main fragments with 17 trained
  turns in total and the final answer trained in the last fragment.
- The **auxiliary** (security-monitor/classifier) calls are the only trajectories
  whose `loss_mask` is entirely zero — by design, not a missing assistant
  target. Their assistant verdicts (`<block>no`, …) are masked at capture time.

To train auxiliary calls as well, either capture differently or post-process:

```bash
# capture-side: do not mask any request class (rerun needed)
uv run swebench-infer-harnesses ... --harness-permission-mode auto --harness-include-auxiliary
# equivalent explicit form
uv run swebench-infer-harnesses ... --harness-permission-mode auto --exclude-request-classes ""

# post-hoc: recompute loss masks for an existing capture
uv run harness-flatten-histories histories.jsonl -o flat.jsonl --train-excluded
```

Verified against Claude Code 2.1.331: with auto mode and a network Bash command
that the classifier reviews, one run produced 12 requests — 2 `main` turns and
10 `auxiliary` classifier calls — and the export contained the 10 auxiliary
turns plus the main trajectory. Actions auto-approved by mode rules (for example
simple file edits in the working directory) may not trigger a classifier call.

**Session titles**: a `claude -p` run started from a shell/script does not
generate a session title, so these entry points see no naming request. If
Claude Code does emit an auxiliary request (titles, summaries, classifiers — in
this or another harness), it is captured exactly like the classifier traffic and
follows the same `--exclude-request-classes` rules. To make auxiliary traffic
trainable end-to-end, run with `--exclude-request-classes ""` (capture nothing
as excluded) and flatten without `--exclude` filtering.

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
5. The harness stream-json log is streamed line-by-line to the output directory
   (line-buffered, so `tail -f` shows events live while the run is in progress;
   stderr is interleaved); the proxy is finalized (`POST /rollout/finish`) and
   the export is validated.
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
| `tools` | OpenAI function-tool list resolved for the main trajectory (directly usable by a trainer) |
| `tool_sets` | deduplicated raw tool-list variants seen in the session (provenance; per-turn mapping lives in the export's `turns[].tool_set`) |
| `trajectories`, `request_log`, `stats` | the rollout export's trajectory data |
| `test_result.git_patch` | candidate patch (also in `output.jsonl`) |
| `evaluation` | judge verdict (`true`/`false`/`null`), when a judge ran |
| `run` | harness exit code, duration, timeout flag, stream-json `result` event |
| `rollout_export` | relative path to the trajectory export |
| `repository` | repo path, base commit, ignored pre-existing untracked files |
| `warnings` | non-fatal harness/proxy problems (e.g. no captures) |

## Building training data: flattening histories

`harness-flatten-histories` turns per-instance history dumps into JSONL records
that carry **OpenAI Chat-Completions `messages` and `tools`** directly.

Collect the history files first (a real JSONL file, or plain `cat` of the
pretty-printed JSON files — the reader handles both):

```bash
# either real JSONL ...
jq -c . eval_outputs/.../harnesses/claude-code/*.history.json > histories.jsonl
# ... or a concatenation of the pretty-printed documents
cat eval_outputs/.../harnesses/claude-code/*.history.json > histories.jsonl

uv run harness-flatten-histories histories.jsonl -o flat.jsonl
```

Two output shapes:

```bash
# default: one loss-masked record per captured trajectory fragment
uv run harness-flatten-histories histories.jsonl -o flat-fragments.jsonl

# one linear record per history row (the 6 fragments of a fragmented Claude Code
# session become a single 25-message trajectory)
uv run harness-flatten-histories histories.jsonl -o flat-stitched.jsonl --mode stitch

# same, but restore the model's original tool-call arguments where Claude Code
# normalized them before replaying history (e.g. stripped "cd /testbed &&")
uv run harness-flatten-histories histories.jsonl --mode stitch --stitch-verbatim

# split samples whenever the harness changed its tool list mid-trajectory
# (one record per contiguous tool-set phase, context duplicated)
uv run harness-flatten-histories histories.jsonl -o flat-tools.jsonl --split-tool-sets

# train already-masked auxiliary calls (auto-mode classifier/security monitor):
# keeps excluded trajectories and recomputes loss_mask so assistant messages are
# targets (implies --include-excluded)
uv run harness-flatten-histories histories.jsonl -o flat-aux.jsonl --train-excluded

# keep assistant content exactly as the API returned it (do not re-attach a
# single configured stop sequence, e.g. the auto-mode classifier's </block>)
uv run harness-flatten-histories histories.jsonl --include-excluded \
  --no-restore-stop-sequences

# fail on format/consistency problems (useful in CI)
uv run harness-flatten-histories histories.jsonl --strict
```

Each output record contains at least:

| Key | Meaning |
| --- | --- |
| `messages` | strict OpenAI Chat-Completions messages (`role`, `content`, assistant `tool_calls` with JSON-string `function.arguments`, `tool_call_id`) |
| `tools` | one OpenAI function-tool list, resolved from the trajectory's `tool_sets`/`turn.tool_set`; `[]` for requests that carried no tools (auto-mode classifier/security monitor) |
| `loss_mask`, `reasoning_content` | message-aligned arrays from the capture (`reasoning_content` is kept out of `messages` by design) |
| `loss_mask` sum / `trained_turns` | which turns of the record are training targets |
| `instance_id`, `attempt`, `harness`, `harness_version`, `model`, `instruction` | provenance |
| `test_result.git_patch`, `evaluation`, `rollout_export`, `stats` | patch, judge verdict, raw capture path |
| `trajectory_id`, `parent_trajectory_id`, `path_turn_indices`, `turn_indices` | tree position / request-log order (per-fragment mode) |
| `fragmented`, `num_fragments`, `trajectory_ids` | fragment count (stitch mode) |
| `restored_stop_sequences` | present when a single configured stop sequence was re-attached to a reply (see below) |

Notes:

- The capture is one fragment per trajectory leaf; a single linear Claude Code
  conversation can produce several fragments because the client normalizes
  replayed tool inputs (Bash `cd <cwd> &&` stripping, Edit defaults). Use
  `--mode stitch` (optionally `--stitch-verbatim`) for one trajectory per
  session, or keep fragments as separate loss-masked samples.
- Tool lists are per request: the history dump keeps deduplicated variants in
  `trajectories[].tool_sets` with a `turns[].tool_set` index. The flattened
  `tools` field is the variant used by the trajectory's last turn (falling back
  to the first variant), or `[]` when the request was tool-less. The root-level
  `tools` in `*.history.json` is already the resolved list for the main
  trajectory; `tool_sets` keeps all variants for provenance (older dumps without
  per-trajectory sets may still expose the legacy list-of-lists `tools`).
- `--split-tool-sets` handles trajectories whose tool list changed mid-way: the
  capture intentionally keeps them as one trajectory (tools are not part of the
  message-prefix identity), so the flattener emits one record per contiguous
  tool-set phase. Each record has `source: "trajectory-split"`, the phase's
  `tools`/`tool_set_index`, duplicated preceding context, only the phase's turns
  as training targets, and `split_index`/`split_count`/
  `split_from_trajectory_id` for reassembly.
- `tool.message` ids are checked against preceding assistant calls; trailing
  assistant tool calls without results are allowed (that continuation lives in
  the next fragment).
- `--include-excluded` keeps trajectories masked out of training
  (`exclude_from_training`, e.g. auxiliary calls); they are dropped by default.
- `--train-excluded` does the same but recomputes `loss_mask` from the messages
  (assistant = 1) and sets `loss_mask_recomputed: true`, for using auxiliary
  turns as training data without re-capturing.
- **Stop sequences are re-attached.** Chat-Completions and the Anthropic
  Messages API strip the matched stop sequence from the returned content.
  Claude Code's auto-mode classifier sends `stop: ["</block>"]` and gets back
  `<block>no`, so training on that reply would omit the closing token the model
  actually generated. The flattener appends a stop sequence back when the turn
  (a) configured exactly one string stop sequence, (b) finished with
  `finish_reason == "stop"`, and (c) the assistant reply is non-empty text
  without tool calls and does not already end with it. Each change is recorded
  in `restored_stop_sequences: [{message_index, turn_index, stop_sequence}]`.
  Multiple stop sequences are ambiguous (the API does not report which matched)
  and are left verbatim; use `--no-restore-stop-sequences` to disable the
  normalization entirely (the raw capture and `*.history.json` are never
  modified — this is a flatten-time, training-oriented change).
- A summary (`histories=… emitted=… warnings=…`) is printed to stderr; warnings
  never change stdout, which is pure JSONL.

## Streamlining fragmented histories

Claude Code replays the conversation on every request but normalizes the replayed
tool arguments first (strips a leading `cd <cwd> &&` from Bash, adds
`replace_all: false` and trims trailing whitespace in Edit). Those replays no
longer match the stored assistant responses, so one logical conversation can be
exported as many prefix fragments (6 fragments for a 23-request run). The
`harness-streamline-histories` tool merges them **in the history JSON**, next to
the flattening step:

- fragments are grouped when one message list is a prefix of the other modulo
  exactly those replay normalizations (Bash `cd` prefix, Edit `replace_all` and
  trailing whitespace); this covers the main conversation, sub-agents and other
  side conversations;
- a merged trajectory reuses the longest fragment's messages, restores the
  tool-call arguments the model originally generated (recovered from the
  captured responses, so the `cd <cwd> &&` prefix is back), unions the fragments'
  trained turns into `loss_mask`, re-interns `tool_sets`, and records the merge
  under `streamlined`;
- auxiliary requests (auto-mode classifier/security monitor) and
  compaction/summarization trajectories are kept unchanged (never merged);
- the root `messages`/`tools` come from the latest main-agent conversation
  trajectory (auxiliary and compaction trajectories are excluded);
- divergent conversation branches that are not prefixes of the chosen canonical
  fragment are kept as separate trajectories (no data is silently dropped);
- output is written next to the input as `<id>.history.streamlined.json`; the
  original history is never modified and existing outputs are skipped unless
  `--overwrite` is passed.

```bash
# one claude-code dir
uv run harness-streamline-histories --dir \
  eval_outputs/.../harnesses/claude-code

# recursively discover directories for a job-array list
uv run harness-streamline-histories --root /path/to/run --list > dirs.txt

# dry-run / include per-attempt histories
uv run harness-streamline-histories --dir ... --dry-run
uv run harness-streamline-histories --dir ... --include-attempts
```

### Distributed SLURM job array

`scripts/streamline_histories_slurm.sh` processes one job directory per array
task. It is meant to run with the **run root** (the directory containing the
`{i}` job directories) as its working directory; `SLURM_ARRAY_TASK_ID` selects
`{i}` (zero-padded names like `00000` are handled, otherwise the i-th sorted
sub-directory is used):

```bash
cd logs/dsv41f1017-high-synth-cc-swerebenchv2-osr-all-run2
sbatch --array=0-$(($(find . -maxdepth 1 -mindepth 1 -type d | wc -l) - 1)) \
  /path/to/repo/scripts/streamline_histories_slurm.sh          # extra flags OK

# or one task per precomputed claude-code directory
uv run harness-streamline-histories --root "$PWD" --list > dirs.txt
sbatch --array=0-$(($(wc -l < dirs.txt) - 1)) \
  /path/to/repo/scripts/streamline_histories_slurm.sh dirs.txt
```

The launcher forwards extra arguments (`--overwrite`, `--dry-run`, ...), runs the
module via `uv run`, and falls back to the active Python interpreter when `uv`
is not on `PATH`. Run `uv sync` once on the cluster so the project environment
is available; the job itself needs no GPU and only reads/writes the run tree.

Reference effect: the `unidata__netcdf-c-1692` history went from 6 fragment
trajectories (11 MB) to 1 merged trajectory with 23 trained turns (4.4 MB), with
22 restored tool-call messages and the root messages taken from the last (23rd)
trajectory.

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

## Debugging what the manager sends upstream

The **effective request body** is always captured per turn in the trajectory
export (`turns[].request`), including prompt messages, tools and the forced
sampling parameters. To also inspect what the proxy actually puts on the wire
(headers + body) in its own log, enable upstream request logging:

```bash
uv run swebench-infer-harnesses .llm_config/model.json \
  --rollout-log-upstream full \
  --rollout-log-upstream-max-chars 50000      # 0 = unlimited (default 20000)
```

- `--rollout-log-upstream summary` logs method/URL, model, message/tool counts,
  header *names* and body keys (cheap, no prompt content).
- `--rollout-log-upstream full` additionally logs the redacted headers and the
  JSON request body. Credentials (`Authorization`, `x-api-key`, `api-key`,
  cookies, ...) are always replaced with `<redacted>`; the body can be truncated
  with `--rollout-log-upstream-max-chars`.

Lines appear in
`<eval_output_dir>/logs/<instance>.attempt<N>.rollout-manager.log` as
`upstream request: POST ...` entries. Note that the logged body contains prompt
and tool-result content, so treat these logs with the same care as the
trajectory exports.

## Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| `Cannot use --dangerously-skip-permissions as root` | The adapter sets `IS_SANDBOX=1`; if you removed it, restore it or run as non-root |
| `rollout-manager` not installed | Run `uv sync` (it is a core dependency) |
| `harness binary ... does not run in the task container` | Wrong `--harness-platform`, musl vs glibc image, or a non-Linux container |
| `Cannot create /opt/harness` | The image user cannot write `/opt`; pass `--harness-install-dir /tmp/harness-dist` |
| Upstream 400 / unsupported params | The backend must accept Chat-Completions with function tools; try `--harness-model` pinning, `--harness-bare`, or disable provider betas (`--harness-header` / LLM-config headers) |
| `rollout-manager did not become healthy ... (timed out)` while its log shows it running | The host-side control request was routed through an `http_proxy`/`all_proxy` in your shell. Control traffic bypasses proxies (and the manager subprocess gets `NO_PROXY` for loopback), so update to the current revision; the error now prints a TCP probe and the proxy environment. If it persists, check the host firewall for the Docker bridge and `--rollout-bind-host` |
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
