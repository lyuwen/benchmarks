# Rollout Manager (agent rollout capture)

A standalone request adapter that captures **one agent session** and exports the
**series of trajectories** that session produced. It is designed to be launched
just before a harness starts and stopped when that harness finishes — one manager
process is one session, not a multi-tenant service.

```text
harness (Claude Code / Codex / OpenHands / any SDK)
      │  OpenAI Chat-Completions | Anthropic Messages | OpenAI Responses
      ▼
rollout manager (one process, one session)
      │  captures every request/response, assembles the trajectory series
      │  OpenAI Chat-Completions (JSON passthrough)
      ▼
model backend (SGLang, vLLM, external API)
```

Unlike `slime.agent.adapters`, this module **never assembles a chat template and
never tokenizes**. It forwards the chat request as JSON in OpenAI
Chat-Completions style; the model backend owns tokenization. What it keeps from
the slime agent stack is the useful half: format conversion from the Anthropic
Messages and OpenAI Responses APIs, and a trajectory manager that stitches
sequential requests together when one is the prefix of the other.

It has no `slime` imports and no third-party dependency beyond `aiohttp`. The
directory is a standalone Python project (`rollout_manager` package + its own
`pyproject.toml`), so it can be installed or copied out of this repository and
run on its own.

## Layout

```text
examples/rollout_manager/          # standalone project root
├── pyproject.toml                 # package metadata + console script + tool configs
├── README.md
├── rollout_manager/               # the installable package
│   ├── __init__.py
│   ├── __main__.py                # python -m rollout_manager
│   ├── server.py                  # request adapter, CLI, SSE rendering
│   ├── capture.py                 # single-session capture, export, purpose labels
│   ├── trajectory.py              # prefix tree -> trajectory series, masks, request log
│   └── convert.py                 # Anthropic Messages / OpenAI Responses <-> chat completions
└── tests/                         # CPU tests against the installed package
```

## Why a series of trajectories

An agent resends its whole conversation on every turn, so request N+1's message
list is usually request N's plus the assistant reply plus an observation. The
manager routes each request into a prefix tree: an extending request mounts only
its new tail, and a divergence starts a new branch.

A single session therefore produces several root-to-leaf **trajectories**:

* the main conversation;
* a **context compaction** (summary call and post-compaction continuation);
* **sub-agent** runs — a fresh system prompt starts a new root in the same tree;
* rewritten histories / retries — a fork at the divergence point.

The export is the ordered set of those trajectories. The `trajectories` array is
in tree order and is **not** the session timeline; `request_log` is the explicit
capture-ordered timeline.

## Request purpose and compaction

Compaction is **never inferred from request shape**. A tools-less request is just
a tools-less request. The manager recognises compaction only from Claude Code's
explicit gateway hint headers:

| header | meaning |
| --- | --- |
| `x-claude-code-request-class` | `main`, `subagent`, `workflow`, `compaction`, or `auxiliary` |
| `x-claude-code-compaction` | present on the summarization request; `auto`/`manual`/`reactive` |
| `x-claude-code-context-compacted` | present once on the first main request after a compaction |
| `x-claude-code-agent-id` / `-agent-type` / `-parent-agent-id` | sub-agent lineage |
| `x-claude-code-prompt-id` | groups requests serving one user prompt |
| `x-claude-code-session-id` | Claude Code's session id (kept separately from our export id) |

For a custom `ANTHROPIC_BASE_URL`, Claude Code sends these only when
`CLAUDE_CODE_GATEWAY_HINT_HEADERS=1` is set. Enable it for capture:

```json
{"env": {"CLAUDE_CODE_GATEWAY_HINT_HEADERS": "1"}}
```

How a signalled compaction is recorded:

1. **The compaction request carries the tools list.** It stays a normal
   continuation. Its turn is labelled `compaction=true`, and the trajectory gets
   `contains_compaction=true`, but it is not split.
2. **The compaction request omits the tools list.** It is recorded as its own
   trajectory: `kind="compaction"`, the full conversation in its prompt is
   context (`loss_mask=0`), and only the final summary response is trainable.

Requests with `request_class=auxiliary` (session titles, classifiers, security
monitors) are ordinary branches, not compactions. They are labelled per turn and
can be kept out of the training mask:

```bash
--exclude-request-classes auxiliary
```

Excluded turns still appear in the export (`excluded=true`, `loss_mask=0`); a
trajectory whose turns are all excluded gets `exclude_from_training=true`.

## Orchestration

The service is meant to be started and stopped by whatever launches the harness.
Install the standalone package once (from this directory), then use the console
script or `python -m rollout_manager`:

```bash
cd examples/rollout_manager
pip install -e .
```

Run the manager:

```bash
RUN_DIR=/tmp/agent-run-42
SESSION_ID=run-42
OUTPUT="$RUN_DIR/session.json"

mkdir -p "$RUN_DIR"
rollout-manager \
    --upstream http://127.0.0.1:30000/v1 \
    --output "$OUTPUT" \
    --session-id "$SESSION_ID" \
    --metadata '{"rollout_id": 42, "task": "swe-verified/1234"}' \
    --host 127.0.0.1 --port 30001 &
MANAGER_PID=$!

# Wait until the adapter is ready, then run the harness against it.
until curl -sf http://127.0.0.1:30001/healthz >/dev/null; do sleep 0.2; done
ANTHROPIC_BASE_URL=http://127.0.0.1:30001 \
ANTHROPIC_AUTH_TOKEN=unused \
    <harness command>

# Finalize explicitly (optional: SIGTERM also finalizes on shutdown), then stop.
curl -sf -X POST http://127.0.0.1:30001/rollout/finish >/dev/null
kill "$MANAGER_PID"
wait "$MANAGER_PID" 2>/dev/null || true
```

`--session-id` falls back to `$ROLLOUT_SESSION_ID`, then to a generated run id.
Backend routing uses a separate per-launch key, not the session id.

Agent APIs and base URLs:

| agent API | base URL |
| --- | --- |
| OpenAI Chat-Completions | `http://<host>:30001/v1` |
| Anthropic Messages | `http://<host>:30001` |
| OpenAI Responses | `http://<host>:30001/v1` |

The upstream may be SGLang (`http://127.0.0.1:30000/v1`), vLLM
(`http://127.0.0.1:8000/v1`) or any external OpenAI-compatible API
(`--upstream-api-key` and `--model` override are supported).

### Capturing a Claude Code session

Claude Code speaks the Anthropic Messages API, so point `ANTHROPIC_BASE_URL` at
this service — no `/v1` suffix. The upstream model must support tool calling
(serve it with the model's tool-call parser / chat template).

```bash
pip install -e "examples/rollout_manager[test]"

mkdir -p /tmp/cc-run
rollout-manager \
    --upstream http://127.0.0.1:30000/v1 \
    --output /tmp/cc-run/session.json \
    --session-id cc-001 \
    --model your-served-model \
    --exclude-request-classes auxiliary \
    --host 127.0.0.1 --port 30001 &
MANAGER_PID=$!
until curl -sf http://127.0.0.1:30001/healthz >/dev/null; do sleep 0.2; done

NO_PROXY=127.0.0.1,localhost \
ANTHROPIC_BASE_URL=http://127.0.0.1:30001 \
ANTHROPIC_AUTH_TOKEN=capture \
ANTHROPIC_MODEL=your-served-model \
CLAUDE_CODE_GATEWAY_HINT_HEADERS=1 \
claude -p "Fix the failing test" --dangerously-skip-permissions --output-format stream-json --verbose

curl -s -X POST http://127.0.0.1:30001/rollout/finish
kill "$MANAGER_PID"
```

Notes:

* `CLAUDE_CODE_GATEWAY_HINT_HEADERS=1` is required for compaction recognition and
  auxiliary/purpose labels on a custom base URL.
* `--model` overrides the model name on every upstream request, so Claude Code's
  aliases all hit the served model.
* `ANTHROPIC_AUTH_TOKEN` sends `Authorization: Bearer <token>`; any value works
  (the proxy does not validate auth). Client headers are forwarded upstream
  unless `--upstream-api-key` overrides `Authorization`.
* Claude Code streams and calls `/v1/messages/count_tokens`; both are supported.
  Token counting is a character-based estimate.
* One manager process is one session; sub-agents and background requests share
  the session and appear as extra trajectories.

#### Overriding an existing Claude Code setup

`--settings` is the highest non-managed settings source and overrides user,
project, and project-local settings for one session:

```json /tmp/capture-settings.json
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://127.0.0.1:30001",
    "ANTHROPIC_AUTH_TOKEN": "capture",
    "CLAUDE_CODE_GATEWAY_HINT_HEADERS": "1",
    "CLAUDE_CODE_USE_BEDROCK": "",
    "CLAUDE_CODE_USE_VERTEX": "",
    "CLAUDE_CODE_USE_FOUNDRY": ""
  }
}
```

```bash
env -u ANTHROPIC_API_KEY -u ANTHROPIC_AUTH_TOKEN -u ANTHROPIC_BASE_URL \
    command claude -p "Fix the failing test" \
    --settings /tmp/capture-settings.json \
    --model your-served-model \
    --dangerously-skip-permissions
```

`--model` overrides both the `model` setting and `ANTHROPIC_MODEL`; `command claude`
bypasses shell aliases. Organization-managed settings cannot be overridden; check
`/status` or `claude doctor`.

## Export format

One JSON document per session at `--output`. Schema version is **2**.

```jsonc
{
  "type": "rollout_session",
  "version": 2,
  "session_id": "run-42",
  "created_at": 1766380000.1,
  "updated_at": 1766380012.4,
  "finished": true,
  "finish_reason": "explicit",
  "model": "qwen",
  "metadata": {"rollout_id": 42},
  "stats": {
    "requests": 3,
    "turns": 3,
    "trajectories": 3,
    "compaction_requests": 1,
    "excluded_turns": 0,
    "request_classes": {"main": 2, "compaction": 1}
  },
  "request_log": [
    {
      "turn_index": 1,
      "trajectory_id": "run-42:1",
      "protocol": "chat",
      "finish_reason": "stop",
      "model": "qwen",
      "response_id": "chatcmpl-...",
      "message_count": 1,
      "timestamp": 1766380001.7,
      "compaction": false,
      "standalone": false,
      "request_class": "main",
      "prompt_id": "prompt-1"
    }
  ],
  "trajectories": [
    {
      "trajectory_id": "run-42:2",
      "trajectory_index": 0,
      "first_turn_index": 1,
      "leaf_turn_index": 2,
      "kind": "conversation",          // "compaction" for a split summary call
      "contains_compaction": false,
      "exclude_from_training": false,
      "parent_trajectory_id": null,
      "branch_message_index": null,
      "branch_turn_index": null,
      "path_turn_indices": [1, 2],
      "messages": [
        // strict OpenAI Chat-Completions messages
        {"role": "user", "content": "run it"},
        {
          "role": "assistant",
          "content": "",
          "tool_calls": [
            {
              "id": "call_abc",
              "type": "function",
              "function": {"name": "bash", "arguments": "{\"cmd\": \"ls\"}"}
            }
          ]
        },
        {"role": "tool", "tool_call_id": "call_abc", "content": "42"},
        {"role": "assistant", "content": "done"}
      ],
      "loss_mask": [0, 1, 0, 1],       // aligned with messages
      "reasoning_content": [null, null, null, null],  // aligned; provider-specific
      "tool_sets": [ /* deduplicated OpenAI function-tool lists, incl. strict */ ],
      "turns": [
        {
          "message_index": 1,
          "turn_index": 1,
          "trained": true,
          "excluded": false,
          "compaction": false,
          "standalone": false,
          "request_class": "main",
          "compaction_trigger": null,
          "context_compacted": null,
          "agent_id": null,
          "agent_type": null,
          "parent_agent_id": null,
          "prompt_id": "prompt-1",
          "client_session_id": null,
          "finish_reason": "tool_calls",
          "response_id": "chatcmpl-...",
          "model": "qwen",
          "protocol": "chat",
          "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
          "tool_set": 0,
          "timestamp": 1766380001.7,
          "latency_ms": 812.4,
          "request": {                 // exact effective upstream request
            "model": "qwen",
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 4096,
            "stream": false,
            "tool_choice": "auto",
            "tools": [/* ... */],
            "messages": [/* exact prompt sent upstream, native roles/content */],
            "message_count": 1
          },
          "response": {                // original upstream assistant message
            "role": "assistant",
            "content": null,
            "tool_calls": [
              {
                "id": "call_abc",
                "type": "function",
                "function": {"name": "bash", "arguments": "{\"cmd\": \"ls\"}"}
              }
            ]
          }
        }
      ]
    }
  ]
}
```

Compatibility notes:

* `messages` are strict OpenAI Chat-Completions messages. Tool-call ids,
  `tool_call_id`, and JSON-encoded `function.arguments` strings are preserved —
  including truncated/invalid argument strings, which are never rewritten.
  Prefix matching compares role/content plus tool-call ids, so two identical calls
  with different ids cannot mount each other's results. Native roles (including
  `developer`) and content shapes are preserved.
* `loss_mask` and provider-specific `reasoning_content` are **not** fields on the
  messages; they are separate arrays aligned with `messages`.
* The effective upstream request (`turns[].request`, including the exact
  prompt `messages` sent upstream plus `message_count`) and the original upstream
  response (`turns[].response`) are retained for faithful replay. Native
  Chat-Completions roles (`developer`, etc.) and content shapes are preserved in
  both `messages` and `turns[].request.messages`; normalization happens only in
  `message_identity()` for history matching. Because the prompt is retained per
  turn, v2 exports are larger than v1.
* `request_log` is the ordered session timeline. `trajectories` is tree order and
  is not chronological.
* `turns[].trained` agrees with `loss_mask`; a shared prefix is trained by the
  first leaf that exports it and is context in siblings.

### Compaction examples

**Scenario A — signalled compaction that carries tools** (not split):

```text
trajectory 0  kind=conversation  contains_compaction=true
              path_turn_indices=[1, 2]  loss_mask=[0, 1, 0, 1]
              (the second turn is labelled compaction=true)

trajectory 1  kind=conversation  path_turn_indices=[3]  loss_mask=[0, 0, 1]
```

**Scenario B — signalled compaction without tools** (split and marked):

```text
trajectory 0  kind=conversation  path_turn_indices=[1]      loss_mask=[0, 1]

trajectory 1  kind=compaction    path_turn_indices=[2]
              parent=trajectory 0 @ turn 1
              loss_mask=[0, 0, 0, 1]   # only the summary response is trained

trajectory 2  kind=conversation  path_turn_indices=[3]      loss_mask=[0, 0, 1]
```

An `auxiliary` request (title/classifier/security monitor) never triggers a
compaction split, even when it is tools-less.

## Write lifecycle

| event | write |
| --- | --- |
| after every captured turn | full export with `"finished": false` (default; disable with `--no-snapshot-every-turn`) |
| `POST /rollout/finish` | drain in-flight turns, final export with `"finished": true`, `"finish_reason": "explicit"` |
| service shutdown (SIGINT/SIGTERM) | final export with `"finish_reason": "shutdown"` if not already finished |

Writes are atomic (`.<name>.tmp` + `os.replace`). `finish` is idempotent.

## Endpoints

| endpoint | purpose |
| --- | --- |
| `POST /rollout/finish` | drain in-flight turns, write the final export, return its summary. Optional body: `{"metadata": {...}}`. |
| `GET /rollout/status` | session state: `finished`, `requests`, `turns`, `trajectories`, `compaction_requests`, `excluded_turns`, `routing_key`, `output`. |
| `GET /healthz` | readiness probe. |
| `GET /v1/models` | proxied to the upstream. |

## CLI

```text
--upstream URL              OpenAI-compatible base URL (required)
--upstream-api-key KEY      Bearer token for the upstream
--model NAME                Force a model name on upstream requests
--output PATH               Export path (default <output-dir>/<session-id>.json)
--output-dir DIR            Directory for the default export path
--session-id ID             Session id (default $ROLLOUT_SESSION_ID, else generated)
--metadata JSON             Session metadata stored in the export
--exclude-request-classes   Comma-separated request classes masked out of training
                            (e.g. auxiliary); they remain in the export
--host HOST --port PORT     Bind address (default 0.0.0.0:30001)
--request-timeout-sec SEC   Upstream read timeout (default 900)
--finish-wait-sec SEC       How long finish waits for in-flight turns (30)
--max-turns N               Optional session turn cap (returns 429 past it)
--max-connections N         Upstream connection pool size (512)
--chars-per-token N         Token estimate divisor for count_tokens (4.0)
--response-cache-size N     Responses kept for previous_response_id chaining (256)
--no-snapshot-every-turn    Only write the export on finish
--log-level LEVEL           Logging level (INFO)
```

## Protocol notes

* **Client request headers are forwarded transparently** to the upstream,
  including `Authorization`, `X-Api-Key`, cookies, Claude Code
  `x-claude-code-*` headers, and tracing headers. Only hop-by-hop/transport
  headers the proxy must own are replaced (`Host`, `Content-Length`,
  `Transfer-Encoding`, `Connection`, `Keep-Alive`, `TE`, `Trailer`, `Upgrade`,
  `Expect`, `Proxy-*`, `Accept-Encoding`) along with `Accept`/`Content-Type`. An
  explicit `--upstream-api-key` overrides a forwarded `Authorization`.
* The proxy sets `X-SMG-Routing-Key` and `X-Session-ID` on every upstream request
  (including `/v1/models`) to a **fresh routing key generated at service launch**
  (`rollout-<hex>`). Client-supplied values are replaced; the key is logged and
  returned by `/rollout/status` and `/rollout/finish`.
* **Chat-Completions** requests are forwarded untouched except `stream` is forced
  to `false` (and `stream_options` dropped), so the complete reply is captured;
  if the client asked for streaming, the reply is re-rendered as SSE afterwards.
* **Anthropic Messages** requests are converted to Chat-Completions and replies
  are converted back. Anthropic-only fields with no chat equivalent are consumed;
  unknown top-level/`extra_body` extensions are forwarded as-is.
* **OpenAI Responses** requests are converted to Chat-Completions (`instructions`
  → system, `input` items grouped, flat tools mapped). `reasoning.effort` becomes
  `reasoning_effort`, `text.format` becomes `response_format`, and `store`,
  `metadata`, `user`, `service_tier`, `prompt_cache_key` are forwarded. Both
  full-history `input` and `previous_response_id` chaining are supported.
* Streaming to the client is emitted after the turn is realized upstream.

## Request body parameters

**Chat-Completions** is forwarded wholesale: `temperature`, `top_p`, `top_k`,
`max_tokens`/`max_completion_tokens`, `reasoning_effort`, `seed`, penalties,
`stop`, `response_format`, `tools`, `logprobs`, provider extensions all pass
through. Only `stream` (forced `false`), `stream_options` (removed) and `--model`
override differ. A literal `extra_body` object is flattened.

**Anthropic Messages**: `temperature`, `top_p`, `top_k`, `max_tokens`,
`stop_sequences`→`stop`, `tools`, `tool_choice` are mapped; other top-level /
`extra_body` extension fields are forwarded.

**OpenAI Responses**: mapped fields are listed in Protocol notes; other
top-level / `extra_body` extension fields are forwarded.

The effective upstream request is retained per turn in `turns[].request`, and the
original upstream assistant message in `turns[].response`.

## Turning a capture into training rows

The export is tokenizer-free. Render `trajectory.messages` with the training chat
template; `messages` is already strict OpenAI format. Then:

1. Train only assistant messages whose aligned `trajectory.loss_mask[i] == 1`
   (content, reasoning, and tool-call argument tokens). A tool result's
   `tool_call_id` now always matches an assistant tool-call id in the same
   trajectory; duplicate call shapes with different ids stay on separate
   branches.
2. `trajectory.reasoning_content[i]` holds provider reasoning for message `i`;
   it is not part of the strict message.
3. Skip `trajectory.exclude_from_training` trajectories, and/or use
   `turns[].request_class` plus `--exclude-request-classes` to keep auxiliary
   classifier/title/security-monitor calls out of the agent-behavior dataset.
4. Use `request_log` when session timeline matters; `trajectories` is tree order.

Example inspection:

```bash
jq '.stats' session.json
jq -r '.request_log[] | "\(.turn_index)\t\(.request_class)\t\(.trajectory_id)"' session.json
jq -c '.trajectories[] | {kind, path_turn_indices, loss_mask}' session.json
```

## Tests

The tests are CPU-only and import the installed package. Install the project in
editable mode once, then run pytest:

```bash
cd examples/rollout_manager
pip install -e ".[test]"
python -m pytest -q
```

They cover strict OpenAI message schema/replay equality, tool IDs and argument
string preservation (including truncation), explicit compaction signalling,
auxiliary labeling/exclusion, request/response retention, the ordered request
log, deep non-recursive history traversal, protocol conversions, streaming,
headers, routing keys and body parameters.

## Limitations

* Schema v1 captures (including an existing `rollout-captures/*.json` produced by
  older builds) are not migrated; they lack strict tool ids/strings and the v2
  mask/request/log fields. Re-capture with this version.
* Compaction recognition and request-purpose labels require the harness to send
  the `x-claude-code-*` gateway hint headers (`CLAUDE_CODE_GATEWAY_HINT_HEADERS=1`
  for Claude Code on a custom base URL). Without them there is no heuristic
  fallback.
* Text-only: image/file content is flattened to a marker.
* Responses host tool types (web search, computer use, ...) are dropped; only
  function tools are captured.
* `count_tokens` is a character-based estimate.
* Native Chat-Completions content shapes are preserved for messages captured
  from the chat wire. Protocol conversions (Anthropic/Responses) still render
  content as text, and multimodal blocks are flattened for converted requests.
* A tool result with no `tool_call_id` on the wire keeps an empty string.
