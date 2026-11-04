# Condenser wiring, tool-result pruning, and history export

**Status:** implemented on `feat/condenser-wiring` and applied to
`benchmarks/swebench/run_infer.py`, `benchmarks/scaleswe/run_infer.py` and its
`run_infer-no-think-no-task.py` variant, `benchmarks/swerebenchv2/run_infer.py` and
its no-think variant, and `benchmarks/z021data/run_infer.py` and its no-think
variant. The remaining `run_infer*.py` tasks still use the historical pipeline.
The SDK side is pinned in `vendor/software-agent-sdk`
(`feat/replay-summarizing-condenser`, which merges `feat/tool-result-pruner`).

## Defaults

Every feature is opt-in. With no flags and no environment variables set, model
behavior and the exported `messages` list are identical to before this branch.

| Feature | Switch | Default |
|---|---|---|
| Condenser | `--condenser {none,legacy,replay}` | `none` (no condensation) |
| Condenser config | `--condenser-config '<inline-json>'` | pydantic defaults (replay) / `{"max_size": 80, "keep_first": 4}` (legacy) |
| Tool-result pruning | `OH_TOOL_RESULT_PRUNER=1` plus optional budgets | off |
| Smart fake user | `--smart-fake-user` | off (plain fake user response) |
| Dev SDK bind mount | `--bind-dev-sdk` | off |

## Enabling

Condensation with a manual token trigger (80K tokens, keep 20K verbatim):

```bash
uv run python benchmarks/swebench/run_infer.py ... \
    --condenser replay \
    --condenser-config '{"max_tokens": 80000, "retain_tokens": 20000, "retain_ratio": null}'
```

Or a window ratio (`threshold_ratio * agent LLM max_input_tokens`):

```bash
--condenser replay --condenser-config '{"threshold_ratio": 0.6104, "retain_ratio": 0.15}'
```

`retain_tokens` requires `retain_ratio: null` (they are mutually exclusive), and
`retain_ratio` must be less than `threshold_ratio`. The legacy condenser also
triggers on event count, so give it a large `max_size` if you only want the token
trigger:

```bash
--condenser legacy --condenser-config '{"max_tokens": 80000, "max_size": 10000, "keep_first": 4}'
```

Tool-result pruning (character budgets; `OH_*` variables are auto-forwarded into
docker/flex workspaces, and must be listed in `forward_env` for remote workspaces):

```bash
export OH_TOOL_RESULT_PRUNER=1
export OH_TOOL_RESULT_PRUNER_THRESHOLD_CHARS=8192
export OH_TOOL_RESULT_PRUNER_HEAD_CHARS=4096
export OH_TOOL_RESULT_PRUNER_TAIL_CHARS=1024
```

Smart fake user (classifies the agent's last message with a non-streaming LLM copy;
falls back to the plain fake response on classifier error):

```bash
--smart-fake-user
```

## History export contract

`{instance.id}.history.json` contains the historical keys (`instance_id`,
`messages`, `model`, `tools`, `temperature`, `top_p`, `test_result`, and
`evaluation` when a judge ran) plus:

- `trajectories` — always present;
- `pruning` — only present when tool-result pruning was active in that run.

### `messages` (full transcript)

The historical full event-dump transcript: every LLM-convertible event, including
events that a condensation later forgot. Two annotations are added only when the
corresponding feature is on:

- **Pruning on:** over-budget tool messages become
  `{"role": "tool", ..., "content": "<pruned head+marker+tail>",
  "original_content": "<full original text>"}`.
- **Condensation on:** at each `Condensation` event's position in the append-only
  log, a marker entry is inserted:

  ```json
  {
    "role": "context",
    "content": "<summary>",
    "kind": "Condensation",
    "id": "...",
    "timestamp": "...",
    "source": "environment",
    "summary_offset": 2,
    "forgotten_event_ids": ["..."],
    "llm_response_id": "..."
  }
  ```

  Pre-compaction events are not removed; the marker only annotates where
  compaction happened. `"role": "context"` is not an OpenAI role, so filter it out
  before feeding `messages` to an API client.

### `trajectories` (model views)

An ordered list alternating model-view segments and compaction records:

```text
segment 0 (before compaction 1)
compaction 1 (trigger events, summarizer input, summarizer output, result)
segment 1 (after compaction 1, up to compaction 2)
...
segment N (after the last compaction)
```

- A segment entry is `{"type": "trajectory", "messages": [...]}`: the accumulated
  model view at the end of the segment (earlier requests in the segment are
  prefixes of it), built through `View.from_events`, so forgetting, summary
  insertion, and unmatched-tool-call filtering are applied. In general it can
  differ from `messages` (which is the raw dump) even without condensation.
- A compaction entry is
  `{"type": "compaction", "trigger": "request"|"pressure", "trigger_events": [...],
  "request": [<summarizer input messages>], "response": <summarizer output message>,
  "result": {<Condensation event without the request/response messages>}}`.
  `request`/`response` are `null` on legacy logs written before the SDK recorded
  them. A trailing condensation request with no following condensation is appended
  as an entry with `request`/`response`/`result` set to `null`.
- With no condensation the list contains a single `trajectory` entry.
- The final model view is `trajectories[-1].messages`.

### `pruning`

Present only when pruning was active, describing the effective budgets resolved
from the recorded policy:

```json
{"enabled": true, "threshold_chars": 8192, "head_chars": 4096,
 "tail_chars": 1024, "marker": "\n\n[... tool result middle pruned ...]\n\n"}
```

### Raw events (`output.jsonl` → `history`)

- `SystemPromptEvent` always carries `tool_result_pruning`
  (`{"enabled": false}` when pruning is off). This is the policy the agent actually
  applied, recorded at conversation start, so the export cannot diverge from the
  container environment (which matters for remote workspaces).
- `Condensation` events carry the summarizer `request_messages` and
  `response_message` once a condensation runs; on runs without condensation the
  event never exists.

## Compatibility guarantees

- **Defaults are a no-op for model behavior.** No condenser, no pruning, no smart
  fake user, and the plain fake-user code path is behavior-preserving.
- **`messages` is identical to the historical dump by default** (covered by
  `test_build_messages_matches_plain_dump_without_features`).
- **File-level differences are additive:** `trajectories` is always added, `pruning`
  only when enabled, and raw `SystemPromptEvent`s gain the
  `tool_result_pruning` field. No existing key or message changes when the features
  are off.
- With features on, files are intentionally not byte-identical to historical runs.

## Operational notes

- **Streaming:** the condenser LLM copy and the smart fake-user classifier copy are
  forced to `stream=False` (auxiliary calls have no token callback). The agent LLM
  itself is untouched.
- **Timeouts:** the condenser copy inherits the agent LLM's `timeout` (default
  300s), `num_retries` (5, exponential backoff) and reasoning/thinking settings.
  Long compactions can therefore be slow; raise `"timeout"` in the LLM config if
  needed.
- **Classifier logging:** a failed classification logs a warning (stderr + the
  per-instance `.log` file) and falls back to the plain fake response; the
  unparseable-answer debug message is filtered out by the instance handler levels.
- **Corpus caveat:** keeping `original_content` and pre-compaction events makes
  files larger when the features are on (accepted by design).

## Migration from the inline export

Before this branch each `run_infer*.py` built the history dump inline:

```python
# old shape, still used by the not-yet-migrated tasks
convertible_events = [
    e for e in conversation.state.events if isinstance(e, LLMConvertibleEvent)
]
msgs = LLMConvertibleEvent.events_to_messages(convertible_events)
messages = [
    m.model_copy(update={"send_reasoning_content": True}).to_chat_dict()
    for m in msgs
]
dump_data = {"instance_id": instance.id, "messages": messages, ...}
```

`benchmarks/swebench/run_infer.py` now delegates to the shared helper:

```python
# new shape (swebench)
pruning_config = resolve_pruning_config(conversation.state.events)
messages = build_messages(conversation.state.events, pruning_config)
...
dump_data = {
    ...,
    "messages": messages,
    "trajectories": build_trajectories(conversation.state.events, pruning_config),
}
pruning = pruning_metadata(pruning_config)
if pruning is not None:
    dump_data["pruning"] = pruning
```

`build_messages` is a drop-in replacement for the inline loop: with pruning and
condensation off it produces exactly the inline output (covered by
`test_build_messages_matches_plain_dump_without_features`), while the
`trajectories` key is additive.

Adoption status in this branch:

| Task file | Export |
|---|---|
| `benchmarks/swebench/run_infer.py` | migrated to `benchmarks/utils/trajectory.py` |
| `benchmarks/scaleswe/run_infer.py` | migrated |
| `benchmarks/scaleswe/run_infer-no-think-no-task.py` | migrated |
| `benchmarks/swerebenchv2/run_infer.py` | migrated |
| `benchmarks/swerebenchv2/run_infer-no-think-no-task.py` | migrated |
| `benchmarks/z021data/run_infer.py` | migrated |
| `benchmarks/z021data/run_infer-no-think-no-task.py` | migrated |
| `benchmarks/scaleswe/run_infer-dsh.py` | inline (no fake-user path by design) |
| `benchmarks/swerebench-leaderboard/run_infer.py` | inline |
| `benchmarks/swesmith/run_infer.py` | inline |
| `benchmarks/swebenchpro/run_infer.py` | inline |

The transition is intentionally spread over separate commits so the migrated task
can be validated before the rest are converted; the helper is import-only from the
task scripts and does not change the evaluation pipeline.

## Rollout to other benchmarks

`benchmarks/swebench/run_infer.py`, `benchmarks/scaleswe/run_infer.py` (and its
no-think variant), `benchmarks/swerebenchv2/run_infer.py` (and its no-think
variant), and `benchmarks/z021data/run_infer.py` (and its no-think variant) are
wired. To adopt another task, mirror the three pieces below:

1. import `resolve_pruning_config`, `build_messages`, `build_trajectories`,
   `pruning_metadata` from `benchmarks.utils.trajectory`;
2. after the conversation finishes, resolve the policy once and replace the manual
   event-to-messages dump with `build_messages(events, pruning_config)`;
3. add `trajectories` and (conditionally) `pruning` to that task's `dump_data`
   before writing `{instance.id}.history.json`.

`messages` semantics stay unchanged in every task; the same contract applies.

## Tests

```bash
uv run pytest benchmarks/utils/test_trajectory.py benchmarks/utils/test_condenser.py \
    tests/test_fake_user_response.py
```
