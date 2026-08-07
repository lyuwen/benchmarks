# Triple conversation-history capture design

## Problem

Today the OpenAI chat-messages history for an evaluation instance is written
only at the very end of `evaluate_instance` in the per-benchmark `run_infer.py`
(e.g. `benchmarks/scaleswe/run_infer.py:428`), after `git diff` extraction and
optional judging. It is built from the local `conversation.state.events` cache.

Two consequences:

1. **Only successful runs get history.** Any exception before line 428 — agent
   crash, timeout, remote/websocket error, judge failure — aborts the method
   and no `*.history.json` is written.
2. **Sync-hell exposure.** `conversation.state.events` for a `RemoteConversation`
   is populated asynchronously by the WebSocket callback thread. The first
   `MessageEvent` (and potentially others) can be lost if the socket
   subscription is not yet ready, so even successful runs can have incomplete
   history. (See `CLAUDE.local.md` "Remote conversation event capture gotcha".)

Meanwhile the agent server already persists the authoritative event log
incrementally, event-by-event, via the SDK's `EventLog.append()` into
`/workspace/conversations/<id>/events/` inside the container. That store is
immune to both problems above.

## Goal

Add a **third, authoritative leg** to history capture and promote the existing
offline converter so that **history exists even when a run fails**. Each
instance ends up with three independent records, all in the same schema.

### Shared schema

Every leg emits:

```json
{
  "instance_id": "...",
  "messages": [],
  "tools": [],
  "model": "...",
  "temperature": 0.0,
  "top_p": 1.0
}
```

## The three legs

### Leg 1 — server-side incremental writer (NEW, SDK submodule)

- **Where:** `openhands-agent-server`, wired in `EventService.start()` where
  callbacks are already registered (`event_service.py:421` per-event wrapper /
  `:435` `set_on_state_change`).
- **Mechanism:** a helper `_maybe_dump_messages()` attached to the existing
  state-change path. On each state change it checks `execution_status`; when it
  reaches a **terminal** state (`FINISHED`, `ERROR`, `STUCK`, `PAUSED`, or the
  `IDLE` reached after a finished run) it:
  1. reads `self._conversation.state.events`,
  2. runs `LLMConvertibleEvent.events_to_messages` + `Message.to_chat_dict`
     (matching `run_infer.py`, with `send_reasoning_content=True`),
  3. collects tools from the `SystemPromptEvent`, and `model` / `temperature` /
     `top_p` from the agent LLM config,
  4. writes `messages.json` into `self.conversation_dir`
     (`/workspace/conversations/<id>/messages.json`) using temp-file + atomic
     rename.
- **Delivery:** because it lives inside `conversation_dir`, it is automatically
  included when `_capture_conversation_archive` tars `/workspace/conversations`
  — no extra download. Rides along with the events on both success and failure.
- **Cost:** near-zero (option C) — re-converts only on terminal transitions,
  not per event.
- **Crash safety:** the server already forces `RUNNING -> ERROR` on restart
  (`event_service.py:446`), which is a terminal transition the writer catches.
- **Submodule handling:** change to `vendor/software-agent-sdk`, on its own
  branch, committed in the submodule separately from the root repo (per
  `CLAUDE.local.md` development guide).

### Leg 2 — archive-converted version (PROMOTED utility)

- **Move:** `benchmarks/scaleswe/convert_events_to_messages.py` ->
  `benchmarks/utils/convert_events_to_messages.py` (shared, next to
  `evaluation.py`).
- **New capability:** accept a `.tar.gz` archive as input in addition to an
  unpacked conversation dir. Archive mode:
  1. extract to a `tempfile.TemporaryDirectory`,
  2. locate `workspace/conversations/<id>/` inside (glob for the dir containing
     `events/`),
  3. run existing `convert()` on it,
  4. clean up.
- **Public API** (callable from `evaluation.py`, not only CLI):
  - `convert_archive(tar_path: Path, instance_id: str | None = None) -> dict[str, Any]`
  - `convert(conversation_dir: Path, instance_id: str | None = None) -> dict[str, Any]` (existing)
- **CLI:** retained and backward compatible, gains an archive-path mode.
- **Back-compat shim:** `benchmarks/scaleswe/convert_events_to_messages.py`
  becomes a thin re-export from the new location so existing imports/CLI keep
  working. The `build/lib/...` copy in the SDK tree is a build artifact and is
  ignored.
- **Bug fixed while promoting:** current lines 129-131 have a `raise`
  immediately before dead `skipped.append(...)` code, so the intended
  "report and continue" path is unreachable and one bad event file aborts the
  whole conversion. Make it actually skip-and-report — relevant because failed
  runs are more likely to contain a half-written final event.

### Leg 3 — event-sync version (EXISTING, unchanged)

- The current `{instance_id}.history.json` built from
  `conversation.state.events` in each `run_infer.py`. Left as-is.

## Wiring in `benchmarks/utils/evaluation.py`

- **Leg 2** is wired right after `_capture_conversation_archive` succeeds
  (`evaluation.py:618`): call `convert_archive()` on the just-downloaded
  `conversations/{instance_id}.tar.gz` and write
  `conversations/{instance_id}.archive-history.json`. Wrapped in its own
  try/except so it never breaks the run.
- **Leg 1** needs no benchmark-side work — it is already in the tar.
  Optionally, `evaluation.py` extracts the in-tar `messages.json` to
  `conversations/{instance_id}.server-history.json` so all three sit side by
  side without manual untarring.

### Final artifacts in `eval_output_dir`

| File | Leg | Source |
|------|-----|--------|
| `{instance_id}.history.json` | 3 | event-sync (existing) |
| `conversations/{instance_id}.tar.gz` | — | raw archive (existing; now contains `messages.json`) |
| `conversations/{instance_id}.server-history.json` | 1 | extracted from tar |
| `conversations/{instance_id}.archive-history.json` | 2 | re-converted from tar |

## Error handling

All three legs are independent and failure-isolated:

- Leg 1 writes on terminal transitions with atomic rename — partial writes are
  impossible.
- Legs 2 and 3 each run in their own try/except in `evaluation.py` — a failure
  in one is logged and skipped, never aborting the instance or the other legs.

## Testing

- Unit test for `convert_archive()` using a small fixture `.tar.gz` (a few event
  JSONs + `meta.json`) -> asserts the 6-key schema and message count.
- Unit test for the skip-and-report fix: a conversation dir with one corrupt
  event file still converts the rest.
- SDK-side unit test for the terminal-state writer: drive an `EventService`
  (or the helper directly) through a terminal transition, assert `messages.json`
  appears with correct content.
- Cross-check test: leg 1 and leg 2 produce identical output for the same event
  set (they share the SDK conversion, so they should match).

## Out of scope

- Changing the leg-3 event-sync dump or the WebSocket readiness race itself
  (mitigated elsewhere by the prepare-step gap workaround).
- Per-event server-side rewrite cadence (rejected in favor of terminal-only,
  option C).
