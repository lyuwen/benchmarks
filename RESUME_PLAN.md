# Resume & Replay Mechanism

## Overview

When an evaluation run is interrupted (crash, OOM, network failure, manual stop), partially-completed instances can be resumed mid-conversation without restarting from scratch.

## Architecture

Three independent, composable layers:

### Layer 1: Event Persistence (`benchmarks/utils/event_persistence.py`)

The orchestrator writes each event to local disk incrementally via a `Conversation` callback.

- `EventFilePersistence` registers as a callback on the Conversation
- Events are written to `{eval_output_dir}/persist/{instance_id}/events/event-{idx:05d}-{uuid}.json`
- A `resume_meta.json` sidecar tracks `conversation_id`, `event_count`, `tools_used`, `last_timestamp`
- Writes are atomic (tmp + rename) and exception-safe
- Events persist on the orchestrator side, surviving container crashes

### Layer 2: Workspace Restoration (`benchmarks/utils/replayer.py`)

Replays side-effecting actions into a fresh container to restore filesystem state.

- `WorkspaceReplayer.replay(events)` pairs ActionEvents with ObservationEvents
- Terminal commands: re-executed, skipping read-only commands (`cat`, `ls`, `grep`, `git log`, etc.) and failed non-env commands
- File editor operations: uses `FileEditorObservation.new_content` (the actual file content post-edit) instead of re-implementing str_replace/insert/undo_edit logic
- ApplyPatch: re-applies patches via `git apply`
- Non-side-effecting events (Think, Finish, TaskTracker, browser) are skipped
- Returns a `ReplayReport` with counts of replayed/skipped/errored actions

### Layer 3: State Injection (`benchmarks/utils/resume.py`)

Writes persisted events + `base_state.json` into the container's persistence directory so the SDK's native `ConversationState.create()` resume path kicks in.

- `ResumeManager` orchestrates detection, workspace restore, and state injection
- Writes to `/workspace/conversations/{conversation_id_hex}/` inside the container
- `base_state.json` has `execution_status: "IDLE"`, serialized agent, LocalWorkspace pointing to `/workspace`
- Event files written via base64-encode + python one-liner through `workspace.execute_command()`
- The original `conversation_id` is reused so the SDK's persistence lookup finds the injected files

## Usage

```bash
# Resume all instances that have persisted state
python -m benchmarks.swebench.run_infer --resume ...

# Resume a specific instance
python -m benchmarks.swebench.run_infer --resume-instance django__django-12345 ...
```

Works with all three benchmarks: swebench, swesmith, swerebench-leaderboard.

## Flow

### During normal operation
1. Each event fires the `EventFilePersistence.callback`
2. Events accumulate in `persist/{instance_id}/events/`
3. `resume_meta.json` is updated with each event

### On resume
1. `evaluation.py::_process_one_mp` detects persisted events
2. Creates a fresh workspace (new container with testbed)
3. After testbed copy + git reset:
   - Workspace restoration: replays terminal commands and file edits
   - State injection: writes events + base_state.json into container
4. `Conversation(conversation_id=original_uuid)` finds injected state
5. Instruction is skipped (already in event history)
6. `run_conversation_with_fake_user_response()` continues where it left off

## Key Design Decisions

1. **Orchestrator-side persistence**: The SDK's LocalConversation persistence only works with LocalWorkspace. Since benchmarks use RemoteWorkspace, events are persisted from the orchestrator's callback.

2. **Server-side injection for resume**: Events + base_state.json are injected INTO the container, then the SDK's native `ConversationState.create()` handles resume -- exact message history, no approximation.

3. **Observation-based file replay**: Uses `FileEditorObservation.new_content` instead of re-implementing edit operations. Deterministic regardless of intermediate states.

4. **Persist always, resume on demand**: Every instance gets event persistence. The `--resume` flag only controls whether existing state is *used* on startup.

## Files

| File | Role |
|------|------|
| `benchmarks/utils/event_persistence.py` | Incremental event writer + loader |
| `benchmarks/utils/replayer.py` | Workspace filesystem restoration |
| `benchmarks/utils/resume.py` | ResumeManager: detection, restore, inject |
| `benchmarks/utils/evaluation.py` | Resume detection in worker loop |
| `benchmarks/{swebench,swesmith,swerebench-leaderboard}/run_infer.py` | Resume flow in evaluate_instance |
