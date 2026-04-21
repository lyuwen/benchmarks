# Resume & Replay Mechanism Plan

This document outlines the strategy for implementing a robust resume and replay mechanism for interrupted agentic trajectories in evaluation benchmarks.

## 1. Goal
When an evaluation run is interrupted (e.g., due to a crash, network failure, or manual stop), we want to be able to resume the inference for a specific instance, restore its filesystem state, and continue the conversation with the **exact** history.

## 2. Architecture

### 2.1 Structured Event Persistence
- **Change:** Modify `benchmarks/utils/evaluation.py` to use a per-instance `persistence_dir` within the evaluation output folder.
- **SDK Integration:** By passing `persistence_dir` to `Conversation`, the `LocalConversation` will automatically save `base_state.json` and individual `events/*.json` files.
- **Location:** `{eval_output_dir}/persist/{instance_id}/`.

### 2.2 Workspace State Restoration (Replay)
Since workspaces (Docker or Remote) are ephemeral, we must restore the filesystem and environment state upon resumption.
- **Utility:** `benchmarks/utils/replayer.py`.
- **Logic:**
    - Iterate through the `EventLog` from the persistence directory.
    - Filter for "side-effecting" actions: `TerminalAction` and `FileEditorAction`.
    - Sequentially re-execute these against the fresh `RemoteWorkspace`.
    - Skip non-side-effecting actions (e.g., `view`, `ls`) to optimize restoration time.

### 2.3 Context Resumption
By using the SDK's native `ConversationState.create(persistence_dir=...)`:
- The `Conversation` object is reconstructed with all previous messages and observations.
- When `conversation.run()` is called, the `Agent` sees the entire history in its context window and continues from the last turn.

## 3. Implementation Steps

1.  **Modify `benchmarks/utils/args_parser.py`**:
    - Add `--resume` (bool) and `--resume-instance` (str) flags.
2.  **Create `benchmarks/utils/replayer.py`**:
    - Implement `ReplayManager` to handle workspace restoration.
3.  **Update `benchmarks/utils/evaluation.py`**:
    - Integrate `persistence_dir` into the `evaluate_instance` flow.
    - Implement logic to detect existing persistence data.
    - Call the `ReplayManager` if resumption is requested.
4.  **Verification**:
    - Add a test case that simulates an interruption and verifies exact state restoration.

## 4. Key Advantages
- **Exact History:** No summaries or approximations. The agent's prompt context is identical to the moment before interruption.
- **Filesystem Integrity:** Every successful command and file edit is re-applied, ensuring the environment is perfectly synced.
- **Native SDK Support:** Leverages existing `ConversationState` persistence features for maximum compatibility.
