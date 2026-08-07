# Triple conversation-history capture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ensure OpenAI chat-messages history is captured for every evaluation instance — including failed runs — via three independent legs that all emit the same schema.

**Architecture:** Add a server-side incremental writer inside the agent server that dumps `messages.json` next to the persisted events on terminal state transitions (leg 1); promote the existing offline event->messages converter to a shared `benchmarks/utils` utility that can also read a `.tar.gz` archive (leg 2); wire leg 2 into `evaluation.py` right after the archive is downloaded, and extract leg 1's in-tar file to a sibling JSON. The existing event-sync dump in `run_infer.py` (leg 3) is unchanged.

**Tech Stack:** Python 3, pytest, the vendored `openhands` SDK (`openhands-sdk`, `openhands-agent-server`), `uv` for running under the project environment.

## Global Constraints

- Run all Python under the project environment: `uv run python ...` / `uv run pytest ...` so the `openhands` packages import.
- SDK changes live in the submodule `vendor/software-agent-sdk`; commit them in the submodule on its own branch, separately from the root repo. Use `git -C vendor/software-agent-sdk <command>`.
- Shared history schema (exact keys, all legs): `{"instance_id", "messages", "tools", "model", "temperature", "top_p"}`.
- Message conversion must match `run_infer.py`: keep only `LLMConvertibleEvent`, call `LLMConvertibleEvent.events_to_messages(...)`, then for each message `msg.model_copy(update={"send_reasoning_content": True}).to_chat_dict()`.
- Never let a history-capture failure abort an instance: each leg runs in its own try/except and logs on failure.
- Do not touch `vendor/software-agent-sdk/openhands-agent-server/build/lib/...` — it is a build artifact.

---

### Task 1: Promote converter to `benchmarks/utils` with a back-compat shim

**Files:**
- Create: `benchmarks/utils/convert_events_to_messages.py` (moved content)
- Modify: `benchmarks/scaleswe/convert_events_to_messages.py` (becomes a shim)
- Test: `benchmarks/utils/test_convert_events_to_messages.py`

**Interfaces:**
- Produces: `convert(conversation_dir: Path, instance_id: str | None = None) -> dict[str, Any]`, `load_events(conversation_dir: Path) -> list[Event]`, `events_to_chat_messages(events: list[Event]) -> list[dict[str, Any]]`, `collect_tools(events: list[Event]) -> list[dict[str, Any]]`, `main() -> None` — all now importable from `benchmarks.utils.convert_events_to_messages`.

- [ ] **Step 1: Move the file with git**

```bash
git mv benchmarks/scaleswe/convert_events_to_messages.py benchmarks/utils/convert_events_to_messages.py
```

- [ ] **Step 2: Recreate the old path as a shim**

Create `benchmarks/scaleswe/convert_events_to_messages.py`:

```python
#!/usr/bin/env python
"""Back-compat shim. The converter now lives in benchmarks.utils.

Kept so existing imports and CLI invocations keep working:
    uv run python benchmarks/scaleswe/convert_events_to_messages.py <dir>
"""

from __future__ import annotations

from benchmarks.utils.convert_events_to_messages import (  # noqa: F401
    collect_tools,
    convert,
    convert_archive,
    events_to_chat_messages,
    load_events,
    main,
)

if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Write a failing test for the moved import + dir conversion**

Create `benchmarks/utils/test_convert_events_to_messages.py`. This test builds a
tiny on-disk conversation dir with one `MessageEvent` and asserts the 6-key
schema. `_make_conversation_dir` is reused by later tasks.

```python
import json
from pathlib import Path

import pytest

from benchmarks.utils.convert_events_to_messages import convert


def _make_conversation_dir(root: Path) -> Path:
    conv = root / "workspace" / "conversations" / "abc123"
    events = conv / "events"
    events.mkdir(parents=True)
    (conv / "meta.json").write_text(
        json.dumps(
            {
                "id": "abc123",
                "tool_module_qualnames": {},
                "agent": {"llm": {"model": "gpt-x", "temperature": 0.0, "top_p": 1.0}},
            }
        )
    )
    msg_event = {
        "kind": "MessageEvent",
        "id": "e1",
        "source": "user",
        "llm_message": {"role": "user", "content": [{"type": "text", "text": "hi"}]},
    }
    (events / "event-00000-e1.json").write_text(json.dumps(msg_event))
    return conv


def test_convert_dir_emits_schema(tmp_path):
    conv = _make_conversation_dir(tmp_path)
    dump = convert(conv)
    assert set(dump) == {
        "instance_id",
        "messages",
        "tools",
        "model",
        "temperature",
        "top_p",
    }
    assert dump["instance_id"] == "abc123"
    assert dump["model"] == "gpt-x"
    assert any(m.get("role") == "user" for m in dump["messages"])
```

- [ ] **Step 4: Run the test to verify import works and it passes**

Run: `uv run pytest benchmarks/utils/test_convert_events_to_messages.py::test_convert_dir_emits_schema -v`
Expected: PASS (the moved module imports; `convert` works on a dir).

- [ ] **Step 5: Commit**

```bash
git add benchmarks/utils/convert_events_to_messages.py benchmarks/scaleswe/convert_events_to_messages.py benchmarks/utils/test_convert_events_to_messages.py
git commit -m "refactor: promote convert_events_to_messages to benchmarks/utils with shim"
```

---

### Task 2: Fix the skip-and-report dead code in `load_events`

**Files:**
- Modify: `benchmarks/utils/convert_events_to_messages.py` (the `raise` at the old lines 129-131)
- Test: `benchmarks/utils/test_convert_events_to_messages.py`

**Interfaces:**
- Consumes: `convert`, `_make_conversation_dir` from Task 1.
- Produces: `load_events` that skips and reports a corrupt event file instead of raising.

- [ ] **Step 1: Write the failing test**

Add to `benchmarks/utils/test_convert_events_to_messages.py`:

```python
def test_convert_skips_corrupt_event_file(tmp_path):
    conv = _make_conversation_dir(tmp_path)
    # A second event file whose "kind" cannot deserialize.
    (conv / "events" / "event-00001-bad.json").write_text(
        json.dumps({"kind": "NotARealEventKind", "id": "bad"})
    )
    dump = convert(conv)  # must not raise
    # The good user message still made it through.
    assert any(m.get("role") == "user" for m in dump["messages"])
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest benchmarks/utils/test_convert_events_to_messages.py::test_convert_skips_corrupt_event_file -v`
Expected: FAIL — `load_events` re-raises on the bad event (the unreachable `skipped.append`).

- [ ] **Step 3: Fix the dead-code block in `load_events`**

In `benchmarks/utils/convert_events_to_messages.py`, replace the broken except
body:

```python
        try:
            events.append(Event.model_validate(data))
        except Exception as exc:  # noqa: BLE001 - report and continue
            raise
            reason = str(exc).splitlines()[0] if str(exc) else repr(exc)
            skipped.append((path.name, reason))
```

with:

```python
        try:
            events.append(Event.model_validate(data))
        except Exception as exc:  # noqa: BLE001 - report and continue
            reason = str(exc).splitlines()[0] if str(exc) else repr(exc)
            skipped.append((path.name, reason))
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest benchmarks/utils/test_convert_events_to_messages.py -v`
Expected: PASS (both tests).

- [ ] **Step 5: Commit**

```bash
git add benchmarks/utils/convert_events_to_messages.py benchmarks/utils/test_convert_events_to_messages.py
git commit -m "fix: skip-and-report corrupt event files instead of aborting conversion"
```

---

### Task 3: Add `convert_archive()` and archive CLI mode

**Files:**
- Modify: `benchmarks/utils/convert_events_to_messages.py`
- Test: `benchmarks/utils/test_convert_events_to_messages.py`

**Interfaces:**
- Consumes: `convert`, `_make_conversation_dir` (Task 1).
- Produces: `convert_archive(tar_path: Path, instance_id: str | None = None) -> dict[str, Any]`.

- [ ] **Step 1: Write the failing test**

Add to `benchmarks/utils/test_convert_events_to_messages.py`:

```python
import tarfile

from benchmarks.utils.convert_events_to_messages import convert_archive


def test_convert_archive_from_targz(tmp_path):
    conv = _make_conversation_dir(tmp_path)
    tar_path = tmp_path / "abc123.tar.gz"
    # Archive the whole "workspace/conversations" tree, mirroring the runtime tar.
    ws_root = tmp_path / "workspace"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(ws_root, arcname="workspace")
    dump = convert_archive(tar_path)
    assert dump["instance_id"] == "abc123"
    assert dump["model"] == "gpt-x"
    assert any(m.get("role") == "user" for m in dump["messages"])
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest benchmarks/utils/test_convert_events_to_messages.py::test_convert_archive_from_targz -v`
Expected: FAIL with `ImportError`/`cannot import name 'convert_archive'`.

- [ ] **Step 3: Implement `convert_archive`**

Add these imports near the top of `benchmarks/utils/convert_events_to_messages.py` (after the existing `import re`):

```python
import tarfile
import tempfile
```

Add the function just above `def main()`:

```python
def _find_conversation_dir(root: Path) -> Path:
    """Locate the conversation dir (the one containing an 'events/' subdir)."""
    for events_dir in root.rglob("events"):
        if events_dir.is_dir():
            return events_dir.parent
    raise FileNotFoundError(f"No conversation 'events/' dir found under {root}")


def convert_archive(
    tar_path: Path, instance_id: str | None = None
) -> dict[str, Any]:
    """Convert a conversation .tar.gz archive into the history dump dict.

    The archive is the one produced by the benchmark's
    ``_capture_conversation_archive`` (a gzip tar of ``workspace/conversations``).
    """
    tar_path = Path(tar_path).expanduser().resolve()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(tmp_root)  # noqa: S202 - trusted, self-produced archive
        conversation_dir = _find_conversation_dir(tmp_root)
        return convert(conversation_dir, instance_id=instance_id)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest benchmarks/utils/test_convert_events_to_messages.py -v`
Expected: PASS (all three tests).

- [ ] **Step 5: Add archive support to the CLI**

In `main()`, replace the positional-arg handling. Change the argument help and
the dispatch so a `.tar.gz`/`.tgz` path routes to `convert_archive`. Replace:

```python
    conversation_dir = args.conversation_dir.expanduser().resolve()
    if not conversation_dir.is_dir():
        parser.error(f"Not a directory: {conversation_dir}")

    dump = convert(conversation_dir, instance_id=args.instance_id)
```

with:

```python
    src = args.conversation_dir.expanduser().resolve()
    if src.is_file() and src.name.endswith((".tar.gz", ".tgz")):
        dump = convert_archive(src, instance_id=args.instance_id)
    elif src.is_dir():
        dump = convert(src, instance_id=args.instance_id)
    else:
        parser.error(f"Not a conversation dir or .tar.gz archive: {src}")
```

- [ ] **Step 6: Commit**

```bash
git add benchmarks/utils/convert_events_to_messages.py benchmarks/utils/test_convert_events_to_messages.py
git commit -m "feat: convert_events_to_messages can read a .tar.gz archive"
```

---

### Task 4: Wire leg 2 (archive-converted) + leg 1 extraction into `evaluation.py`

**Files:**
- Modify: `benchmarks/utils/evaluation.py` (`_capture_conversation_archive`, ends around line 196; callsite at line 618)
- Test: `benchmarks/utils/test_history_capture_wiring.py`

**Interfaces:**
- Consumes: `convert_archive` (Task 3); existing `_capture_conversation_archive(self, workspace, instance)`.
- Produces: after a successful archive download, sibling files
  `conversations/{instance_id}.archive-history.json` (leg 2) and
  `conversations/{instance_id}.server-history.json` (leg 1, extracted from tar,
  when present). New method `_derive_history_from_archive(self, conv_tar_path: Path, instance) -> None`.

- [ ] **Step 1: Write the failing test**

Create `benchmarks/utils/test_history_capture_wiring.py`. It calls the new
derive method directly against a fixture tar and asserts both sibling files.

```python
import json
import tarfile
import types
from pathlib import Path

from benchmarks.utils.evaluation import Evaluation


def _make_archive_with_messages(tmp_path: Path) -> Path:
    conv = tmp_path / "workspace" / "conversations" / "abc123"
    (conv / "events").mkdir(parents=True)
    (conv / "meta.json").write_text(
        json.dumps(
            {
                "id": "abc123",
                "tool_module_qualnames": {},
                "agent": {"llm": {"model": "gpt-x", "temperature": 0.0, "top_p": 1.0}},
            }
        )
    )
    (conv / "events" / "event-00000-e1.json").write_text(
        json.dumps(
            {
                "kind": "MessageEvent",
                "id": "e1",
                "source": "user",
                "llm_message": {
                    "role": "user",
                    "content": [{"type": "text", "text": "hi"}],
                },
            }
        )
    )
    # Server-side leg-1 file already present inside the archive.
    (conv / "messages.json").write_text(
        json.dumps({"instance_id": "abc123", "messages": [], "tools": [],
                    "model": "gpt-x", "temperature": 0.0, "top_p": 1.0})
    )
    tar_path = tmp_path / "abc123.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(tmp_path / "workspace", arcname="workspace")
    return tar_path


def test_derive_history_from_archive(tmp_path):
    tar_path = _make_archive_with_messages(tmp_path)
    instance = types.SimpleNamespace(id="abc123")
    # Call the unbound method with a minimal fake self (only logger use).
    Evaluation._derive_history_from_archive(
        types.SimpleNamespace(), tar_path, instance
    )
    conv_dir = tar_path.parent
    archive_hist = json.loads((conv_dir / "abc123.archive-history.json").read_text())
    server_hist = json.loads((conv_dir / "abc123.server-history.json").read_text())
    assert archive_hist["instance_id"] == "abc123"
    assert any(m.get("role") == "user" for m in archive_hist["messages"])
    assert server_hist["instance_id"] == "abc123"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest benchmarks/utils/test_history_capture_wiring.py -v`
Expected: FAIL with `AttributeError: ... '_derive_history_from_archive'`.

- [ ] **Step 3: Add the `_derive_history_from_archive` method**

In `benchmarks/utils/evaluation.py`, add this method right after
`_capture_conversation_archive` (after its final `except` block near line 196).
Add `import tarfile` and `from benchmarks.utils.convert_events_to_messages import convert_archive` at the top of the file if not already imported.

```python
    def _derive_history_from_archive(
        self,
        conv_tar_path: Path,
        instance: EvalInstance,
    ) -> None:
        """Produce leg-1 and leg-2 history JSONs from the downloaded archive.

        Leg 2 (archive-converted): re-run the SDK conversion over the events in
        the tar and write ``{id}.archive-history.json``. Leg 1 (server-side):
        extract the in-tar ``messages.json`` (written during the run on terminal
        state) to ``{id}.server-history.json`` if present. Both are best-effort
        and never raise.
        """
        conv_dir = conv_tar_path.parent

        # Leg 2: archive-converted.
        try:
            dump = convert_archive(conv_tar_path, instance_id=instance.id)
            out = conv_dir / f"{instance.id}.archive-history.json"
            out.write_text(json.dumps(dump, indent=2))
            logger.info("[child] Wrote archive-converted history for %s", instance.id)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "[child] Failed archive-converted history for %s: %s", instance.id, e
            )

        # Leg 1: extract server-side messages.json from the tar, if present.
        try:
            with tarfile.open(conv_tar_path, "r:gz") as tar:
                member = next(
                    (m for m in tar.getmembers()
                     if m.name.endswith("messages.json")),
                    None,
                )
                if member is None:
                    logger.debug(
                        "[child] No server-side messages.json in archive for %s",
                        instance.id,
                    )
                    return
                fobj = tar.extractfile(member)
                if fobj is None:
                    return
                data = fobj.read()
            out = conv_dir / f"{instance.id}.server-history.json"
            out.write_bytes(data)
            logger.info("[child] Wrote server-side history for %s", instance.id)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "[child] Failed server-side history for %s: %s", instance.id, e
            )
```

- [ ] **Step 4: Call it after a successful archive download**

In `_capture_conversation_archive`, in the `if result.success:` branch (around
line 179), after the existing `logger.info(... "Saved conversation archive" ...)`
call, add:

```python
                self._derive_history_from_archive(conv_tar_path, instance)
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `uv run pytest benchmarks/utils/test_history_capture_wiring.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add benchmarks/utils/evaluation.py benchmarks/utils/test_history_capture_wiring.py
git commit -m "feat: derive archive-converted and server-side history from conversation tar"
```

---

### Task 5: Server-side incremental writer (SDK submodule, leg 1)

**Files:**
- Modify: `vendor/software-agent-sdk/openhands-agent-server/openhands/agent_server/event_service.py`
- Test: `vendor/software-agent-sdk/openhands-agent-server/tests/test_messages_dump.py`

**Interfaces:**
- Consumes: existing `EventService` with `self._conversation`, `self.conversation_dir`, `_publish_state_update` (line 626), `ConversationExecutionStatus` (imported).
- Produces: `EventService._dump_messages_json(self) -> None` that writes `messages.json` into `self.conversation_dir` atomically; called from `_publish_state_update` when the status is terminal.

- [ ] **Step 1: Write the failing test**

Create `vendor/software-agent-sdk/openhands-agent-server/tests/test_messages_dump.py`.
It builds a real `LocalConversation` with a persistence dir, appends a user
`MessageEvent`, sets a terminal status, calls the dump helper, and asserts the
file + schema.

```python
import json
from pathlib import Path

import pytest

from openhands.agent_server.event_service import EventService
from openhands.sdk.conversation.state import ConversationExecutionStatus


@pytest.mark.asyncio
async def test_dump_messages_json_writes_schema(tmp_path, monkeypatch):
    # Build a minimal EventService whose _conversation exposes the pieces the
    # dumper reads. We reuse the real dumper against a hand-built state.
    svc = EventService.__new__(EventService)
    conv_dir = tmp_path / "conv"
    (conv_dir).mkdir()
    monkeypatch.setattr(
        type(svc), "conversation_dir", property(lambda self: conv_dir), raising=False
    )

    class _LLM:
        model = "gpt-x"
        temperature = 0.0
        top_p = 1.0

    class _Agent:
        def get_all_llms(self):
            return [_LLM()]

    from openhands.sdk.event.llm_convertible.message import MessageEvent
    from openhands.sdk.llm.message import Message, TextContent

    class _State:
        events = [
            MessageEvent(
                source="user",
                llm_message=Message(role="user", content=[TextContent(text="hi")]),
            )
        ]

    class _Conv:
        agent = _Agent()
        state = _State()
        _state = _State()

    svc._conversation = _Conv()
    svc._dump_messages_json()

    out = conv_dir / "messages.json"
    assert out.is_file()
    dump = json.loads(out.read_text())
    assert set(dump) == {
        "instance_id", "messages", "tools", "model", "temperature", "top_p"
    }
    assert dump["model"] == "gpt-x"
    assert any(m.get("role") == "user" for m in dump["messages"])
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd vendor/software-agent-sdk/openhands-agent-server && uv run pytest tests/test_messages_dump.py -v`
Expected: FAIL with `AttributeError: ... '_dump_messages_json'`.

- [ ] **Step 3: Implement `_dump_messages_json` and the terminal gate**

In `event_service.py`, add these imports near the existing SDK imports (after
line 24):

```python
from openhands.sdk.event.base import LLMConvertibleEvent
from openhands.sdk.event.llm_convertible.system import SystemPromptEvent
from openhands.sdk.tool.tool import ToolDefinition
```

Add a module-level constant after `logger = get_logger(__name__)` (line ~32):

```python
_TERMINAL_STATUSES = (
    ConversationExecutionStatus.FINISHED,
    ConversationExecutionStatus.ERROR,
    ConversationExecutionStatus.STUCK,
    ConversationExecutionStatus.PAUSED,
    ConversationExecutionStatus.IDLE,
)
```

Add the method (e.g. just before `_publish_state_update`, around line 626):

```python
    def _dump_messages_json(self) -> None:
        """Write messages.json next to the persisted events (best-effort).

        Mirrors the benchmark's *.history.json schema. Called on terminal state
        transitions so a failed/crashed run still leaves a messages record
        inside the conversation dir (and thus inside the downloaded archive).
        """
        if not self._conversation:
            return
        try:
            state = self._conversation.state
            convertible = [
                e for e in state.events if isinstance(e, LLMConvertibleEvent)
            ]
            msgs = LLMConvertibleEvent.events_to_messages(convertible)
            messages = [
                m.model_copy(update={"send_reasoning_content": True}).to_chat_dict()
                for m in msgs
            ]

            tools: list = []
            for event in state.events:
                if isinstance(event, SystemPromptEvent):
                    for tool in event.tools:
                        if isinstance(tool, ToolDefinition):
                            tools.append(tool.to_openai_tool())
                    break

            llms = self._conversation.agent.get_all_llms()
            llm = llms[0] if llms else None

            dump = {
                "instance_id": self.conversation_dir.name,
                "messages": messages,
                "tools": tools,
                "model": getattr(llm, "model", None),
                "temperature": getattr(llm, "temperature", None),
                "top_p": getattr(llm, "top_p", None),
            }

            self.conversation_dir.mkdir(parents=True, exist_ok=True)
            target = self.conversation_dir / "messages.json"
            tmp = target.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(dump, indent=2))
            tmp.replace(target)  # atomic
        except Exception as e:  # noqa: BLE001 - never break the run
            logger.warning("Failed to dump messages.json: %s", e)
```

Add `import json` at the top if not present (it is used elsewhere; verify).

- [ ] **Step 4: Call it from `_publish_state_update` on terminal status**

In `_publish_state_update` (line 626), inside the `with state:` block, after
building `state_update_event`, add:

```python
            if state.execution_status in _TERMINAL_STATUSES:
                self._dump_messages_json()
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `cd vendor/software-agent-sdk/openhands-agent-server && uv run pytest tests/test_messages_dump.py -v`
Expected: PASS.

- [ ] **Step 6: Commit in the submodule**

```bash
git -C vendor/software-agent-sdk add openhands-agent-server/openhands/agent_server/event_service.py openhands-agent-server/tests/test_messages_dump.py
git -C vendor/software-agent-sdk commit -m "feat: dump messages.json on terminal conversation state"
```

---

### Task 6: Cross-check leg 1 and leg 2 produce identical output

**Files:**
- Test: `benchmarks/utils/test_convert_events_to_messages.py`

**Interfaces:**
- Consumes: `convert` (Task 1); the server-side `_dump_messages_json` logic is
  duplicated in the SDK, so this test asserts parity of the *conversion* against
  a hand-written expected messages list rather than importing the SDK server.

- [ ] **Step 1: Write the parity test**

Add to `benchmarks/utils/test_convert_events_to_messages.py`. Both legs use
`LLMConvertibleEvent.events_to_messages` + `to_chat_dict`, so converting the same
event dir twice must be deterministic and equal.

```python
def test_convert_is_deterministic(tmp_path):
    conv = _make_conversation_dir(tmp_path)
    first = convert(conv)
    second = convert(conv)
    assert first == second
    assert first["messages"] == second["messages"]
```

- [ ] **Step 2: Run the test to verify it passes**

Run: `uv run pytest benchmarks/utils/test_convert_events_to_messages.py::test_convert_is_deterministic -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add benchmarks/utils/test_convert_events_to_messages.py
git commit -m "test: assert conversion determinism for leg parity"
```

---

### Task 7: Full-suite verification

- [ ] **Step 1: Run the benchmark-side test suite for the touched files**

Run: `uv run pytest benchmarks/utils/test_convert_events_to_messages.py benchmarks/utils/test_history_capture_wiring.py -v`
Expected: PASS (all tests).

- [ ] **Step 2: Run the SDK-side test**

Run: `cd vendor/software-agent-sdk/openhands-agent-server && uv run pytest tests/test_messages_dump.py -v`
Expected: PASS.

- [ ] **Step 3: Sanity-check the CLI archive mode still works end-to-end**

Run: `uv run python benchmarks/scaleswe/convert_events_to_messages.py --help`
Expected: help text prints (shim import chain resolves).
