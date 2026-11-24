"""Shared fixtures for mini-swe-agent adapter tests."""

import os
import tempfile
from collections.abc import Callable
from pathlib import Path

import pytest

from benchmarks.utils.tests.fakes import FakeWorkspace
from openhands.sdk.workspace.models import CommandResult


# mini-swe-agent writes to a user config directory at import time; point it at a
# throwaway directory so tests work in sandboxes with a read-only home.
os.environ.setdefault("MSWEA_SILENT_STARTUP", "1")
os.environ.setdefault(
    "MSWEA_GLOBAL_CONFIG_DIR", str(Path(tempfile.mkdtemp(prefix="mswea-cfg-")))
)


@pytest.fixture
def fake_workspace_factory() -> Callable[..., FakeWorkspace]:
    """Return a factory so tests can script per-command responses."""

    def _make(
        responder: Callable[[str, object], CommandResult] | None = None,
    ) -> FakeWorkspace:
        return FakeWorkspace(responder=responder)

    return _make
