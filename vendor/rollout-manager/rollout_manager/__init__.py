"""Standalone agent-rollout capture service.

A request adapter you point one agent run at instead of the model backend. It
proxies Chat-Completions (plus Anthropic Messages and OpenAI Responses) to any
OpenAI-compatible upstream, records every turn of that single session, and
exports the series of trajectories the session produced -- including the branches
created by context compaction and sub-agents.

Unlike ``slime.agent.adapters`` this package never assembles a chat template: the
request is forwarded as JSON and the model backend tokenizes it. See ``README.md``.
"""

from __future__ import annotations

from .capture import RolloutCapture, SessionClosed
from .trajectory import TrajectoryManager, TurnRecord

__all__ = ["RolloutCapture", "SessionClosed", "TrajectoryManager", "TurnRecord"]
