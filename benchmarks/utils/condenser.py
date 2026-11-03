"""Conversation condenser backends for benchmark runs.

``--condenser`` selects how conversation history is condensed while an evaluation
instance runs:

- ``none``: no condenser (the historical benchmark default).
- ``legacy``: the SDK's event-count based ``LLMSummarizingCondenser``.
- ``replay``: the token-budgeted ``ReplaySummarizingCondenser``, which condenses by
  replaying the real messages of the forgotten span instead of display previews.

The replay backend ships with newer SDK builds, so it is resolved at first use
rather than imported directly: the harness routinely runs against an SDK mounted
into the agent-server image that may be ahead of the vendored submodule used for
local development.
"""

import argparse
import importlib
import json
from typing import Any, Literal, Protocol, cast

from openhands.sdk import LLM
from openhands.sdk.context.condenser import CondenserBase, LLMSummarizingCondenser


CondenserKind = Literal["none", "legacy", "replay"]

CONDENSER_KINDS: tuple[CondenserKind, ...] = ("none", "legacy", "replay")

CONDENSER_LLM_USAGE_ID = "condenser"
"""Usage id of the condenser LLM copy, so summary cost is attributable."""

LEGACY_CONDENSER_DEFAULTS: dict[str, Any] = {"max_size": 80, "keep_first": 4}
"""Defaults matching the SDK preset condenser configuration."""

REPLAY_CONDENSER_CLASS_NAME = "ReplaySummarizingCondenser"


class _CondenserFactory(Protocol):
    """Constructor shape shared by the condensers built from configuration."""

    def __call__(self, *, llm: LLM, **kwargs: Any) -> CondenserBase: ...


def add_condenser_args(parser: argparse.ArgumentParser) -> None:
    """Add condenser-related arguments to an argparse parser."""
    parser.add_argument(
        "--condenser",
        type=str,
        default="none",
        choices=list(CONDENSER_KINDS),
        help=(
            "Conversation condenser backend (default: 'none'). "
            "'legacy' uses LLMSummarizingCondenser (event-count thresholds). "
            "'replay' uses ReplaySummarizingCondenser (token budgets against the "
            "model context window, and summarizes by replaying real messages). "
            "'replay' requires an SDK that provides that backend."
        ),
    )
    parser.add_argument(
        "--condenser-config",
        type=parse_condenser_config,
        default=None,
        help=(
            "Inline JSON object of extra keyword arguments for the selected "
            "condenser, e.g. '{\"threshold_ratio\": 0.7}'"
        ),
    )


def parse_condenser_config(value: str) -> dict[str, Any]:
    """Parse ``--condenser-config`` inline JSON into constructor keyword arguments."""
    try:
        config = json.loads(value)
    except json.JSONDecodeError as e:
        raise argparse.ArgumentTypeError(f"invalid JSON: {e}") from e
    if not isinstance(config, dict):
        raise argparse.ArgumentTypeError("condenser config must be a JSON object")
    return config


def get_condenser(
    kind: CondenserKind,
    llm: LLM,
    config: dict[str, Any] | None = None,
) -> CondenserBase | None:
    """Build the condenser selected by the evaluation metadata.

    A fresh condenser (and a deep copy of the agent's LLM with its own usage id) is
    created per conversation so summary cost is attributed to the current instance
    instead of accumulating across evaluation instances.

    Args:
        kind: Backend selection from ``--condenser``.
        llm: The agent's LLM.
        config: Extra constructor keyword arguments from ``--condenser-config``.

    Returns:
        The condenser instance, or None when condensation is disabled.

    Raises:
        RuntimeError: If the replay backend is requested but the running SDK does not
            provide it.
        ValueError: If the kind is unknown.
    """
    if kind == "none":
        return None
    if kind not in CONDENSER_KINDS:
        raise ValueError(f"Unknown condenser kind: {kind!r}")

    kwargs = dict(config or {})
    # Condenser completions are synchronous: they provide no token callback, so
    # inheriting streaming from the agent LLM would fail with
    # "Streaming requires an on_token callback".
    condenser_llm = llm.model_copy(
        update={"usage_id": CONDENSER_LLM_USAGE_ID, "stream": False}, deep=True
    )
    if kind == "legacy":
        return LLMSummarizingCondenser(
            llm=condenser_llm, **{**LEGACY_CONDENSER_DEFAULTS, **kwargs}
        )

    replay_condenser = _load_replay_condenser()
    return replay_condenser(llm=condenser_llm, **kwargs)


def _load_replay_condenser() -> _CondenserFactory:
    """Resolve the replay backend from whichever SDK this process is running."""
    module = importlib.import_module("openhands.sdk.context.condenser")
    replay_condenser = getattr(module, REPLAY_CONDENSER_CLASS_NAME, None)
    if replay_condenser is None:
        raise RuntimeError(
            "--condenser replay requires an SDK that provides "
            f"openhands.sdk.context.condenser.{REPLAY_CONDENSER_CLASS_NAME}. "
            "Update the vendored SDK, or mount an SDK that includes it into the "
            "agent-server image."
        )
    return cast(_CondenserFactory, replay_condenser)
