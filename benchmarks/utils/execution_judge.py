"""
Execution-based judge framework.

Execution-based judges assess agent output by running tests against the
produced patch in a containerized environment, providing ground-truth
evaluation at the cost of additional compute.

Unlike Critics (which gate retry logic), Judges are invoked once after
the agent finishes and their result is stored alongside the conversation
history for later analysis.
"""

from __future__ import annotations

import abc
import importlib
import logging
from argparse import ArgumentParser, Namespace
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ExecutionBasedJudge(BaseModel, abc.ABC):
    """Abstract base for judges that assess patches by executing tests.

    Subclasses must implement ``judge`` which receives the instance ID,
    git patch, and instance data, and returns a boolean indicating whether
    the patch resolves the issue.

    Configuration fields (dataset name, docker settings, timeouts, etc.)
    should be declared as Pydantic fields on concrete subclasses.
    """

    timeout: int = Field(
        default=1800,
        description="Per-instance evaluation timeout in seconds",
    )

    @abc.abstractmethod
    def judge(
        self,
        instance_id: str,
        git_patch: str,
        instance_data: dict[str, Any],
    ) -> bool:
        """Evaluate a patch by executing tests in a container.

        Args:
            instance_id: Unique identifier for the evaluation instance.
            git_patch: The git diff produced by the agent.
            instance_data: Dataset row dict with benchmark-specific fields
                (image URL, test spec info, repo, version, etc.).

        Returns:
            True if the patch resolves the issue, False otherwise.
        """
        ...


_JUDGE_REGISTRY: dict[str, tuple[str, str]] = {
    "swebench": (
        "benchmarks.swebench.judge",
        "SWEBenchJudge",
    ),
    "scaleswe": (
        "benchmarks.scaleswe.judge",
        "ScaleSWEJudge",
    ),
}


def add_judge_args(parser: ArgumentParser) -> None:
    """Add judge-related arguments to an argparse parser."""
    parser.add_argument(
        "--judge",
        type=str,
        default=None,
        help=(
            "Name of the execution-based judge to run after the agent "
            "finishes (default: None — no execution evaluation). "
            "Available: "
            + ", ".join(sorted(_JUDGE_REGISTRY))
            + ". "
            "The judge result is saved in the history JSON file."
        ),
    )
    parser.add_argument(
        "--judge-timeout",
        type=int,
        default=1800,
        help="Per-instance judge timeout in seconds (default: 1800).",
    )


def create_judge(args: Namespace) -> ExecutionBasedJudge | None:
    """Create a judge from parsed argparse arguments.

    Returns None if no judge was requested.
    """
    if not getattr(args, "judge", None):
        return None

    name = args.judge
    if name not in _JUDGE_REGISTRY:
        raise ValueError(
            f"Unknown judge: {name}. "
            f"Available: {', '.join(sorted(_JUDGE_REGISTRY))}"
        )

    kwargs: dict[str, Any] = {}
    timeout = getattr(args, "judge_timeout", None)
    if timeout is not None:
        kwargs["timeout"] = timeout

    module_path, class_name = _JUDGE_REGISTRY[name]
    module = importlib.import_module(module_path)
    judge_class = getattr(module, class_name)
    judge = judge_class(**kwargs)
    logger.info("Created judge: %s with timeout: %s", name, kwargs.get("timeout"))
    return judge
