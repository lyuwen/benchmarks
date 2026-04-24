"""
Execution-based evaluator framework.

Execution-based evaluators assess agent output by running tests against the
produced patch in a containerized environment, providing ground-truth
evaluation at the cost of additional compute.

Unlike Critics (which gate retry logic), Evaluators are invoked once after
the agent finishes and their result is stored alongside the conversation
history for later analysis.
"""

from __future__ import annotations

import abc
import importlib
import json
import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ExecutionBasedEvaluator(BaseModel, abc.ABC):
    """Abstract base for evaluators that assess patches by executing tests.

    Subclasses must implement ``evaluate`` which receives the instance ID,
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
    def evaluate(
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


_EVALUATOR_REGISTRY: dict[str, tuple[str, str]] = {
    "swebench": (
        "benchmarks.swebench.evaluator",
        "SWEBenchEvaluator",
    ),
    "scaleswe": (
        "benchmarks.scaleswe.evaluator",
        "ScaleSWEEvaluator",
    ),
}


def add_evaluator_args(parser: ArgumentParser) -> None:
    """Add evaluator-related arguments to an argparse parser."""
    parser.add_argument(
        "--evaluator",
        type=str,
        default=None,
        help=(
            "Name of the execution-based evaluator to run after the agent "
            "finishes (default: None — no execution evaluation). "
            "Available: "
            + ", ".join(sorted(_EVALUATOR_REGISTRY))
            + ". "
            "The evaluation result is saved in the history JSON file."
        ),
    )
    parser.add_argument(
        "--evaluator-config",
        type=str,
        default=None,
        help="Path to JSON config file with evaluator parameters.",
    )


def create_evaluator(args: Namespace) -> ExecutionBasedEvaluator | None:
    """Create an evaluator from parsed argparse arguments.

    Returns None if no evaluator was requested.
    """
    if not getattr(args, "evaluator", None):
        return None

    name = args.evaluator
    if name not in _EVALUATOR_REGISTRY:
        raise ValueError(
            f"Unknown evaluator: {name}. "
            f"Available: {', '.join(sorted(_EVALUATOR_REGISTRY))}"
        )

    kwargs: dict[str, Any] = {}
    config_path = getattr(args, "evaluator_config", None)
    if config_path:
        p = Path(config_path)
        if not p.exists():
            raise ValueError(f"Evaluator config file not found: {config_path}")
        with open(p) as f:
            kwargs = json.load(f)
        logger.info("Loaded evaluator config from %s: %s", config_path, list(kwargs.keys()))

    module_path, class_name = _EVALUATOR_REGISTRY[name]
    module = importlib.import_module(module_path)
    evaluator_class = getattr(module, class_name)
    evaluator = evaluator_class(**kwargs)
    logger.info("Created evaluator: %s with args: %s", name, kwargs)
    return evaluator
