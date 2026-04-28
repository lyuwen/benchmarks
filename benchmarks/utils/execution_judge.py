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
import logging
from argparse import ArgumentParser, Namespace
from collections.abc import Callable
from typing import Any, TypeVar

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
    ) -> bool | None:
        """Evaluate a patch by executing tests in a container.

        Args:
            instance_id: Unique identifier for the evaluation instance.
            git_patch: The git diff produced by the agent.
            instance_data: Dataset row dict with benchmark-specific fields
                (image URL, test spec info, repo, version, etc.).

        Returns:
            True if the patch resolves the issue, False if not, None if
            the judge could not run (e.g. missing dependency or error).
        """
        ...


# Global judge registry
_JUDGE_REGISTRY: dict[str, type[ExecutionBasedJudge]] = {}

T = TypeVar("T", bound=ExecutionBasedJudge)


def register_judge(name: str) -> Callable[[type[T]], type[T]]:
    """Decorator to register a judge class.

    Usage:
        @register_judge("swebench")
        class SWEBenchJudge(ExecutionBasedJudge):
            ...

    Args:
        name: The name to register the judge under (e.g., "swebench", "scaleswe").

    Returns:
        A decorator that registers the judge class.
    """

    def decorator(cls: type[T]) -> type[T]:
        if name in _JUDGE_REGISTRY:
            logger.warning(
                "Judge '%s' is already registered, overwriting with %s",
                name,
                cls.__name__,
            )
        _JUDGE_REGISTRY[name] = cls
        logger.debug("Registered judge '%s': %s", name, cls.__name__)
        return cls

    return decorator


def get_registered_judges() -> dict[str, type[ExecutionBasedJudge]]:
    """Get all registered judges.

    Returns:
        Dictionary mapping judge names to their classes.
    """
    return _JUDGE_REGISTRY.copy()


def add_judge_args(parser: ArgumentParser, default_judge: str | None = None) -> None:
    """Add judge-related arguments to an argparse parser.

    Args:
        parser: The argument parser to add arguments to.
        default_judge: Default judge name for this benchmark (e.g. "swebench").
            When set, ``--judge`` acts as a flag that enables the default judge.
            An explicit name can still be passed: ``--judge scaleswe``.
    """
    parser.add_argument(
        "--judge",
        nargs="?",
        const=default_judge,
        default=None,
        help=(
            "Enable execution-based judge after the agent finishes. "
            + (f"Defaults to '{default_judge}' for this benchmark. " if default_judge else "")
            + "Available judges depend on which benchmark modules have been imported. "
            "The judge result is saved in the history JSON file."
        ),
    )
    parser.add_argument(
        "--judge-timeout",
        type=int,
        default=1800,
        help="Per-instance judge timeout in seconds (default: 1800).",
    )
    parser.add_argument(
        "--judge-rm-image",
        action="store_true",
        help="Remove Docker image after each judge evaluation (default: False).",
    )
    parser.add_argument(
        "--judge-force-rebuild",
        action="store_true",
        help="Force rebuild Docker images for judge evaluation (default: False).",
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
            f"Available: {', '.join(sorted(_JUDGE_REGISTRY))}. "
            "Make sure the judge module has been imported."
        )

    kwargs: dict[str, Any] = {}

    # Common arguments
    timeout = getattr(args, "judge_timeout", None)
    if timeout is not None:
        kwargs["timeout"] = timeout

    rm_image = getattr(args, "judge_rm_image", None)
    if rm_image is not None:
        kwargs["rm_image"] = rm_image

    force_rebuild = getattr(args, "judge_force_rebuild", None)
    if force_rebuild is not None:
        kwargs["force_rebuild"] = force_rebuild

    # Judge-specific arguments
    if name == "scaleswe":
        docker_image_prefix = getattr(args, "docker_image_prefix", None)
        if docker_image_prefix is not None:
            kwargs["docker_image_prefix"] = docker_image_prefix

    judge_class = _JUDGE_REGISTRY[name]
    judge = judge_class(**kwargs)
    logger.info("Created judge: %s with config: %s", name, kwargs)
    return judge
