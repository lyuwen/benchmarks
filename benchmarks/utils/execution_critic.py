"""
Execution-based critic abstract base class.

Execution-based critics evaluate agent output by actually running tests
against the produced patch in a containerized environment, rather than
simply inspecting the event history or patch contents.  This provides
ground-truth evaluation at the cost of additional compute.
"""

from __future__ import annotations

import abc
from collections.abc import Sequence
from typing import Any

from pydantic import Field

from openhands.sdk.critic.base import CriticBase, CriticResult
from openhands.sdk.event import LLMConvertibleEvent


class ExecutionBasedCritic(CriticBase, abc.ABC):
    """Abstract base for critics that evaluate patches by executing tests.

    Subclasses must implement ``evaluate_with_context`` which receives
    the instance ID and optional instance data in addition to the patch.
    The standard ``evaluate`` method (required by CriticBase) falls back
    to a simple non-empty-patch check when instance context is unavailable.

    Configuration fields (dataset name, docker settings, timeouts, etc.)
    should be declared as Pydantic fields on concrete subclasses and can
    be passed via ``--critic-config`` JSON file.
    """

    timeout: int = Field(
        default=1800,
        description="Per-instance evaluation timeout in seconds",
    )

    # -- CriticBase interface (fallback without instance context) -----------

    def evaluate(
        self,
        events: Sequence[LLMConvertibleEvent],
        git_patch: str | None = None,
    ) -> CriticResult:
        """Fallback evaluation when no instance context is available.

        Execution-based critics need instance-level metadata (image, test
        spec, etc.) to run tests.  When called through the standard
        interface without that context, we can only check whether a patch
        exists.
        """
        if not git_patch or not git_patch.strip():
            return CriticResult(score=0.0, message="Empty or missing git patch")
        return CriticResult(
            score=0.5,
            message=(
                "Execution-based critic requires instance context; "
                "patch exists but could not be verified"
            ),
        )

    # -- Extended interface with instance context --------------------------

    @abc.abstractmethod
    def evaluate_with_context(
        self,
        instance_id: str,
        git_patch: str,
        instance_data: dict[str, Any] | None = None,
    ) -> CriticResult:
        """Evaluate a patch by executing tests in a container.

        Args:
            instance_id: Unique identifier for the evaluation instance.
            git_patch: The git diff produced by the agent.
            instance_data: Optional dataset row dict with benchmark-specific
                fields (image URL, test spec info, repo, version, etc.).
                When ``None``, the critic should attempt to load instance
                data from its configured dataset.

        Returns:
            CriticResult with score 1.0 for resolved, 0.0 for unresolved.
        """
        ...
