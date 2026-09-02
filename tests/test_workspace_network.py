"""Tests for central network-mode resolution in the evaluation layer."""

import pytest

from benchmarks.utils.workspace_network import resolve_network_policy
from openhands.workspace.docker.network_policy import WorkspaceNetworkPolicy


def test_unset_env_is_public():
    assert resolve_network_policy("docker", env={}).mode == "public"


def test_env_mode_is_resolved():
    policy = resolve_network_policy("docker", env={"OH_NETWORK_MODE": "no-network"})
    assert policy.mode == "no-network"


def test_invalid_env_mode_raises():
    with pytest.raises(ValueError):
        resolve_network_policy("docker", env={"OH_NETWORK_MODE": "typo-here"})


def test_remote_workspace_rejects_non_public_mode():
    """Remote has no local container to isolate: must fail closed, not open."""
    with pytest.raises(ValueError, match="remote"):
        resolve_network_policy("remote", env={"OH_NETWORK_MODE": "no-network"})


def test_remote_workspace_allows_public():
    assert resolve_network_policy("remote", env={}).mode == "public"


def test_explicit_policy_conflicting_with_env_is_rejected():
    with pytest.raises(ValueError, match="conflict"):
        resolve_network_policy(
            "docker",
            explicit=WorkspaceNetworkPolicy(mode="strict-no-network"),
            env={"OH_NETWORK_MODE": "no-network"},
        )


def test_explicit_policy_matching_env_is_accepted():
    policy = resolve_network_policy(
        "docker",
        explicit=WorkspaceNetworkPolicy(mode="no-network"),
        env={"OH_NETWORK_MODE": "no-network"},
    )
    assert policy.mode == "no-network"


def test_explicit_policy_without_env_is_used():
    policy = resolve_network_policy(
        "docker", explicit=WorkspaceNetworkPolicy(mode="strict-no-network"), env={}
    )
    assert policy.mode == "strict-no-network"


def test_metadata_carries_resolved_mode_and_digest():
    from benchmarks.utils.models import EvalMetadata
    from openhands.sdk import LLM
    from openhands.sdk.critic import PassCritic

    metadata = EvalMetadata(
        llm=LLM(model="test-model"),
        dataset="test",
        dataset_split="test",
        max_iterations=1,
        eval_output_dir="/tmp/test-network-meta",
        critic=PassCritic(),
        network_mode="no-network",
        network_policy_digest="abc123",
    )
    dumped = metadata.model_dump(mode="json")
    assert dumped["network_mode"] == "no-network"
    assert dumped["network_policy_digest"] == "abc123"
