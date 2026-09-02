"""End-to-end egress enforcement tests. Requires a working Docker daemon.

Every test here drives real containers, a real docker bridge and a real
nftables ruleset. That is deliberate: the two most serious defects in this
feature (a rules file mode that ``--cap-drop ALL`` could not read, and an
orphan reconciler that trusted a manifest-supplied path) both survived a
large mocked unit suite and were only caught by execution.

All tests carry the ``docker`` marker and are excluded from the default run
by ``addopts = "-m 'not docker'"``.
"""

import base64
import json
import os
import re
import socket
import stat
import subprocess
import time
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest

from openhands.workspace.docker import egress_runtime
from openhands.workspace.docker.egress_runtime import (
    EgressRuntime,
    reconcile_orphans,
    start_egress_sidecar,
)
from openhands.workspace.docker.network_policy import (
    NetworkMode,
    WorkspaceNetworkPolicy,
)
from openhands.workspace.docker.nftables_renderer import policy_digest, render_rules


pytestmark = pytest.mark.docker

SIDECAR_IMAGE = "openhands-egress-static:dev"
PROBE_IMAGE = "alpine:3.20"
LABEL_MANAGED = "workspace.managed=true"

# Candidate public hosts for the "is a routable address blocked?" test. The
# reachable one is discovered at runtime and used as a literal IP thereafter,
# because under policy there is no DNS left to resolve a name with.
PUBLIC_CANDIDATES = ("dl-cdn.alpinelinux.org", "example.com", "github.com")
PUBLIC_PORT = 80

# busybox/BIND nslookup print an answer section only when a name was actually
# resolved; the "Address: 127.0.0.11#53" server echo appears either way, so
# matching bare "Address" would be a false positive.
_RESOLVED = re.compile(r"^Name:\s*example\.com", re.MULTILINE)

NAIVE_PORT_BASED_RULES = """table inet t {
  chain output {
    type filter hook output priority filter;
    policy drop;
    ip daddr 127.0.0.11 udp dport 53 reject
    oifname "lo" accept
    ct state established,related accept
    ip daddr 10.0.0.0/8 accept
    reject with icmpx type admin-prohibited
  }
}
"""


def _docker(*args: str, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", *args], capture_output=True, text=True, timeout=timeout
    )


def _free_port() -> int:
    """Reserve a currently-free loopback port and hand back its number."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture(scope="session", autouse=True)
def require_docker() -> None:
    try:
        probe = _docker("version", "--format", "{{.Server.Os}}", timeout=30)
    except (OSError, subprocess.SubprocessError) as exc:
        pytest.skip(f"docker daemon unavailable: {exc}")
    if probe.returncode != 0:
        pytest.skip(f"docker daemon unavailable: {probe.stderr.strip()}")


@pytest.fixture(scope="session")
def sidecar_image() -> str:
    """The prebuilt egress sidecar image, or a skip explaining how to build it."""
    if _docker("image", "inspect", SIDECAR_IMAGE, timeout=60).returncode != 0:
        pytest.skip(
            f"{SIDECAR_IMAGE} not built; run: docker build -t {SIDECAR_IMAGE} "
            "vendor/software-agent-sdk/openhands/workspace/docker/egress_images/static/"
        )
    return SIDECAR_IMAGE


@pytest.fixture(scope="session")
def reachable_public_ip() -> str:
    """A public IP this host can actually reach, discovered without any policy.

    Hardcoding an address risks a false pass: a destination that is dead or
    firewalled off looks exactly like a destination the policy blocked.
    """
    name = f"egress-test-net-{uuid.uuid4().hex[:8]}"
    if _docker("network", "create", "--label", LABEL_MANAGED, name).returncode != 0:
        pytest.skip("cannot create docker network for reachability discovery")
    try:
        probe = _run_in_policy_container(
            name,
            None,
            "for h in "
            + " ".join(PUBLIC_CANDIDATES)
            + "; do ip=$(getent ahostsv4 $h 2>/dev/null | head -1 | awk '{print $1}');"
            f' [ -n "$ip" ] || continue; nc -z -w4 "$ip" {PUBLIC_PORT}'
            ' && echo "REACHABLE $ip" && break; done',
        )
    finally:
        _docker("network", "rm", name)
    for line in probe.stdout.splitlines():
        if line.startswith("REACHABLE "):
            return line.split(None, 1)[1].strip()
    pytest.skip(
        "no reachable public address without a policy; enforcement cannot be "
        "distinguished from an unreachable destination"
    )


@pytest.fixture
def probe_network() -> Iterator[str]:
    """A user-defined bridge, which is what gives a container 127.0.0.11.

    Containers on the default bridge inherit the host resolvers instead, so
    the DNS pair below would not exercise docker's embedded resolver at all.
    """
    name = f"egress-test-net-{uuid.uuid4().hex[:8]}"
    created = _docker("network", "create", "--label", LABEL_MANAGED, name)
    if created.returncode != 0:
        pytest.skip(f"cannot create docker network: {created.stderr.strip()}")
    try:
        yield name
    finally:
        _docker("network", "rm", name)


@pytest.fixture
def state_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect egress state at the module level, never at the real root.

    ``STATE_ROOT`` is bound once at import, so setting the environment
    variable alone would have no effect on an already-imported module; it is
    set as well so that any subprocess re-reading it agrees.
    """
    root = tmp_path / "egress-state"
    root.mkdir(mode=0o700)
    monkeypatch.setenv("OH_EGRESS_STATE_ROOT", str(root))
    monkeypatch.setattr(egress_runtime, "STATE_ROOT", root)
    return root


@pytest.fixture
def sidecars() -> Iterator[list[EgressRuntime]]:
    """Collects started runtimes and tears them down even on assertion failure."""
    started: list[EgressRuntime] = []
    try:
        yield started
    finally:
        for runtime in started:
            runtime.cleanup()


def _run_in_policy_container(
    network: str, rules_text: str | None, script: str
) -> subprocess.CompletedProcess[str]:
    """Apply ``rules_text`` inside a NET_ADMIN container, then run ``script``.

    The ruleset travels as base64 so that nftables comments, braces and
    newlines cannot be reinterpreted by the shell. Packages are installed
    before the policy is applied, since afterwards there is no egress.
    """
    apply_rules = ""
    if rules_text is not None:
        encoded = base64.b64encode(rules_text.encode("utf-8")).decode("ascii")
        apply_rules = (
            f"echo {encoded} | base64 -d > /tmp/r.nft && nft -f /tmp/r.nft && "
        )
    return _docker(
        "run",
        "--rm",
        "--name",
        f"egress-test-{uuid.uuid4().hex[:8]}",
        "--network",
        network,
        "--cap-drop",
        "ALL",
        "--cap-add",
        "NET_ADMIN",
        "--security-opt",
        "no-new-privileges=true",
        "--label",
        LABEL_MANAGED,
        PROBE_IMAGE,
        "sh",
        "-c",
        "apk add --no-cache nftables bind-tools >/dev/null 2>&1 && "
        f"{apply_rules}{script}",
        timeout=180,
    )


def _shipped_rules(mode: NetworkMode = "no-network") -> str:
    return render_rules(WorkspaceNetworkPolicy(mode=mode))


# --------------------------------------------------------------------------
# The DNS regression pair
# --------------------------------------------------------------------------


def test_4a_embedded_resolver_blocked_by_shipped_policy(probe_network: str) -> None:
    """The address-only resolver drop must stop external name resolution."""
    result = _run_in_policy_container(
        probe_network,
        _shipped_rules(),
        "nslookup example.com 127.0.0.11 2>&1 || true",
    )
    combined = result.stdout + result.stderr
    assert not _RESOLVED.search(combined), (
        f"embedded resolver leaked external DNS:\n{combined}"
    )
    assert "timed out" in combined or "no servers could be reached" in combined, (
        f"expected the resolver query to fail, got:\n{combined}"
    )


def test_4b_port_based_resolver_rule_is_insufficient(probe_network: str) -> None:
    """Regression guard: docker's nat-OUTPUT DNAT defeats a dport 53 match.

    This test asserts that an INSECURE rule still works, which is intentional.
    Docker DNATs 127.0.0.11:53 in the nat OUTPUT chain (priority -100), which
    runs before the filter OUTPUT hook (priority 0) our chain is attached to.
    By the time our chain sees the packet the destination port has already
    been rewritten, so `udp dport 53` never matches and the query escapes.
    That is precisely why the shipped rule matches on address only.

    If this test starts FAILING (i.e. the naive rule begins working), docker's
    resolver behaviour has changed -- re-verify the shipped address-only rule
    before anyone relaxes it. Do not "fix" this test by deleting it.
    """
    result = _run_in_policy_container(
        probe_network,
        NAIVE_PORT_BASED_RULES,
        "nslookup example.com 127.0.0.11 2>&1 || true",
    )
    combined = result.stdout + result.stderr
    assert _RESOLVED.search(combined), (
        "The naive dport-53 rule unexpectedly blocked DNS. Docker's DNAT "
        f"behavior may have changed; re-verify the shipped address-only rule.\n"
        f"{combined}"
    )


# --------------------------------------------------------------------------
# Traffic enforcement
# --------------------------------------------------------------------------


def test_public_ip_is_blocked(probe_network: str, reachable_public_ip: str) -> None:
    """A routable public address must be unreachable under the shipped policy."""
    reach = (
        f"nc -z -w4 {reachable_public_ip} {PUBLIC_PORT} && echo REACHED || echo BLOCKED"
    )

    control = _run_in_policy_container(probe_network, None, reach)
    assert "REACHED" in control.stdout, (
        f"{reachable_public_ip} unreachable without a policy; the discovery "
        f"fixture and this run disagree:\n{control.stdout}"
    )

    result = _run_in_policy_container(probe_network, _shipped_rules(), reach)
    assert "BLOCKED" in result.stdout, (
        f"public address reachable under no-network policy:\n{result.stdout}"
    )


def test_loopback_still_works(probe_network: str) -> None:
    """The policy must not break in-namespace loopback traffic."""
    result = _run_in_policy_container(
        probe_network,
        _shipped_rules(),
        "(nc -l -p 9999 >/dev/null &) ; sleep 1 ; "
        "echo ping | nc -w 2 127.0.0.1 9999 && echo LOOPBACK_OK",
    )
    assert "LOOPBACK_OK" in result.stdout, (
        f"loopback broken by policy:\n{result.stdout}\n{result.stderr}"
    )


def test_strict_no_network_blocks_ten_slash_eight(probe_network: str) -> None:
    """strict-no-network drops the internal baseline, in the text and in fact."""
    rules = _shipped_rules("strict-no-network")
    assert "10.0.0.0/8" not in rules

    result = _run_in_policy_container(
        probe_network,
        rules,
        "nc -z -w4 10.0.0.1 80 && echo REACHED || echo BLOCKED",
    )
    assert "BLOCKED" in result.stdout, (
        f"10.0.0.0/8 reachable under strict-no-network:\n{result.stdout}"
    )


def test_no_network_still_permits_the_internal_baseline(probe_network: str) -> None:
    """no-network keeps 10.0.0.0/8: the rule is present and the chain accepts it.

    A policy that also severed the LLM proxy would be indistinguishable from
    strict-no-network, so this pins the difference between the two modes.
    """
    rules = _shipped_rules()
    assert "ip daddr 10.0.0.0/8 accept" in rules

    result = _run_in_policy_container(
        probe_network,
        rules,
        "nft list table inet workspace_egress",
    )
    assert result.returncode == 0, result.stderr
    assert "ip daddr 10.0.0.0/8 accept" in result.stdout


def test_resolver_drop_precedes_loopback_accept_in_applied_chain(
    probe_network: str,
) -> None:
    """Ordering survives the round trip through nft, not just the renderer.

    `oifname "lo" accept` above the resolver drop would re-open DNS, since
    127.0.0.11 is reached over the loopback interface.
    """
    result = _run_in_policy_container(
        probe_network, _shipped_rules(), "nft list table inet workspace_egress"
    )
    assert result.returncode == 0, result.stderr
    applied = result.stdout
    resolver_at = applied.find("127.0.0.11 drop")
    loopback_at = applied.find('oifname "lo" accept')
    assert resolver_at != -1 and loopback_at != -1, applied
    assert resolver_at < loopback_at, (
        f"resolver drop must precede the loopback accept:\n{applied}"
    )


# --------------------------------------------------------------------------
# Sidecar lifecycle
# --------------------------------------------------------------------------


def test_sidecar_starts_and_verifies_its_own_policy(
    sidecar_image: str, state_root: Path, sidecars: list[EgressRuntime]
) -> None:
    """A real sidecar reaches readiness, which means it verified its ruleset.

    This is also the regression test for the 0o600 rules file: a sidecar
    running --cap-drop ALL has no CAP_DAC_OVERRIDE, so an unreadable rules
    file makes readiness unreachable and this test fail.
    """
    policy = WorkspaceNetworkPolicy(mode="no-network")
    runtime = start_egress_sidecar(policy, host_port=_free_port(), image=sidecar_image)
    sidecars.append(runtime)

    assert runtime.sidecar_id is not None
    assert runtime.network_id is not None
    assert runtime.is_alive()
    assert runtime.policy_digest == policy_digest(render_rules(policy))

    logs = _docker("logs", runtime.sidecar_id)
    assert "policy applied and verified" in logs.stdout + logs.stderr

    assert runtime.rules_path is not None
    rules_mode = stat.S_IMODE(runtime.rules_path.stat().st_mode)
    assert rules_mode & stat.S_IROTH, (
        f"rules file mode {rules_mode:o} is unreadable without CAP_DAC_OVERRIDE"
    )
    assert stat.S_IMODE(runtime.rules_path.parent.stat().st_mode) == 0o700, (
        "the world-readable rules file must sit in a private directory"
    )


def test_sidecar_cleanup_removes_every_resource(
    sidecar_image: str, state_root: Path, sidecars: list[EgressRuntime]
) -> None:
    """cleanup() must leave no container, no network and no state behind."""
    runtime = start_egress_sidecar(
        WorkspaceNetworkPolicy(mode="strict-no-network"),
        host_port=_free_port(),
        image=sidecar_image,
    )
    sidecars.append(runtime)
    sidecar_id = runtime.sidecar_id
    network_id = runtime.network_id
    assert sidecar_id is not None and network_id is not None

    runtime.cleanup()

    assert _docker("inspect", sidecar_id).returncode != 0
    assert _docker("network", "inspect", network_id).returncode != 0
    assert not runtime.manifest_path.exists()
    assert runtime.rules_path is not None
    assert not runtime.rules_path.exists()


def test_sidecar_publishes_ports_on_loopback_only(
    sidecar_image: str, state_root: Path, sidecars: list[EgressRuntime]
) -> None:
    """Published ports must bind 127.0.0.1, never 0.0.0.0."""
    host_port = _free_port()
    runtime = start_egress_sidecar(
        WorkspaceNetworkPolicy(mode="no-network"),
        host_port=host_port,
        extra_ports=True,
        image=sidecar_image,
    )
    sidecars.append(runtime)
    assert runtime.sidecar_id is not None

    inspected = _docker(
        "inspect", "-f", "{{json .NetworkSettings.Ports}}", runtime.sidecar_id
    )
    assert inspected.returncode == 0, inspected.stderr
    ports: dict[str, list[dict[str, str]] | None] = json.loads(inspected.stdout)

    published = {
        binding["HostPort"]: binding["HostIp"]
        for bindings in ports.values()
        if bindings
        for binding in bindings
    }
    assert set(published) == {str(host_port), str(host_port + 1), str(host_port + 2)}
    assert set(published.values()) == {"127.0.0.1"}


def test_public_mode_starts_no_sidecar() -> None:
    """public mode applies no policy at all, so nothing may be allocated."""
    policy = WorkspaceNetworkPolicy(mode="public")
    assert policy.requires_sidecar is False
    assert render_rules(policy) == ""


# --------------------------------------------------------------------------
# Orphan reconciliation
# --------------------------------------------------------------------------


def _write_manifest(root: Path, workspace_id: str, payload: dict[str, object]) -> Path:
    directory = root / workspace_id
    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory / "manifest.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    return manifest


def test_reconcile_reclaims_a_dead_controllers_container(state_root: Path) -> None:
    """A real container named by a dead controller's manifest is removed."""
    workspace_id = f"ws-{uuid.uuid4().hex[:12]}"
    name = f"{workspace_id}-egress"
    started = _docker(
        "run",
        "-d",
        "--name",
        name,
        "--label",
        LABEL_MANAGED,
        PROBE_IMAGE,
        "sleep",
        "300",
    )
    assert started.returncode == 0, started.stderr
    container_id = started.stdout.strip()
    try:
        rules_path = state_root / workspace_id / "rules.nft"
        rules_path.parent.mkdir(parents=True, exist_ok=True)
        rules_path.write_text("", encoding="utf-8")
        _write_manifest(
            state_root,
            workspace_id,
            {
                "workspace_id": workspace_id,
                # pid 2**22 is above the default pid_max, so it cannot exist.
                "controller_id": "deadbeef-4194304-abcd1234",
                "network_id": None,
                "sidecar_id": container_id,
                "rules_path": str(rules_path),
                "policy_digest": None,
                "status": "active",
                "lease_expires_at": time.time() - 600,
            },
        )

        assert reconcile_orphans() == [workspace_id]
        assert _docker("inspect", container_id).returncode != 0
        assert not (state_root / workspace_id).exists()
    finally:
        _docker("rm", "-f", container_id)


def test_reconcile_refuses_a_manifest_pointing_outside_the_state_root(
    state_root: Path, tmp_path: Path
) -> None:
    """Regression: a manifest path must never direct deletion off the root.

    ``rules_path`` comes from a file the reconciler did not write, so an
    escaping path has to abort the whole entry rather than wipe the victim.
    """
    victim = tmp_path / "victim"
    victim.mkdir()
    (victim / "precious.txt").write_text("keep me", encoding="utf-8")

    workspace_id = f"ws-{uuid.uuid4().hex[:12]}"
    _write_manifest(
        state_root,
        workspace_id,
        {
            "workspace_id": workspace_id,
            "controller_id": "deadbeef-4194304-abcd1234",
            "network_id": None,
            "sidecar_id": None,
            "rules_path": str(victim / "precious.txt"),
            "policy_digest": None,
            "status": "active",
            "lease_expires_at": time.time() - 600,
        },
    )

    assert reconcile_orphans() == []
    assert (victim / "precious.txt").read_text(encoding="utf-8") == "keep me"
    assert victim.is_dir()


def test_reconcile_leaves_a_live_controller_alone(
    sidecar_image: str, state_root: Path, sidecars: list[EgressRuntime]
) -> None:
    """This process is alive, so its own sidecar must survive a reconcile pass."""
    runtime = start_egress_sidecar(
        WorkspaceNetworkPolicy(mode="no-network"),
        host_port=_free_port(),
        image=sidecar_image,
    )
    sidecars.append(runtime)

    assert reconcile_orphans(now=time.time() + 10_000) == []
    assert runtime.is_alive()


def test_reconcile_ignores_a_world_writable_state_root(state_root: Path) -> None:
    """Manifests from a directory other users can write are never acted upon."""
    workspace_id = f"ws-{uuid.uuid4().hex[:12]}"
    _write_manifest(
        state_root,
        workspace_id,
        {
            "workspace_id": workspace_id,
            "controller_id": "deadbeef-4194304-abcd1234",
            "network_id": None,
            "sidecar_id": None,
            "rules_path": None,
            "policy_digest": None,
            "status": "active",
            "lease_expires_at": time.time() - 600,
        },
    )
    os.chmod(state_root, 0o777)
    try:
        assert reconcile_orphans() == []
        assert (state_root / workspace_id / "manifest.json").exists()
    finally:
        os.chmod(state_root, 0o700)


# --------------------------------------------------------------------------
# Hygiene
# --------------------------------------------------------------------------


def test_no_managed_docker_residue_remains() -> None:
    """Ordered last: the suite must not leak labelled containers or networks."""
    containers = _docker("ps", "-a", "--filter", f"label={LABEL_MANAGED}", "-q")
    networks = _docker("network", "ls", "--filter", f"label={LABEL_MANAGED}", "-q")
    assert containers.stdout.strip() == "", (
        f"leaked managed containers: {containers.stdout}"
    )
    assert networks.stdout.strip() == "", f"leaked managed networks: {networks.stdout}"
