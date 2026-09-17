# Docker Workspace Egress Control (Step 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add optional, per-workspace egress control to the Docker workspace launcher via an nftables sidecar that owns the network namespace, so a workspace can reach only `10.0.0.0/8` plus explicitly allowed destinations, with no host-level firewall state.

**Architecture:** A prebuilt Alpine+nftables sidecar container starts first, applies a host-rendered rules file, verifies it against a digest, and publishes readiness. The workspace container then joins that namespace with `--network container:<id>` and drops `NET_ADMIN`/`NET_RAW`. Mode is resolved centrally in the benchmark evaluation layer (validation, persistence, remote rejection) with an SDK-side default as an enforcement backstop. Every non-`public` mode fails closed.

**Tech Stack:** Python 3.12, Pydantic v2, `uv` workspace, pytest, Docker CLI via `execute_command` (subprocess, argv arrays — never shell strings), nftables, Alpine.

**Spec:** `docs/superpowers/specs/2026-09-02-docker-egress-step1-design.md` (commit `756e9a8`). Read it before starting. Where this plan and the spec disagree, the spec wins.

## Global Constraints

- **Two git repositories.** The launcher lives in the submodule `vendor/software-agent-sdk` (remote `git@github.com:lyuwen/software-agent-sdk.git`, currently detached at `492f5036`). SDK work is committed on branch `network-modes` **inside the submodule**; the root repo (branch `network-modes`) commits the updated submodule pin. Use `git -C vendor/software-agent-sdk <cmd>` — never `cd` into it (the shell keeps state between tool calls and this causes confusion).
- **Single uv workspace.** The submodule packages are `[tool.uv.workspace]` members of the root `pyproject.toml`. Run all tests from the repo root with `uv run pytest`; there is no separate venv to activate.
- **Never use mypy.** Type checking is `pyright` in strict mode, via pre-commit.
- Ruff: `target-version = py312`, `line-length = 88`, `select = ["E","F","I"]`, isort `known-first-party = ["benchmarks","openhands"]`. pycodestyle runs at `--max-line-length=88`.
- Every commit message ends with the trailer `Co-authored-by: openhands <openhands@all-hands.dev>`.
- **Only commit relevant files.** Never `git add -A` or `git add .`.
- All Docker invocations are built as **argv lists** passed to `execute_command(cmd: list[str], ...) -> subprocess.CompletedProcess`. Passing a `str` makes it run under `shell=True` — never do that with any value derived from configuration or task input.
- **Fail closed everywhere.** No code path may fall back to unrestricted networking, privileged exec, or reinterpret an invalid mode as `public`.
- The mandatory destination baseline is exactly `10.0.0.0/8`. Caller entries union with it and can never replace or narrow it.
- `host-allowlist` and `public-bootstrap` are Step 2 values: parse them only to reject them with a message naming Step 2.

## File Structure

**SDK — `vendor/software-agent-sdk/openhands-workspace/openhands/workspace/docker/`**

| File | Responsibility |
|---|---|
| `network_policy.py` (new) | `AllowedEndpoint`, `WorkspaceNetworkPolicy`, mode parsing/validation, baseline resolution. Pure data — no Docker, no I/O. |
| `nftables_renderer.py` (new) | Resolved policy → canonical rules text + SHA-256 digest. Pure functions. |
| `egress_runtime.py` (new) | Sidecar lifecycle: network create + subnet guard, sidecar start, readiness, verification, manifest I/O, locked idempotent cleanup, liveness watcher, reconciliation. |
| `egress_images/static/Dockerfile` (new) | Digest-pinned Alpine + nftables, build-time only. |
| `egress_images/static/entrypoint.sh` (new) | Apply rules, verify, publish readiness, trap signals, stay alive. |
| `workspace.py` (modify) | Extract `_build_run_args()`; add `network_policy` field; add the egress branch; fix `--memory`. |
| `flex_workspace.py` (modify) | Consume the shared builder; delete the duplicated construction. |

**Benchmarks — `benchmarks/`**

| File | Responsibility |
|---|---|
| `utils/workspace_network.py` (new) | Layer 1: resolve/validate `OH_NETWORK_MODE`, reject `remote` + non-public, produce the policy for `EvalMetadata`. |
| `utils/models.py` (modify) | `EvalMetadata.network_policy` + digest fields. |
| `utils/evaluation.py` (modify) | Resolve policy before the metadata write; SIGTERM/SIGINT handler in the child; bounded-deadline pool shutdown. |
| `utils/network_isolation.py` (**delete**) | Replaced. |
| `swebench/run_infer.py` (modify) | Remove the import (`:19`) and call (`:293`). |

**Tests** — SDK: `vendor/software-agent-sdk/tests/workspace/test_network_policy.py`, `test_nftables_renderer.py`, `test_egress_runtime.py`, `test_docker_static_egress.py` (docker-marked). Benchmarks: `tests/test_workspace_network.py`, `tests/test_egress_e2e.py` (docker-marked).

## Task Ordering

Tasks 1–3 are pure and independent of Docker. Task 4 is the refactor that Task 5 depends on. Tasks 6–8 are the benchmark layer and reconciliation. Task 9 removes the old mechanism. Task 10 is the docker-gated end-to-end suite, and Task 11 verifies everything and moves the submodule pin.

---

### Task 0: Branch setup

**Files:** none (git state only)

**Interfaces:**
- Produces: submodule on branch `network-modes`; a `docker` pytest marker other tasks use.

- [ ] **Step 1: Create the SDK working branch**

The submodule is in detached HEAD. Create the branch at the current commit so the pin stays reproducible:

```bash
cd /home/lfu/git-projects/benchmarks-main/.claude/worktrees/network-modes
git -C vendor/software-agent-sdk rev-parse HEAD
git -C vendor/software-agent-sdk checkout -b network-modes
git -C vendor/software-agent-sdk status -sb | head -2
```

Expected: branch `network-modes` at `492f50368fee514f682a8060d53115844ef08b2c`.

- [ ] **Step 2: Register the `docker` pytest marker**

The root `pyproject.toml` has **no** `[tool.pytest.ini_options]` section. Append one so docker tests are excluded by default:

```toml
[tool.pytest.ini_options]
testpaths = ["tests", "benchmarks"]
python_files = ["test_*.py"]
markers = [
    "docker: requires a working Docker daemon (excluded by default; run with -m docker)",
]
addopts = "-m 'not docker'"
```

- [ ] **Step 3: Verify the marker works**

```bash
uv run pytest tests/ --collect-only -q 2>&1 | tail -3
```

Expected: collection succeeds, no "unknown marker" warnings.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "$(printf 'test: register docker pytest marker for egress tests\n\nCo-authored-by: openhands <openhands@all-hands.dev>')"
```

---

### Task 1: Network policy model

**Files:**
- Create: `vendor/software-agent-sdk/openhands-workspace/openhands/workspace/docker/network_policy.py`
- Test: `vendor/software-agent-sdk/tests/workspace/test_network_policy.py`

**Interfaces:**
- Produces:
  - `INTERNAL_BASELINE: IPv4Network` — `ip_network("10.0.0.0/8")`
  - `NetworkMode = Literal["public","static-allowlist","no-network","strict-no-network"]`
  - `class AllowedEndpoint(BaseModel)` — `destination: IPv4Network | IPv6Network`, `protocol: Literal["tcp","udp"] | None`, `ports: tuple[int, ...]`, `description: str | None`
  - `class WorkspaceNetworkPolicy(BaseModel)` — `mode: NetworkMode`, `allowed_endpoints: tuple[AllowedEndpoint, ...]`; method `resolved_endpoints() -> tuple[AllowedEndpoint, ...]`
  - `parse_network_mode(raw: str | None) -> NetworkMode`
  - `policy_from_env(env: Mapping[str, str] | None = None) -> WorkspaceNetworkPolicy`

- [ ] **Step 1: Write the failing tests**

Create `vendor/software-agent-sdk/tests/workspace/test_network_policy.py`:

```python
"""Tests for the workspace network policy model."""

from ipaddress import ip_network

import pytest
from pydantic import ValidationError

from openhands.workspace.docker.network_policy import (
    INTERNAL_BASELINE,
    AllowedEndpoint,
    WorkspaceNetworkPolicy,
    parse_network_mode,
    policy_from_env,
)


def test_baseline_is_ten_slash_eight():
    assert INTERNAL_BASELINE == ip_network("10.0.0.0/8")


def test_unset_mode_is_public():
    assert parse_network_mode(None) == "public"
    assert parse_network_mode("") == "public"
    assert parse_network_mode("   ") == "public"


def test_invalid_mode_is_rejected_not_coerced():
    with pytest.raises(ValueError, match="OH_NETWORK_MODE"):
        parse_network_mode("no-netwrok")


def test_step2_modes_rejected_naming_step2():
    for mode in ("host-allowlist", "public-bootstrap"):
        with pytest.raises(ValueError, match="Step 2"):
            parse_network_mode(mode)


def test_no_network_resolves_to_exactly_the_baseline():
    policy = WorkspaceNetworkPolicy(mode="no-network")
    resolved = policy.resolved_endpoints()
    assert len(resolved) == 1
    assert resolved[0].destination == INTERNAL_BASELINE
    assert resolved[0].protocol is None
    assert resolved[0].ports == ()


def test_strict_no_network_resolves_to_no_destinations():
    assert WorkspaceNetworkPolicy(mode="strict-no-network").resolved_endpoints() == ()


def test_caller_entries_union_with_baseline_and_cannot_replace_it():
    policy = WorkspaceNetworkPolicy(
        mode="static-allowlist",
        allowed_endpoints=(AllowedEndpoint(destination=ip_network("192.0.2.0/24")),),
    )
    destinations = [e.destination for e in policy.resolved_endpoints()]
    assert INTERNAL_BASELINE in destinations
    assert ip_network("192.0.2.0/24") in destinations


def test_narrowing_the_baseline_still_yields_the_full_baseline():
    """A caller entry for a 10.x subnet must not shrink the /8."""
    policy = WorkspaceNetworkPolicy(
        mode="static-allowlist",
        allowed_endpoints=(
            AllowedEndpoint(destination=ip_network("10.1.2.0/24"), protocol="tcp", ports=(443,)),
        ),
    )
    unrestricted = [
        e.destination
        for e in policy.resolved_endpoints()
        if e.protocol is None and e.ports == ()
    ]
    assert INTERNAL_BASELINE in unrestricted


def test_no_network_rejects_caller_endpoints():
    with pytest.raises(ValidationError, match="no-network"):
        WorkspaceNetworkPolicy(
            mode="no-network",
            allowed_endpoints=(AllowedEndpoint(destination=ip_network("192.0.2.0/24")),),
        )


def test_public_rejects_allowlist_fields():
    with pytest.raises(ValidationError, match="public"):
        WorkspaceNetworkPolicy(
            mode="public",
            allowed_endpoints=(AllowedEndpoint(destination=ip_network("192.0.2.0/24")),),
        )


def test_protocol_requires_nonempty_ports():
    with pytest.raises(ValidationError, match="ports"):
        AllowedEndpoint(destination=ip_network("192.0.2.0/24"), protocol="tcp", ports=())


def test_ports_require_protocol():
    with pytest.raises(ValidationError, match="protocol"):
        AllowedEndpoint(destination=ip_network("192.0.2.0/24"), ports=(443,))


@pytest.mark.parametrize("port", [0, 65536, -1])
def test_port_range_enforced(port):
    with pytest.raises(ValidationError):
        AllowedEndpoint(destination=ip_network("192.0.2.0/24"), protocol="tcp", ports=(port,))


@pytest.mark.parametrize(
    "dest", ["224.0.0.0/4", "169.254.0.0/16", "0.0.0.0/0", "255.255.255.255/32"]
)
def test_unsafe_destinations_rejected(dest):
    with pytest.raises(ValidationError):
        AllowedEndpoint(destination=ip_network(dest))


def test_env_unset_gives_public():
    assert policy_from_env({}).mode == "public"


def test_env_invalid_raises():
    with pytest.raises(ValueError):
        policy_from_env({"OH_NETWORK_MODE": "bogus"})
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
uv run pytest vendor/software-agent-sdk/tests/workspace/test_network_policy.py -q
```

Expected: collection error — `ModuleNotFoundError: openhands.workspace.docker.network_policy`.

- [ ] **Step 3: Implement the model**

Create `vendor/software-agent-sdk/openhands-workspace/openhands/workspace/docker/network_policy.py`:

```python
"""Typed network policy for Docker workspace egress control.

Pure data and validation: no Docker calls and no filesystem I/O, so this
module is cheap to unit test.
"""

import os
from collections.abc import Mapping
from ipaddress import IPv4Network, IPv6Network, ip_network
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


INTERNAL_BASELINE: IPv4Network = ip_network("10.0.0.0/8")
"""Mandatory internal destination: LLM proxy and package mirrors live here."""

NetworkMode = Literal[
    "public", "static-allowlist", "no-network", "strict-no-network"
]

_STEP2_MODES = frozenset({"host-allowlist", "public-bootstrap"})
_VALID_MODES = frozenset({"public", "static-allowlist", "no-network", "strict-no-network"})

MAX_ENDPOINTS = 64
MAX_PORTS_PER_ENDPOINT = 32

ENV_VAR = "OH_NETWORK_MODE"


def parse_network_mode(raw: str | None) -> NetworkMode:
    """Parse an OH_NETWORK_MODE value.

    Unset/empty means "public" (preserving historical unrestricted behavior).
    Anything unrecognized is an error — never silently coerced to public.
    """
    if raw is None or not raw.strip():
        return "public"
    value = raw.strip().lower()
    if value in _STEP2_MODES:
        raise ValueError(
            f"{ENV_VAR}={value!r} requires Step 2 (GOST hostname allowlisting), "
            "which is not implemented. Use one of: "
            f"{', '.join(sorted(_VALID_MODES))}."
        )
    if value not in _VALID_MODES:
        raise ValueError(
            f"Invalid {ENV_VAR}={raw!r}. Expected one of: "
            f"{', '.join(sorted(_VALID_MODES))}."
        )
    return value  # type: ignore[return-value]


class AllowedEndpoint(BaseModel):
    """One allowed destination, optionally narrowed to a protocol and ports."""

    model_config = ConfigDict(frozen=True)

    destination: IPv4Network | IPv6Network
    protocol: Literal["tcp", "udp"] | None = None
    ports: tuple[int, ...] = ()
    description: str | None = None

    @field_validator("destination")
    @classmethod
    def _reject_unsafe_destinations(
        cls, v: IPv4Network | IPv6Network
    ) -> IPv4Network | IPv6Network:
        if v.is_multicast:
            raise ValueError(f"multicast destination not allowed: {v}")
        if v.is_link_local:
            raise ValueError(f"link-local destination not allowed: {v}")
        if v.is_unspecified or int(v.network_address) == 0:
            raise ValueError(f"unspecified destination not allowed: {v}")
        if isinstance(v, IPv4Network) and v.broadcast_address == v.network_address:
            if str(v.network_address) == "255.255.255.255":
                raise ValueError(f"broadcast destination not allowed: {v}")
        return v

    @field_validator("ports")
    @classmethod
    def _validate_ports(cls, v: tuple[int, ...]) -> tuple[int, ...]:
        if len(v) > MAX_PORTS_PER_ENDPOINT:
            raise ValueError(f"too many ports: {len(v)} > {MAX_PORTS_PER_ENDPOINT}")
        for port in v:
            if not 1 <= port <= 65535:
                raise ValueError(f"port out of range 1..65535: {port}")
        return tuple(sorted(set(v)))

    @model_validator(mode="after")
    def _protocol_and_ports_agree(self) -> "AllowedEndpoint":
        if self.protocol is not None and not self.ports:
            raise ValueError(
                "a non-empty 'ports' list is required when 'protocol' is set"
            )
        if self.protocol is None and self.ports:
            raise ValueError("'ports' requires an explicit 'protocol'")
        return self

    def sort_key(self) -> tuple[int, str, str, tuple[int, ...]]:
        """Deterministic ordering key for stable rule rendering."""
        return (
            self.destination.version,
            str(self.destination),
            self.protocol or "",
            self.ports,
        )


class WorkspaceNetworkPolicy(BaseModel):
    """The egress policy applied to one workspace."""

    model_config = ConfigDict(frozen=True)

    mode: NetworkMode = "public"
    allowed_endpoints: tuple[AllowedEndpoint, ...] = Field(default=())

    @model_validator(mode="after")
    def _mode_accepts_fields(self) -> "WorkspaceNetworkPolicy":
        if self.allowed_endpoints:
            if self.mode == "public":
                raise ValueError(
                    "mode 'public' must not carry allowed_endpoints; it applies "
                    "no policy at all"
                )
            if self.mode == "no-network":
                raise ValueError(
                    "mode 'no-network' must not carry caller-supplied "
                    "allowed_endpoints; it resolves to the fixed 10.0.0.0/8 baseline"
                )
            if self.mode == "strict-no-network":
                raise ValueError(
                    "mode 'strict-no-network' must not carry allowed_endpoints; "
                    "it resolves to no external destinations"
                )
        if len(self.allowed_endpoints) > MAX_ENDPOINTS:
            raise ValueError(
                f"too many endpoints: {len(self.allowed_endpoints)} > {MAX_ENDPOINTS}"
            )
        return self

    @property
    def requires_sidecar(self) -> bool:
        return self.mode != "public"

    def resolved_endpoints(self) -> tuple[AllowedEndpoint, ...]:
        """Final destination set: baseline unioned with caller entries.

        Caller entries are additive only. The unrestricted 10.0.0.0/8 baseline
        is always present except in 'strict-no-network'.
        """
        if self.mode in ("public", "strict-no-network"):
            return ()

        baseline = AllowedEndpoint(
            destination=INTERNAL_BASELINE,
            description="internal-llm-proxy-and-package-mirrors",
        )
        merged: dict[tuple, AllowedEndpoint] = {baseline.sort_key(): baseline}
        for endpoint in self.allowed_endpoints:
            merged.setdefault(endpoint.sort_key(), endpoint)
        return tuple(sorted(merged.values(), key=lambda e: e.sort_key()))


def policy_from_env(
    env: Mapping[str, str] | None = None,
) -> WorkspaceNetworkPolicy:
    """Build a policy from OH_NETWORK_MODE. Raises on an invalid value."""
    source = os.environ if env is None else env
    return WorkspaceNetworkPolicy(mode=parse_network_mode(source.get(ENV_VAR)))
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
uv run pytest vendor/software-agent-sdk/tests/workspace/test_network_policy.py -q
```

Expected: all pass.

- [ ] **Step 5: Commit (in the submodule)**

```bash
git -C vendor/software-agent-sdk add \
  openhands-workspace/openhands/workspace/docker/network_policy.py \
  tests/workspace/test_network_policy.py
git -C vendor/software-agent-sdk commit -m "$(printf 'feat(workspace): typed network policy model for egress control\n\nCo-authored-by: openhands <openhands@all-hands.dev>')"
```

---

### Task 2: nftables renderer

**Files:**
- Create: `vendor/software-agent-sdk/openhands-workspace/openhands/workspace/docker/nftables_renderer.py`
- Test: `vendor/software-agent-sdk/tests/workspace/test_nftables_renderer.py`

**Interfaces:**
- Consumes: `WorkspaceNetworkPolicy`, `AllowedEndpoint`, `INTERNAL_BASELINE` from Task 1.
- Produces:
  - `DOCKER_EMBEDDED_RESOLVER = "127.0.0.11"`
  - `TABLE_NAME = "workspace_egress"`
  - `render_rules(policy: WorkspaceNetworkPolicy) -> str`
  - `policy_digest(rules_text: str) -> str` (SHA-256 hex)

**Critical context — read before implementing.** The rule `ip daddr 127.0.0.11 drop` must come **before** `oifname "lo" accept`, and must match on **address only**. A `udp dport 53` match does **not** work: Docker DNATs `127.0.0.11:53` in the `nat` OUTPUT chain (priority −100), which runs before this `filter` OUTPUT hook (priority 0), so the port is already rewritten. This was verified empirically; see spec §2.7. Without the address-only rule, a container resolves arbitrary external names through Docker's resolver despite `policy drop`.

- [ ] **Step 1: Write the failing tests**

Create `vendor/software-agent-sdk/tests/workspace/test_nftables_renderer.py`:

```python
"""Tests for nftables rule rendering."""

from ipaddress import ip_network

from openhands.workspace.docker.network_policy import (
    AllowedEndpoint,
    WorkspaceNetworkPolicy,
)
from openhands.workspace.docker.nftables_renderer import (
    DOCKER_EMBEDDED_RESOLVER,
    TABLE_NAME,
    policy_digest,
    render_rules,
)


def _lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def test_renders_inet_table_with_drop_policy():
    text = render_rules(WorkspaceNetworkPolicy(mode="no-network"))
    assert f"table inet {TABLE_NAME}" in text
    assert "type filter hook output priority filter;" in text
    assert "policy drop;" in text


def test_resolver_drop_precedes_loopback_accept():
    """Ordering is load-bearing: see spec 2.7."""
    lines = _lines(render_rules(WorkspaceNetworkPolicy(mode="no-network")))
    resolver_idx = next(
        i for i, ln in enumerate(lines) if DOCKER_EMBEDDED_RESOLVER in ln
    )
    loopback_idx = next(i for i, ln in enumerate(lines) if 'oifname "lo"' in ln)
    assert resolver_idx < loopback_idx


def test_resolver_rule_matches_address_only_not_port():
    """A dport 53 match is defeated by docker's nat-OUTPUT DNAT."""
    lines = _lines(render_rules(WorkspaceNetworkPolicy(mode="no-network")))
    resolver_rule = next(ln for ln in lines if DOCKER_EMBEDDED_RESOLVER in ln)
    assert "dport" not in resolver_rule
    assert "53" not in resolver_rule
    assert resolver_rule.endswith("drop")


def test_no_network_emits_baseline_accept():
    text = render_rules(WorkspaceNetworkPolicy(mode="no-network"))
    assert "ip daddr 10.0.0.0/8 accept" in text


def test_strict_no_network_has_no_baseline():
    text = render_rules(WorkspaceNetworkPolicy(mode="strict-no-network"))
    assert "10.0.0.0/8" not in text
    assert 'oifname "lo" accept' in text
    assert "ct state established,related accept" in text


def test_final_reject_present():
    text = render_rules(WorkspaceNetworkPolicy(mode="no-network"))
    assert "reject with icmpx type admin-prohibited" in _lines(text)[-3]


def test_protocol_and_ports_rendered():
    policy = WorkspaceNetworkPolicy(
        mode="static-allowlist",
        allowed_endpoints=(
            AllowedEndpoint(
                destination=ip_network("192.0.2.0/24"), protocol="tcp", ports=(443, 80)
            ),
        ),
    )
    text = render_rules(policy)
    assert "ip daddr 192.0.2.0/24 tcp dport { 80, 443 } accept" in text


def test_ipv6_endpoint_uses_ip6_keyword():
    policy = WorkspaceNetworkPolicy(
        mode="static-allowlist",
        allowed_endpoints=(AllowedEndpoint(destination=ip_network("2001:db8::/32")),),
    )
    assert "ip6 daddr 2001:db8::/32 accept" in render_rules(policy)


def test_rendering_is_deterministic_regardless_of_input_order():
    a = AllowedEndpoint(destination=ip_network("192.0.2.0/24"))
    b = AllowedEndpoint(destination=ip_network("198.51.100.0/24"))
    first = render_rules(
        WorkspaceNetworkPolicy(mode="static-allowlist", allowed_endpoints=(a, b))
    )
    second = render_rules(
        WorkspaceNetworkPolicy(mode="static-allowlist", allowed_endpoints=(b, a))
    )
    assert first == second
    assert policy_digest(first) == policy_digest(second)


def test_duplicate_endpoints_deduplicated():
    e = AllowedEndpoint(destination=ip_network("192.0.2.0/24"))
    text = render_rules(
        WorkspaceNetworkPolicy(mode="static-allowlist", allowed_endpoints=(e, e))
    )
    assert text.count("ip daddr 192.0.2.0/24 accept") == 1


def test_digest_changes_when_policy_changes():
    one = render_rules(WorkspaceNetworkPolicy(mode="no-network"))
    two = render_rules(WorkspaceNetworkPolicy(mode="strict-no-network"))
    assert policy_digest(one) != policy_digest(two)


def test_public_mode_renders_nothing():
    assert render_rules(WorkspaceNetworkPolicy(mode="public")) == ""
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
uv run pytest vendor/software-agent-sdk/tests/workspace/test_nftables_renderer.py -q
```

Expected: `ModuleNotFoundError: ...nftables_renderer`.

- [ ] **Step 3: Implement the renderer**

Create `vendor/software-agent-sdk/openhands-workspace/openhands/workspace/docker/nftables_renderer.py`:

```python
"""Render a validated network policy into an nftables ruleset.

The rules text is built only from values that have already passed the
validation in network_policy.py. No caller-provided string is ever
interpolated as a raw nftables fragment.
"""

import hashlib
from ipaddress import IPv6Network

from .network_policy import AllowedEndpoint, WorkspaceNetworkPolicy


TABLE_NAME = "workspace_egress"

DOCKER_EMBEDDED_RESOLVER = "127.0.0.11"
"""Docker's embedded DNS resolver.

Traffic to this address must be dropped by ADDRESS ONLY, before the loopback
accept. Docker DNATs 127.0.0.11:53 in the nat OUTPUT chain (priority -100),
which runs before this filter OUTPUT hook (priority 0), so a `dport 53` match
never fires and external names remain resolvable despite `policy drop`.
Verified empirically; see the design spec section 2.7.
"""


def _render_endpoint(endpoint: AllowedEndpoint) -> str:
    family = "ip6" if isinstance(endpoint.destination, IPv6Network) else "ip"
    parts = [f"{family} daddr {endpoint.destination}"]
    if endpoint.protocol is not None:
        ports = ", ".join(str(p) for p in endpoint.ports)
        parts.append(f"{endpoint.protocol} dport {{ {ports} }}")
    parts.append("accept")
    return " ".join(parts)


def render_rules(policy: WorkspaceNetworkPolicy) -> str:
    """Render the canonical ruleset. Returns "" for public mode (no sidecar)."""
    if not policy.requires_sidecar:
        return ""

    lines = [
        f"table inet {TABLE_NAME} {{",
        "    chain output {",
        "        type filter hook output priority filter;",
        "        policy drop;",
        "",
        "        # Docker's embedded resolver forwards external queries from the",
        "        # host namespace. Match on address only -- a dport 53 match is",
        "        # defeated by docker's nat-OUTPUT DNAT. See spec 2.7.",
        f"        ip daddr {DOCKER_EMBEDDED_RESOLVER} drop",
        "",
        '        oifname "lo" accept',
        "        ct state established,related accept",
        "",
    ]
    lines.extend(
        f"        {_render_endpoint(e)}" for e in policy.resolved_endpoints()
    )
    lines.extend(
        [
            "",
            "        reject with icmpx type admin-prohibited",
            "    }",
            "}",
            "",
        ]
    )
    return "\n".join(lines)


def policy_digest(rules_text: str) -> str:
    """SHA-256 of the canonical rules text, used to verify the applied ruleset."""
    return hashlib.sha256(rules_text.encode("utf-8")).hexdigest()
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
uv run pytest vendor/software-agent-sdk/tests/workspace/test_nftables_renderer.py -q
```

Expected: all pass. If `test_final_reject_present` fails on the index, print `_lines(text)[-4:]` and adjust the assertion to locate the reject line rather than changing the renderer's output.

- [ ] **Step 5: Commit**

```bash
git -C vendor/software-agent-sdk add \
  openhands-workspace/openhands/workspace/docker/nftables_renderer.py \
  tests/workspace/test_nftables_renderer.py
git -C vendor/software-agent-sdk commit -m "$(printf 'feat(workspace): nftables renderer with embedded-resolver block\n\nThe resolver rule matches on address only and precedes the loopback\naccept: docker DNATs 127.0.0.11:53 in nat OUTPUT (prio -100) before the\nfilter hook (prio 0), so a dport 53 match never fires.\n\nCo-authored-by: openhands <openhands@all-hands.dev>')"
```

---

### Task 3: Sidecar image

**Files:**
- Create: `vendor/software-agent-sdk/openhands-workspace/openhands/workspace/docker/egress_images/static/Dockerfile`
- Create: `vendor/software-agent-sdk/openhands-workspace/openhands/workspace/docker/egress_images/static/entrypoint.sh`

**Interfaces:**
- Produces: image build context; readiness file path `/tmp/workspace-egress.ready`; rules mount path `/etc/workspace-egress/rules.nft`; expected digest via env `EGRESS_POLICY_DIGEST`.

- [ ] **Step 1: Resolve the pinned base image digest**

Do not invent a digest. Resolve the real one:

```bash
docker pull alpine:3.20
docker inspect --format '{{index .RepoDigests 0}}' alpine:3.20
```

Use the printed `alpine@sha256:...` value verbatim in the `FROM` line below.

- [ ] **Step 2: Write the entrypoint**

Create `.../egress_images/static/entrypoint.sh`:

```sh
#!/bin/sh
# Workspace egress sidecar entrypoint.
#
# Owns the network namespace the workspace container joins. Applies the
# host-rendered nftables policy, verifies it, then publishes readiness and
# stays alive. Any failure exits non-zero WITHOUT publishing readiness, so
# the controller fails closed.
set -eu

RULES_FILE="/etc/workspace-egress/rules.nft"
READY_FILE="/tmp/workspace-egress.ready"
TABLE_NAME="workspace_egress"

cleanup() {
    rm -f "$READY_FILE" 2>/dev/null || true
}
trap cleanup EXIT

on_signal() {
    cleanup
    exit 0
}
trap on_signal TERM INT HUP

if [ ! -f "$RULES_FILE" ]; then
    echo "egress: missing rules file $RULES_FILE" >&2
    exit 1
fi

if ! nft -f "$RULES_FILE"; then
    echo "egress: failed to apply ruleset" >&2
    exit 1
fi

# Verify the applied ruleset rather than trusting that nft exited 0.
applied="$(nft list table inet "$TABLE_NAME" 2>/dev/null || true)"
if [ -z "$applied" ]; then
    echo "egress: table inet $TABLE_NAME absent after apply" >&2
    exit 1
fi

for required in \
    "policy drop" \
    "ip daddr 127.0.0.11 drop" \
    "ct state established,related accept"
do
    if ! printf '%s\n' "$applied" | grep -qF "$required"; then
        echo "egress: verification failed, missing: $required" >&2
        exit 1
    fi
done

# The resolver drop must precede the loopback accept, or DNS escapes.
resolver_line="$(printf '%s\n' "$applied" | grep -n 'daddr 127.0.0.11' | head -1 | cut -d: -f1)"
loopback_line="$(printf '%s\n' "$applied" | grep -n 'oifname "lo"' | head -1 | cut -d: -f1)"
if [ -n "$loopback_line" ] && [ "$resolver_line" -ge "$loopback_line" ]; then
    echo "egress: resolver drop must precede loopback accept" >&2
    exit 1
fi

echo "egress: policy applied and verified"
touch "$READY_FILE"

# Stay alive as the namespace owner, remaining responsive to signals.
while true; do
    sleep 1 &
    wait $!
done
```

- [ ] **Step 3: Write the Dockerfile**

Create `.../egress_images/static/Dockerfile`, substituting the digest from Step 1:

```dockerfile
# Minimal nftables sidecar. All packages are installed at BUILD time; the
# runtime entrypoint must never install anything.
FROM alpine@sha256:REPLACE_WITH_DIGEST_FROM_STEP_1

RUN apk add --no-cache nftables

COPY --chmod=755 entrypoint.sh /usr/local/bin/egress-entrypoint

ENTRYPOINT ["/usr/local/bin/egress-entrypoint"]
```

- [ ] **Step 4: Build and verify the image behaves correctly**

```bash
cd /home/lfu/git-projects/benchmarks-main/.claude/worktrees/network-modes/vendor/software-agent-sdk/openhands-workspace/openhands/workspace/docker/egress_images/static
docker build -t openhands-egress-static:dev .
cd /home/lfu/git-projects/benchmarks-main/.claude/worktrees/network-modes

# Valid rules -> readiness
mkdir -p /tmp/egress-check
cat > /tmp/egress-check/rules.nft <<'NFT'
table inet workspace_egress {
    chain output {
        type filter hook output priority filter;
        policy drop;
        ip daddr 127.0.0.11 drop
        oifname "lo" accept
        ct state established,related accept
        ip daddr 10.0.0.0/8 accept
        reject with icmpx type admin-prohibited
    }
}
NFT
cid=$(docker run -d --rm --cap-drop ALL --cap-add NET_ADMIN \
  --security-opt no-new-privileges=true --read-only \
  --tmpfs /tmp:rw,noexec,nosuid,nodev,size=1m \
  --mount type=bind,src=/tmp/egress-check/rules.nft,dst=/etc/workspace-egress/rules.nft,readonly \
  openhands-egress-static:dev)
sleep 2
docker exec "$cid" test -f /tmp/workspace-egress.ready && echo "READY_OK"
docker stop -t 5 "$cid" >/dev/null && echo "STOPPED_CLEANLY"

# Invalid rules -> non-zero exit, no readiness
printf 'this is not nftables\n' > /tmp/egress-check/bad.nft
docker run --rm --cap-drop ALL --cap-add NET_ADMIN \
  --mount type=bind,src=/tmp/egress-check/bad.nft,dst=/etc/workspace-egress/rules.nft,readonly \
  openhands-egress-static:dev >/dev/null 2>&1; echo "invalid_rules_exit=$?"
```

Expected: `READY_OK`, `STOPPED_CLEANLY`, and `invalid_rules_exit=1`.

- [ ] **Step 5: Commit**

```bash
git -C vendor/software-agent-sdk add \
  openhands-workspace/openhands/workspace/docker/egress_images/static/Dockerfile \
  openhands-workspace/openhands/workspace/docker/egress_images/static/entrypoint.sh
git -C vendor/software-agent-sdk commit -m "$(printf 'feat(workspace): prebuilt nftables egress sidecar image\n\nCo-authored-by: openhands <openhands@all-hands.dev>')"
```

---

### Task 4: Extract shared run-argument construction

**Files:**
- Modify: `vendor/software-agent-sdk/openhands-workspace/openhands/workspace/docker/workspace.py` (add `_build_run_args`; fix `--memory` at `:219-220` and `:237`)
- Modify: `vendor/software-agent-sdk/openhands-workspace/openhands/workspace/docker/flex_workspace.py` (consume the builder; `:242-294`)
- Test: `vendor/software-agent-sdk/tests/workspace/test_run_args.py` (new)

**Interfaces:**
- Produces on `DockerWorkspace`:
  - `_build_run_args(self, image: str, *, extra_flags: list[str] | None = None, entrypoint: list[str] | None = None, command: list[str] | None = None, container_name: str) -> list[str]`

**Why this comes before the egress branch:** `FlexWorkspace._start_container` currently duplicates ~100 lines of `DockerWorkspace._start_container`. Adding the egress branch to both would create two enforcement paths — exactly what the spec forbids. Extract first.

**Behavior must not change**, with one deliberate exception: both files today append a hardcoded `--memory=14g` *after* `--memory {memory_limit}`, so `memory_limit` and `OH_WORKSPACE_MEMORY_LIMIT` are silently inert. Remove the hardcoded flag **and** change `memory_limit`'s default from `"13g"` to `"14g"`, so the effective limit stays 14g and the env var becomes functional.

- [ ] **Step 1: Write the failing tests**

Create `vendor/software-agent-sdk/tests/workspace/test_run_args.py`:

```python
"""Tests for shared docker run-argument construction."""

from unittest.mock import patch

import pytest

from openhands.workspace import DockerWorkspace, FlexWorkspace


@pytest.fixture
def docker_ws():
    with patch.object(DockerWorkspace, "_start_container"):
        ws = DockerWorkspace(server_image="test:latest")
    ws.host_port = 30000
    return ws


@pytest.fixture
def flex_ws():
    with patch.object(FlexWorkspace, "_start_container"):
        ws = FlexWorkspace(base_image="base:latest")
    ws.host_port = 30000
    return ws


def _memory_flags(args: list[str]) -> list[str]:
    return [
        args[i + 1] if args[i] == "--memory" else args[i].split("=", 1)[1]
        for i in range(len(args))
        if args[i] == "--memory" or args[i].startswith("--memory=")
    ]


def test_memory_limit_default_is_14g(docker_ws):
    assert docker_ws.memory_limit == "14g"


def test_exactly_one_memory_flag_is_emitted(docker_ws):
    args = docker_ws._build_run_args("img:latest", container_name="c")
    assert _memory_flags(args) == ["14g"]


def test_memory_limit_is_honoured(docker_ws):
    docker_ws.memory_limit = "7g"
    args = docker_ws._build_run_args("img:latest", container_name="c")
    assert _memory_flags(args) == ["7g"]


def test_flex_emits_one_memory_flag(flex_ws):
    args = flex_ws._build_run_args("img:latest", container_name="c")
    assert _memory_flags(args) == ["14g"]


def test_args_are_argv_list_not_shell_string(docker_ws):
    args = docker_ws._build_run_args("img:latest", container_name="c")
    assert isinstance(args, list)
    assert all(isinstance(a, str) for a in args)
    assert args[:3] == ["docker", "run", "-d"]


def test_ports_published_on_all_interfaces_in_public_mode(docker_ws):
    args = docker_ws._build_run_args("img:latest", container_name="c")
    assert "30000:8000" in args


def test_container_name_is_used(docker_ws):
    args = docker_ws._build_run_args("img:latest", container_name="my-name")
    assert args[args.index("--name") + 1] == "my-name"


def test_entrypoint_and_command_appended_after_image(docker_ws):
    args = docker_ws._build_run_args(
        "img:latest",
        container_name="c",
        entrypoint=["/bin/python"],
        command=["-m", "server"],
    )
    assert args[args.index("--entrypoint") + 1] == "/bin/python"
    image_idx = args.index("img:latest")
    assert args[image_idx + 1 :] == ["-m", "server"]
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
uv run pytest vendor/software-agent-sdk/tests/workspace/test_run_args.py -q
```

Expected: `AttributeError: ... no attribute '_build_run_args'`, and `test_memory_limit_default_is_14g` fails with `'13g' != '14g'`.

- [ ] **Step 3: Change the memory default in `workspace.py`**

At `workspace.py:119-122`, change the default from `"13g"` to `"14g"`:

```python
    memory_limit: str = Field(
        default_factory=lambda: os.getenv("OH_WORKSPACE_MEMORY_LIMIT", "14g"),
        description="Docker container memory limit.",
    )
```

- [ ] **Step 4: Add `_build_run_args` to `DockerWorkspace`**

Insert this method into `DockerWorkspace` (place it directly above `_start_container`). It is the single construction path both launchers use:

```python
    def _build_run_args(
        self,
        image: str,
        *,
        extra_flags: list[str] | None = None,
        entrypoint: list[str] | None = None,
        command: list[str] | None = None,
        container_name: str,
    ) -> list[str]:
        """Build the full `docker run` argv for this workspace.

        Shared by DockerWorkspace and FlexWorkspace so there is exactly one
        place where run arguments -- and therefore network enforcement -- are
        constructed. Always returns an argv list; never a shell string.
        """
        flags: list[str] = list(extra_flags or [])

        for key in self.forward_env:
            if key in os.environ:
                flags += ["-e", f"{key}={os.environ[key]}"]
        for key, val in os.environ.items():
            if key.startswith("OH_") and key not in self.forward_env:
                flags += ["-e", f"{key}={val}"]

        if self.mount_dir:
            flags += ["-v", f"{self.mount_dir}:/workspace"]
            logger.info(
                f"Mounting host dir {self.mount_dir} to container path /workspace"
            )
        for volume in self.bind_volumes:
            flags += ["-v", volume]

        if self.memory_limit:
            flags += ["--memory", self.memory_limit]

        flags += self._build_port_args()

        if self.enable_gpu:
            flags += ["--gpus", "all"]

        run_cmd = [
            "docker",
            "run",
            "-d",
            "--platform",
            self.platform,
            "--rm",
            "--name",
            container_name,
            *flags,
        ]
        if entrypoint:
            run_cmd += ["--entrypoint", *entrypoint]
        run_cmd.append(image)
        if command:
            run_cmd += command
        return run_cmd

    def _build_port_args(self) -> list[str]:
        """Publish the agent-server port (and optional VSCode/VNC ports).

        Overridden when a sidecar owns the namespace: in that case the sidecar
        publishes the ports and the workspace must publish none.
        """
        ports = ["-p", f"{self.host_port}:8000"]
        if self.extra_ports:
            ports += [
                "-p",
                f"{self.host_port + 1}:8001",  # VSCode
                "-p",
                f"{self.host_port + 2}:8002",  # Desktop VNC
            ]
        return ports
```

- [ ] **Step 5: Rewrite `DockerWorkspace._start_container` to use the builder**

Replace the flag-building block (currently `workspace.py:197-255`, from `# Prepare Docker run flags` through the end of the `run_cmd = [...]` literal) with:

```python
        run_cmd = self._build_run_args(
            image,
            container_name=f"agent-server-{uuid.uuid4()}",
            command=["--host", "0.0.0.0", "--port", "8000"],
        )
```

Everything before (`self._image_name`, port selection, docker availability check) and after (`proc = execute_command(run_cmd)` onward) stays unchanged. **Delete the hardcoded `flags += ["--memory=14g"]` line entirely** — do not re-add it.

- [ ] **Step 6: Rewrite `FlexWorkspace._start_container` to use the builder**

In `flex_workspace.py`, replace the flag-building block (currently `:242-294`, from `# Prepare Docker run flags` through `flags += ["--memory=14g"]`) and the `run_cmd = [...]` literal (`:297-316`) with:

```python
        run_cmd = self._build_run_args(
            image,
            container_name=f"agent-server-{uuid.uuid4()}",
            extra_flags=[
                "--volumes-from",
                self._plugin_container_name,
                "-e",
                f"PATH={combined_path}",
                "-e",
                f"LD_LIBRARY_PATH={combined_lib}",
                "-e",
                "UV_PYTHON_INSTALL_DIR=/agent-server/uv-managed-python",
                "-w",
                "/",
            ],
            entrypoint=["/agent-server/.venv/bin/python"],
            command=[
                "-m",
                "openhands.agent_server",
                "--host",
                "0.0.0.0",
                "--port",
                "8000",
            ],
        )
```

Note this also fixes a second latent bug: flex never applied `memory_limit` at all. It now does, via the shared builder.

- [ ] **Step 7: Run the full workspace test suite**

```bash
uv run pytest vendor/software-agent-sdk/tests/workspace/ -q
```

Expected: the new `test_run_args.py` passes and all pre-existing workspace tests still pass. If a pre-existing test asserted on `13g` or on duplicate memory flags, update it and note the change in the commit message.

- [ ] **Step 8: Commit**

```bash
git -C vendor/software-agent-sdk add \
  openhands-workspace/openhands/workspace/docker/workspace.py \
  openhands-workspace/openhands/workspace/docker/flex_workspace.py \
  tests/workspace/test_run_args.py
git -C vendor/software-agent-sdk commit -m "$(printf 'refactor(workspace): share docker run-arg construction; fix --memory\n\nFlexWorkspace duplicated ~100 lines of DockerWorkspace startup. Extract\n_build_run_args so there is one construction path for the egress branch.\n\nBoth files appended a hardcoded --memory=14g AFTER --memory {memory_limit},\nmaking memory_limit and OH_WORKSPACE_MEMORY_LIMIT silently inert (flex\nnever applied memory_limit at all). Remove the duplicate and default\nmemory_limit to 14g: the env var now works and the effective limit is\nunchanged.\n\nCo-authored-by: openhands <openhands@all-hands.dev>')"
```

---

### Task 5: Egress runtime and workspace wiring

**Files:**
- Create: `vendor/software-agent-sdk/openhands-workspace/openhands/workspace/docker/egress_runtime.py`
- Modify: `vendor/software-agent-sdk/openhands-workspace/openhands/workspace/docker/workspace.py`
- Test: `vendor/software-agent-sdk/tests/workspace/test_egress_runtime.py`

**Interfaces:**
- Consumes: `WorkspaceNetworkPolicy` (Task 1), `render_rules`/`policy_digest` (Task 2), `_build_run_args`/`_build_port_args` (Task 4).
- Produces:
  - `STATE_ROOT: Path` — `Path(os.getenv("OH_EGRESS_STATE_ROOT", "/var/tmp/openhands-egress"))`
  - `class EgressRuntime` — attrs `workspace_id: str`, `controller_id: str`, `network_id: str | None`, `sidecar_id: str | None`, `rules_path: Path | None`, `policy_digest: str | None`; methods `cleanup() -> None`, `is_alive() -> bool`
  - `start_egress_sidecar(policy, *, host_port, extra_ports, image) -> EgressRuntime`
  - `subnets_overlap_allowlist(subnets, policy) -> bool`
  - `controller_id() -> str` — `f"{boot_id}-{pid}-{random}"`
- Adds to `DockerWorkspace`: field `network_policy: WorkspaceNetworkPolicy` (default from `policy_from_env()`), private `_egress: EgressRuntime | None`.

`reconcile_orphans()` is added to this same module in Task 8.

- [ ] **Step 1: Write the failing tests**

Create `vendor/software-agent-sdk/tests/workspace/test_egress_runtime.py`:

```python
"""Tests for the egress sidecar runtime (docker calls faked)."""

import threading
from ipaddress import ip_network
from unittest.mock import Mock, patch

import pytest

from openhands.workspace import DockerWorkspace
from openhands.workspace.docker.egress_runtime import (
    EgressRuntime,
    subnets_overlap_allowlist,
)
from openhands.workspace.docker.network_policy import (
    AllowedEndpoint,
    WorkspaceNetworkPolicy,
)


def test_public_policy_is_default_and_needs_no_sidecar():
    with patch.object(DockerWorkspace, "_start_container"):
        ws = DockerWorkspace(server_image="test:latest")
    assert ws.network_policy.mode == "public"
    assert not ws.network_policy.requires_sidecar


def test_env_var_sets_policy_without_explicit_argument(monkeypatch):
    monkeypatch.setenv("OH_NETWORK_MODE", "no-network")
    with patch.object(DockerWorkspace, "_start_container"):
        ws = DockerWorkspace(server_image="test:latest")
    assert ws.network_policy.mode == "no-network"


def test_invalid_env_var_raises_rather_than_defaulting_public(monkeypatch):
    monkeypatch.setenv("OH_NETWORK_MODE", "nonsense")
    with pytest.raises(ValueError):
        with patch.object(DockerWorkspace, "_start_container"):
            DockerWorkspace(server_image="test:latest")


def test_workspace_drops_net_admin_and_net_raw_in_sidecar_mode():
    with patch.object(DockerWorkspace, "_start_container"):
        ws = DockerWorkspace(
            server_image="test:latest",
            network_policy=WorkspaceNetworkPolicy(mode="no-network"),
        )
    ws.host_port = 30000
    ws._egress = Mock(sidecar_id="side123")
    args = ws._build_run_args("img:latest", container_name="c")
    assert args[args.index("--cap-drop") : args.index("--cap-drop") + 2] == [
        "--cap-drop",
        "NET_ADMIN",
    ]
    assert "NET_RAW" in args
    assert "--network" in args
    assert "container:side123" in args
    assert "no-new-privileges=true" in args


def test_workspace_publishes_no_ports_when_sidecar_owns_namespace():
    """Docker rejects -p on a container using container: network mode."""
    with patch.object(DockerWorkspace, "_start_container"):
        ws = DockerWorkspace(
            server_image="test:latest",
            network_policy=WorkspaceNetworkPolicy(mode="no-network"),
        )
    ws.host_port = 30000
    ws._egress = Mock(sidecar_id="side123")
    assert ws._build_port_args() == []


def test_subnet_overlap_detected():
    policy = WorkspaceNetworkPolicy(mode="no-network")
    assert subnets_overlap_allowlist([ip_network("10.5.0.0/16")], policy) is True
    assert subnets_overlap_allowlist([ip_network("172.18.0.0/16")], policy) is False


def test_subnet_overlap_checks_caller_endpoints_too():
    policy = WorkspaceNetworkPolicy(
        mode="static-allowlist",
        allowed_endpoints=(AllowedEndpoint(destination=ip_network("192.0.2.0/24")),),
    )
    assert subnets_overlap_allowlist([ip_network("192.0.2.128/25")], policy) is True


def test_cleanup_is_idempotent():
    runtime = EgressRuntime(
        workspace_id="ws1", controller_id="ctrl1", sidecar_id="side1", network_id="net1"
    )
    with patch(
        "openhands.workspace.docker.egress_runtime.execute_command"
    ) as mock_exec:
        mock_exec.return_value = Mock(returncode=0, stdout="", stderr="")
        runtime.cleanup()
        first = mock_exec.call_count
        runtime.cleanup()
        runtime.cleanup()
    assert mock_exec.call_count == first, "repeat cleanup must be a no-op"


def test_cleanup_is_threadsafe():
    runtime = EgressRuntime(
        workspace_id="ws1", controller_id="ctrl1", sidecar_id="side1", network_id="net1"
    )
    with patch(
        "openhands.workspace.docker.egress_runtime.execute_command"
    ) as mock_exec:
        mock_exec.return_value = Mock(returncode=0, stdout="", stderr="")
        threads = [threading.Thread(target=runtime.cleanup) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        calls = mock_exec.call_count
    assert calls <= 4, f"expected one cleanup pass, saw {calls} docker calls"


def test_cleanup_continues_after_one_resource_fails():
    """Best-effort, not fail-fast: a failed stop must not skip network removal."""
    runtime = EgressRuntime(
        workspace_id="ws1", controller_id="ctrl1", sidecar_id="side1", network_id="net1"
    )
    with patch(
        "openhands.workspace.docker.egress_runtime.execute_command"
    ) as mock_exec:
        def fail_on_stop(cmd, *a, **kw):
            if "stop" in cmd or "rm" in cmd:
                return Mock(returncode=1, stdout="", stderr="boom")
            return Mock(returncode=0, stdout="", stderr="")

        mock_exec.side_effect = fail_on_stop
        runtime.cleanup()
        issued = [" ".join(c.args[0]) for c in mock_exec.call_args_list]
    assert any("network" in cmd and "rm" in cmd for cmd in issued)
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
uv run pytest vendor/software-agent-sdk/tests/workspace/test_egress_runtime.py -q
```

Expected: `ModuleNotFoundError: ...egress_runtime`.

- [ ] **Step 3: Implement `egress_runtime.py`**

Create the module. Key requirements: a `threading.Lock` plus a `_cleaned` flag makes `cleanup()` idempotent and thread-safe; cleanup is best-effort across resources (each step wrapped so a failure does not skip later steps); the manifest is written atomically (temp file + `os.replace`).

```python
"""Lifecycle for the per-workspace nftables egress sidecar."""

import json
import os
import secrets
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from ipaddress import IPv4Network, IPv6Network
from pathlib import Path

from openhands.sdk.logger import get_logger
from openhands.sdk.utils.command import execute_command

from .network_policy import WorkspaceNetworkPolicy
from .nftables_renderer import policy_digest, render_rules


logger = get_logger(__name__)

STATE_ROOT = Path(os.getenv("OH_EGRESS_STATE_ROOT", "/var/tmp/openhands-egress"))
READY_PATH = "/tmp/workspace-egress.ready"
RULES_MOUNT = "/etc/workspace-egress/rules.nft"
DEFAULT_SIDECAR_IMAGE = os.getenv(
    "OH_EGRESS_IMAGE", "openhands-egress-static:dev"
)
READY_TIMEOUT_SECONDS = 30.0
STOP_TIMEOUT_SECONDS = 10
LEASE_STALE_SECONDS = 120.0

LABEL_MANAGED = "workspace.managed=true"


def controller_id() -> str:
    """Stable-per-process controller identity: boot id, pid, and a nonce.

    A differing boot id means the host rebooted, so the controller is dead.
    The nonce guards against pid reuse within one boot.
    """
    try:
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError:
        boot = "noboot"
    return f"{boot[:8]}-{os.getpid()}-{secrets.token_hex(4)}"


def subnets_overlap_allowlist(
    subnets: list[IPv4Network | IPv6Network], policy: WorkspaceNetworkPolicy
) -> bool:
    """True if any bridge subnet intersects an allowed destination.

    An overlapping bridge would place the gateway and host services inside the
    allowlist and shadow the internal service range, so startup must abort.
    """
    for endpoint in policy.resolved_endpoints():
        for subnet in subnets:
            if subnet.version != endpoint.destination.version:
                continue
            if subnet.overlaps(endpoint.destination):
                return True
    return False


@dataclass
class EgressRuntime:
    """Resources acquired for one workspace's egress boundary."""

    workspace_id: str
    controller_id: str
    network_id: str | None = None
    sidecar_id: str | None = None
    rules_path: Path | None = None
    policy_digest: str | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _cleaned: bool = field(default=False, repr=False)

    @property
    def manifest_path(self) -> Path:
        return STATE_ROOT / self.workspace_id / "manifest.json"

    def write_manifest(self, status: str = "active") -> None:
        """Atomically record acquired resources so a later pass can reconcile."""
        directory = self.manifest_path.parent
        directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "workspace_id": self.workspace_id,
            "controller_id": self.controller_id,
            "network_id": self.network_id,
            "sidecar_id": self.sidecar_id,
            "rules_path": str(self.rules_path) if self.rules_path else None,
            "policy_digest": self.policy_digest,
            "status": status,
            "lease_expires_at": time.time() + LEASE_STALE_SECONDS,
        }
        fd, tmp = tempfile.mkstemp(dir=str(directory), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
            os.replace(tmp, self.manifest_path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    def is_alive(self) -> bool:
        """Whether the sidecar container is still running."""
        if not self.sidecar_id:
            return False
        proc = execute_command(
            ["docker", "inspect", "-f", "{{.State.Running}}", self.sidecar_id],
            print_output=False,
        )
        return proc.stdout.strip() == "true"

    def cleanup(self) -> None:
        """Release every acquired resource. Idempotent and thread-safe.

        Best-effort across resources: a failure removing one resource must not
        prevent the removal of the others.
        """
        with self._lock:
            if self._cleaned:
                return
            self._cleaned = True

        errors: list[str] = []

        def attempt(label: str, cmd: list[str]) -> None:
            try:
                proc = execute_command(cmd, print_output=False)
                if proc.returncode != 0:
                    errors.append(f"{label}: {proc.stderr.strip()}")
            except Exception as exc:  # noqa: BLE001 - aggregate and continue
                errors.append(f"{label}: {exc}")

        if self.sidecar_id:
            attempt(
                f"stop sidecar {self.sidecar_id}",
                ["docker", "stop", "-t", str(STOP_TIMEOUT_SECONDS), self.sidecar_id],
            )
            attempt(
                f"remove sidecar {self.sidecar_id}",
                ["docker", "rm", "-f", self.sidecar_id],
            )
        if self.network_id:
            attempt(
                f"remove network {self.network_id}",
                ["docker", "network", "rm", self.network_id],
            )
        if self.rules_path:
            try:
                self.rules_path.unlink(missing_ok=True)
                parent = self.rules_path.parent
                if parent != STATE_ROOT and parent.is_dir():
                    for leftover in parent.iterdir():
                        leftover.unlink(missing_ok=True)
                    parent.rmdir()
            except OSError as exc:
                errors.append(f"remove rules {self.rules_path}: {exc}")

        if errors:
            logger.warning(
                "egress cleanup for %s completed with errors: %s",
                self.workspace_id,
                "; ".join(errors),
            )
        else:
            self.manifest_path.unlink(missing_ok=True)
```

Then add the startup transaction to the same module. Every failure path calls
`runtime.cleanup()` before re-raising, and the manifest is rewritten after each
acquisition:

```python
def _network_subnets(network_id: str) -> list[IPv4Network | IPv6Network]:
    """Read the IPAM subnets docker assigned to a network."""
    proc = execute_command(
        [
            "docker", "network", "inspect", network_id,
            "--format", "{{range .IPAM.Config}}{{.Subnet}} {{end}}",
        ],
        print_output=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"failed to inspect network {network_id}: {proc.stderr}")
    return [ip_network(token) for token in proc.stdout.split() if token]


def start_egress_sidecar(
    policy: WorkspaceNetworkPolicy,
    *,
    host_port: int,
    extra_ports: bool = False,
    image: str = DEFAULT_SIDECAR_IMAGE,
) -> EgressRuntime:
    """Create the network and sidecar that will own the workspace namespace.

    Fails closed: any error tears down whatever was already acquired and
    re-raises. The caller must never proceed to start a workspace after this
    raises.
    """
    workspace_id = f"ws-{uuid.uuid4().hex[:12]}"
    runtime = EgressRuntime(workspace_id=workspace_id, controller_id=controller_id())

    try:
        # 1. Rules file, private to this controller.
        rules_text = render_rules(policy)
        runtime.policy_digest = policy_digest(rules_text)
        directory = STATE_ROOT / workspace_id
        directory.mkdir(parents=True, exist_ok=True)
        os.chmod(directory, 0o700)
        rules_path = directory / "rules.nft"
        rules_path.write_text(rules_text, encoding="utf-8")
        os.chmod(rules_path, 0o600)
        runtime.rules_path = rules_path
        runtime.write_manifest()

        # 2. Dedicated bridge.
        net_proc = execute_command(
            [
                "docker", "network", "create",
                "--label", LABEL_MANAGED,
                "--label", f"workspace.id={workspace_id}",
                "--label", f"workspace.controller={runtime.controller_id}",
                f"{workspace_id}-network",
            ],
            print_output=False,
        )
        if net_proc.returncode != 0:
            raise RuntimeError(f"failed to create network: {net_proc.stderr}")
        runtime.network_id = net_proc.stdout.strip()
        runtime.write_manifest()

        # 3. A bridge inside the allowlist would put the gateway and host
        #    services inside the policy. Abort rather than weaken the boundary.
        subnets = _network_subnets(runtime.network_id)
        if subnets_overlap_allowlist(subnets, policy):
            raise RuntimeError(
                f"docker allocated bridge subnets {subnets} which overlap the "
                "resolved allowlist; refusing to start. Configure the daemon's "
                "default-address-pools to a range outside the allowlist."
            )

        # 4. Sidecar owns the namespace and publishes ports on loopback only.
        publish = ["-p", f"127.0.0.1:{host_port}:8000"]
        if extra_ports:
            publish += [
                "-p", f"127.0.0.1:{host_port + 1}:8001",
                "-p", f"127.0.0.1:{host_port + 2}:8002",
            ]
        run_proc = execute_command(
            [
                "docker", "run", "-d",
                "--name", f"{workspace_id}-egress",
                "--init", "--restart", "no",
                "--stop-timeout", str(STOP_TIMEOUT_SECONDS),
                "--network", f"{workspace_id}-network",
                "--cap-drop", "ALL",
                "--cap-add", "NET_ADMIN",
                "--security-opt", "no-new-privileges=true",
                "--read-only",
                "--tmpfs", "/tmp:rw,noexec,nosuid,nodev,size=1m",
                *publish,
                "--label", LABEL_MANAGED,
                "--label", f"workspace.id={workspace_id}",
                "--label", f"workspace.controller={runtime.controller_id}",
                "--label", "workspace.role=egress",
                "--mount",
                f"type=bind,src={rules_path},dst={RULES_MOUNT},readonly",
                image,
            ],
            print_output=False,
        )
        if run_proc.returncode != 0:
            raise RuntimeError(f"failed to start egress sidecar: {run_proc.stderr}")
        runtime.sidecar_id = run_proc.stdout.strip()
        runtime.write_manifest()

        # 5. Readiness is published only after the sidecar verifies its own
        #    ruleset, so this is also policy verification.
        deadline = time.time() + READY_TIMEOUT_SECONDS
        while time.time() < deadline:
            probe = execute_command(
                ["docker", "exec", runtime.sidecar_id, "test", "-f", READY_PATH],
                print_output=False,
            )
            if probe.returncode == 0:
                logger.info(
                    "egress sidecar %s ready for %s (mode=%s digest=%s)",
                    runtime.sidecar_id[:12],
                    workspace_id,
                    policy.mode,
                    runtime.policy_digest[:12] if runtime.policy_digest else "-",
                )
                return runtime
            if not runtime.is_alive():
                logs = execute_command(
                    ["docker", "logs", runtime.sidecar_id], print_output=False
                )
                raise RuntimeError(
                    f"egress sidecar exited before readiness:\n"
                    f"{logs.stdout}\n{logs.stderr}"
                )
            time.sleep(0.2)

        logs = execute_command(
            ["docker", "logs", runtime.sidecar_id], print_output=False
        )
        raise RuntimeError(
            f"egress sidecar not ready within {READY_TIMEOUT_SECONDS}s:\n"
            f"{logs.stdout}\n{logs.stderr}"
        )
    except BaseException:
        runtime.cleanup()
        raise
```

Add `from ipaddress import ip_network` to the module imports for `_network_subnets`.

- [ ] **Step 4: Wire the policy into `DockerWorkspace`**

In `workspace.py`, add the import and field, and override port/network behavior:

```python
from .egress_runtime import EgressRuntime, start_egress_sidecar
from .network_policy import WorkspaceNetworkPolicy, policy_from_env
```

Add the field beside the others:

```python
    network_policy: WorkspaceNetworkPolicy = Field(
        default_factory=policy_from_env,
        description=(
            "Egress policy. Defaults from OH_NETWORK_MODE; 'public' preserves "
            "unrestricted networking. Non-public modes start an nftables "
            "sidecar that owns the network namespace."
        ),
    )
```

Add the private attr: `_egress: EgressRuntime | None = PrivateAttr(default=None)`.

Change `_build_port_args` so a sidecar-owned namespace publishes nothing:

```python
    def _build_port_args(self) -> list[str]:
        if self._egress is not None:
            # The sidecar owns the namespace and publishes the ports; docker
            # rejects -p on a container using container: network mode.
            return []
        ports = ["-p", f"{self.host_port}:8000"]
        if self.extra_ports:
            ports += [
                "-p",
                f"{self.host_port + 1}:8001",
                "-p",
                f"{self.host_port + 2}:8002",
            ]
        return ports
```

In `_build_run_args`, immediately before building `run_cmd`, append the sidecar flags:

```python
        if self._egress is not None and self._egress.sidecar_id:
            flags += [
                "--network",
                f"container:{self._egress.sidecar_id}",
                "--cap-drop",
                "NET_ADMIN",
                "--cap-drop",
                "NET_RAW",
                "--security-opt",
                "no-new-privileges=true",
            ]
```

In `_start_container`, start the sidecar before the workspace runs, and tear it down if anything fails:

```python
        if self.network_policy.requires_sidecar:
            self._egress = start_egress_sidecar(
                self.network_policy,
                host_port=self.host_port,
                extra_ports=self.extra_ports,
            )
        try:
            ...  # existing run + health-wait body
        except BaseException:
            if self._egress is not None:
                self._egress.cleanup()
                self._egress = None
            raise
```

Finally, in `cleanup()`, release the sidecar after the workspace container:

```python
        if self._egress is not None:
            self._egress.cleanup()
            self._egress = None
```

- [ ] **Step 5: Run the tests**

```bash
uv run pytest vendor/software-agent-sdk/tests/workspace/ -q
```

Expected: all pass, including the pre-existing workspace tests.

- [ ] **Step 6: Commit**

```bash
git -C vendor/software-agent-sdk add \
  openhands-workspace/openhands/workspace/docker/egress_runtime.py \
  openhands-workspace/openhands/workspace/docker/workspace.py \
  tests/workspace/test_egress_runtime.py
git -C vendor/software-agent-sdk commit -m "$(printf 'feat(workspace): nftables egress sidecar lifecycle\n\nTransactional startup with rollback, locked idempotent cleanup, bridge\nsubnet overlap guard, and loopback-only port publication.\n\nCo-authored-by: openhands <openhands@all-hands.dev>')"
```

---

### Task 6: Benchmark-side central resolution

**Files:**
- Create: `benchmarks/utils/workspace_network.py`
- Modify: `benchmarks/utils/models.py` (add fields to `EvalMetadata`)
- Modify: `benchmarks/utils/evaluation.py` (`model_post_init`, `:52-60`)
- Test: `tests/test_workspace_network.py`

**Interfaces:**
- Consumes: `WorkspaceNetworkPolicy`, `parse_network_mode` (Task 1).
- Produces: `resolve_network_policy(workspace_type: str, explicit: WorkspaceNetworkPolicy | None = None, env: Mapping[str,str] | None = None) -> WorkspaceNetworkPolicy`; `EvalMetadata.network_mode: str` and `EvalMetadata.network_policy_digest: str | None`.

**Why this layer exists.** The SDK default alone is insufficient for two verified reasons (spec §2.9): `APIRemoteWorkspace` has no `network_policy` field and no local container, so `OH_NETWORK_MODE=no-network` with `workspace_type=remote` would run **unrestricted** — fail-open; and `metadata.json` is written in `Evaluation.model_post_init` (`evaluation.py:52-60`) long before `prepare_workspace` runs at `:529`, so a policy resolved at construction can never be persisted.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_workspace_network.py`:

```python
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
    from openhands.sdk import LLM
    from openhands.sdk.critic import PassCritic

    from benchmarks.utils.models import EvalMetadata

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
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
uv run pytest tests/test_workspace_network.py -q
```

Expected: `ModuleNotFoundError: benchmarks.utils.workspace_network`.

- [ ] **Step 3: Implement `workspace_network.py`**

Create `benchmarks/utils/workspace_network.py`:

```python
"""Central resolution of the workspace egress policy.

This is the authority for validating OH_NETWORK_MODE and persisting the
resolved policy. It runs during Evaluation setup, before metadata.json is
written, and rejects combinations the SDK layer cannot enforce.
"""

import os
from collections.abc import Mapping

from openhands.sdk.logger import get_logger
from openhands.workspace.docker.network_policy import (
    ENV_VAR,
    WorkspaceNetworkPolicy,
    parse_network_mode,
)


logger = get_logger(__name__)


def resolve_network_policy(
    workspace_type: str,
    explicit: WorkspaceNetworkPolicy | None = None,
    env: Mapping[str, str] | None = None,
) -> WorkspaceNetworkPolicy:
    """Resolve the effective egress policy, or raise.

    Raises:
        ValueError: on an invalid mode, a conflict between an explicit policy
            and the environment, or a non-public mode requested for a remote
            workspace (which has no local container to isolate).
    """
    source = os.environ if env is None else env
    env_mode = parse_network_mode(source.get(ENV_VAR))

    if explicit is not None:
        if source.get(ENV_VAR, "").strip() and explicit.mode != env_mode:
            raise ValueError(
                f"Network policy conflict: explicit mode {explicit.mode!r} does "
                f"not match {ENV_VAR}={env_mode!r}. Set them to the same value "
                "or supply only one."
            )
        policy = explicit
    else:
        policy = WorkspaceNetworkPolicy(mode=env_mode)

    if workspace_type == "remote" and policy.requires_sidecar:
        raise ValueError(
            f"workspace_type='remote' cannot enforce network mode "
            f"{policy.mode!r}: there is no local container to isolate. Use a "
            "docker or flex workspace, or set the mode to 'public'."
        )

    logger.info(
        "Resolved workspace network policy: mode=%s endpoints=%d",
        policy.mode,
        len(policy.resolved_endpoints()),
    )
    return policy
```

- [ ] **Step 4: Add the metadata fields**

In `benchmarks/utils/models.py`, add to `EvalMetadata` (after `workspace_type`):

```python
    network_mode: str = Field(
        default="public",
        description=(
            "Resolved workspace egress mode (from OH_NETWORK_MODE). Persisted "
            "so historical runs do not depend on a changing implicit value."
        ),
    )
    network_policy_digest: str | None = Field(
        default=None,
        description="SHA-256 of the rendered nftables policy, if any.",
    )
```

- [ ] **Step 5: Resolve before the metadata write**

In `benchmarks/utils/evaluation.py`, inside `Evaluation.model_post_init`, resolve **before** the file is written (i.e. before line 57's `metadata_file = ...`):

```python
        from benchmarks.utils.workspace_network import resolve_network_policy
        from openhands.workspace.docker.nftables_renderer import (
            policy_digest,
            render_rules,
        )

        policy = resolve_network_policy(self.metadata.workspace_type)
        self.metadata.network_mode = policy.mode
        self.metadata.network_policy_digest = (
            policy_digest(render_rules(policy)) if policy.requires_sidecar else None
        )
```

- [ ] **Step 6: Run the tests**

```bash
uv run pytest tests/test_workspace_network.py tests/test_workspace_cleanup.py -q
```

Expected: all pass, including the pre-existing cleanup test.

- [ ] **Step 7: Commit**

```bash
git add benchmarks/utils/workspace_network.py benchmarks/utils/models.py \
        benchmarks/utils/evaluation.py tests/test_workspace_network.py
git commit -m "$(printf 'feat(utils): central network policy resolution and persistence\n\nValidates OH_NETWORK_MODE before metadata.json is written and rejects\nnon-public modes for remote workspaces, which have no local container to\nisolate and would otherwise run unrestricted.\n\nCo-authored-by: openhands <openhands@all-hands.dev>')"
```

---

### Task 7: Signal-aware worker shutdown

**Files:**
- Modify: `benchmarks/utils/evaluation.py` (`_process_one_mp` child entry; `_cleanup_pool` at `:426-452`)
- Test: `tests/test_worker_shutdown.py` (new)

**Interfaces:**
- Produces: `install_worker_signal_handlers() -> None`; `_cleanup_pool(..., grace_seconds: float = 30.0)`.

**Why.** `_cleanup_pool` calls `process.terminate()` (`evaluation.py:449`), which sends SIGTERM. Python's default SIGTERM disposition kills the process outright, so the `finally` at `:614-634` that calls `workspace.__exit__` never runs — orphaning containers today. An idempotent `cleanup()` cannot help a process that never calls it. Convert SIGTERM into a normal exception so the existing `finally` runs, then escalate to SIGKILL after a bounded deadline.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_worker_shutdown.py`:

```python
"""Tests for signal-aware worker shutdown."""

import signal

import pytest


def test_handler_converts_sigterm_to_exception():
    from benchmarks.utils.evaluation import install_worker_signal_handlers

    original = signal.getsignal(signal.SIGTERM)
    try:
        install_worker_signal_handlers()
        handler = signal.getsignal(signal.SIGTERM)
        assert callable(handler)
        assert handler not in (signal.SIG_DFL, signal.SIG_IGN)
        with pytest.raises(KeyboardInterrupt):
            handler(signal.SIGTERM, None)
    finally:
        signal.signal(signal.SIGTERM, original)


def test_handler_installed_for_sigint_too():
    from benchmarks.utils.evaluation import install_worker_signal_handlers

    originals = (signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGINT))
    try:
        install_worker_signal_handlers()
        with pytest.raises(KeyboardInterrupt):
            signal.getsignal(signal.SIGINT)(signal.SIGINT, None)
    finally:
        signal.signal(signal.SIGTERM, originals[0])
        signal.signal(signal.SIGINT, originals[1])


def test_cleanup_pool_escalates_to_kill_after_grace(monkeypatch):
    """SIGTERM first so cleanup can run; SIGKILL only after the deadline."""
    from unittest.mock import Mock

    from benchmarks.utils.evaluation import Evaluation

    proc = Mock()
    proc.is_alive.return_value = True  # never exits on its own
    pool = Mock()
    pool._processes = {1: proc}

    Evaluation._cleanup_pool(Mock(), pool, futures=[], wait=False, grace_seconds=0.2)

    assert proc.terminate.called, "must SIGTERM first so cleanup can run"
    assert proc.kill.called, "must escalate to SIGKILL after the deadline"
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
uv run pytest tests/test_worker_shutdown.py -q
```

Expected: `ImportError: cannot import name 'install_worker_signal_handlers'`.

- [ ] **Step 3: Add the handler installer**

In `benchmarks/utils/evaluation.py`, add `import signal` at the top and this module-level function:

```python
def install_worker_signal_handlers() -> None:
    """Make SIGTERM/SIGINT raise so the worker's cleanup `finally` runs.

    ProcessPoolExecutor shutdown sends SIGTERM. Under the default disposition
    the process dies immediately and the `finally` that calls
    workspace.__exit__ never executes, orphaning containers and networks.
    Raising KeyboardInterrupt routes termination through normal unwinding.

    The handler only raises: no Docker or other I/O work happens inside an
    asynchronous signal handler.
    """

    def _raise_on_signal(signum, frame):  # type: ignore[no-untyped-def]
        raise KeyboardInterrupt(f"worker received signal {signum}")

    signal.signal(signal.SIGTERM, _raise_on_signal)
    signal.signal(signal.SIGINT, _raise_on_signal)
```

Call it as the **first statement** of the child entry point `_process_one_mp`, so every worker installs it before doing any work.

- [ ] **Step 4: Make `_cleanup_pool` bound the deadline**

Replace the termination block in `_cleanup_pool` (currently `:443-449`) with:

```python
        if not wait and hasattr(pool, "_processes") and pool._processes:
            processes = list(pool._processes.values())
            for process in processes:
                try:
                    process.terminate()  # SIGTERM: lets cleanup run
                except Exception:
                    pass

            deadline = time.time() + grace_seconds
            while time.time() < deadline:
                if not any(p.is_alive() for p in processes):
                    break
                time.sleep(0.1)

            for process in processes:
                try:
                    if process.is_alive():
                        logger.warning(
                            "Worker %s did not exit within %.1fs; sending SIGKILL",
                            getattr(process, "pid", "?"),
                            grace_seconds,
                        )
                        process.kill()
                except Exception:
                    pass
```

Add `grace_seconds: float = 30.0` to the signature and `import time` if not already present.

- [ ] **Step 5: Run the tests**

```bash
uv run pytest tests/test_worker_shutdown.py tests/test_keyboard_interrupt.py -q
```

Expected: all pass. `test_keyboard_interrupt.py` is pre-existing — if it fails, the handler is interfering with the parent process; ensure handlers are installed only in the child (`_process_one_mp`), never at import time.

- [ ] **Step 6: Commit**

```bash
git add benchmarks/utils/evaluation.py tests/test_worker_shutdown.py
git commit -m "$(printf 'fix(utils): run workspace cleanup on worker SIGTERM\n\nProcessPoolExecutor shutdown sent SIGTERM, whose default disposition\nskipped the finally block that calls workspace.__exit__, orphaning\ncontainers. Workers now convert SIGTERM/SIGINT into KeyboardInterrupt so\nnormal unwinding runs, and the parent escalates to SIGKILL only after a\nbounded grace period.\n\nCo-authored-by: openhands <openhands@all-hands.dev>')"
```

---

### Task 8: Orphan reconciliation

**Files:**
- Modify: `vendor/software-agent-sdk/openhands-workspace/openhands/workspace/docker/egress_runtime.py`
- Modify: `benchmarks/utils/evaluation.py` (call the pass before work starts)
- Test: `vendor/software-agent-sdk/tests/workspace/test_reconcile.py` (new)

**Interfaces:**
- Consumes: `EgressRuntime`, `STATE_ROOT`, `LEASE_STALE_SECONDS`, `LABEL_MANAGED` (Task 5).
- Produces: `controller_is_alive(controller_id: str) -> bool`; `reconcile_orphans(now: float | None = None) -> list[str]` returning the workspace IDs reclaimed.

**Why.** Graceful cleanup cannot run after `SIGKILL`, a crash, or host failure, so reconciliation is mandatory rather than optional (spec §4.4). Bias hard toward leaving things alone: a leaked container is recoverable, deleting a live workspace is not.

- [ ] **Step 1: Write the failing tests**

Create `vendor/software-agent-sdk/tests/workspace/test_reconcile.py`:

```python
"""Tests for orphan reconciliation."""

import json
import os
import time
from unittest.mock import Mock, patch

import pytest

from openhands.workspace.docker import egress_runtime
from openhands.workspace.docker.egress_runtime import (
    controller_is_alive,
    reconcile_orphans,
)


@pytest.fixture
def state_root(tmp_path, monkeypatch):
    monkeypatch.setattr(egress_runtime, "STATE_ROOT", tmp_path)
    return tmp_path


def _write_manifest(root, workspace_id, controller_id, lease_expires_at):
    directory = root / workspace_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "workspace_id": workspace_id,
                "controller_id": controller_id,
                "network_id": f"{workspace_id}-net",
                "sidecar_id": f"{workspace_id}-side",
                "rules_path": str(directory / "rules.nft"),
                "policy_digest": "deadbeef",
                "status": "active",
                "lease_expires_at": lease_expires_at,
            }
        )
    )
    (directory / "rules.nft").write_text("table inet workspace_egress {}\n")
    return directory


def test_live_pid_is_alive():
    assert controller_is_alive(f"boot1234-{os.getpid()}-abcd1234") is True


def test_dead_pid_is_not_alive():
    # PID 0 is never a real user process.
    assert controller_is_alive("boot1234-0-abcd1234") is False


def test_different_boot_id_is_dead():
    assert controller_is_alive(f"ffffffff-{os.getpid()}-abcd1234") is False


def test_malformed_controller_id_is_treated_as_alive():
    """Undeterminable liveness must NOT delete: leaking beats destroying."""
    assert controller_is_alive("garbage") is True


def test_stale_manifest_from_dead_controller_is_reclaimed(state_root):
    _write_manifest(state_root, "ws-dead", "boot1234-0-abcd1234", time.time() - 999)
    with patch.object(egress_runtime, "execute_command") as mock_exec:
        mock_exec.return_value = Mock(returncode=0, stdout="", stderr="")
        reclaimed = reconcile_orphans()
    assert "ws-dead" in reclaimed
    assert not (state_root / "ws-dead").exists()


def test_live_lease_is_preserved(state_root):
    _write_manifest(
        state_root, "ws-live", f"boot1234-{os.getpid()}-abcd1234", time.time() + 999
    )
    with patch.object(egress_runtime, "execute_command") as mock_exec:
        mock_exec.return_value = Mock(returncode=0, stdout="", stderr="")
        reclaimed = reconcile_orphans()
    assert reclaimed == []
    assert (state_root / "ws-live").exists()


def test_expired_lease_but_live_controller_is_preserved(state_root):
    """A busy controller may let its lease lapse; the process is authoritative."""
    _write_manifest(
        state_root, "ws-busy", f"boot1234-{os.getpid()}-abcd1234", time.time() - 999
    )
    with patch.object(egress_runtime, "execute_command") as mock_exec:
        mock_exec.return_value = Mock(returncode=0, stdout="", stderr="")
        assert reconcile_orphans() == []
    assert (state_root / "ws-busy").exists()


def test_unreadable_manifest_is_left_alone(state_root):
    directory = state_root / "ws-corrupt"
    directory.mkdir(parents=True)
    (directory / "manifest.json").write_text("{ not json")
    with patch.object(egress_runtime, "execute_command") as mock_exec:
        mock_exec.return_value = Mock(returncode=0, stdout="", stderr="")
        assert reconcile_orphans() == []
    assert directory.exists()


def test_reconcile_removes_container_before_network(state_root):
    _write_manifest(state_root, "ws-order", "boot1234-0-abcd1234", time.time() - 999)
    with patch.object(egress_runtime, "execute_command") as mock_exec:
        mock_exec.return_value = Mock(returncode=0, stdout="", stderr="")
        reconcile_orphans()
        issued = [" ".join(c.args[0]) for c in mock_exec.call_args_list]
    container_idx = next(i for i, c in enumerate(issued) if "ws-order-side" in c)
    network_idx = next(i for i, c in enumerate(issued) if "ws-order-net" in c)
    assert container_idx < network_idx


def test_one_failure_does_not_stop_the_scan(state_root):
    _write_manifest(state_root, "ws-a", "boot1234-0-abcd1234", time.time() - 999)
    _write_manifest(state_root, "ws-b", "boot1234-0-abcd1234", time.time() - 999)
    with patch.object(egress_runtime, "execute_command") as mock_exec:
        def flaky(cmd, *a, **kw):
            if "ws-a-side" in " ".join(cmd):
                raise RuntimeError("docker exploded")
            return Mock(returncode=0, stdout="", stderr="")

        mock_exec.side_effect = flaky
        reclaimed = reconcile_orphans()
    assert "ws-b" in reclaimed
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
uv run pytest vendor/software-agent-sdk/tests/workspace/test_reconcile.py -q
```

Expected: `ImportError: cannot import name 'controller_is_alive'`.

- [ ] **Step 3: Implement reconciliation**

Append to `egress_runtime.py`:

```python
def controller_is_alive(cid: str) -> bool:
    """Whether the controller that created a resource still exists.

    Returns True whenever liveness cannot be determined. Reclaiming a live
    controller's workspace destroys a running evaluation; leaving a leaked
    container costs disk until the next pass. Bias to leaving it alone.
    """
    parts = cid.split("-")
    if len(parts) != 3:
        return True  # unparseable: do not touch
    boot, pid_text, _nonce = parts
    try:
        current_boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()[:8]
    except OSError:
        return True  # cannot compare: do not touch
    if boot != current_boot:
        return False  # host rebooted: the process cannot exist
    try:
        pid = int(pid_text)
    except ValueError:
        return True
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, owned by another user
    return True


def reconcile_orphans(now: float | None = None) -> list[str]:
    """Remove managed resources whose controller is gone. Returns reclaimed IDs.

    Runs before a worker accepts work. Only acts on a complete, readable
    manifest whose controller is provably dead AND whose lease has expired.
    Continues scanning after individual failures.
    """
    current = time.time() if now is None else now
    reclaimed: list[str] = []
    if not STATE_ROOT.is_dir():
        return reclaimed

    for directory in sorted(STATE_ROOT.iterdir()):
        manifest_file = directory / "manifest.json"
        if not manifest_file.is_file():
            continue
        try:
            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
            workspace_id = manifest["workspace_id"]
            cid = manifest["controller_id"]
            lease = float(manifest["lease_expires_at"])
        except (OSError, ValueError, KeyError, TypeError) as exc:
            logger.warning("skipping unreadable manifest %s: %s", manifest_file, exc)
            continue

        if lease > current or controller_is_alive(cid):
            continue

        logger.info("reconciling orphaned workspace %s (controller %s)", workspace_id, cid)
        runtime = EgressRuntime(
            workspace_id=workspace_id,
            controller_id=cid,
            network_id=manifest.get("network_id"),
            sidecar_id=manifest.get("sidecar_id"),
            rules_path=Path(manifest["rules_path"]) if manifest.get("rules_path") else None,
        )
        try:
            runtime.cleanup()  # container before network, best-effort
            manifest_file.unlink(missing_ok=True)
            if directory.is_dir() and not any(directory.iterdir()):
                directory.rmdir()
            reclaimed.append(workspace_id)
        except Exception as exc:  # noqa: BLE001 - keep scanning
            logger.warning("reconciliation failed for %s: %s", workspace_id, exc)

    return reclaimed
```

- [ ] **Step 4: Call it before work starts**

In `benchmarks/utils/evaluation.py`, at the end of `Evaluation.model_post_init` (after the metadata write), reclaim anything a previous crashed run left behind:

```python
        try:
            from openhands.workspace.docker.egress_runtime import reconcile_orphans

            reclaimed = reconcile_orphans()
            if reclaimed:
                logger.info(
                    "Reclaimed %d orphaned egress workspace(s): %s",
                    len(reclaimed),
                    ", ".join(reclaimed),
                )
        except Exception as exc:  # noqa: BLE001 - never block startup
            logger.warning("Orphan reconciliation skipped: %s", exc)
```

- [ ] **Step 5: Run the tests**

```bash
uv run pytest vendor/software-agent-sdk/tests/workspace/test_reconcile.py tests/ -q
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git -C vendor/software-agent-sdk add \
  openhands-workspace/openhands/workspace/docker/egress_runtime.py \
  tests/workspace/test_reconcile.py
git -C vendor/software-agent-sdk commit -m "$(printf 'feat(workspace): lease-based orphan reconciliation\n\nGraceful cleanup cannot run after SIGKILL, so a pre-work pass reclaims\nmanaged resources whose controller is provably dead and whose lease has\nexpired. Undeterminable liveness leaves resources untouched.\n\nCo-authored-by: openhands <openhands@all-hands.dev>')"
git add benchmarks/utils/evaluation.py
git commit -m "$(printf 'feat(utils): reconcile orphaned egress resources before work\n\nCo-authored-by: openhands <openhands@all-hands.dev>')"
```

---

### Task 9: Remove the legacy mechanism

**Files:**
- Delete: `benchmarks/utils/network_isolation.py`
- Modify: `benchmarks/swebench/run_infer.py` (remove import at `:19`, call at `:293`)

**Interfaces:** removes `maybe_disable_network` and all `OH_DISABLE_NETWORK` handling. No compatibility mapping and no privileged-exec fallback may remain.

**Migration hazard to document.** The old flag allowed only loopback + established — it had **no** `10.0.0.0/8` exception. Its equivalent is therefore **`strict-no-network`**, *not* `no-network` (which additionally permits all of `10.0.0.0/8`). Migrating by name similarity silently widens isolation. This must appear in the commit message.

- [ ] **Step 1: Confirm the call sites before deleting**

```bash
cd /home/lfu/git-projects/benchmarks-main/.claude/worktrees/network-modes
grep -rn "network_isolation\|maybe_disable_network\|OH_DISABLE_NETWORK" \
  --include=*.py --include=*.md . | grep -v vendor/software-agent-sdk/.git
```

Expected: only `benchmarks/utils/network_isolation.py` plus `run_infer.py:19` and `:293`. If any other call site appears, remove it too.

- [ ] **Step 2: Remove the import and the call**

In `benchmarks/swebench/run_infer.py`, delete the line at `:19`:

```python
from benchmarks.utils.network_isolation import maybe_disable_network
```

and the call at `:293`, leaving the surrounding `prepare_workspace` return intact:

```python
        maybe_disable_network(workspace)
        return workspace
```

becomes:

```python
        return workspace
```

- [ ] **Step 3: Delete the module**

```bash
git rm benchmarks/utils/network_isolation.py
```

- [ ] **Step 4: Verify nothing references it**

```bash
grep -rn "network_isolation\|maybe_disable_network\|OH_DISABLE_NETWORK" \
  --include=*.py . | grep -v vendor/software-agent-sdk/.git || echo "CLEAN"
uv run pytest tests/ -q
```

Expected: `CLEAN`, and the test suite passes.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/swebench/run_infer.py
git commit -m "$(printf 'refactor: replace OH_DISABLE_NETWORK with OH_NETWORK_MODE\n\nRemoves the privileged-exec iptables mechanism entirely; the sidecar\nlauncher is now the single enforcement path.\n\nMIGRATION: OH_DISABLE_NETWORK allowed only loopback and established\ntraffic, with NO 10.0.0.0/8 exception. Its equivalent is\nOH_NETWORK_MODE=strict-no-network, NOT no-network -- no-network also\npermits all of 10.0.0.0/8, so migrating by name similarity would\nsilently widen isolation.\n\nCo-authored-by: openhands <openhands@all-hands.dev>')"
```

---

### Task 10: Docker-gated end-to-end suite

**Files:**
- Create: `tests/test_egress_e2e.py`

**Interfaces:** consumes everything above. All tests carry `@pytest.mark.docker` and are excluded from the default run.

- [ ] **Step 1: Write the end-to-end tests**

Create `tests/test_egress_e2e.py`. The DNS pair is the most important: 4a proves the shipped rule works, 4b proves the naive port-based rule does **not**, locking in why the address-only match is required.

```python
"""End-to-end egress enforcement tests. Requires a working Docker daemon."""

import subprocess
import uuid

import pytest

from openhands.workspace.docker.nftables_renderer import render_rules
from openhands.workspace.docker.network_policy import WorkspaceNetworkPolicy


pytestmark = pytest.mark.docker

SIDECAR_IMAGE = "openhands-egress-static:dev"


def _run_in_policy_container(rules_text: str, script: str) -> subprocess.CompletedProcess:
    """Apply rules_text inside a NET_ADMIN container, then run script."""
    name = f"egress-test-{uuid.uuid4().hex[:8]}"
    return subprocess.run(
        [
            "docker", "run", "--rm", "--name", name,
            "--cap-drop", "ALL", "--cap-add", "NET_ADMIN",
            "alpine:3.20", "sh", "-c",
            "apk add --no-cache nftables bind-tools >/dev/null 2>&1 && "
            f"printf '%s' {rules_text!r} > /tmp/r.nft && nft -f /tmp/r.nft && {script}",
        ],
        capture_output=True, text=True, timeout=120,
    )


def test_4a_embedded_resolver_blocked_by_shipped_policy():
    """The address-only resolver drop must stop external name resolution."""
    rules = render_rules(WorkspaceNetworkPolicy(mode="no-network"))
    result = _run_in_policy_container(
        rules, "nslookup example.com 127.0.0.11 2>&1 || true"
    )
    assert "Address" not in result.stdout or "timed out" in result.stdout, (
        f"embedded resolver leaked external DNS:\n{result.stdout}"
    )


def test_4b_port_based_resolver_rule_is_insufficient():
    """Regression guard: docker's nat-OUTPUT DNAT defeats a dport 53 match.

    If this test starts FAILING (i.e. the naive rule begins working), docker's
    resolver behavior changed -- re-verify the shipped rule before relaxing it.
    """
    naive = """table inet t {
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
    result = _run_in_policy_container(
        naive, "nslookup example.com 127.0.0.11 2>&1 || true"
    )
    assert "Address" in result.stdout, (
        "The naive dport-53 rule unexpectedly blocked DNS. Docker's DNAT "
        "behavior may have changed; re-verify the shipped address-only rule."
    )


def test_public_ip_is_blocked():
    rules = render_rules(WorkspaceNetworkPolicy(mode="no-network"))
    result = _run_in_policy_container(
        rules, "wget -q -T4 -O- http://93.184.216.34/ 2>&1 || echo BLOCKED"
    )
    assert "BLOCKED" in result.stdout


def test_loopback_still_works():
    rules = render_rules(WorkspaceNetworkPolicy(mode="no-network"))
    result = _run_in_policy_container(
        rules,
        "(nc -l -p 9999 &) ; sleep 1 ; echo ping | nc -w 2 127.0.0.1 9999 && echo LOOPBACK_OK",
    )
    assert "LOOPBACK_OK" in result.stdout


def test_strict_no_network_blocks_ten_slash_eight():
    rules = render_rules(WorkspaceNetworkPolicy(mode="strict-no-network"))
    assert "10.0.0.0/8" not in rules
```

- [ ] **Step 2: Confirm they are excluded from the default run**

```bash
uv run pytest tests/ -q --collect-only 2>&1 | grep -c "test_egress_e2e" || echo "EXCLUDED_OK"
```

Expected: `EXCLUDED_OK` (the `addopts = "-m 'not docker'"` from Task 0 filters them).

- [ ] **Step 3: Run them explicitly**

```bash
docker build -t openhands-egress-static:dev \
  vendor/software-agent-sdk/openhands-workspace/openhands/workspace/docker/egress_images/static/
uv run pytest tests/test_egress_e2e.py -m docker -v
```

Expected: all pass. `test_4b` passing means the naive rule still leaks — which is the documented, expected behavior.

- [ ] **Step 4: Commit**

```bash
git add tests/test_egress_e2e.py
git commit -m "$(printf 'test: docker-gated egress enforcement suite\n\nIncludes the DNS regression pair: 4a proves the shipped address-only\nresolver drop blocks external resolution, 4b proves the naive dport-53\nrule does not, so the ordering/matching requirement cannot regress.\n\nCo-authored-by: openhands <openhands@all-hands.dev>')"
```

---

### Task 11: Full verification and submodule pin

**Files:** Modify: root repo submodule pointer for `vendor/software-agent-sdk`

- [ ] **Step 1: Run the whole default suite**

```bash
cd /home/lfu/git-projects/benchmarks-main/.claude/worktrees/network-modes
uv run pytest tests/ vendor/software-agent-sdk/tests/workspace/ -q
```

Expected: all pass, no docker tests collected.

- [ ] **Step 2: Run the docker-gated suite**

```bash
uv run pytest -m docker -v
```

Expected: all pass.

- [ ] **Step 3: Run pre-commit on every changed file**

```bash
uv run pre-commit run --files $(git diff --name-only network-modes@{u} 2>/dev/null || \
  git log --name-only --pretty=format: -20 | sort -u | grep '\.py$' | tr '\n' ' ')
```

Expected: ruff-format, ruff-check, pycodestyle, and **pyright (strict)** all pass. Fix any type errors — do not add `# type: ignore` without a specific reason.

- [ ] **Step 4: Verify public mode is unchanged**

```bash
OH_NETWORK_MODE= uv run python -c "
from unittest.mock import patch
from openhands.workspace import DockerWorkspace
with patch.object(DockerWorkspace, '_start_container'):
    ws = DockerWorkspace(server_image='test:latest')
ws.host_port = 30000
args = ws._build_run_args('test:latest', container_name='c')
assert ws.network_policy.mode == 'public', ws.network_policy.mode
assert '--network' not in args, 'public mode must not set --network'
assert '30000:8000' in args, 'public mode must publish on all interfaces'
mem = [a for a in args if a == '--memory' or a.startswith('--memory=')]
assert len(mem) == 1, f'expected one --memory flag, got {mem}'
print('PUBLIC_MODE_UNCHANGED')
"
```

Expected: `PUBLIC_MODE_UNCHANGED`.

- [ ] **Step 5: Push the SDK branch and pin it**

```bash
git -C vendor/software-agent-sdk log --oneline network-modes -6
git -C vendor/software-agent-sdk push -u origin network-modes
git add vendor/software-agent-sdk
git commit -m "$(printf 'chore: bump software-agent-sdk for egress control\n\nCo-authored-by: openhands <openhands@all-hands.dev>')"
```

If `push` fails on credentials, stop and report — do not rewrite history or change the remote.

- [ ] **Step 6: Confirm a clean tree**

```bash
git status --short
git log --oneline -8
```

Expected: no unexpected modified or untracked files.

---

## Deferred / follow-up

Not in this plan; tracked so they are not silently lost:

- **Periodic reconciliation.** `reconcile_orphans()` runs before work (Task 8); the periodic (30s) and shutdown passes from spec §4.4, plus lease refresh while a workspace is active, still need wiring into the controller loop. Until then a long-running controller's stale resources are reclaimed at the next worker start rather than continuously.
- **O3 — image distribution.** The sidecar image is built locally as `openhands-egress-static:dev`. CI build, publication, and worker pre-pull are unresolved. Note `/etc/docker/daemon.json` declares an insecure registry (`cr-ee.registry.cn-hangzhou-zjy-d01...`) which may be the intended target — confirm before wiring CI.
- **O4 — IP-only mirror reachability (blocks canary).** D7 removes DNS entirely, so every internal mirror and the LLM proxy must be reachable by IP and must not redirect outside `10.0.0.0/8`. Verify before enabling a non-`public` mode in production. If a mirror is hostname-only, revisit D7 rather than widening the allowlist.
- **IPv6 coverage.** The renderer emits `ip6` rules, but no e2e test exercises them; the default bridge lacks IPv6. A skipped test is not evidence of enforcement.
