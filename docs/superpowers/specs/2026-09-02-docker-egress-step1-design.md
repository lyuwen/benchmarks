# Docker Workspace Egress Control — Step 1 Design

Date: 2026-09-02
Status: Approved for planning (revised after spec review)
Source plan: `docker-workspace-egress-control-implementation-plan.md` (§1–§4, §6, §7)
Review resolved: `spec-review.md` — findings 1–6 all resolved; see §1.2

This spec is a **delta on the source plan**. It records the scope cut, the
decisions taken, and the findings from inspecting and empirically probing the
codebase that the source plan did not account for. Where this spec and the
source plan disagree, this spec wins.

---

## 1. Scope

**In scope:** Step 1 only — the static nftables sidecar with a `10.0.0.0/8`
destination baseline.

**Out of scope:** Step 2 (GOST / hostname allowlisting / runtime phase
switching). `host-allowlist` and `public-bootstrap` are rejected as unsupported
values in Step 1.

### 1.1 Decisions

| # | Decision | Choice | Rationale |
|---|---|---|---|
| D1 | How the policy reaches the workspace | **Two-layer:** central resolution in the evaluation layer (validate, persist, reject remote/non-public) **plus** an SDK-side `network_policy` field defaulting from `OH_NETWORK_MODE` as defense-in-depth | 30+ workspace construction sites exist. Explicit passing at each is fail-open on omission. Central resolution fixes the remote fail-open and the metadata-timing gap (review finding 4); the SDK default keeps enforcement automatic with zero call-site edits. Neither layer alone is sufficient — see §4.5. |
| D2 | Lifecycle machinery | **Lean per-workspace atomic JSON manifest + lease.** Reconcile before work, periodically, and at shutdown | *Revised after review finding 1.* The label-only sweep could not identify the generated rules directory and left SIGKILL residue indefinitely if no later worker started. One atomic JSON manifest per workspace closes both gaps without becoming a framework. |
| D3 | Port binding | `127.0.0.1` in sidecar modes only; `public` keeps `0.0.0.0` | Loopback is required for the sidecar boundary. Changing `public` too would alter behavior for every historical run, violating the repo's backward-compatibility principle (`AGENTS.md`). Verified safe: §2.5. |
| D4 | `--memory` | **Remove the duplicate flag AND set `memory_limit`'s default to `14g`** | *Revised after review finding 6.* Makes `OH_WORKSPACE_MEMORY_LIMIT` functional while keeping the effective limit at 14g, so benchmark comparability is preserved and no OOM behavior shifts. |
| D5 | SDK change delivery | Branch `network-modes` in `vendor/software-agent-sdk` (remote `git@github.com:lyuwen/software-agent-sdk.git`), then bump the root submodule pin | Matches the root repo's branch name. |
| D6 | Target environment | Linux Docker daemon — **verified**, see §2.6 | Confirms source plan §4.4's local-daemon constraint. |
| D7 | Name resolution | **No DNS.** Block the embedded resolver outright; internal services are reached by IP inside `10.0.0.0/8` | *Added after review finding 3.* Source plan §4.1 warns DNS is an egress and exfiltration channel. IP-only is the only option with no residual channel. See §2.7 and §4.3. |
| D8 | Bridge subnet | Inspect the created network's subnets and **abort before starting the sidecar** if they overlap the resolved allowlist | *Added after review finding 5.* Latent today but cheap insurance; see §2.8. |
| D9 | Worker shutdown | Child worker installs a SIGTERM/SIGINT handler so the existing cleanup `finally` runs; parent waits a bounded deadline, then SIGKILLs | *Added after review finding 2.* An idempotent `cleanup()` is useless if the process never calls it. |

### 1.2 Review disposition

| Finding | Disposition |
|---|---|
| 1 — deferred reconciliation contradicts cleanup guarantee | **Accepted.** D2 revised to lean manifest + lease; §4.4 rewritten; §7 acceptance now matches the mechanism. |
| 2 — SIGTERM cleanup cannot work as specified | **Accepted.** D9 added; §4.4 and test 14 updated. |
| 3 — embedded DNS bypasses the policy | **Accepted (vulnerability), remedy corrected.** Confirmed empirically (§2.7). The review's suggested port-53 rule **does not work**; the rule must match on address only. D7 added. |
| 4 — SDK-only resolution fails open for remote | **Accepted, both halves confirmed** (§2.9). D1 revised to two-layer; §4.5 added. |
| 5 — bridge IPAM overlap | **Accepted as latent risk** (§2.8). D8 added. |
| 6 — memory fix | **Accepted, better variant taken.** D4 revised: remove duplicate *and* default to 14g. |

---

## 2. Findings from inspection and probing

### 2.1 The refactor is a genuine prerequisite

`FlexWorkspace._start_container`
(`vendor/software-agent-sdk/openhands-workspace/openhands/workspace/docker/flex_workspace.py:153-340`)
is a ~100-line copy-paste of `DockerWorkspace._start_container`
(`.../docker/workspace.py:156-281`): port allocation, docker availability
check, env forwarding, mount/volume flags, port publication, GPU flags, run
invocation, log thread, host assignment, and health wait are all duplicated.

Adding the egress branch without extracting this first would create **two
independent enforcement paths** — precisely what source plan §1 forbids.
Extraction is therefore step one, not cleanup.

### 2.2 Existing lifecycle is not cleanup-safe

- `DockerWorkspace.cleanup()` (`workspace.py:356-379`) has **no lock** and is
  reachable from `__exit__`, `__del__`, and the harness. Concurrent or repeated
  invocation is not currently safe.
- `cleanup()` issues `docker stop` only and relies on `--rm` for removal.
- The harness kills pool workers with `process.terminate()`
  (`benchmarks/utils/evaluation.py:449`). Default SIGTERM disposition skips the
  `finally` block at `evaluation.py:614-634` that calls `workspace.__exit__`.
  **Orphaned containers are already possible today**, independent of this
  feature. Addressed by D9.

### 2.3 Latent bug in the code being extracted

Both launchers append a hardcoded `--memory=14g` *after* the configurable
`--memory {memory_limit}`:

- `workspace.py:219-220` then `workspace.py:237`
- `flex_workspace.py:294` (flex never applies `memory_limit` at all)

Docker honors the last `--memory`, so the `memory_limit` field and
`OH_WORKSPACE_MEMORY_LIMIT` are silently inert.

**Resolution (D4):** remove the hardcoded `--memory=14g` and change
`memory_limit`'s default from `13g` to `14g`. The environment variable becomes
functional; the effective limit is unchanged at 14g; no OOM behavior shifts and
historical comparability is preserved.

### 2.4 Migration hazard: `OH_DISABLE_NETWORK` is NOT `no-network`

`benchmarks/utils/network_isolation.py:52-58` allows only **loopback +
established/related** and drops everything else. It has no `10.0.0.0/8`
exception.

The equivalent new mode is **`strict-no-network`**, not `no-network` —
`no-network` additionally permits all of `10.0.0.0/8`. Migrating by name
similarity would **silently widen** isolation. This must appear in the removal
commit message and in the `OH_NETWORK_MODE` documentation.

### 2.5 Loopback publication does not break the remote conversation

Verified: the agent-server client is the host-side evaluation process and
already reaches the container over loopback —
`host = "http://localhost:{host_port}"` (`workspace.py:273`) and the health
probe `http://127.0.0.1:{host_port}/health` (`workspace.py:313`). There is no
`DOCKER_HOST`, no `docker.sock` mount, and no `host.docker.internal` anywhere
in the repository, so the controller is not containerized and the daemon is
local.

Under the nftables policy the agent-server API keeps working by design: the
OUTPUT chain is `policy drop`, but response packets on a host-initiated inbound
connection match `ct state established,related accept`.

**Known incompatibility (documented, not currently triggered):** a
*containerized* controller reaching the workspace via the docker bridge gateway
would not work under loopback publication. D3 keeps `public` on `0.0.0.0`, so
this is unaffected today.

### 2.6 Environment verified (resolves O1)

- Host: Linux, kernel `6.17.0-35-generic`, `x86_64`, storage driver `overlay2`.
- Docker daemon: `OSType=linux`, context `default` on `unix:///var/run/docker.sock`.
- **Functional probe passed:** an `inet` table with an `output` filter hook was
  created and listed successfully inside a container run with
  `--cap-drop ALL --cap-add NET_ADMIN --security-opt no-new-privileges=true`.
  `NET_RAW` was **not** required, confirming source plan §6's guidance that
  Step 1 should not receive it.

### 2.7 Docker embedded DNS bypasses the policy — confirmed empirically

Running the spec's exact §4.3 policy (`policy drop`, `oifname "lo" accept`,
`ct state established,related accept`, `ip daddr 10.0.0.0/8 accept`, final
reject) in a container on a user-defined bridge:

| Probe | Result |
|---|---|
| `nslookup example.com 127.0.0.11` | **SUCCEEDED** — returned real external A records |
| `nslookup example.com 8.8.8.8` | Correctly failed (host unreachable) |
| `wget http://93.184.216.34/` | Correctly failed (host unreachable) |

The container's `resolv.conf` reports `nameserver 127.0.0.11` and
`ExtServers: [host(127.0.0.53)]`: dockerd forwards the query **from the host
namespace**, so the container's OUTPUT chain never sees the external traffic.
This is a live data-exfiltration channel.

**The review's suggested remedy does not work.** Adding
`ip daddr 127.0.0.11 udp dport 53 reject` and
`ip daddr 127.0.0.11 tcp dport 53 reject` **before** the loopback accept was
tested and external DNS still resolved. The cause is netfilter hook ordering:
Docker's resolver DNATs `127.0.0.11:53` in the `nat` OUTPUT chain
(priority −100), which runs **before** the `filter` OUTPUT hook (priority 0).
By the time the filter chain sees the packet the destination port has been
rewritten to the resolver's ephemeral port, so a `dport 53` match never fires.

**The remedy that works**, verified: match on address alone —
`ip daddr 127.0.0.11 drop`, placed before the loopback accept. Retested:
embedded-resolver DNS timed out via both the explicit server and the default
`resolv.conf` path, while ordinary loopback TCP (`nc` to `127.0.0.1:9999`)
still worked, so local fixtures such as the httpbin workaround
(`benchmarks/utils/httpbin_fix.py`) are unaffected.

Both the naive port-based rule and the working address-based rule become
regression tests (§5.2 tests 4a/4b) so this cannot silently regress.

### 2.8 Bridge subnet overlap is latent, not active

`docker info` reports `DefaultAddressPools = null` and `/etc/docker/daemon.json`
sets no `default-address-pools`; a probe network was allocated
`172.18.0.0/16`, which does not overlap `10.0.0.0/8`.

The risk is real but configuration-dependent: a daemon whose pools include
`10.0.0.0/8` would place the bridge gateway and host services **inside** the
allowlist, shadowing the internal service range. D8 adds an explicit
pre-sidecar overlap check rather than relying on the current daemon config.

### 2.9 SDK-only resolution fails open for remote — confirmed

Both halves of review finding 4 verified:

- `APIRemoteWorkspace` (`.../remote_api/workspace.py:19`) extends
  `RemoteWorkspace` and has no `network_policy` field and no local container.
  An SDK default factory on the Docker policy field therefore never runs, so
  `OH_NETWORK_MODE=no-network` with `workspace_type=remote` would run
  **unrestricted** — fail-open.
- `metadata.json` is written at `evaluation.py:57-60` during setup, while
  `prepare_workspace` is not called until `evaluation.py:529`. Policy resolved
  at workspace construction therefore **cannot** be persisted into metadata.

Both are fixed by the central resolution layer in §4.5.

### 2.10 Tooling constraints

- Pre-commit runs **pyright (strict)**, ruff format, ruff check, and
  pycodestyle at `--max-line-length=88`. `AGENTS.md`: **never use mypy**.
- Ruff: `target-version = py312`, `line-length = 88`, `select = ["E","F","I"]`,
  isort `known-first-party = ["benchmarks","openhands"]`.
- `pytest` has **no marker configuration**. A `docker` marker must be
  introduced (§5.2).
- Commits carry `Co-authored-by: openhands <openhands@all-hands.dev>`; only
  relevant files are committed.

---

## 3. Configuration model

Implemented in `openhands-workspace`, Pydantic v2, mirroring existing SDK style.

```python
class AllowedEndpoint(BaseModel):
    destination: IPv4Network | IPv6Network
    protocol: Literal["tcp", "udp"] | None = None
    ports: tuple[int, ...] = ()
    description: str | None = None


class WorkspaceNetworkPolicy(BaseModel):
    mode: Literal[
        "public", "static-allowlist", "no-network", "strict-no-network"
    ] = "public"
    allowed_endpoints: tuple[AllowedEndpoint, ...] = ()
```

`host-allowlist` and `public-bootstrap` are rejected with a message naming
Step 2.

### 3.1 Resolution rules

- The mandatory baseline is `10.0.0.0/8`, no protocol or port restriction.
- `static-allowlist`: baseline ∪ caller `allowed_endpoints`. Additions never
  replace or narrow the baseline.
- `no-network`: exactly the baseline. Caller endpoints are **rejected**, not
  ignored.
- `strict-no-network`: no external destinations. Caller endpoints rejected.
- `public`: no sidecar; allowlist fields rejected.
- If `protocol` is given, a non-empty port list in `1..65535` is required.
  Ports without a protocol are rejected.
- Reject multicast, broadcast, link-local, and unspecified destinations.
- Deduplicate and deterministically sort. Cap endpoint and port counts.

### 3.2 `OH_NETWORK_MODE`

- Unset or empty → `public` (preserves today's behavior exactly).
- Unknown / misspelled / Step-2-only values → **hard error**. Never coerced to
  `public`.
- Explicit policy + `OH_NETWORK_MODE` must **match**; mismatch is an error, not
  a silent precedence rule.

---

## 4. Architecture

### 4.1 Topology

Sidecar starts first and owns the network namespace; the workspace joins with
`--network container:<egress-id>`. Per-workspace dedicated bridge, whose
subnets are checked for allowlist overlap before the sidecar starts (D8). All
workspace ports (8000, plus 8001/8002 when `extra_ports`) are published **by
the sidecar** on `127.0.0.1`.

The workspace container gets `--cap-drop NET_ADMIN`, `--cap-drop NET_RAW`, and
`--security-opt no-new-privileges=true`. It never receives privileged mode, host
networking, the Docker socket, or `SYS_ADMIN`.

Conflicting workspace arguments (`--network`, `-p`, `--dns`, `--add-host`,
`--hostname`, `--expose`) are rejected in sidecar modes — Docker does not
support them with `container:` network mode.

### 4.2 Code structure

Shared extraction first:

```
_start_container(Docker) ─┐
                          ├─► _build_run_args()   (single construction path)
_start_container(Flex)   ─┘   + EgressRuntime      (single enforcement path)
```

`DockerDevWorkspace` inherits this with no separate path.

New in `vendor/software-agent-sdk/openhands-workspace/openhands/workspace/docker/`:

- `network_policy.py` — models and resolution rules of §3.
- `nftables_renderer.py` — validated structured config → rules file + canonical
  digest. Never accepts raw nftables fragments.
- `egress_runtime.py` — sidecar lifecycle: start, readiness, verification,
  rollback, locked idempotent cleanup, liveness watcher, manifest I/O.
- `egress_images/static/{Dockerfile,entrypoint.sh}` — prebuilt image,
  digest-pinned Alpine base, nftables installed at build time, no runtime
  package installation.

New in the benchmarks repo:

- `benchmarks/utils/workspace_network.py` — central mode resolution (§4.5).

Removed: `benchmarks/utils/network_isolation.py` and its single call site at
`benchmarks/swebench/run_infer.py:293` (import at `:19`). No compatibility
mapping and no privileged-exec fallback are retained.

### 4.3 Rendered policy

An `inet` table so one policy covers IPv4 and IPv6. **Rule order is
load-bearing** — the resolver drop must precede the loopback accept:

```nft
table inet workspace_egress {
    chain output {
        type filter hook output priority filter;
        policy drop;

        # D7: block Docker's embedded resolver. Address-only match —
        # a dport 53 match does NOT work, because Docker DNATs
        # 127.0.0.11:53 in nat OUTPUT (priority -100), before this
        # filter hook (priority 0), rewriting the port. See §2.7.
        ip daddr 127.0.0.11 drop

        oifname "lo" accept
        ct state established,related accept
        ip daddr 10.0.0.0/8 accept
        reject with icmpx type admin-prohibited
    }
}
```

Under D7 there is no name resolution: internal services are addressed by IP
inside `10.0.0.0/8`. Package managers and installation tooling must be
configured with IP-based mirror URLs, and any mirror must serve content itself
rather than redirecting outside the allowlist.

The rules file is written to a workspace-owned directory beneath the fixed state
root with restrictive permissions and bind-mounted read-only. After `nft -f`,
the applied ruleset is verified **against the recorded canonical digest** —
checking only that a table and chain exist is insufficient. Readiness is
published only after that verification passes.

All docker commands are built as argv arrays; no user-supplied value is ever
concatenated into a shell string.

### 4.4 Failure, cleanup, and reconciliation

**Fail closed, always.** Missing image, rule-application failure, readiness
timeout, subnet overlap, or a failed post-start probe aborts workspace creation.
There is never a retry with unrestricted networking and never a privileged-exec
fallback.

**Transactional startup.** Each acquired resource is recorded in the manifest as
it is created: state dir + rules file → network (subnet checked) → sidecar
(started, verified) → workspace → health. Any failure or cancellation
synchronously unwinds completed steps in reverse. Cleanup does not rely on
`__del__` or Docker garbage collection.

**Manifest (D2).** One atomically written JSON file per workspace under a
single fixed, controller-owned state root, holding: controller ID, workspace
runtime ID, sidecar and workspace container IDs, network ID, rules path, policy
digest, creation time, and lease expiry. Written atomically and updated after
every acquisition. Every managed container and network carries the same
controller and workspace IDs as labels. All generated paths stay beneath the
fixed state root, and IDs are validated before deletion — never reconcile a
path taken from a label or task input.

**Cleanup** is one idempotent, lock-guarded operation; a second or concurrent
call is harmless. Order: mark manifest `cleaning` → stop/remove workspace →
stop/remove sidecar → remove network → delete rules directory → verify absence
by exact ID → remove manifest. It is **best-effort across resources, not
fail-fast**: failing to stop the workspace must not skip sidecar, network, and
rules cleanup. Transient Docker conflicts retry with short bounded backoff;
already-absent resources count as success; remaining failures are aggregated and
logged with their managed IDs. If anything remains, the manifest is kept with
failure details and an expired lease for the next pass.

**Liveness watcher** owned by the workspace runtime detects sidecar exit, marks
the workspace failed, and stops it. The sidecar is never auto-restarted into an
unknown policy state.

**Reconciliation** runs once before a worker accepts work, periodically while
the controller is alive, and once at normal shutdown. Under a controller-wide
lock it removes only resources whose complete managed identity matches a
manifest with a lease stale beyond the grace period, in the same
container-before-network order as cleanup, continuing after individual
failures. Live leases, unrelated objects, and partially labelled objects are
never touched.

**Worker shutdown (D9).** The child worker installs SIGTERM/SIGINT handlers
that raise, so the existing `finally` at `evaluation.py:614-634` runs
`workspace.__exit__`. The parent's `_cleanup_pool` sends SIGTERM, waits a
bounded deadline, then escalates to SIGKILL. Docker work never happens inside
an async signal handler — the handler signals a normal shutdown path.

**Timing defaults** (host-operator settings, validated and logged, shorter
values injectable in tests): refresh lease every 5s, reconcile every 30s, lease
stale after 2min, 10s graceful container stop, 30s bounded worker shutdown.

Both containers use `--restart no`.

### 4.5 Two-layer mode resolution (D1)

Neither layer alone is sufficient, so both exist:

**Layer 1 — evaluation layer** (`benchmarks/utils/workspace_network.py`, called
during `Evaluation` setup **before** `metadata.json` is written at
`evaluation.py:57`):

- Parse and validate `OH_NETWORK_MODE`; hard-error on unknown or Step-2 values.
- Reconcile against any explicit policy; error on mismatch.
- **Reject `workspace_type=remote` with any non-`public` mode** — closes the
  fail-open path of §2.9.
- Persist the resolved policy and its digest into `EvalMetadata`
  (`benchmarks/utils/models.py`) so it lands in `metadata.json`.

**Layer 2 — SDK** (`network_policy` field on `DockerWorkspace` with a
`default_factory` reading `OH_NETWORK_MODE`):

- Enforces automatically for Docker/Flex/Dev with zero call-site edits, so a
  benchmark that never learned about the feature still gets enforcement.
- An explicitly passed policy overrides the default.

Layer 1 is the authority for validation and persistence; Layer 2 is the
enforcement backstop.

---

## 5. Verification

### 5.1 Unit (no docker; run everywhere)

- IPv4/IPv6 address, CIDR, protocol, and port validation; rejection of
  malformed and unsafe values.
- Deterministic rule rendering, dedup, and stable ordering.
- **The resolver-drop rule is emitted before the loopback accept** — ordering is
  asserted, not just presence.
- `no-network` resolves to exactly the `10.0.0.0/8` baseline.
- `strict-no-network` resolves to no external destinations.
- Caller entries union with, and cannot replace or narrow, the baseline.
- Endpoint/port caps; rejection of multicast/broadcast/link-local/unspecified.
- The workspace never receives `NET_ADMIN` or `NET_RAW`.
- Identical policy behavior and lifecycle wiring for `DockerWorkspace` and
  `FlexWorkspace`, asserted on shared-builder output.
- `OH_NETWORK_MODE`: unset/empty → `public`; invalid → error; Step-2 values →
  error naming Step 2; explicit-policy conflict → error.
- **Layer 1 rejects `remote` + non-`public`**, and the resolved policy + digest
  appear in `EvalMetadata` before `metadata.json` is written.
- Subnet-overlap guard rejects a bridge subnet intersecting the allowlist.
- Rollback from every partial-startup failure point (faked docker layer).
- `cleanup()` is idempotent and safe under concurrent invocation.
- One resource's cleanup error does not prevent cleanup of the rest.
- Reconciliation distinguishes live-lease, stale-managed, and unrelated objects.
- Manifest writes are atomic and survive truncation/partial-write simulation.
- `public` mode produces byte-identical run arguments to today, except the
  removed duplicate `--memory` (D4).

### 5.2 Docker-gated tests

`pyproject.toml` gains a `docker` pytest marker (none exists today); these are
excluded from the default run and executed in a dedicated Linux CI job.

**Image tests:** valid rules produce readiness; invalid rules exit nonzero
without readiness; readiness only after verification; SIGTERM/SIGINT/SIGHUP
exit cleanly with no orphan children; no runtime package installation.

**End-to-end** (local fixture servers for allowed and denied targets, so tests
never depend on public internet):

1. Destinations inside `10.0.0.0/8` succeed across TCP, UDP, and ICMP.
2. Other private ranges fail unless explicitly added.
3. A public IPv4 address fails unless explicitly added.
4. **DNS regression pair, from §2.7:**
   - **4a** With the resolver-drop rule, `nslookup` via `127.0.0.11` and via the
     default `resolv.conf` path both fail, while loopback TCP still works.
   - **4b** With a `dport 53`-only rule substituted, external resolution
     **succeeds** — proving the naive remedy is insufficient and locking in why
     the address-only match is required.
5. IPv6 and direct QUIC cannot bypass the policy.
6. Loopback and established response traffic keep working — including the
   agent-server API through the sidecar's published port.
7. The workspace cannot list, flush, or replace nftables policy.
8. Two concurrent workspaces get independent namespaces and policies.
9. Sidecar failure prevents workspace startup.
10. Sidecar death invalidates the workspace.
11. Cleanup removes both containers, the network, the rules directory, and the
    manifest.
12. Agent-server ports reachable from host loopback but **not** from a remote
    host or an unrelated container.
13. Failure after sidecar startup leaves no residue.
14. **SIGINT/SIGTERM at each startup stage** leaves no managed resources within
    the bounded shutdown deadline (exercises D9 end to end).
15. After `SIGKILL` of the controller, resources persist only until lease
    expiry; the next reconciliation pass removes containers, network, rules
    directory, and manifest, while preserving live-lease and unrelated objects.
16. A non-`public` policy on an unsupported or remote Docker daemon fails closed.
17. Bridge subnet overlapping `10.0.0.0/8` aborts startup with no residue.

The CI job records the Docker Engine, kernel, nftables, IPv4, and IPv6
combination exercised. A test skipped because the bridge lacks IPv6 does not
count as IPv6 enforcement.

---

## 6. Open items

- **O1 — RESOLVED.** Linux `6.17.0-35-generic`, daemon `OSType=linux` on the
  default unix socket; functional `inet`/`NET_ADMIN` probe passed without
  `NET_RAW` (§2.6).
- **O2 — RESOLVED.** Remote `git@github.com:lyuwen/software-agent-sdk.git`,
  currently detached at `492f5036`; work lands on branch `network-modes` (D5).
- **O3 — Sidecar image distribution.** Source plan §4.3 assumes an internal
  registry with pre-pulled workers. Build/publish/pre-pull mechanics and the
  digest-pinned Alpine base must be settled during planning. Note
  `/etc/docker/daemon.json` declares an insecure registry, which may be the
  intended publication target — confirm before wiring CI.
- **O4 — IP-only mirror reachability.** D7 removes DNS entirely, so every
  internal mirror and the LLM proxy must be reachable by IP and must not
  redirect outside `10.0.0.0/8`. Confirm the deployment's mirror URLs satisfy
  this before enabling a non-`public` mode in the canary (source plan §7 step 7).
  If any mirror is hostname-only, revisit D7 rather than widening the allowlist.

---

## 7. Acceptance

Step 1 is done when: enforcement is selectable per workspace; the
privileged-exec implementation and all `OH_DISABLE_NETWORK` handling are gone;
unset/empty `OH_NETWORK_MODE` preserves unrestricted behavior while every
non-public mode fails closed **including `workspace_type=remote`**; the `public`
path is behaviorally unchanged and the effective memory limit is still 14g; the
sidecar image is prebuilt with no runtime installation; the `10.0.0.0/8`
baseline is always reachable in allowlist modes and cannot be replaced by caller
entries; destinations outside the resolved allowlist are blocked across TCP,
UDP, ICMP, and QUIC; **the Docker embedded resolver cannot resolve external
names, with both the working and the naive rule variants covered by
regression tests**; a bridge subnet overlapping the allowlist aborts startup;
the workspace cannot modify policy; unhealthy sidecars and unsupported hosts
fail closed; concurrent workspaces stay isolated; graceful exit and
SIGTERM/SIGINT within the bounded deadline leave no managed containers,
networks, rules directories, manifests, or threads; lease-based reconciliation
removes stale managed resources after an uncatchable failure without touching
live or unrelated resources; and all §5 tests pass.
