# Fix httpbin flakiness on the inference end — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** For `psf/requests` inference instances, stand up a local httpbin server inside the openhands agent-server workspace and redirect `httpbin.org` / `www.google.co.uk` to it, so the agent's own test runs are deterministic instead of hitting the flaky public service.

**Architecture:** A new shared helper `benchmarks/utils/httpbin_fix.py` returns the ordered shell commands (adapted from `thirdparty/SWE-bench`'s hardened offline version, but `sudo`-wrapped for the non-root `openhands` workspace user). `run_infer.py` calls it in `prepare_workspace`, gated on `repo == "psf/requests"`, running the commands non-fatally so a setup failure degrades to external httpbin rather than aborting the instance.

**Tech Stack:** Python 3, pytest, OpenHands SDK `RemoteWorkspace.execute_command`, gunicorn/gevent + httpbin inside the container.

## Global Constraints

- Workspace container runs as **`openhands` (uid 10001)**, NOT root; it has **passwordless sudo** (`NOPASSWD:ALL`). Every privileged step MUST be `sudo`-wrapped.
- Target the testbed interpreter explicitly: `/opt/miniconda3/envs/testbed/bin/python` (its env is root-owned).
- Pinned versions: `httpbin[mainapp]==0.10.2`, `pytest-httpbin==2.1.0`.
- Cert dir: `/tmp/swebench_httpbin_certs`; cert SAN MUST include `DNS:httpbin.org`.
- Hosts redirect line: `127.0.0.1    httpbin.org www.google.co.uk`.
- Profile persistence file: `/etc/profile.d/swebench_httpbin.sh`.
- Gating: only `instance.data.get("repo") == "psf/requests"`; all other repos untouched.
- httpbin setup is **non-fatal** — log a warning on failure, never `raise`.
- Source of truth for the approach: `thirdparty/SWE-bench/swebench/harness/test_spec/python.py` (`make_httpbin_setup_commands`).

---

### Task 1: Shared helper `benchmarks/utils/httpbin_fix.py`

**Files:**
- Create: `benchmarks/utils/httpbin_fix.py`
- Test: `benchmarks/utils/test_httpbin_fix.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `make_httpbin_setup_commands() -> list[str]` — ordered shell commands to install httpbin, generate a SAN cert, trust it, launch gunicorn on 80/443, edit `/etc/hosts`, and persist env exports to `/etc/profile.d`.

- [ ] **Step 1: Write the failing test**

```python
# benchmarks/utils/test_httpbin_fix.py
"""Test the httpbin setup command list for the inference workspace.

The workspace runs as the non-root `openhands` user (uid 10001) with
passwordless sudo, so every privileged step must be sudo-wrapped. See
docs/superpowers/specs/2026-07-24-fix-httpbin-inference-design.md.
"""

import importlib.util

SPEC = importlib.util.spec_from_file_location(
    "httpbin_fix", "benchmarks/utils/httpbin_fix.py"
)
assert SPEC is not None and SPEC.loader is not None
httpbin_fix = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(httpbin_fix)


def test_returns_nonempty_command_list():
    cmds = httpbin_fix.make_httpbin_setup_commands()
    assert isinstance(cmds, list) and cmds
    assert all(isinstance(c, str) for c in cmds)


def test_pip_installs_pinned_versions_into_testbed():
    joined = "\n".join(httpbin_fix.make_httpbin_setup_commands())
    assert "httpbin[mainapp]==0.10.2" in joined
    assert "pytest-httpbin==2.1.0" in joined
    assert "/opt/miniconda3/envs/testbed/bin/python" in joined


def test_cert_has_httpbin_org_san():
    joined = "\n".join(httpbin_fix.make_httpbin_setup_commands())
    assert "subjectAltName=DNS:httpbin.org" in joined
    assert "/tmp/swebench_httpbin_certs" in joined


def test_trusts_cert_both_paths():
    joined = "\n".join(httpbin_fix.make_httpbin_setup_commands())
    assert "REQUESTS_CA_BUNDLE" in joined
    assert "CURL_CA_BUNDLE" in joined
    assert "requests.certs.where()" in joined


def test_hosts_and_profile_redirect():
    joined = "\n".join(httpbin_fix.make_httpbin_setup_commands())
    assert "httpbin.org www.google.co.uk" in joined
    assert "/etc/hosts" in joined
    assert "/etc/profile.d/swebench_httpbin.sh" in joined


def test_privileged_steps_are_sudo_wrapped():
    cmds = httpbin_fix.make_httpbin_setup_commands()
    joined = "\n".join(cmds)
    # gunicorn on privileged ports, hosts, profile.d, cacert append, pip -> sudo
    assert "sudo" in joined
    for c in cmds:
        if "/etc/hosts" in c or "/etc/profile.d" in c:
            assert "sudo" in c, f"privileged step not sudo-wrapped: {c!r}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest benchmarks/utils/test_httpbin_fix.py -v`
Expected: FAIL — `FileNotFoundError` / module import error (`httpbin_fix.py` does not exist yet).

- [ ] **Step 3: Write minimal implementation**

```python
# benchmarks/utils/httpbin_fix.py
"""Local-httpbin setup for `psf/requests` inference instances.

Some `psf/requests` tests hit the public `httpbin.org` / `www.google.co.uk`,
which is slow or unreachable and makes the agent's own iterate-and-verify loop
flaky. This module returns shell commands that stand up a LOCAL httpbin server
inside the workspace container and redirect those hosts to 127.0.0.1.

Adapted from the hardened offline-harness version in
``thirdparty/SWE-bench/swebench/harness/test_spec/python.py``
(``make_httpbin_setup_commands``), which is the source of truth for the
approach. KEY DIFFERENCE: the offline harness runs as root; the inference
workspace runs as the non-root ``openhands`` user (uid 10001) with passwordless
sudo, so every privileged step here is ``sudo``-wrapped and pip/gunicorn target
the root-owned testbed interpreter explicitly.
"""

from __future__ import annotations

# Explicit testbed interpreter (its conda env is root-owned).
TESTBED_PY = "/opt/miniconda3/envs/testbed/bin/python"
CERT_DIR = "/tmp/swebench_httpbin_certs"
CERT = f"{CERT_DIR}/cert.pem"
KEY = f"{CERT_DIR}/key.pem"
PROFILE_D = "/etc/profile.d/swebench_httpbin.sh"


def make_httpbin_setup_commands() -> list[str]:
    """Return the ordered shell commands to set up local httpbin.

    Intended to be run one-by-one via ``workspace.execute_command`` in the
    openhands agent-server workspace. Privileged steps use ``sudo`` (available
    passwordless as the ``openhands`` user).
    """
    return [
        # 1. Install httpbin + gunicorn/gevent into the root-owned testbed env.
        f"sudo -E {TESTBED_PY} -m pip install "
        "'httpbin[mainapp]==0.10.2' 'pytest-httpbin==2.1.0'",
        # 2. Generate a self-signed cert valid for httpbin.org (SAN, not just CN).
        f"mkdir -p {CERT_DIR}",
        "openssl req -x509 -newkey rsa:2048 -nodes "
        f"-keyout {KEY} -out {CERT} -days 3650 "
        "-subj '/CN=httpbin.org' "
        "-addext 'subjectAltName=DNS:httpbin.org,DNS:localhost,IP:127.0.0.1'",
        # 3. Trust it for env-var-aware requests calls (shell-local export).
        f"export REQUESTS_CA_BUNDLE={CERT}",
        f"export CURL_CA_BUNDLE={CERT}",
        # 4. Trust it for Session.send() which bypasses the env var: append to
        #    the root-owned bundle that requests.certs.where() resolves to.
        f"sudo bash -c 'cat {CERT} >> "
        f'"$({TESTBED_PY} -c \\"import requests; print(requests.certs.where())\\")"\'',
        # 5. Launch detached gunicorn: HTTP on :80, HTTPS on :443 (privileged).
        f"sudo bash -c '(nohup {TESTBED_PY} -m gunicorn -b 127.0.0.1:80 "
        "-k gevent httpbin:app > /dev/null 2>&1 &)'",
        f"sudo bash -c '(nohup {TESTBED_PY} -m gunicorn -b 127.0.0.1:443 "
        f"--certfile={CERT} --keyfile={KEY} -k gevent httpbin:app "
        "> /dev/null 2>&1 &)'",
        # 6. Let gunicorn bind before tests race it.
        "sleep 2",
        # 7. Redirect the external hosts to the local server.
        'echo "127.0.0.1    httpbin.org www.google.co.uk" | sudo tee -a /etc/hosts',
        # 8. Persist exports so the agent's SEPARATE login shell inherits them.
        "printf '%s\\n' "
        f"'export REQUESTS_CA_BUNDLE={CERT}' "
        f"'export CURL_CA_BUNDLE={CERT}' "
        f"| sudo tee {PROFILE_D}",
    ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest benchmarks/utils/test_httpbin_fix.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: py_compile the new module**

Run: `python -m py_compile benchmarks/utils/httpbin_fix.py`
Expected: no output, exit 0.

- [ ] **Step 6: Commit**

```bash
git add benchmarks/utils/httpbin_fix.py benchmarks/utils/test_httpbin_fix.py
git commit -m "feat: add sudo-wrapped local-httpbin setup helper for inference

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: Wire the helper into `prepare_workspace`

**Files:**
- Modify: `benchmarks/swebench/run_infer.py` (import block near line 16; injection after the env_setup loop at line 273, before `return workspace` at line 274)

**Interfaces:**
- Consumes: `make_httpbin_setup_commands()` from Task 1.
- Produces: no new symbols; a side effect during `prepare_workspace` for `psf/requests` instances.

- [ ] **Step 1: Add the import**

Add alongside the existing `from benchmarks.utils.mirror_config import get_mirror_env_commands` line:

```python
from benchmarks.utils.httpbin_fix import make_httpbin_setup_commands
```

- [ ] **Step 2: Inject the gated, non-fatal setup**

Insert immediately after the `env_setup_commands` loop (after `logger.debug(f"Ran env setup command '{cmd}': {res.stdout}")`) and BEFORE `return workspace`:

```python
        # psf/requests tests hit the flaky public httpbin.org; stand up a local
        # httpbin so the agent's own test runs are deterministic. Non-fatal:
        # on failure we degrade to external httpbin rather than losing the
        # instance. See benchmarks/utils/httpbin_fix.py.
        if instance.data.get("repo") == "psf/requests":
            for cmd in make_httpbin_setup_commands():
                res = workspace.execute_command(cmd)
                if res.exit_code != 0:
                    logger.warning(
                        "httpbin setup command failed (continuing): %r: %s",
                        cmd,
                        res.stderr,
                    )
                else:
                    logger.debug("Ran httpbin setup command '%s': %s", cmd, res.stdout)
```

- [ ] **Step 3: Verify `instance` is in scope**

Run: `sed -n '129,140p' benchmarks/swebench/run_infer.py`
Expected: the `def prepare_workspace(self, instance: EvalInstance, ...)` signature includes an `instance` parameter. If the parameter is named differently, use that name in Step 2.

- [ ] **Step 4: py_compile the modified file**

Run: `python -m py_compile benchmarks/swebench/run_infer.py`
Expected: no output, exit 0.

- [ ] **Step 5: Import-smoke the module**

Run: `python -c "import benchmarks.swebench.run_infer"`
Expected: exits 0 (no ImportError for `make_httpbin_setup_commands`).

- [ ] **Step 6: Commit**

```bash
git add benchmarks/swebench/run_infer.py
git commit -m "feat: inject local-httpbin setup for psf/requests inference instances

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3: Manual container verification (documented, run-once)

**Files:** none (verification only). Record results in the commit message / PR body.

**Interfaces:** Consumes the behavior from Tasks 1–2.

- [ ] **Step 1: Print the command list for inspection**

Run:
```bash
python -c "from benchmarks.utils.httpbin_fix import make_httpbin_setup_commands; print(chr(10).join(make_httpbin_setup_commands()))"
```
Expected: the ordered commands, each privileged one prefixed with `sudo`.

- [ ] **Step 2: Exercise the commands inside a psf/requests agent-server image**

Run (substitute the actual psf/requests agent-server image tag available in the registry):
```bash
img="<psf__requests agent-server image>"
docker run --rm --entrypoint bash "$img" -lc '
  set -e
  cmds=$(python3 - <<"PY"
from benchmarks.utils.httpbin_fix import make_httpbin_setup_commands
print("\n".join(make_httpbin_setup_commands()))
PY
) 2>/dev/null || true
  # If benchmarks pkg is not on the image, paste the printed commands from Step 1 here instead.
  whoami
  sudo -n true && echo SUDO_OK
'
```
Expected: `whoami` → `openhands`; `SUDO_OK` printed. (If the `benchmarks` package is not present on the image, paste the Step 1 output directly.)

- [ ] **Step 3: Confirm the end state**

Inside the same container, after running the setup commands, run:
```bash
grep 'httpbin.org' /etc/hosts
bash -lc 'echo "CA=$REQUESTS_CA_BUNDLE"'
curl -sS https://httpbin.org/get -o /dev/null -w '%{http_code}\n'
```
Expected: `/etc/hosts` shows the redirect line; `CA=/tmp/swebench_httpbin_certs/cert.pem` (proves `/etc/profile.d` is sourced by the login shell); `curl` returns `200` against the local HTTPS server with a trusted cert.

- [ ] **Step 4: Record the outcome**

Note the observed results (user, sudo, hosts line, CA var, curl 200) in the PR description. No commit needed for this task.

---

## Self-Review

**Spec coverage:**
- Shared helper `benchmarks/utils/httpbin_fix.py` → Task 1. ✓
- Gated, non-fatal injection in `prepare_workspace` → Task 2. ✓
- sudo-wrapping / testbed interpreter / profile.d persistence → Task 1 impl + Global Constraints. ✓
- Unit test asserting critical lines + sudo-wrapping → Task 1 Step 1. ✓
- Manual container verification → Task 3. ✓
- Judge end (verify-only, no code change) → intentionally NOT a task; spec marks it already-covered. Called out here so it isn't mistaken for a gap.

**Placeholder scan:** The only intentional `<placeholder>` is the concrete psf/requests image tag in Task 3 (resolved at run time from the registry) — flagged inline, not a code placeholder.

**Type consistency:** `make_httpbin_setup_commands()` name/signature/return type (`list[str]`) identical across the helper (Task 1), its test (Task 1), and the import + call site (Task 2). ✓
