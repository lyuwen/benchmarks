# Mirror the psf/requests httpbin fix onto the inference end

**Date:** 2026-07-24
**Branch (worktree):** `feat/fix-httpbin`
**Reference:** `thirdparty/SWE-bench/HTTPBIN_FIX_REPORT.md`

## Problem

The `psf__requests-*` instances in SWE-bench_Verified run tests that make live
HTTP/HTTPS requests to `httpbin.org` (and one redirect to `www.google.co.uk`).
That public service is frequently slow or unreachable, producing false negatives
unrelated to the model's patch.

The fix — stand up a **local** httpbin server inside the container and redirect
`httpbin.org` to `127.0.0.1` via `/etc/hosts`, with a TLS cert valid for the
`httpbin.org` hostname — is documented in
`thirdparty/SWE-bench/HTTPBIN_FIX_REPORT.md`.

This fix is already applied on the **judge/grading** path but not on the
**inference** path. This spec mirrors it onto inference.

## Current state (what already exists)

### Judge end — already covered, verify only

`benchmarks/swebench/judge.py` (`SWEBenchJudge`) grades patches via
`run_instance` / `make_test_spec` imported from the installed `swebench`
package. That package resolves to the editable install at
`/home/lfu/git-projects/SWE-bench`, currently on branch `feat/fix-httpbin`,
which is byte-identical to the in-repo `thirdparty/SWE-bench`. Both already
contain `make_httpbin_setup_commands()` and inject it for `psf/requests` inside
`make_eval_script_list_py(...)` (`swebench/harness/test_spec/python.py`).

Therefore execution-based judging of `psf/requests` already stands up the local
httpbin server. **No judge code change is required** — this spec only verifies
and documents that path.

### Inference end — the gap to fix

`benchmarks/swebench/run_infer.py` has no httpbin handling. The agent prompt
(`benchmarks/swebench/prompts/default.j2`) explicitly instructs the agent to run
the test suite during its session (Phase 2 "RUNNING", Phase 7 "VERIFICATION",
Phase 8 "FINAL REVIEW"). For `psf/requests`, those tests hit the flaky public
`httpbin.org`, so the agent's own iterate-and-verify loop is unreliable.

Relevant facts confirmed (openhands agent-server workspace image, **not** the
bare testbed image):

- The workspace container runs as user **`openhands` (uid 10001)**, not root.
  The `openhands` user has **passwordless sudo** (`NOPASSWD:ALL`, granted in the
  agent-server Dockerfile). `openssl` is present.
- The testbed conda env (`/opt/miniconda3/envs/testbed`) and the requests CA
  bundle (`requests.certs.where()`) are **root-owned**, and ports 80/443 and
  `/etc/hosts` / `/etc/profile.d` require root. Therefore nearly every setup step
  must be run through `sudo`:

  | Step | Needs `sudo` | Why |
  |------|:---:|------|
  | `pip install httpbin ...` (into testbed) | yes | testbed env is root-owned |
  | append cert to `requests.certs.where()` | yes | root-owned CA file |
  | `gunicorn` bind :80 / :443 | yes | privileged ports |
  | write `/etc/hosts` | yes | root-owned |
  | write `/etc/profile.d/*.sh` | yes | root-owned |
  | generate cert under `/tmp/...` | no | writable by `openhands` |
  | `export REQUESTS_CA_BUNDLE=...` in a shell | no | shell-local |

- `run_infer.py` already has an `env_setup_commands` hook, applied run-wide in
  `prepare_workspace` via `workspace.execute_command(cmd)`
  (`get_mirror_env_commands() + ["export PIP_CACHE_DIR=~/.cache/pip"]`).
- That setup shell is **not** the same shell the agent later uses. Filesystem
  and process side effects (`/etc/hosts` edit, appended `cacert.pem`, background
  gunicorn) persist regardless, but per-shell `export`s do **not** carry into the
  agent's separate terminal.

## Goals / Non-goals

**Goals**

- For `psf/requests` inference instances only, stand up a local httpbin server
  and make both the setup shell and the agent's later terminal shell trust it
  and resolve `httpbin.org` / `www.google.co.uk` locally.
- Keep one shared definition of the setup commands for maintainability.

**Non-goals**

- No change to grading/judge logic, patch application, or the test command.
- No change to any repo other than `psf/requests`.
- No Docker image rebuild (setup runs at container runtime).

## Design

### 1. Shared helper: `benchmarks/utils/httpbin_fix.py`

New module exposing `make_httpbin_setup_commands() -> list[str]`, adapted from
the hardened version in
`thirdparty/SWE-bench/swebench/harness/test_spec/python.py`. **Key difference
from the offline-harness version:** that version runs as root, so it needs no
`sudo`. The inference workspace runs as the non-root `openhands` user, so every
privileged step here is wrapped in `sudo` (passwordless sudo is available in the
agent-server image).

1. `sudo -E <testbed_python> -m pip install 'httpbin[mainapp]==0.10.2' 'pytest-httpbin==2.1.0'`
   — installed into the **testbed** conda env (root-owned), targeting its
   interpreter explicitly rather than relying on the active `python`.
2. Generate self-signed cert with `subjectAltName=DNS:httpbin.org,DNS:localhost,IP:127.0.0.1`
   under `/tmp/swebench_httpbin_certs` (no sudo — `/tmp` is writable by `openhands`).
3. `export REQUESTS_CA_BUNDLE=<cert>` and `export CURL_CA_BUNDLE=<cert>` (shell-local).
4. Append the cert to `requests.certs.where()` via `sudo` (root-owned file;
   covers `Session.send()` which bypasses the env var).
5. Launch detached gunicorn on `127.0.0.1:80` (HTTP) and `127.0.0.1:443`
   (HTTPS, `-k gevent`) via `sudo` (privileged ports). Because the server runs
   under sudo/root it must reference the testbed interpreter's gunicorn
   explicitly and preserve `REQUESTS_CA_BUNDLE`-relevant paths.
6. `sleep 2` to let servers bind.
7. `echo "127.0.0.1    httpbin.org www.google.co.uk" | sudo tee -a /etc/hosts`
   (root-owned).

**Inference-specific addition:** the helper also persists the env exports to
`/etc/profile.d/swebench_httpbin.sh` (written via `sudo tee`) so the agent's
separate terminal shell inherits `REQUESTS_CA_BUNDLE` / `CURL_CA_BUNDLE`:

```
printf '%s\n' \
  'export REQUESTS_CA_BUNDLE=/tmp/swebench_httpbin_certs/cert.pem' \
  'export CURL_CA_BUNDLE=/tmp/swebench_httpbin_certs/cert.pem' \
  | sudo tee /etc/profile.d/swebench_httpbin.sh
```

Re-implemented in-repo (rather than importing from the `swebench` package) so
`run_infer.py` takes no hard runtime dependency on `swebench` internals, and
because the inference version diverges (sudo wrapping, explicit testbed
interpreter, profile.d persistence) from the root-context offline version. The
module docstring points at `thirdparty/SWE-bench` as the origin of the approach.

### 2. Injection point: `prepare_workspace` in `run_infer.py`

After the workspace is created and after the existing `env_setup_commands` loop,
add gated injection:

```python
if instance.data.get("repo") == "psf/requests":
    for cmd in make_httpbin_setup_commands():
        res = workspace.execute_command(cmd)
        if res.exit_code != 0:
            logger.warning(
                "httpbin setup command failed (continuing): %r: %s",
                cmd, res.stderr,
            )
```

- **Gated** on `repo == "psf/requests"`; a mixed-repo run leaves all other
  instances untouched.
- Runs **before** `evaluate_instance` (before the conversation starts), so the
  local server, hosts redirect, and certs are in place when the agent first runs
  tests.
- **Non-fatal**: unlike the mirror-env loop (which `raise`s), a failed httpbin
  setup command logs a warning and continues. Rationale: httpbin setup is a
  reliability aid; if it partially fails the agent can still work (falling back
  to external httpbin) rather than losing the whole instance.
- Because exports are persisted to `/etc/profile.d`, the agent's later terminal
  shell (which runs `conda activate testbed`, a login-ish shell that sources
  `/etc/profile.d/*.sh`) inherits the CA-bundle vars. The direct
  `execute_command` exports additionally cover the immediate setup context.

### 3. Import

`from benchmarks.utils.httpbin_fix import make_httpbin_setup_commands` at the top
of `run_infer.py`, alongside the existing
`from benchmarks.utils.mirror_config import get_mirror_env_commands`.

## Verification

1. `python -m py_compile benchmarks/utils/httpbin_fix.py benchmarks/swebench/run_infer.py`.
2. Manual container check on the **agent-server** image for a `psf/requests`
   instance (run as the `openhands` user, not root): run the generated commands,
   confirm the `sudo`-wrapped steps succeed, `gunicorn` binds 80/443, `/etc/hosts`
   is updated, and a fresh `bash -lc 'echo $REQUESTS_CA_BUNDLE'` shows the cert
   path via `/etc/profile.d`. (Confirmed during design: `openhands` uid 10001 has
   passwordless sudo; `/etc/hosts`, `/etc/profile.d`, and the root-owned testbed
   env are all reachable via `sudo`.)
3. Unit test `benchmarks/utils/test_httpbin_fix.py` asserting the command list
   contains the critical lines (SAN cert with `DNS:httpbin.org`, both
   `REQUESTS_CA_BUNDLE` and the `requests.certs.where()` append, the `/etc/hosts`
   redirect including `www.google.co.uk`, the `/etc/profile.d` persistence, and
   that every privileged step is `sudo`-wrapped).

## Files touched

| File | Change |
|------|--------|
| `benchmarks/utils/httpbin_fix.py` | New — shared `make_httpbin_setup_commands()` + `/etc/profile.d` persistence |
| `benchmarks/swebench/run_infer.py` | Import helper; gated per-instance injection in `prepare_workspace` |
| `benchmarks/utils/test_httpbin_fix.py` | New — unit test for the command list |

## Out of scope / follow-ups

- Mirroring to other benchmarks (swerebenchv2, swesmith, scaleswe) — those tasks
  have their own `run_infer.py`; if they evaluate `psf/requests` they can reuse
  `benchmarks/utils/httpbin_fix.py` the same way. Not done here.
