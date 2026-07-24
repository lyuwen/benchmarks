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

Relevant facts confirmed:

- The container runs as **root** (`uid=0`); `/etc/hosts` is writable and
  `openssl` is present in the testbed conda env — the fix's requirements hold.
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

New module exposing `make_httpbin_setup_commands() -> list[str]`, ported
verbatim (hardened version) from
`thirdparty/SWE-bench/swebench/harness/test_spec/python.py`:

1. `pip install 'httpbin[mainapp]==0.10.2' 'pytest-httpbin==2.1.0'`
2. Generate self-signed cert with `subjectAltName=DNS:httpbin.org,DNS:localhost,IP:127.0.0.1`
   under `/tmp/swebench_httpbin_certs`.
3. `export REQUESTS_CA_BUNDLE=<cert>` and `export CURL_CA_BUNDLE=<cert>`.
4. Append the cert to `requests.certs.where()` (covers `Session.send()` which
   bypasses the env var).
5. Launch detached gunicorn on `127.0.0.1:80` (HTTP) and `127.0.0.1:443`
   (HTTPS, `-k gevent`).
6. `sleep 2` to let servers bind.
7. `echo "127.0.0.1    httpbin.org www.google.co.uk" >> /etc/hosts`.

**Inference-specific addition:** the helper (or the caller) also persists the
env exports to `/etc/profile.d/swebench_httpbin.sh` so the agent's separate
terminal shell inherits `REQUESTS_CA_BUNDLE` / `CURL_CA_BUNDLE`. Concretely, the
command list also writes:

```
cat > /etc/profile.d/swebench_httpbin.sh <<'EOF'
export REQUESTS_CA_BUNDLE=/tmp/swebench_httpbin_certs/cert.pem
export CURL_CA_BUNDLE=/tmp/swebench_httpbin_certs/cert.pem
EOF
```

Re-implemented in-repo (rather than importing from the `swebench` package) so
`run_infer.py` takes no hard runtime dependency on `swebench` internals. The two
definitions are intentionally kept in sync; the module docstring points at
`thirdparty/SWE-bench` as the source of truth.

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
2. Manual container check on a `psf/requests` image (e.g.
   `sweb.eval.x86_64.psf__requests-2317`): run the ported commands, confirm
   gunicorn binds 80/443, `/etc/hosts` is updated, and a fresh `bash -lc 'echo
   $REQUESTS_CA_BUNDLE'` shows the cert path via `/etc/profile.d`.
3. Unit test `benchmarks/utils/test_httpbin_fix.py` asserting the command list
   contains the critical lines (SAN cert with `DNS:httpbin.org`, both
   `REQUESTS_CA_BUNDLE` and the `requests.certs.where()` append, the `/etc/hosts`
   redirect including `www.google.co.uk`, and the `/etc/profile.d` persistence).

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
