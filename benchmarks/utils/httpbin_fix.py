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
        # 3. Belt-and-suspenders shell-local exports. Each command runs in its
        #    own fresh shell, so these have no consumer within this setup
        #    sequence; persistence to the agent's shell is carried by the
        #    /etc/profile.d write (step 8) and the cacert append (step 4).
        f"export REQUESTS_CA_BUNDLE={CERT}",
        f"export CURL_CA_BUNDLE={CERT}",
        # 4. Trust it for Session.send() which bypasses the env var: append to
        #    the root-owned bundle that requests.certs.where() resolves to.
        f"sudo bash -c 'DST=$({TESTBED_PY} -c \"import requests; print(requests.certs.where())\"); "
        f'cat {CERT} >> "$DST"\'',
        # 5. Launch detached gunicorn: HTTP on :80, HTTPS on :443 (privileged).
        f"sudo bash -c '(nohup {TESTBED_PY} -m gunicorn -b 127.0.0.1:80 "
        "-k gevent httpbin:app > /dev/null 2>&1 &)'",
        f"sudo bash -c '(nohup {TESTBED_PY} -m gunicorn -b 127.0.0.1:443 "
        f"--certfile={CERT} --keyfile={KEY} -k gevent httpbin:app "
        "> /dev/null 2>&1 &)'",
        # 6. Poll the local server, and ONLY write the /etc/hosts redirect once
        #    it responds. The gunicorn launches above are backgrounded and
        #    always exit 0, so a fixed sleep + unconditional redirect would
        #    point httpbin.org at a dead local port (connection-refused) if the
        #    server never bound -- worse than the flaky-external baseline. On
        #    timeout this returns non-zero (call site logs a warning) and leaves
        #    httpbin.org resolving externally.
        "sudo bash -c 'URL=http://127.0.0.1:80/get; "
        "for i in $(seq 1 15); do "
        f'{TESTBED_PY} -c "import sys,urllib.request; urllib.request.urlopen(sys.argv[1], timeout=2)" "$URL" >/dev/null 2>&1 '
        '&& { echo "127.0.0.1    httpbin.org www.google.co.uk" >> /etc/hosts; exit 0; }; '
        "sleep 1; done; "
        "echo httpbin-local-server-did-not-start >&2; exit 1'",
        # 8. Persist exports so the agent's SEPARATE login shell inherits them.
        "printf '%s\\n' "
        f"'export REQUESTS_CA_BUNDLE={CERT}' "
        f"'export CURL_CA_BUNDLE={CERT}' "
        f"| sudo tee {PROFILE_D}",
    ]
