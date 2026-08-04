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
