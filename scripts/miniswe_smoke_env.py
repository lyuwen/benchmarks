#!/usr/bin/env python3
"""Live smoke test for the mini-swe-agent workspace adapter.

Exercises ``benchmarks.utils.miniswe.WorkspaceEnvironment`` against a real
agent-server container: cwd handling, shell/conda activation, multiline quoting,
stderr/exit-code mapping and submission-marker detection. No model endpoint is
needed, so this validates the backend half of the port independently of the LLM.

Usage:
    uv run python scripts/miniswe_smoke_env.py \\
        --image <agent-server-image> \\
        --cwd /testbed

The image is always supplied by the caller (no registry default is baked in).
For SWE-bench, any pre-built ``<agent-server>:<sha>-<instance>-source-minimal``
image works; the cwd defaults to ``/testbed`` and ``BASH_ENV=/root/.bashrc``
activates the testbed conda environment.
"""

import argparse
import json
import sys

from benchmarks.utils.miniswe import (
    SUBMISSION_MARKER,
    WorkspaceEnvironment,
    require_miniswe,
)
from openhands.sdk import get_logger
from openhands.workspace import DockerWorkspace


logger = get_logger(__name__)

SAMPLE_PATCH = "diff --git a/x b/x\n--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a\n+b\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="agent-server image to run")
    parser.add_argument("--cwd", default="/testbed", help="task working directory")
    parser.add_argument(
        "--bash-env-file",
        default="none",
        help=(
            "file exported as BASH_ENV; off by default because agent-server "
            "images run as a non-root user and already export the task PATH"
        ),
    )
    args = parser.parse_args()
    require_miniswe()
    from minisweagent.exceptions import Submitted

    bash_env_file = None if args.bash_env_file.lower() == "none" else args.bash_env_file
    workspace = DockerWorkspace(
        server_image=args.image, working_dir="/workspace", forward_env=[]
    )
    with workspace:
        env = WorkspaceEnvironment(
            workspace,
            cwd=args.cwd,
            timeout=120,
            bash_env_file=bash_env_file,
        )

        output = env.execute({"command": "pwd && python -V"})
        logger.info("cwd/python: %s", json.dumps(output))
        assert output["returncode"] == 0, output
        assert args.cwd in output["output"], output

        output = env.execute(
            {
                "command": (
                    "cat > /tmp/miniswe_smoke.py <<'PYEOF'\n"
                    'print("it\'s fine")\n'
                    "PYEOF\n"
                    "python /tmp/miniswe_smoke.py"
                )
            }
        )
        logger.info("multiline: %s", json.dumps(output))
        assert output["returncode"] == 0, output
        assert "it's fine" in output["output"], output

        output = env.execute({"command": "echo smoke-stderr >&2 ; exit 3"})
        logger.info("nonzero: %s", json.dumps(output))
        assert output["returncode"] == 3, output
        assert "smoke-stderr" in output["output"], output

        try:
            env.execute(
                {
                    "command": (
                        "printf 'diff --git a/x b/x\\n--- a/x\\n+++ b/x\\n"
                        "@@ -1 +1 @@\\n-a\\n+b\\n' > /tmp/miniswe_smoke.patch && "
                        f"echo {SUBMISSION_MARKER} && cat /tmp/miniswe_smoke.patch"
                    )
                }
            )
        except Submitted as exc:
            submission = exc.messages[0]["extra"]["submission"]
            logger.info("submission: %r", submission)
            assert submission == SAMPLE_PATCH, (submission, SAMPLE_PATCH)
        else:
            raise AssertionError(
                "submission marker was not detected over the transport"
            )

    print("live workspace-environment smoke PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
