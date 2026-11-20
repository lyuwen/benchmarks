#!/usr/bin/env python3
"""Run the updated judge for one prepared work item (one SLURM array task).

Exits 0 when a boolean verdict was produced, 1 when the judge could not produce
one (the sbatch wrapper retries those indices).
"""

import argparse
import json
import logging
import sys

from benchmarks.swerebenchv2.rescore import run_one


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Re-run the SWE-rebench-V2 judge for a single history index."
    )
    parser.add_argument("--workdir", required=True, help="Prepared work directory")
    parser.add_argument(
        "--index", required=True, type=int, help="SLURM_ARRAY_TASK_ID (history index)"
    )
    parser.add_argument(
        "--docker-image-prefix",
        default=None,
        help="Registry/namespace prefix for instance images (same as inference run)",
    )
    parser.add_argument(
        "--timeout", type=int, default=1800, help="Per-instance judge timeout (seconds)"
    )
    parser.add_argument(
        "--rm-image",
        action="store_true",
        help="Remove the instance image after judging (recommended for large arrays)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    result = run_one(
        args.workdir,
        args.index,
        docker_image_prefix=args.docker_image_prefix,
        timeout=args.timeout,
        rm_image=args.rm_image,
    )
    print(json.dumps(result), flush=True)
    return 0 if result["evaluation"] is not None else 1


if __name__ == "__main__":
    sys.exit(main())
