#!/usr/bin/env python3
"""Split histories + dataset into one work item per instance for re-scoring.

Run from the repository root (with it on PYTHONPATH), e.g.:

    PYTHONPATH=$PWD python scripts/swerebenchv2_rescore/prepare_rescore.py \
        --histories /path/to/histories.jsonl \
        --dataset /path/to/SWE-rebench-V2-go.jsonl \
        --workdir /path/to/rescore-workdir
"""

import argparse
import logging
import sys

from benchmarks.swerebenchv2.rescore import prepare_workdir


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare per-instance work items for SWE-rebench-V2 re-scoring."
    )
    parser.add_argument("--histories", required=True, help="Path to histories.jsonl")
    parser.add_argument(
        "--dataset", required=True, help="Path to the SWE-rebench-V2 task JSONL"
    )
    parser.add_argument(
        "--workdir", required=True, help="Directory to write work items into"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    count = prepare_workdir(args.histories, args.dataset, args.workdir)
    print(f"prepared {count} work items in {args.workdir}")
    print(
        "submit the array with:\n"
        f"  sbatch --array=0-{count - 1}%64 "
        "--export=ALL,WORKDIR=...,REPO_DIR=...,DOCKER_IMAGE_PREFIX=... "
        "scripts/swerebenchv2_rescore/rescore.sbatch"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
