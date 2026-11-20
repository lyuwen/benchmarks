#!/usr/bin/env python3
"""Merge re-scored verdicts into a new histories JSONL file (never in place).

Run from the repository root (with it on PYTHONPATH), e.g.:

    PYTHONPATH=$PWD python scripts/swerebenchv2_rescore/merge_rescore.py \
        --workdir /path/to/rescore-workdir \
        --histories /path/to/histories.jsonl \
        --output /path/to/histories.rescored.jsonl \
        --report-json /path/to/rescore-report.json
"""

import argparse
import logging
import sys

from benchmarks.swerebenchv2.rescore import merge_histories


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Merge re-scored evaluations back into a histories JSONL file."
    )
    parser.add_argument("--workdir", required=True, help="Prepared work directory")
    parser.add_argument("--histories", required=True, help="Original histories.jsonl")
    parser.add_argument(
        "--output", required=True, help="New histories JSONL file to write"
    )
    parser.add_argument(
        "--report-json",
        default=None,
        help="Where to write the transition report (default: <workdir>/report.json)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and print the report without writing the output file",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    report_path = args.report_json or f"{args.workdir.rstrip('/')}/report.json"
    report = merge_histories(
        args.workdir,
        args.histories,
        args.output,
        report_path,
        dry_run=args.dry_run,
    )
    counts = report["counts"]
    print(
        "re-scored histories: "
        f"total={counts['total']} updated={counts['updated']} "
        f"recovered={counts['recovered']} regressed={counts['regressed']} "
        f"unchanged={counts['unchanged']} missing_result={counts['missing_result']} "
        f"errors={counts['errors']}"
    )
    if report["missing_indices"] or report["error_indices"]:
        rerun = sorted(report["missing_indices"] + report["error_indices"])
        print(
            f"{len(rerun)} index(es) still need a verdict, e.g. "
            f"{rerun[:20]}{' ...' if len(rerun) > 20 else ''}\n"
            "resubmit them with: sbatch --array="
            f"{','.join(str(i) for i in rerun[:100])}%32 ..."
        )
    if report["dry_run"]:
        print("dry run: no output file written")
    else:
        print(f"wrote {report['output']} (report: {report_path})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
