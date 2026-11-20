#!/bin/bash
#
# One-command re-scoring pipeline for a recorded SWE-rebench-V2 run:
#
#   1. split histories.jsonl + task JSONL into one work item per instance
#   2. submit a SLURM array (one task per instance, capped concurrency)
#   3. submit a merge job that writes histories.rescored.jsonl once the array
#      finishes (dependency afterany, so it also runs if some tasks failed)
#
# Run this from the directory that holds histories.jsonl; that directory is
# used as WORKDIR by default and receives instances/, tasks/, results/, logs/
# and the re-scored histories, so no `cd`/workdir juggling is needed:
#
#   cd .../logs/<run-dir>            # contains histories.jsonl
#   DATASET=/path/to/SWE-rebench-V2-go.jsonl \
#     /path/to/repo/scripts/swerebenchv2_rescore/submit_rescore.sh
#
# Overridable environment variables:
#   DATASET (required)   SWE-rebench-V2 task JSONL (no default)
#   WORKDIR              scratch/output dir (default: current directory)
#   HISTORIES            input histories (default: $WORKDIR/histories.jsonl)
#   REPO_DIR             repo checkout (default: two levels above this script)
#   DOCKER_IMAGE_PREFIX  instance image registry prefix (same as inference run)
#   CONCURRENCY          SLURM array %N cap (default: 32)
#   JUDGE_TIMEOUT        per-instance judge timeout seconds (default: 1800)
#   JUDGE_RM_IMAGE       1 = remove the instance image after judging (default)
#   ATTEMPTS             attempts per instance in the array task (default: 2)
#   OUTPUT_HISTORIES     default: $WORKDIR/histories.rescored.jsonl
#   REPORT_JSON          default: $WORKDIR/report.json
#   DRY_RUN              1 = prepare + print the sbatch commands, do not submit
#   SLURM_PARTITION / SLURM_ACCOUNT  optional sbatch arguments

set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=${REPO_DIR:-$(cd "$HERE/../.." && pwd)}

# --- configurable paths ------------------------------------------------------
# Default to the current directory so the script can be launched from the run
# output directory without changing any workdir.
WORKDIR=${WORKDIR:-$PWD}
HISTORIES=${HISTORIES:-$WORKDIR/histories.jsonl}
DOCKER_IMAGE_PREFIX=${DOCKER_IMAGE_PREFIX:-cr-ee.registry.cn-jswx-xuelang-d01.res.cloud.wuxi-yqgcy.cn/sandbox/swerebenchv2}
: "${DATASET:?set DATASET to the SWE-rebench-V2 task JSONL, e.g. /mnt/users/lfu/llm/datasets/SWE-rebench-V2/openswe-solvable-full/SWE-rebench-V2-go.jsonl}"

# --- configurable runtime ----------------------------------------------------
PYTHON=${PYTHON:-python}
CONCURRENCY=${CONCURRENCY:-32}
JUDGE_TIMEOUT=${JUDGE_TIMEOUT:-1800}
JUDGE_RM_IMAGE=${JUDGE_RM_IMAGE:-1}
ATTEMPTS=${ATTEMPTS:-2}
OUTPUT_HISTORIES=${OUTPUT_HISTORIES:-$WORKDIR/histories.rescored.jsonl}
REPORT_JSON=${REPORT_JSON:-$WORKDIR/report.json}
DRY_RUN=${DRY_RUN:-0}

# Resolve paths before any `cd`: relative paths are relative to the caller's cwd.
abspath() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s\n' "$PWD/$1" ;;
  esac
}
REPO_DIR=$(abspath "$REPO_DIR")
WORKDIR=$(abspath "$WORKDIR")
HISTORIES=$(abspath "$HISTORIES")
DATASET=$(abspath "$DATASET")
OUTPUT_HISTORIES=$(abspath "$OUTPUT_HISTORIES")
REPORT_JSON=$(abspath "$REPORT_JSON")

# Optional SLURM settings (passed through when set)
SLURM_ARGS=()
if [[ -n "${SLURM_PARTITION:-}" ]]; then
  SLURM_ARGS+=(--partition="$SLURM_PARTITION")
fi
if [[ -n "${SLURM_ACCOUNT:-}" ]]; then
  SLURM_ARGS+=(--account="$SLURM_ACCOUNT")
fi

echo "== config =="
printf '  REPO_DIR           = %s\n' "$REPO_DIR"
printf '  WORKDIR            = %s\n' "$WORKDIR"
printf '  HISTORIES          = %s\n' "$HISTORIES"
printf '  DATASET            = %s\n' "$DATASET"
printf '  OUTPUT_HISTORIES   = %s\n' "$OUTPUT_HISTORIES"
printf '  REPORT_JSON        = %s\n' "$REPORT_JSON"

if [[ ! -f "$HISTORIES" ]]; then
  echo "error: histories file not found: $HISTORIES" >&2
  echo "       run from the directory containing histories.jsonl or set HISTORIES=..." >&2
  exit 1
fi
if [[ ! -f "$DATASET" ]]; then
  echo "error: dataset file not found: $DATASET" >&2
  echo "       set DATASET=/path/to/SWE-rebench-V2-go.jsonl" >&2
  exit 1
fi

export REPO_DIR WORKDIR HISTORIES DATASET DOCKER_IMAGE_PREFIX PYTHON
export JUDGE_TIMEOUT JUDGE_RM_IMAGE ATTEMPTS OUTPUT_HISTORIES REPORT_JSON

mkdir -p "$WORKDIR/logs"

echo "== [1/3] preparing work items =="
(
  cd "$REPO_DIR"
  PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}" \
    "$PYTHON" scripts/swerebenchv2_rescore/prepare_rescore.py \
      --histories "$HISTORIES" \
      --dataset "$DATASET" \
      --workdir "$WORKDIR"
)
count=$(cat "$WORKDIR/count")
last=$((count - 1))

if [[ "$DRY_RUN" == "1" ]]; then
  echo "== DRY_RUN=1: not submitting =="
  echo "  sbatch --array=0-${last}%${CONCURRENCY} --export=ALL \\"
  echo "    --output=$WORKDIR/logs/rescore-%A_%a.out $HERE/rescore.sbatch"
  echo "  sbatch --dependency=afterany:<array-job-id> --export=ALL \\"
  echo "    --output=$WORKDIR/logs/merge-%j.out $HERE/merge_rescore.sbatch"
  exit 0
fi

echo "== [2/3] submitting array 0-${last}%${CONCURRENCY} =="
array_job=$(sbatch --parsable ${SLURM_ARGS[@]+"${SLURM_ARGS[@]}"} --export=ALL \
  --array="0-${last}%${CONCURRENCY}" \
  --output="$WORKDIR/logs/rescore-%A_%a.out" \
  --error="$WORKDIR/logs/rescore-%A_%a.err" \
  "$HERE/rescore.sbatch")
echo "array job id: $array_job"

echo "== [3/3] submitting merge (dependency: afterany:${array_job}) =="
merge_job=$(sbatch --parsable ${SLURM_ARGS[@]+"${SLURM_ARGS[@]}"} --export=ALL \
  --dependency="afterany:${array_job}" \
  --output="$WORKDIR/logs/merge-%j.out" \
  --error="$WORKDIR/logs/merge-%j.err" \
  "$HERE/merge_rescore.sbatch")
echo "merge job id: $merge_job"

echo
echo "outputs once the merge job completes:"
echo "  re-scored histories: $OUTPUT_HISTORIES"
echo "  transition report:   $REPORT_JSON"
echo
echo "If the report lists index(es) without a verdict, re-submit just those with:"
echo "  sbatch --array=<comma-separated-indices>%${CONCURRENCY} \\"
echo "    --export=ALL,WORKDIR=$WORKDIR,REPO_DIR=$REPO_DIR,DOCKER_IMAGE_PREFIX=$DOCKER_IMAGE_PREFIX \\"
echo "    $HERE/rescore.sbatch"
echo "then re-run the merge job."
