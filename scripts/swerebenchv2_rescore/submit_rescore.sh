#!/bin/bash
#
# One-command re-scoring pipeline for a recorded SWE-rebench-V2 run:
#
#   1. split histories.jsonl + task JSONL into one work item per instance
#   2. submit a SLURM array (one task per instance, capped concurrency)
#   3. submit a merge job that writes histories.rescored.jsonl once the array
#      finishes (dependency afterany, so it also runs if some tasks failed)
#
# Usage:
#   scripts/swerebenchv2_rescore/submit_rescore.sh
#
# Everything is overridable through environment variables, e.g.:
#   WORKDIR=... HISTORIES=... DATASET=... DOCKER_IMAGE_PREFIX=... CONCURRENCY=32 \
#     scripts/swerebenchv2_rescore/submit_rescore.sh
#
# Defaults match the cluster layout used by the SWE-rebench-V2 go runs.

set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=${REPO_DIR:-$(cd "$HERE/../.." && pwd)}

# --- configurable paths ------------------------------------------------------
WORKDIR=${WORKDIR:-$REPO_DIR/logs/rescore-swerebenchv2-go}
HISTORIES=${HISTORIES:-/mnt/users/lfu/llm/datasets/SWE-rebench-V2/openswe-solvable-full/histories.jsonl}
DATASET=${DATASET:-/mnt/users/lfu/llm/datasets/SWE-rebench-V2/openswe-solvable-full/SWE-rebench-V2-go.jsonl}
DOCKER_IMAGE_PREFIX=${DOCKER_IMAGE_PREFIX:-cr-ee.registry.cn-jswx-xuelang-d01.res.cloud.wuxi-yqgcy.cn/sandbox/swerebenchv2}

# --- configurable runtime ----------------------------------------------------
PYTHON=${PYTHON:-python}
CONCURRENCY=${CONCURRENCY:-32}
JUDGE_TIMEOUT=${JUDGE_TIMEOUT:-1800}
JUDGE_RM_IMAGE=${JUDGE_RM_IMAGE:-1}
ATTEMPTS=${ATTEMPTS:-2}
OUTPUT_HISTORIES=${OUTPUT_HISTORIES:-$WORKDIR/histories.rescored.jsonl}
REPORT_JSON=${REPORT_JSON:-$WORKDIR/report.json}

# Optional SLURM settings (passed through when set)
SLURM_ARGS=()
if [[ -n "${SLURM_PARTITION:-}" ]]; then
  SLURM_ARGS+=(--partition="$SLURM_PARTITION")
fi
if [[ -n "${SLURM_ACCOUNT:-}" ]]; then
  SLURM_ARGS+=(--account="$SLURM_ACCOUNT")
fi

export REPO_DIR WORKDIR HISTORIES DATASET DOCKER_IMAGE_PREFIX PYTHON
export JUDGE_TIMEOUT JUDGE_RM_IMAGE ATTEMPTS OUTPUT_HISTORIES REPORT_JSON

mkdir -p "$WORKDIR/logs" "$REPO_DIR/logs"

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
