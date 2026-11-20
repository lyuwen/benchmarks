#!/bin/bash
#SBATCH --job-name=swerebenchv2-rescore
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#
# One SLURM array task per history index: re-runs the updated judge for a
# single instance and writes <WORKDIR>/results/<index>.json.
#
# Submit it through scripts/swerebenchv2_rescore/submit_rescore.sh (which runs
# the prepare step, computes the array bounds and chains the merge job), or
# manually after prepare_rescore.py has written <WORKDIR>/count:
#
#   N=$(cat "$WORKDIR/count")
#   sbatch --array=0-$((N-1))%32 \
#     --export=ALL,WORKDIR=...,REPO_DIR=...,DOCKER_IMAGE_PREFIX=... \
#     scripts/swerebenchv2_rescore/rescore.sbatch
#
# Required environment variables:
#   WORKDIR              prepared work directory (instances/, tasks/, results/)
#   REPO_DIR             repository checkout root
#   DOCKER_IMAGE_PREFIX  instance image registry/namespace prefix (as in inference)
# Optional:
#   PYTHON               python interpreter (default: python)
#   JUDGE_TIMEOUT        per-instance judge timeout in seconds (default: 1800)
#   JUDGE_RM_IMAGE       1 to remove each instance image after judging (default: 1)
#   ATTEMPTS             attempts per instance before the array task fails (default: 2)

set -euo pipefail

: "${WORKDIR:?set WORKDIR (run prepare_rescore.py first)}"
: "${REPO_DIR:?set REPO_DIR to the repository checkout}"
: "${DOCKER_IMAGE_PREFIX:?set DOCKER_IMAGE_PREFIX to the instance image registry prefix}"
PYTHON="${PYTHON:-python}"
JUDGE_TIMEOUT="${JUDGE_TIMEOUT:-1800}"
JUDGE_RM_IMAGE="${JUDGE_RM_IMAGE:-1}"
ATTEMPTS="${ATTEMPTS:-2}"
INDEX="${SLURM_ARRAY_TASK_ID:?this script must run as a slurm array job}"

mkdir -p "$WORKDIR/results" "$WORKDIR/logs"
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO_DIR"

args=(
  --workdir "$WORKDIR"
  --index "$INDEX"
  --docker-image-prefix "$DOCKER_IMAGE_PREFIX"
  --timeout "$JUDGE_TIMEOUT"
)
if [[ "$JUDGE_RM_IMAGE" == "1" ]]; then
  args+=(--rm-image)
fi

rc=1
for attempt in $(seq 1 "$ATTEMPTS"); do
  echo "[$(date -Is)] index=$INDEX attempt=$attempt/$ATTEMPTS"
  if "$PYTHON" scripts/swerebenchv2_rescore/run_one.py "${args[@]}"; then
    rc=0
    break
  fi
  echo "[$(date -Is)] attempt $attempt failed for index=$INDEX" >&2
  sleep $((attempt * 20))
done

if [[ "$rc" -ne 0 ]]; then
  echo "[$(date -Is)] index=$INDEX: no verdict after $ATTEMPTS attempt(s)" >&2
fi
exit "$rc"
