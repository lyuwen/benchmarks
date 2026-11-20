#!/bin/bash
#SBATCH --job-name=swerebenchv2-rescore-merge
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#
# Collects <WORKDIR>/results/*.json and writes a new histories JSONL file with
# the re-scored `evaluation` values. Chained after the array by
# submit_rescore.sh with `--dependency=afterany:<array-job-id>`, so it runs even
# when some array tasks failed (the report lists the indices to re-run).
#
# Required environment variables:
#   WORKDIR   prepared work directory
#   HISTORIES original histories JSONL
#   REPO_DIR  repository checkout root
# Optional:
#   PYTHON              python interpreter (default: python)
#   OUTPUT_HISTORIES    output path (default: <WORKDIR>/histories.rescored.jsonl)
#   REPORT_JSON         report path (default: <WORKDIR>/report.json)

set -euo pipefail

: "${WORKDIR:?set WORKDIR}"
: "${HISTORIES:?set HISTORIES}"
: "${REPO_DIR:?set REPO_DIR to the repository checkout}"
PYTHON="${PYTHON:-python}"
OUTPUT_HISTORIES="${OUTPUT_HISTORIES:-$WORKDIR/histories.rescored.jsonl}"
REPORT_JSON="${REPORT_JSON:-$WORKDIR/report.json}"

export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO_DIR"

exec "$PYTHON" scripts/swerebenchv2_rescore/merge_rescore.py \
  --workdir "$WORKDIR" \
  --histories "$HISTORIES" \
  --output "$OUTPUT_HISTORIES" \
  --report-json "$REPORT_JSON"
