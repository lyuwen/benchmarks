#!/usr/bin/env bash
# Launch harness-streamline-histories as a SLURM job array.
#
# Each array task streamlines one run directory {i}: it reads the final
# <id>.history.json (per-attempt files are ignored), merges the fragment
# trajectories split by Claude Code's replayed-tool normalization (leading
# "cd <cwd> &&" and Edit defaults), restores the tool arguments the model
# originally generated, keeps auxiliary/compaction trajectories unchanged and
# writes <id>.history.streamlined.json next to the input.
#
# The job is expected to run with the run root (the directory containing the
# {i} job directories) as its working directory; SLURM_ARRAY_TASK_ID selects
# the job directory. Zero-padded names such as 00000 are handled, and if no
# direct match exists the i-th sorted sub-directory is used.
#
#   cd logs/dsv41f1017-high-synth-cc-swerebenchv2-osr-all-run2
#   sbatch --array=0-$(($(find . -maxdepth 1 -mindepth 1 -type d | wc -l) - 1)) \
#       /path/to/repo/scripts/streamline_histories_slurm.sh
#
# Extra arguments are forwarded to the python CLI, e.g. --overwrite/--dry-run.
#
# Alternative modes:
#   # root given explicitly (first argument) instead of the working directory
#   /path/to/repo/scripts/streamline_histories_slurm.sh <run-root> --overwrite
#   # precomputed list of claude-code directories (absolute paths)
#   uv run harness-streamline-histories --root <run-root> --list > dirs.txt
#   sbatch --array=0-$(($(wc -l < dirs.txt) - 1)) \
#       /path/to/repo/scripts/streamline_histories_slurm.sh dirs.txt
#
# Local test without SLURM:
#   SLURM_ARRAY_TASK_ID=0 ./scripts/streamline_histories_slurm.sh --dry-run
set -euo pipefail

INVOKE_DIR="$(pwd)"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
ARGS=("$@")
EXTRA=()
DIRECTORY=""
RUN_ROOT=""

if [[ ${#ARGS[@]} -gt 0 && -f "${ARGS[0]}" ]]; then
    # Mode: first argument is a file with one directory per line.
    LIST="${ARGS[0]}"
    [[ "${LIST}" = /* ]] || LIST="${INVOKE_DIR}/${LIST}"
    DIRECTORY="$(sed -n "$((TASK_ID + 1))p" "${LIST}")"
    EXTRA=("${ARGS[@]:1}")
    if [[ -z "${DIRECTORY}" ]]; then
        echo "No directory for array task ${TASK_ID} in ${LIST}" >&2
        exit 1
    fi
elif [[ -n "${STREAMLINE_ROOT:-}" ]]; then
    RUN_ROOT="${STREAMLINE_ROOT}"
    EXTRA=("$@")
elif [[ ${#ARGS[@]} -gt 0 && -d "${INVOKE_DIR}/${ARGS[0]}" ]]; then
    RUN_ROOT="${INVOKE_DIR}/${ARGS[0]}"
    EXTRA=("${ARGS[@]:1}")
else
    # Default: the working directory is the run root; all arguments are flags.
    RUN_ROOT="${INVOKE_DIR}"
    EXTRA=("$@")
fi

if [[ -z "${DIRECTORY}" ]]; then
    # Map the array task id to a job directory: exact name, zero-padded name
    # (e.g. 00000), then the i-th sorted sub-directory.
    if [[ -d "${RUN_ROOT}/${TASK_ID}" ]]; then
        DIRECTORY="${RUN_ROOT}/${TASK_ID}"
    elif [[ -d "${RUN_ROOT}/$(printf '%05d' "${TASK_ID}")" ]]; then
        DIRECTORY="${RUN_ROOT}/$(printf '%05d' "${TASK_ID}")"
    else
        NAME="$(
            find "${RUN_ROOT}" -maxdepth 1 -mindepth 1 -type d -printf '%f\n' \
                | LC_ALL=C sort \
                | sed -n "$((TASK_ID + 1))p"
        )"
        if [[ -z "${NAME}" ]]; then
            echo "No job directory for array task ${TASK_ID} under ${RUN_ROOT}" >&2
            exit 1
        fi
        DIRECTORY="${RUN_ROOT}/${NAME}"
    fi
fi

[[ "${DIRECTORY}" = /* ]] || DIRECTORY="${INVOKE_DIR}/${DIRECTORY}"
echo "[slurm task ${TASK_ID}] streamlining ${DIRECTORY}" >&2
cd "${REPO_ROOT}"
# Run the module (works even before the console-script entry point is installed).
MODULE="benchmarks.utils.harnesses.streamline"
if command -v uv >/dev/null 2>&1; then
    exec uv run python -m "${MODULE}" --dir "${DIRECTORY}" "${EXTRA[@]}"
fi
# Cluster environments usually provide uv; fall back to the active interpreter
# (with the project's dependencies installed and PYTHONPATH set to the repo root).
cd "${REPO_ROOT}"
exec python -m "${MODULE}" --dir "${DIRECTORY}" "${EXTRA[@]}"
