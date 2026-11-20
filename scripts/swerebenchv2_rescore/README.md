# SWE-rebench-V2 re-scoring (SLURM array)

Re-score an existing `histories.jsonl` with the current execution judge (binary
artifacts stripped, subset resolution, no `--network host`), one instance per
SLURM array task, and write a new histories file with updated `evaluation`
values. No inference/agent run is repeated — only the judge (Docker test run).

## Quick start (cluster defaults)

```bash
cd /mnt/users/lfu/llm/projects/openhands/flex-benchmarks-dev-v2

scripts/swerebenchv2_rescore/submit_rescore.sh
```

This does three steps:

1. `prepare_rescore.py` — one pass over `histories.jsonl` + the task JSONL,
   writing `<WORKDIR>/instances/<i>.json`, `<WORKDIR>/tasks/<i>.json`,
   `manifest.jsonl` and `count`. Index = line number in `histories.jsonl`.
2. `rescore.sbatch` — `sbatch --array=0-$((N-1))%32` (one task per history
   index). Each task runs the judge for that instance and writes
   `<WORKDIR>/results/<i>.json` with `evaluation: true|false|null`.
3. `merge_rescore.sbatch` — submitted with `--dependency=afterany:<array>`;
   streams `histories.jsonl`, replaces only the `evaluation` field, and writes
   `<WORKDIR>/histories.rescored.jsonl` plus `<WORKDIR>/report.json`
   (recovered / regressed / unchanged / missing / error counts + index lists).

`afterany` (not `afterok`) is deliberate: a few array tasks may run out of
retries, and the merge still runs and lists the indices to re-submit.

## Overrides

```bash
WORKDIR=/path/to/workdir \
HISTORIES=/path/to/histories.jsonl \
DATASET=/path/to/SWE-rebench-V2-go.jsonl \
DOCKER_IMAGE_PREFIX=cr-ee.../swerebenchv2 \
CONCURRENCY=32 JUDGE_TIMEOUT=1800 JUDGE_RM_IMAGE=1 ATTEMPTS=2 \
SLURM_PARTITION=<partition> SLURM_ACCOUNT=<account> \
  scripts/swerebenchv2_rescore/submit_rescore.sh
```

| Variable | Default | Meaning |
|---|---|---|
| `WORKDIR` | `<repo>/logs/rescore-swerebenchv2-go` | scratch dir for work items/results |
| `HISTORIES` | cluster `.../openswe-solvable-full/histories.jsonl` | input histories |
| `DATASET` | cluster `.../SWE-rebench-V2-go.jsonl` | task rows (for test_cmd, F2P/P2P, image) |
| `DOCKER_IMAGE_PREFIX` | cluster `.../sandbox/swerebenchv2` | same prefix as the inference run |
| `CONCURRENCY` | `32` | SLURM array `%N` cap |
| `JUDGE_TIMEOUT` | `1800` | per-instance judge timeout (s) |
| `JUDGE_RM_IMAGE` | `1` | remove each instance image after judging |
| `ATTEMPTS` | `2` | retries per instance inside the array task |
| `OUTPUT_HISTORIES` | `<WORKDIR>/histories.rescored.jsonl` | merge output (never in place) |
| `REPORT_JSON` | `<WORKDIR>/report.json` | merge report |

## Retrying index(es) without a verdict

The merge report lists `missing_indices` (never ran) and `error_indices`
(judge returned no verdict). Re-submit just those, then re-run the merge:

```bash
WORKDIR=... REPO_DIR=... DOCKER_IMAGE_PREFIX=... \
  sbatch --array=12,345,678 --export=ALL \
  scripts/swerebenchv2_rescore/rescore.sbatch

WORKDIR=... HISTORIES=... REPO_DIR=... \
  sbatch --export=ALL scripts/swerebenchv2_rescore/merge_rescore.sbatch
```

## Manual steps (equivalent)

```bash
export PYTHONPATH=$PWD
python scripts/swerebenchv2_rescore/prepare_rescore.py \
  --histories "$HISTORIES" --dataset "$DATASET" --workdir "$WORKDIR"

N=$(cat "$WORKDIR/count")
sbatch --array="0-$((N-1))%32" \
  --export=ALL,WORKDIR="$WORKDIR",REPO_DIR="$PWD",DOCKER_IMAGE_PREFIX="$DOCKER_IMAGE_PREFIX" \
  scripts/swerebenchv2_rescore/rescore.sbatch

python scripts/swerebenchv2_rescore/merge_rescore.py \
  --workdir "$WORKDIR" --histories "$HISTORIES" \
  --output "$WORKDIR/histories.rescored.jsonl"
```

## Notes

- Requires Docker on the compute nodes (the judge runs the instance image) and
  the repo checkout on the shared filesystem (including `vendor/`).
- `instances/<i>.json` stores the recorded `git_patch` verbatim; the judge
  strips binary diffs and ground-truth-test-file overlap itself.
- `merge_rescore.py` always writes a **new** file; the original
  `histories.jsonl` is never modified.
- The judge logs per instance (`SWERebenchV2Judge … resolved=…, F2P passed …`)
  into the array task's `.out` file.
