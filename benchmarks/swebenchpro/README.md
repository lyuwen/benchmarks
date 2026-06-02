# SWE-Bench Pro

OpenHands benchmark integration for `ScaleAI/SWE-bench_Pro`.

## Dataset
- Dataset: `ScaleAI/SWE-bench_Pro`
- Split: `test`
- Official harness assets: `benchmarks/swebenchpro/SWE-bench_Pro-os/`
- Official images: `docker.io/jefzda/sweap-images:<dockerhub_tag>`

## Initialize the harness submodule

```bash
git submodule update --init benchmarks/swebenchpro/SWE-bench_Pro-os
```

## Build agent-server images

```bash
uv run benchmarks/swebenchpro/build_images.py \
  --dataset ScaleAI/SWE-bench_Pro \
  --split test \
  --image ghcr.io/openhands/eval-agent-server \
  --target source-minimal
```

## Run inference

```bash
uv run benchmarks/swebenchpro/run_infer.py path/to/llm_config.json \
  --dataset ScaleAI/SWE-bench_Pro \
  --split test \
  --workspace flex
```

### Enable on-the-fly judging

```bash
uv run benchmarks/swebenchpro/run_infer.py path/to/llm_config.json \
  --dataset ScaleAI/SWE-bench_Pro \
  --split test \
  --workspace flex \
  --judge
```

The `--judge` flag enables the `swebenchpro` execution-based judge by default and stores the judge result under `evaluation` in each `*.history.json` file.

## Run offline evaluation

Point `--predictions` at the `output.jsonl` produced inside the structured evaluation output directory created by `run_infer.py`.

```bash
uv run benchmarks/swebenchpro/eval_infer.py \
  --predictions path/to/output.jsonl \
  --dataset ScaleAI/SWE-bench_Pro \
  --split test \
  --report-json eval_report.json
```

## Smoke tests

1. `uv run benchmarks/swebenchpro/build_images.py --dataset ScaleAI/SWE-bench_Pro --split test --n-limit 1 --image ghcr.io/openhands/eval-agent-server --target source-minimal`
2. `uv run benchmarks/swebenchpro/run_infer.py path/to/llm_config.json --dataset ScaleAI/SWE-bench_Pro --split test --workspace docker --n-limit 1 --max-iterations 5`
3. Repeat #2 with `--judge` and check that the per-instance `*.history.json` contains `evaluation`.
4. `uv run benchmarks/swebenchpro/eval_infer.py --predictions path/to/output.jsonl --dataset ScaleAI/SWE-bench_Pro --split test --max-workers 1 --report-json eval_report.json`
