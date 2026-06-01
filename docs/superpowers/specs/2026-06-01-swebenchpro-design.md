# SWE-bench Pro evaluation — design

**Branch:** `feat/swebenchpro`
**Status:** Approved design, ready for implementation plan
**Author:** Lyuwen Fu (with Claude Code)
**Date:** 2026-06-01

## 1. Goal

Add a SWE-bench Pro evaluation harness to `benchmarks/swebenchpro/` that aligns with the existing `benchmarks/scaleswe/` and `benchmarks/swerebenchv2/` style:

- Supports `flex`, `docker`, and `remote` workspaces (default: `flex`).
- Runs inference end-to-end with the OpenHands agent server.
- Provides an **on-the-fly execution-based judge** that scores each agent patch against the official SWE-bench Pro test harness immediately after the agent finishes.
- Provides a separate **offline batch evaluator** (`eval_infer.py`) for scoring previously generated patches.

The implementation must be a standalone benchmark module — it must not subclass `SWEBenchEvaluation` (that's the pattern the upstream `thirdparty/benchmarks/benchmarks/swebenchpro/` uses; we are deliberately diverging to match scaleswe / swerebenchv2).

## 2. Reference material

- Upstream OpenHands implementation: `thirdparty/benchmarks/benchmarks/swebenchpro/` — useful for prompt template, constants, image-tag sanitization logic.
- Official SWE-bench Pro evaluation harness: `thirdparty/SWE-bench_Pro-os/swe_bench_pro_eval.py` — defines the entryscript contract, scoring rule, and per-instance asset layout.
- Style anchors in our repo: `benchmarks/swerebenchv2/` (judge + offline eval pattern), `benchmarks/scaleswe/` (workspace-type handling).
- Dataset: `ScaleAI/SWE-bench_Pro`, split `test`, 731 instances across 11 repos and 4 languages.

## 3. Layout

```
benchmarks/swebenchpro/
├── __init__.py
├── README.md
├── constants.py            # DOCKER_IMAGE_PREFIX, SOURCE_REPO_PATH, harness paths
├── build_images.py         # Build agent-server images for unique base images
├── run_infer.py            # Inference harness (flex/docker/remote)
├── eval_infer.py           # Offline batch evaluator
├── judge.py                # SWEBenchProJudge (ExecutionBasedJudge subclass)
├── _evaluator.py           # Per-instance container-eval logic; shared by judge + eval_infer
├── prompts/
│   └── default.j2          # Ported from upstream swebenchpro
└── SWE-bench_Pro-os/       # git submodule, pinned to a known ref
```

Top-level `.gitmodules` gains:

```
[submodule "benchmarks/swebenchpro/SWE-bench_Pro-os"]
    path = benchmarks/swebenchpro/SWE-bench_Pro-os
    url = https://github.com/scaleapi/SWE-bench_Pro-os.git
```

Pinned to ref `0c64e26f00b9c190432de7fc520c8ceed5c25518` (the same ref the upstream OpenHands swebenchpro uses, documented in `constants.OFFICIAL_HARNESS_REF`).

## 4. Key decisions

| Question | Decision |
|---|---|
| Where does the on-the-fly judge run? | **Sidecar Docker container** — fresh container per instance from `jefzda/sweap-images:<tag>`, isolated from the agent workspace. |
| Where do per-instance harness assets come from? | **Git submodule at `benchmarks/swebenchpro/SWE-bench_Pro-os/`** — pinned, versioned with the code. |
| Offline evaluator? | **Own implementation** using the same `_evaluator.evaluate_instance` helper as the judge. No Modal, no Laminar, no archive download. |
| Repo location inside the container? | **Copy `/app` → `/workspace/repo/`** at start of `evaluate_instance` — keeps `/app` clean as the gold reference and provides the WebSocket timing gap between `Conversation(...)` and `send_message(...)`. |

## 5. Module-by-module spec

### 5.1 `constants.py`

```python
from pathlib import Path
from typing import Final

DOCKER_IMAGE_PREFIX: Final[str] = "docker.io/jefzda/sweap-images"
DEFAULT_DOCKERHUB_USERNAME: Final[str] = "jefzda"
SOURCE_REPO_PATH: Final[str] = "/app"
HARNESS_SUBMODULE_PATH: Final[Path] = Path(__file__).parent / "SWE-bench_Pro-os"
OFFICIAL_HARNESS_REPO: Final[str] = "https://github.com/scaleapi/SWE-bench_Pro-os"
OFFICIAL_HARNESS_REF: Final[str] = "0c64e26f00b9c190432de7fc520c8ceed5c25518"
```

### 5.2 `build_images.py`

Same structure as `benchmarks/swerebenchv2/build_images.py`:

```python
def get_official_docker_image(dockerhub_tag: str) -> str:
    return f"{DOCKER_IMAGE_PREFIX}:{dockerhub_tag.strip()}"

def extract_custom_tag(base_image: str) -> str:
    """Sanitize a SWE-bench Pro dockerhub tag for use as an OpenHands agent-server tag.

    Lowercases, replaces [^a-z0-9_.-] with '-', and hash-truncates names
    longer than 96 chars. Ported from upstream openhands swebenchpro.
    """
    ...

def collect_unique_base_images(dataset, split, n_limit, selected_instances_file):
    df = get_dataset(...)
    return sorted({get_official_docker_image(str(row["dockerhub_tag"]))
                   for _, row in df.iterrows()})

def main(argv): ...  # standard get_build_parser + build_all_images wiring
```

`run_infer.py` imports `get_official_docker_image` and `extract_custom_tag` from here, exactly like v2.

### 5.3 `run_infer.py`

`SWEBenchProEvaluation(Evaluation)` with three fields (matches v2):

- `use_legacy_tools: bool`
- `bind_dev_sdk: bool`
- `judge: ExecutionBasedJudge | None`

**`prepare_instances()`** — identical to v2: `get_dataset(...)`, iterate rows, wrap in `EvalInstance(id=instance_id, data=row.to_dict())`.

**`prepare_workspace(instance, resource_factor=1, forward_env=None)`** — copy v2's three-branch structure (`flex` / `docker` / `remote`) verbatim with these substitutions:

- Base image: `get_official_docker_image(instance.data["dockerhub_tag"])`.
- Custom tag: `extract_custom_tag(official_docker_image)`.

Run `metadata.env_setup_commands` at the end (same pattern).

**`evaluate_instance(instance, workspace)`** — adapted from v2:

1. Build `Agent(llm=..., tools=..., system_prompt_kwargs={"cli_mode": True})`.
2. Construct `Conversation` early (provides the WebSocket timing gap before `send_message`).
3. Set `repo_path = "/workspace/repo/"`; assign to `instance.data["repo_path"]`. Set `base_commit = str(instance.data["base_commit"])`.
4. `mkdir -p /workspace/repo && cp -r /app/. /workspace/repo/` via `workspace.execute_command(...)`.
5. `cd /workspace/repo && git reset --hard`.
6. Log brief git history.
7. Render prompt with `get_instruction(...)`. `conversation.send_message(instruction)`. Run `run_conversation_with_fake_user_response(...)`.
8. `git add -A`, configure git user, `git commit -m 'patch'`.
9. `git diff <base_commit> HEAD` → `git_patch`.
10. If `self.judge is not None`: invoke `self.judge.judge(instance.id, git_patch, instance.data)`, capture result. Wrap in try/except so judge failures never fail the run.
11. Dump conversation history JSON (same format as v2, with `evaluation` field when judge ran).
12. Return `EvalOutput` (same shape).

**`main()`** — copy v2's:

- `get_parser()` + `add_prompt_path_argument(parser, __file__)`.
- `parser.set_defaults(dataset="ScaleAI/SWE-bench_Pro", split="test", workspace="flex")`.
- `--use-legacy-tools`, `--bind-dev-sdk` flags.
- `add_judge_args(parser, default_judge="swebenchpro")`.
- Build `EvalMetadata` with `env_setup_commands=["export PIP_CACHE_DIR=~/.cache/pip"] + OH_*-env vars`.
- Instantiate `SWEBenchProEvaluation` and call `evaluator.run(...)`.

### 5.4 `_evaluator.py`

Private module holding the container-execution logic shared by the judge and the offline evaluator.

```python
def evaluate_instance(
    spec: dict,
    model_patch: str,
    harness_dir: Path,
    *,
    timeout: int = 1800,
    block_network: bool = False,
    docker_platform: str | None = None,
    rm_image: bool = False,
) -> dict:
    """Run a single SWE-bench Pro test execution inside a sidecar container.

    Returns a result dict:
        {
            "instance_id": str,
            "resolved": bool,
            "exit_code": int,
            "passed_tests": list[str],
            "fail_to_pass_total": int,
            "from_fail_to_pass": list[str],
            "pass_to_pass_total": int,
            "failed_from_pass_to_pass": list[str],
            "error": str,   # "" on success, descriptive on failure
        }
    """
```

Internal helpers:

- `_validate_harness_dir(harness_dir)` — raises a clear "submodule not initialized" error if `run_scripts/` is missing.
- `_load_instance_assets(harness_dir, instance_id)` — reads `run_script.sh`, `parser.py`, base/instance Dockerfiles. Raises `FileNotFoundError` if any are missing.
- `_strip_binary_hunks(patch)` — port of upstream helper.
- `_create_entryscript(spec, base_dockerfile, instance_dockerfile)` — ports upstream's `create_entryscript`:
  - Scrapes `ENV` lines from both Dockerfiles → `export ...` block.
  - Resolves `before_repo_set_cmd = spec["before_repo_set_cmd"].strip().split("\n")[-1]`.
  - Resolves `selected = ",".join(eval(spec["selected_test_files_to_run"]))` (accept real list too for safety).
  - Emits the canonical `cd /app; git reset --hard {base_commit}; git checkout {base_commit}; git apply -v /workspace/patch.diff; {before_repo_set_cmd}; bash /workspace/run_script.sh {selected} > /workspace/stdout.log 2> /workspace/stderr.log; python /workspace/parser.py /workspace/stdout.log /workspace/stderr.log /workspace/output.json`.
- `_run_in_container(image, workspace_dir, *, timeout, block_network, docker_platform)` — `docker run --rm --network {host|none} [--platform ...] -v {workspace_dir}:/workspace -w /app {image} /bin/bash -c "bash /workspace/entryscript.sh"`, with `subprocess.run(..., timeout=timeout)`.
- `_score(output_json, spec)` — `passed = {t["name"] for t in tests if t["status"] == "PASSED"}`; `f2p = set(parse(spec["fail_to_pass"]))`; `p2p = set(parse(spec["pass_to_pass"]))`; `resolved = (f2p | p2p) <= passed`.

`parse(value)` accepts either a real list or a stringified list literal (uses `ast.literal_eval` instead of `eval`).

`evaluate_instance` is the single entry point that the judge and offline eval both call.

### 5.5 `judge.py`

```python
@register_judge("swebenchpro")
class SWEBenchProJudge(ExecutionBasedJudge):
    harness_dir: Path = Field(default_factory=lambda: HARNESS_SUBMODULE_PATH)
    rm_image: bool = Field(default=False)
    block_network: bool = Field(default=False)
    docker_platform: str | None = Field(default=None)

    def judge(self, instance_id, git_patch, instance_data) -> bool | None:
        if not git_patch or not git_patch.strip():
            return False
        try:
            result = evaluate_instance(
                instance_data, git_patch, self.harness_dir,
                timeout=self.timeout,
                block_network=self.block_network,
                docker_platform=self.docker_platform,
                rm_image=self.rm_image,
            )
            return bool(result["resolved"])
        except Exception:
            logger.exception("SWEBenchProJudge failed for %s", instance_id)
            return None
```

### 5.6 `eval_infer.py`

Modeled on `benchmarks/swerebenchv2/eval_infer.py`:

```bash
uv run benchmarks/swebenchpro/eval_infer.py \
  --predictions output.jsonl \
  --dataset ScaleAI/SWE-bench_Pro \
  --split test \
  --max-workers 4 \
  --report-json eval_report.json \
  [--harness-dir <path>] \
  [--timeout 1800] \
  [--block-network] \
  [--docker-platform linux/amd64] \
  [--rm-image]
```

Flow:

1. `load_predictions(predictions_path)` — extract `record["test_result"]["git_patch"]` per `instance_id`.
2. `get_dataset(dataset, split)` and intersect on `instance_id`.
3. `ThreadPoolExecutor(max_workers=...)` — submit `evaluate_instance(spec, model_patch, harness_dir, ...)` per matched instance.
4. Aggregate into a report dict (matches v2):

```python
{
    "total": int,
    "resolved": int,
    "errors": int,
    "resolve_rate": float,
    "items": [ ... per-instance _evaluator result dicts ... ],
}
```

5. Write `--report-json`, log accuracy, exit 0.

No Modal, no archive download, no Laminar, no cost report (the latter is an inference-time concern handled elsewhere).

### 5.7 `prompts/default.j2`

Port the upstream OpenHands swebenchpro prompt verbatim (8-phase READING / RUNNING / EXPLORATION / TEST CREATION / FIX ANALYSIS / FIX IMPLEMENTATION / VERIFICATION / FINAL REVIEW). Uses `{{ instance.repo_path }}`, `{{ instance.problem_statement }}`, `{{ instance.base_commit }}` — all already populated by `get_instruction(...)`.

### 5.8 `README.md`

Sections:

1. Dataset / image prefix overview.
2. Submodule init: `git submodule update --init benchmarks/swebenchpro/SWE-bench_Pro-os`.
3. Build agent-server images: `uv run benchmarks/swebenchpro/build_images.py ...`.
4. Run inference: `uv run benchmarks/swebenchpro/run_infer.py <llm_config.json> ...`, with workspace-type and judge notes.
5. Run offline evaluation: `uv run benchmarks/swebenchpro/eval_infer.py --predictions output.jsonl ...`.
6. Smoke tests checklist (see §7).

## 6. Error handling

- *Missing `dockerhub_tag`* — `prepare_workspace` raises `ValueError`. `build_images.py` raises during dataset iteration.
- *Submodule not initialized* — `_evaluator._validate_harness_dir` raises a one-line actionable message pointing at `git submodule update --init`.
- *Missing per-instance assets* — judge returns `None`; offline eval records `{"resolved": False, "error": "missing harness assets: <path>"}`.
- *Patch apply fails / missing `output.json`* — judge returns `None` (the upstream behavior treats this as not-resolved, but our judge result is consumed only for analysis); offline eval records `{"resolved": False, "error": "no output.json (exit_code=N)"}`.
- *Docker pull / timeout* — `subprocess.run(timeout=...)` raises `TimeoutExpired`; judge returns `None`, offline eval records the specific error string.
- *Judge raises during inference* — caught in `evaluate_instance`; logged; `evaluation_result = None`. Agent run never fails because of judge failures.

## 7. Smoke tests (manual; no unit-test framework in this repo for benchmarks)

1. Build path: `uv run benchmarks/swebenchpro/build_images.py --n-limit 1` succeeds.
2. Inference (docker): `uv run benchmarks/swebenchpro/run_infer.py <cfg> --workspace docker --n-limit 1 --max-iterations 5` writes a `git_patch` for one instance.
3. On-the-fly judge: same as (2) with `--judge`; per-instance `*.history.json` contains an `evaluation` field.
4. Offline eval: `uv run benchmarks/swebenchpro/eval_infer.py --predictions <output.jsonl> --max-workers 1` writes a report.
5. Gold-patch sanity: feed the dataset's `patch` field through `eval_infer.py` for one instance; expect `resolved=True`.
6. Flex workspace: repeat (2)+(3) with `--workspace flex`.

## 8. Alignment matrix

| Aspect | scaleswe | swerebenchv2 | **swebenchpro (new)** |
|---|---|---|---|
| Base class | `Evaluation` direct | `Evaluation` direct | `Evaluation` direct |
| Image source field | `image_url` | `image_name` | `dockerhub_tag` |
| Image prefix | dataset-provided | `docker.io/swerebenchv2/...` | `docker.io/jefzda/sweap-images` |
| Repo location in image | dataset `workdir` | `/<repo-leaf>` | `/app` |
| Repo copy target | none | `/workspace/<repo-leaf>/` | `/workspace/repo/` |
| `base_commit` field | `parent_commit` | `base_commit` | `base_commit` |
| Workspaces | flex/docker/remote | flex/docker/remote | flex/docker/remote |
| Default workspace | flex | flex | flex |
| Judge style | sidecar Docker | sidecar Docker | sidecar Docker |
| Judge asset source | `awe_agent` lib | `thirdparty/SWE-rebench-V2/lib` | submodule under `benchmarks/swebenchpro/SWE-bench_Pro-os/` |
| Offline eval | n/a | `eval_infer.py` | `eval_infer.py` (shared `_evaluator.py`) |
| Patch convention | `git diff parent HEAD` | `git diff base HEAD` | `git diff base HEAD` |

## 9. Out of scope

- Modal-based execution (upstream supports both; we standardize on local Docker, matching v2).
- Laminar telemetry hooks (out of scope for benchmarks repo entry).
- A cost-report wrapper for `eval_infer.py` — costs are already accounted for at inference time.
- Re-implementing the phased base-image / builder-image pipeline used by the upstream OpenHands swebenchpro; we delegate to the shared `build_all_images` utility like v2 does.
