# Scale-SWE Notes

## Dataset fields and how they are used

The Scale-SWE dataset in `thirdparty/Scale-SWE/scale-swe-batch1.jsonl` contains these key fields used by the benchmark harness:

- `instance_id`: becomes `EvalInstance.id`
- `image_url`: the per-instance prebuilt Docker image, used as the base image for `FlexWorkspace`
- `workdir`: the repository path inside the container, used as `repo_path`
- `parent_commit`: used as `base_commit` for `git diff <base_commit> HEAD`
- `problem_statement`: passed into the prompt template
- `pre_commands`: run before the agent starts; contains repo cleanup, checkout, branch creation, and git config
- `user` + `repo`: combined into `user/repo` when needed for prompt compatibility
- `language`: informational
- `patch`: gold/reference patch, not used by inference
- `FAIL_TO_PASS`, `PASS_TO_PASS`, `f2p_patch`, `f2p_script`: evaluation-oriented fields, not used by inference

## AweAgent Scale-SWE behavior

Reference implementation: `thirdparty/AweAgent/awe_agent/tasks/scale_swe/task.py`

Important behaviors:

1. Uses `image_url` directly as the container image.
2. Uses `workdir` directly instead of copying `/testbed` into `/workspace/...`.
3. Uses `parent_commit` (fallback `base_commit`) as the diff base.
4. Runs dataset-provided `pre_commands` instead of doing repo setup in code.
5. Uses a simple issue-focused prompt; in this benchmark implementation we keep the same prompt as `benchmarks/swebench/prompts/default.j2` by default, and keep an optional `aweagent.j2` copy for reference.

## Adaptation plan from `benchmarks/swebench/run_infer.py`

The Scale-SWE harness is adapted from SWE-bench with these changes:

1. **Instance loading**
   - Same `get_dataset()` utility.
   - Default dataset points to `thirdparty/Scale-SWE/scale-swe-batch1.jsonl`.

2. **Workspace preparation**
   - Primary workspace type is `flex`.
   - `image_url` supplies the base image directly.
   - Optional `--docker-image-prefix` allows swapping namespace/registry while keeping the image name/tag.

3. **Repo setup**
   - No `/testbed` copy.
   - Use `workdir` as the repo path.
   - Run `pre_commands` before starting the agent.

4. **Prompting**
   - Default prompt is copied from `benchmarks/swebench/prompts/default.j2`.
   - `instance.repo_path` is set from `workdir`.
   - `instance.base_commit` is set from `parent_commit` for template compatibility.

5. **Patch extraction**
   - Same as SWE-bench after the run: `git add -A`, `git commit -m 'patch'`, then `git diff <parent_commit> HEAD`.

## Comparison with related benchmarks

- `benchmarks/swebench/run_infer.py`: derives images from instance ID and copies `/testbed` into `/workspace/repo/`.
- `benchmarks/swesmith/run_infer.py`: uses dataset image field directly, but still follows SWE-bench repo-copy setup.
- `benchmarks/swerebench-leaderboard/run_infer.py`: similar to swesmith, but more minimal (no legacy tools or dev SDK binds).
- `benchmarks/scaleswe/run_infer.py`: closest to SWE-bench structure, but uses Scale-SWE's own `image_url`, `workdir`, and `pre_commands` model.
