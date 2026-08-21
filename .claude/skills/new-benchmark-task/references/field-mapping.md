# Dataset field mapping

Map the new dataset's field names to the SWE-bench concepts the harness expects,
**before** writing code. Getting this wrong causes silent evaluation errors (wrong
diff base, uncomputable F2P, corrupted image names).

## Questions to answer for any new dataset

1. **instance id** — which field uniquely names the row? (`instance_id`)
2. **base commit** — the pre-fix commit the agent starts from and the diff base.
   NOT the fix/merge commit.
3. **image** — which field identifies the Docker image? Is it a bare
   `namespace/repo:tag`, or already registry-qualified? Lowercased?
4. **gold patch** — the reference code fix. Does it combine source + test edits
   in one diff (needs separation) or are they already split?
5. **test artifacts** — the ground-truth test patch and/or a generated test
   script that defines the F2P tests.
6. **test command** — the bare command to run the tests, and any separate setup
   (service startup, env vars) that must run first.
7. **FAIL_TO_PASS / PASS_TO_PASS** — present already, or must be computed? Native
   lists or JSON-encoded strings? (Parse strings with `json.loads`.)
8. **problem statement** — ever empty? What's the fallback text?
9. **workdir** — where the repo lives inside the image (e.g. `/app`, `/testbed`,
   `/workspace`).

## z021data concrete map (dataset: pr_data_1_327.jsonl, 327 rows)

| SWE-bench concept        | z021data field                                             |
|--------------------------|------------------------------------------------------------|
| instance_id              | `instance_id`                                              |
| repo                     | split across `user` + `repo`; full slug in `github_url`   |
| base_commit              | `parent_commit` (NOT `base_commit`; `pr_commit` is the fix)|
| gold patch (src+test)    | `patch` (combined — separate with the validator helper)   |
| test_patch               | test-file part of `patch`, plus `f2p_patch`               |
| generated test body      | `test_script` (== `f2p_script`; validator uses `test_script`)|
| FAIL_TO_PASS/PASS_TO_PASS| `FAIL_TO_PASS` / `PASS_TO_PASS` — **JSON-encoded strings** |
| docker image             | `image_url`, bare `swe_pr_agent/...:<inst>` (prefix at runtime)|
| bare test command        | `test_command`                                            |
| setup / pre-command      | `pre_commands_parsed.setup` (e.g. `redis-server --daemonize`)|
| problem_statement        | `problem_statement`, empty in ~1/3 rows                   |
| problem_statement fallback | `pr_description.pr_body` (+ `pr_title`, `issue_titles/bodies`)|
| workdir                  | `workdir` (`/app`)                                        |
| (absent)                 | no `environment_setup_commit`, no `version`               |

Notes:
- `FAIL_TO_PASS`/`PASS_TO_PASS` node ids are prefixed `tests/test_f2p_generated.py::…`.
- The dataset's `pre_commands` field holds the **wrapped test invocation**
  (`cd /app && conda run -n testbed python -m pytest …`), NOT setup — don't run
  it during inference. The validator strips `conda run` and relies on the image's
  baked env; the judge should do the same via the validator's `_strip_conda_run`.

## Key differences across image-URL benchmarks

| Aspect              | scaleswe                       | z021data                          |
|---------------------|--------------------------------|-----------------------------------|
| Image field         | `image_url`                    | `image_url`                       |
| Image prefix op     | **replace** before last `/`    | **prepend** (guard qualified URL) |
| Base commit field   | `parent_commit`                | `parent_commit`                   |
| Judge backend       | delegates to `awe_agent`       | self-contained, reuses validator  |
| F2P/P2P             | in `awe_agent`                 | validator: `before_failed ∩ after_passed` |
| `pre_commands` field| git checkout/branch setup      | the test invocation (don't run)   |
| Repo setup          | already at workdir, no copy    | already at workdir, checkout base |

## Judge success criterion (z021data / validator)

```
before_failed = result1.failed ∪ result1.errors
before_passed = result1.passed
after_failed  = result2.failed ∪ result2.errors
after_passed  = result2.passed

FAIL_TO_PASS = before_failed ∩ after_passed   # failed before, pass after
PASS_TO_FAIL = before_passed ∩ after_failed   # regression
resolved     = len(FAIL_TO_PASS) > 0 and len(PASS_TO_FAIL) == 0
```

`result1` is the run BEFORE applying the agent's patch; `result2` is AFTER. Both
runs execute `setup ; test_command` and are parsed with `python_logparse`. The
`unsupported` runner (opaque command, no per-test node ids) cannot compute F2P →
skip (return `None`).
