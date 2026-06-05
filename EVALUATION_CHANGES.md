# SWE-bench Pro Evaluation Changes Summary

## Overview
Modified the evaluator to match the official `swe_bench_pro_eval.py` logic while maintaining enhanced robustness features.

## Changes Made

### 1. `_evaluator.py` - Core Evaluation Logic

#### Test Name Normalization (CRITICAL FIX)
- **Before**: Aggressive normalization (timing suffix removal, path transformations)
- **After**: Minimal normalization (whitespace stripping only)
- **Impact**: Fixes gold patch failures caused by test name mismatches

#### Scoring Logic (CRITICAL FIX)
- **Before**: Set difference check with exit_code requirement
- **After**: Subset comparison matching official script
  ```python
  # Official: (f2p | p2p) <= passed_tests
  required_tests = f2p | p2p
  resolved = required_tests <= passed_tests
  ```
- **Impact**: Matches official evaluation behavior exactly

#### Binary Patch Handling
- **Added**: Logging when binary hunks are stripped
  ```python
  if cleaned_patch != git_patch:
      print(f"Stripped binary diff hunks from patch for {instance_id}")
  ```

#### Patch Application Strategy
- **Kept**: Flexible multi-strategy approach with fallbacks
- **Order**: `git apply` → `--reject` → `--3way` → `--ignore-whitespace`
- **Rationale**: More robust than official script's simple approach

#### Container Naming
- **Added**: Unique container names for parallel execution
  ```python
  container_name = f"swebenchpro_eval_{safe_instance_id}_{timestamp}"
  ```
- **Benefits**: Better debugging, prevents conflicts

#### Test Format
- **Changed**: Comma-separated test list (matching official)
- **Changed**: Extract last line of `before_repo_set_cmd` (matching official)

### 2. `eval_infer.py` - Progress Tracking

#### Progress Bar
- **Added**: tqdm-based progress bar with live statistics
- **Displays**: passed count, failed count, error count, pass rate
- **Position**: Sticks to bottom, updates smoothly

#### Statistics Tracking
- **Tracks**: `resolved_count`, `failed_count`, `error_count`
- **Updates**: After each instance completes
- **Shows**: Live pass rate percentage

#### Logger Management
- **Changed**: Temporarily raises log level to WARNING during progress
- **Benefit**: Prevents log interference with progress bar
- **Fallback**: Standard INFO logging if tqdm unavailable

## Compatibility Analysis

### `run_infer.py` - No Changes Required ✓
The inference script generates patches and delegates evaluation to the judge.
No changes needed because:
- Still generates same `git_patch` format
- Still uses same `EvalOutput` structure
- Judge interface unchanged

### `judge.py` - Compatible ✓
The `SWEBenchProJudge` calls `evaluate_instance()` and checks `result["resolved"]`.
Compatible because:
- Return structure still has `resolved` field
- Return structure still has `exit_code` and `error` fields
- Only internal scoring logic changed, not interface

#### Judge Return Value Verification
```python
# Judge expects:
result = evaluate_instance(instance_data, git_patch, ...)
result["resolved"]  # ✓ Still present
result["exit_code"]  # ✓ Still present  
result["error"]      # ✓ Still present (in error cases)
```

#### Evaluator Now Returns
```python
{
    "instance_id": str,
    "resolved": bool,         # ✓ Judge uses this
    "test_result": {
        "git_patch": str
    },
    "tests": list,            # New: full test details
    "exit_code": int,         # ✓ Judge logs this
}
```

### Impact on Existing Workflows

#### Gold Patch Evaluation
- **Before**: Failed due to test name mismatch
- **After**: Should pass (matches official evaluator)

#### Agent Inference
- **No Change**: Same workflow, just more accurate scoring

#### Parallel Evaluation
- **Improved**: Unique container names prevent conflicts

#### Progress Monitoring
- **Improved**: Live statistics instead of log parsing

## Testing Recommendations

1. **Verify Gold Patches Pass**
   ```bash
   python benchmarks/swebenchpro/eval_infer.py \
       --predictions gold_patches.jsonl \
       --dataset ScaleAI/SWE-bench_Pro \
       --split test \
       --max-workers 4
   ```
   Expected: High pass rate matching official evaluator

2. **Check Progress Bar**
   - Verify progress bar displays correctly
   - Verify statistics update in real-time
   - Verify final report matches progress bar stats

3. **Parallel Execution**
   - Run multiple evaluations simultaneously
   - Verify no container name conflicts
   - Check `docker ps` shows unique container names

4. **Judge Integration**
   - Run inference with `--judge swebenchpro`
   - Verify judge returns correct boolean values
   - Check judge log output includes exit_code and error

## Key Takeaway

**The root cause of gold patch failures was aggressive test name normalization.**

The official script uses raw test names from the parser output, but our version was transforming them (stripping timing suffixes, converting dotted paths to pytest format). This caused mismatches between:
- Dataset's `fail_to_pass`/`pass_to_pass` test names
- Parser output test names

By matching the official script's minimal normalization, test names now align correctly.
