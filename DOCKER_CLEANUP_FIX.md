# Docker Image Cleanup Fix - Quick Reference

## Problem
Using `--rm-image` flag with `eval_infer.py` was not actually removing Docker images, causing disk to fill up.

## Root Cause
The `remove_image` parameter was declared in function signatures but never actually used to call `docker rmi`.

## Solution
Added proper Docker image cleanup in the `_run_in_container` function's `finally` block:

```python
finally:
    # Remove Docker image if requested
    if remove_image:
        try:
            subprocess.run(
                ["docker", "rmi", "-f", image],
                check=False, capture_output=True, timeout=60
            )
        except Exception:
            pass
```

## Verification
Check that images are being removed:

```bash
# Before evaluation - count images
docker images | grep sweap-images | wc -l

# Run evaluation with --rm-image
python benchmarks/swebenchpro/eval_infer.py \
    --predictions output.jsonl \
    --rm-image \
    --max-workers 4

# After evaluation - count should be same or lower
docker images | grep sweap-images | wc -l
```

## Monitor Disk Usage During Evaluation
```bash
# Watch disk usage in real-time (separate terminal)
watch -n 5 'df -h / && echo && docker images | grep sweap-images | wc -l'
```

## Clean Up Existing Images Manually
If your disk is already full from previous runs:

```bash
# Remove all sweap-images Docker images
docker images | grep sweap-images | awk '{print $3}' | xargs docker rmi -f

# Or use Docker's built-in cleanup
docker image prune -a --filter "label=sweap-images" -f
```

## Best Practices
1. **Always use `--rm-image`** for evaluation runs to prevent disk fill-up
2. **Monitor disk space** before starting large evaluation batches
3. **Use `--max-workers` wisely** - more workers = more concurrent images = more disk usage
4. **Consider image pre-pulling** for better performance with `--rm-image`

## Trade-offs
- **With `--rm-image`**: Saves disk space but requires re-pulling images for each instance
- **Without `--rm-image`**: Faster evaluation (reuses images) but can fill disk quickly

## Recommended Settings

### Low Disk Space (< 100GB free)
```bash
python benchmarks/swebenchpro/eval_infer.py \
    --predictions output.jsonl \
    --rm-image \
    --max-workers 2  # Lower concurrency to reduce disk pressure
```

### Ample Disk Space (> 500GB free)
```bash
python benchmarks/swebenchpro/eval_infer.py \
    --predictions output.jsonl \
    --max-workers 8  # Can skip --rm-image for speed
```

### Continuous Evaluation (Multiple Batches)
```bash
# Clean up between batches
for batch in batch_*.jsonl; do
    python benchmarks/swebenchpro/eval_infer.py \
        --predictions $batch \
        --rm-image \
        --max-workers 4
    
    # Force cleanup after each batch
    docker image prune -f
done
```
