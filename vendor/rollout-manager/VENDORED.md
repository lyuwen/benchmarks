# Vendored rollout-manager

Snapshot of <https://github.com/lyuwen/rollout-manager> at commit
`380427cfae49ac429c5cb7c5904982c7ab756ca6` ("Correlate tool results by id and
retain native prompts"), vendored into this repository so the harness pipeline
is self-contained: the proxy is installed from a tracked local path instead of
being fetched from GitHub at `uv sync` time.

## Local changes on top of upstream

- `--request-params JSON` (and `$ROLLOUT_REQUEST_PARAMS`): merge (override) a
  fixed set of Chat-Completions parameters into every upstream request *after*
  protocol conversion, so the benchmark's LLM sampling configuration
  (`temperature`, `top_p`, `top_k`, `reasoning_effort`, `seed`, `max_tokens`,
  provider `extra_body` entries, ...) applies to harness traffic that does not
  carry those parameters. Structural fields (`messages`, `stream`,
  `stream_options`, `tools`) are rejected; an explicit `--model` still wins over
  a `request_params` `model` entry. The overlay is visible in the captured
  per-turn `request`, so exports stay self-describing.
- Tests for the overlay and for `--request-params` parsing/validation.

## Updating / upstreaming

Push the same change to the upstream repository, then switch
`pyproject.toml [tool.uv.sources] rollout-manager` back to a pinned git `rev`
and remove this directory plus the `vendor/rollout-manager` entry from
`[tool.uv.workspace]`-adjacent configuration (none) / `uv.lock` regeneration.
Until then, re-sync this snapshot from the upstream clone at the commit above
and re-apply the two local commits when bumping the vendored version.
