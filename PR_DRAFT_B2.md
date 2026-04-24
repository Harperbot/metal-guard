# PR Draft — B2: `subprocess_inference_guard` contextmanager

**Branch**: `b2-subprocess-inference-guard`
**Base**: `main` (`bac5f8a` v0.8.0 — L9 CadenceGuard)
**Commit**: `e6a69d2`
**Status**: LOCAL ONLY — not pushed

## Summary

Adds a new `subprocess_inference_guard(model_id)` contextmanager to `metal_guard.py`. This is a per-inference Metal flush designed for anyone running MLX generation inside a subprocess worker for crash isolation (a common defensive pattern).

Pure addition — zero breaking changes.

## Motivation

Harper's internal fork hit **6 consecutive `IOGPUMemory.cpp:492 "completeMemory() prepare count underflow"` kernel panics in 3 days** while running subprocess-isolated MLX workers. Root cause:

- The parent process typically ran `mx.clear_cache()` + post-gen `mx.eval(result)` as defensive sync hooks.
- These **do NOT reach the subprocess** (separate address space; separate MLX runtime).
- Subprocess worker called `gen_fn(...)` with no per-inference flush.
- GPU command buffer was still in flight when the worker released its buffers → Metal accounting drifted → next load tripped a `prepare_count` underflow → kernel panic.

Harper landed this contextmanager internally (B1) and the panic streak ended. B2 ports it to the public repo so any metal-guard user doing their own subprocess isolation gets the same fix.

Related upstream reports already cited in the code comment block:

- `ml-explore/mlx-lm#1128` — prefill guard design
- `ml-explore/mlx#3186` — subprocess isolation guidance
- `ml-explore/mlx#3346` — kernel panic reproducers

## Design

### Call order (load-bearing)

```
PRE:   mx.clear_cache()      # release prior iter cached buffers
       metal_guard.breadcrumb("SUBPROC_PRE: <model>")
yield                        # generate runs
POST:  mx.synchronize()      # block until GPU command buffer drained
POST:  mx.clear_cache()      # release this iter's buffers
       metal_guard.breadcrumb("SUBPROC_POST: <model>")
```

Reversing POST sync vs clear reproduces the `IOGPUMemory.cpp:492` underflow that this guard exists to prevent.

### Why `synchronize()` instead of `mx.eval(result)`

Generate return types vary by backend:

- `mlx-lm` returns `str`
- `mlx-vlm` returns a `GenerateResult` object

`synchronize()` is return-type agnostic and semantically correct: "wait for all pending GPU ops to complete."

### Best-effort safety

- If `mlx.core` is not importable (test env), the context manager **no-ops** — it does not raise. Production worker can fail loud on its own model `load()` instead of on the guard.
- **PRE hook failure** is logged but does NOT block the body. A broken guard is worse than no guard because it would silently stop all inference.
- **POST hook failure** is logged but never masks a body exception (`finally` semantics).
- Hook errors use `log.warning`; breadcrumb errors use `log.debug` (benign if breadcrumb storage is down).

### Forensics

Breadcrumbs `SUBPROC_PRE: <model>` / `SUBPROC_POST: <model>` go to the standard metal-guard breadcrumb log. Post-mortem of a panic can grep these to confirm whether the guard was engaged on the fatal call and whether PRE/POST paired up (PRE only → panic during generate; both → panic is upstream of the guard).

## Usage

```python
from metal_guard import subprocess_inference_guard

# Inside your subprocess worker:
def worker_main(model_id, ...):
    model, tokenizer = load(model_id)
    while True:
        req = recv_request()
        with subprocess_inference_guard(model_id):
            result = gen_fn(model, tokenizer, prompt=req.prompt, ...)
        send_response(result)
```

## Test Coverage

**9 new unit tests** in `tests/test_subprocess_inference_guard.py`:

1. `test_symbol_exists_at_module_level` — API discoverability
2. `test_pre_calls_clear_cache_before_yield` — PRE ordering
3. `test_post_synchronize_before_clear_cache` — POST ordering (critical)
4. `test_post_runs_even_on_exception` — `finally` semantics on body exception
5. `test_degrades_gracefully_without_mlx` — no-op when `mlx.core` missing
6. `test_pre_failure_does_not_block_body` — broken PRE doesn't stop inference
7. `test_post_failure_does_not_mask_body_exception` — body exception surfaces
8. `test_breadcrumb_emits_pre_and_post` — forensics breadcrumbs present
9. `test_model_id_escapes_cleanly` — edge cases (whitespace, empty string)

**Test results**:

- Baseline main: 139 passing / 18 failing (18 failures are pre-existing env issues — mlx not installed in CI env — unrelated to this PR)
- With B2: **148 passing** / 18 pre-existing failing
- New `test_subprocess_inference_guard.py` file: **9/9 passing**
- Zero regressions in existing tests.

## Breaking Changes

**None.** This is a pure addition:

- New symbol `subprocess_inference_guard` at module level
- No existing function signature changed
- No behavior change unless callers opt in by using the new contextmanager

## Notes for Reviewer

- The monkeypatch in tests sets both `sys.modules["mlx"]` (parent package) and `sys.modules["mlx.core"]` because the public-repo CI env does not have MLX installed; importing `mlx.core as mx` needs both keys present for the mock to resolve. In environments with MLX installed, only `mlx.core` needs monkeypatching.
- The function is appended at the end of `metal_guard.py` after the CircuitBreaker class so the singleton `metal_guard` is fully initialized by the time the decorator runs.

## Release Plan

Suggest tagging this as **v0.9.0** together with the in-flight CadenceGuard `idle_warmup` changes currently preserved on stash (`wip-v0.9-cadence-idle-warmup-preserve-2026-04-23`). Both are additive. Alternatively, ship B2 alone as **v0.8.1** and the CadenceGuard idle-warmup as **v0.9.0** separately.
