# Changelog

All notable changes to **metal-guard** are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.3.1] — 2026-04-13

### Added

- **Cross-process mutual exclusion (Layer 8)** — `acquire_mlx_lock()`,
  `release_mlx_lock()`, `read_mlx_lock()`, and `mlx_exclusive_lock()` context
  manager. File-based lock at `~/.metal-guard/locks/mlx_exclusive.lock`
  prevents concurrent MLX workloads across process boundaries — the root
  cause of IOGPUMemory kernel panics when `mlx_lm.server`, benchmarks, or
  direct `mlx_lm.generate` calls run simultaneously with other MLX processes.
  Stale locks from crashed processes are self-healing (pid liveness check).
  New `MLXLockConflict` exception raised when a live process already holds
  the lock.

  ```python
  from metal_guard import mlx_exclusive_lock

  with mlx_exclusive_lock("my_script"):
      model, tokenizer = mlx_lm.load("mlx-community/gemma-4-31b-it-8bit")
      result = mlx_lm.generate(model, tokenizer, prompt="Hello")
  ```

## [0.3.0] — 2026-04-12

### Added

- **Pre-generate Metal health probe** — `probe_metal_health()` runs a tiny
  `mx.eval(mx.zeros(1))` to verify the Metal command queue is alive before
  starting a long generate call. If the GPU is in a bad state from a prior
  crash (stale command queue, leaked buffers), this crashes at a controlled
  point instead of mid-inference. Costs ~1ms.

- **SIGABRT signal handler** — `install_abort_handler()` installs a
  Python-level `signal.SIGABRT` handler for crash forensics. When MLX's
  `check_error(MTL::CommandBuffer*)` throws a C++ exception from the Metal
  GCD CompletionQueueDispatch queue (which Python cannot catch), the handler
  writes a final breadcrumb and logs at CRITICAL before re-raising for a
  proper crash report. Does not attempt recovery — Metal state is corrupt.
  Observed in production: 2026-04-12 18:30 SIGABRT on Thread 34
  `com.Metal.CompletionQueueDispatch`.

- **6-bit / 3-bit / mxfp4 quantization support** in
  `estimate_model_size_from_name()`. Fixes a bug where 6-bit models
  (e.g. `LFM2-24B-A2B-MLX-6bit`) fell back to fp16 estimation (48 GB
  instead of correct 18 GB), causing spurious `MemoryError` from
  `require_fit`. New multipliers:
  - `6bit` → 0.75 bytes/param
  - `3bit` / `int3` → 0.375 bytes/param
  - `mxfp4` → 0.5 bytes/param (alias for Metal FP4 format)

### Fixed

- `estimate_model_size_from_name` no longer returns wildly inflated
  estimates for mixed-precision MLX models (unsloth UD-MLX, lmstudio
  community 6-bit variants).

### Changed

- Root causes documentation updated to include cause #3: Metal
  CommandBuffer completion error (C++ exception on GCD queue, SIGABRT).

## [0.2.3] — 2026-04-10

### Added

- **Escalated retry in `require_fit`** — two-tier retry strategy with a
  caller-supplied cache-clear callback and configurable cooldown for
  tight-memory ensemble workloads. Fixes the observed OOM path where the
  standard `safe_cleanup` leaves enough stale GPU buffers that a large
  follow-up model still can't fit. Common on M1 Ultra running multi-debater
  ensembles where each KOL sees the full mistral-24B → phi-4-mini →
  gemma-4-26B cycle and the next batch tries to load mistral-24B again
  before Metal has returned pages to the OS.
- `require_fit` new keyword-only parameters:
  - `cache_clear_cb: Callable[[], None] | None = None`
  - `escalated_cooldown_sec: float = 0.0` (opt-in; 0 keeps old behavior)
- 4 new unit tests: cache_clear_cb invocation on escalation, bad-cb
  non-fatal propagation, hopeless-memory still raises, backward-compat
  old-style call.

### Changed

- `require_fit` now accepts additional keyword arguments without breaking
  existing callers. All pre-0.2.3 call sites continue to work unchanged.
- README (en / zh-TW / ja) updated with v0.2.3 section explaining the
  two-tier strategy and opt-in usage.

### Security

- No security impact. Escalated retry only affects the happy path —
  failure still raises `MemoryError` cleanly instead of reaching Metal.

## [0.2.2] — 2026-04-10

### Added

- **Model size estimator** — `MetalGuard.estimate_model_size_from_name()`
  parses param count + quantization hints directly from model names
  (`Mistral-24B-8bit` → 24 GB, `Phi-4-mini-4bit` → 2 GB, etc.). Designed
  to pair with `require_fit` for multi-model ensemble pre-load gating.
  Returns `None` when no hint is parseable so callers can fall back to
  the threshold-based `ensure_headroom` path.
- **AGX driver workaround** — sets `AGX_RELAX_CDM_CTXSTORE_TIMEOUT=1` at
  import time. Suggested by @zcbenz (MLX maintainer) in
  [mlx#3267](https://github.com/ml-explore/mlx/issues/3267) to relax the
  IOGPUFamily command buffer context store timeout and reduce kernel
  panics on long-running GPU workloads. Zero-cost, safe to set
  unconditionally.
- **Additional OOM pattern detection** — `is_metal_oom` now detects the
  `fPendingMemorySet` panic signature reported in
  [mlx#3346](https://github.com/ml-explore/mlx/issues/3346) by
  @yoyaku155, alongside existing `Insufficient Memory` and
  `kIOGPUCommandBufferCallbackErrorOutOfMemory` patterns.

### Fixed

- `estimate_model_size_from_name` uses `\b` word boundary in the
  quantization regex to avoid spurious matches inside longer identifiers
  (follow-up commit `2a7466d`).

## [0.2.1] — 2026-04-10

### Fixed

- Addressed code review findings from the v0.2.0 release.

## [0.2.0] — 2026-04-10

### Added

- **OOM recovery** — catches Metal GPU out-of-memory errors and converts
  them to recoverable `MetalOOMError` instead of crashing the process.
  Addresses [mlx-lm#1015](https://github.com/ml-explore/mlx-lm/issues/1015)
  and [#854](https://github.com/ml-explore/mlx-lm/issues/854).
- **Pre-load memory check** — `ensure_headroom(model_name)` proactively
  unloads cached models when Metal active memory exceeds 75% before
  attempting to load a new one.
- **Periodic Metal flush** — `flush_gpu()` exposed as a lightweight
  keep-memory-bounded primitive for long-running batch workloads.
- **Memory watchdog** — `memory_stats()` returns structured stats
  (active_gb, peak_gb, limit_gb, available_gb, active_pct, peak_pct).
- READMEs in English, Traditional Chinese, and Japanese.

## [0.1.0] — 2026-04-10

### Added

- Initial release.
- **MetalGuard singleton** — thread registry + `wait_for_threads()` to
  bound Metal GPU cleanup on daemon threads.
- **Safe cleanup** — atomic `wait_for_threads → gc.collect → flush_gpu →
  cooldown sleep` primitive; the only correct way to release Metal
  memory.
- **Breadcrumb logging** — crash-safe log append for post-panic forensics.
- **Thread tracking** — `register_thread()` called before `thread.start()`
  to close the μs race window between registration and generate.
- **Module-level `_MLX_CALL_LOCK`** (in `inference.py` client wrapper)
  to serialize in-process MLX backend calls.

[0.3.0]: https://github.com/harper-systems/metal-guard/releases/tag/v0.3.0
[0.2.3]: https://github.com/harper-systems/metal-guard/releases/tag/v0.2.3
[0.2.2]: https://github.com/harper-systems/metal-guard/releases/tag/v0.2.2
[0.2.1]: https://github.com/harper-systems/metal-guard/releases/tag/v0.2.1
[0.2.0]: https://github.com/harper-systems/metal-guard/releases/tag/v0.2.0
[0.1.0]: https://github.com/harper-systems/metal-guard/releases/tag/v0.1.0
