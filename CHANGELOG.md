# Changelog

All notable changes to **metal-guard** are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.7.0] — 2026-04-16

Minor release adding the five R-series defences that were originally
written to Harper's internal fork (`lib/mlx_client`) and validated
against real 4-bit MAGI / bench workloads across March and April
2026. This release rolls them into the open-source single-file
distribution and adds nine new version advisories plus a 7th
documented kernel-panic root-cause class.

### Added (features)

- **R2 — System-level audits.** Two new module-level functions:
  - `audit_wired_limit()` reads `sysctl iogpu.wired_limit_mb` and flags
    explicit overrides exceeding 85 % of unified memory. Per mlx-lm
    maintainer `angeloskath` on `ml-explore/mlx-lm#1047`, too-high
    wired-memory overrides are correlated with IOGPUFamily kernel
    panics. Returns `mode="default" / "override" / "unknown"` with an
    optional `advisory` string.
  - `read_gpu_driver_version()` reads the `IOGPUFamily` kext bundle
    version via `kextstat` (falls back to `ioreg`). Panic reports on
    `ml-explore/mlx#3186` pin the fault to this specific kext, so
    recording the driver revision at startup gives forensic context
    for future crash correlation.
  - `log_system_audit_at_startup()` is the convenience entry point
    for CLI `main()` or FastAPI lifespans.

- **R4 — Prefill allocation guard.** New `ModelDims` dataclass, a
  curated `KNOWN_MODELS` table (Gemma 4 family, Mistral Small 3.2,
  Pixtral, Hermes 3 Llama 3.1, LFM2-VL), plus:
  - `estimate_prefill_peak_alloc_gb(context_tokens, dims)` returns
    the larger of the per-layer attention-score tensor and the
    whole-model KV cache. Scores scale quadratically in context; KV
    linearly.
  - `require_prefill_fit(context_tokens, dims, available_gb,
    single_alloc_ceiling_gb=5.0, headroom_pct=0.30)` raises
    `MetalOOMError` before `mlx_lm.load` / `mlx_vlm.load` runs if the
    estimate exceeds either the hard 5 GB single-allocation ceiling
    (IOGPUFamily state corruption risk per mlx#3186) or the available
    memory headroom. The 131 k × Mistral-Small-3.2-24B × 8-bit case
    that caused a real kernel panic on 2026-04-15 estimates ≈ 30 GB
    and is refused.
  - `describe_prefill_plan(context_tokens, model_id, available_gb)`
    returns a dashboard-safe null-tolerant summary.

- **R5 — Per-request KV cumulative tracker.** New `KVGrowthTracker`
  class plus a module-level `kv_tracker` singleton. The existing
  `MetalGuard.start_kv_cache_monitor` watches *global* Metal pressure
  — a long-running request that steadily grows its KV cache can push
  the device past the IOGPUFamily threshold while the global metric
  still looks fine. `kv_tracker.start(request_id, ceiling_gb=…)` /
  `add_bytes(request_id, bytes)` / `finalize(request_id)` catches
  that specific request before the global metric crosses. Opt-in;
  untracked requests are no-ops.

- **R6 — Process-mode detection.** `detect_process_mode()` classifies
  the current process as `server` / `embedded` / `notebook` / `cli` /
  `subprocess_worker`. `apply_mode_defaults(mode)` returns
  mode-specific timeouts, flush intervals, KV ceilings, and prefill
  allocation caps. `subprocess_worker` mode carries
  `skip_process_lock=True` since the parent already owns the
  cross-process lock. Detection uses `METALGUARD_SUBPROCESS_WORKER=1`
  (set automatically by `MLXSubprocessRunner` in the child), then
  argv inspection for `mlx_lm.server` / `uvicorn` / `gunicorn`, then
  `ipykernel` import for notebook, then script-name heuristics.

- **R7 — Chunked-prefill advisory.** `recommend_chunk_size(
  context_tokens, dims, single_alloc_ceiling_gb=4.0)` binary-searches
  the largest chunk whose estimate fits the ceiling. Purely advisory
  — metal-guard does not chunk on behalf of the caller.

- **R8 — Apple Feedback Assistant panic formatter.**
  `format_panic_for_apple_feedback(forensics, include_breadcrumb=True,
  max_breadcrumb_lines=60)` converts a forensics dict into a
  ready-to-paste Feedback Assistant report mirroring the
  `ml-explore/mlx#3186` (FB22091885) template. Null-tolerant for
  missing fields; optional breadcrumb suppression for
  prompt-sensitive content.

### Added (advisories)

Nine new entries in `_VERSION_ADVISORIES`:

- `Blaizzy/mlx-vlm#967` (`<0.4.5`, **critical**) — TurboQuant fused
  quantize race; decode T=1 silent corruption.
- `Blaizzy/mlx-vlm#1016` (`==0.4.4`, high) — `prefill_attention`
  always returns None after #909; silent full dequantize for
  externally-built TQ caches.
- `Blaizzy/mlx-vlm#1011` (`==0.4.4`, high) — Gemma 4 loading fails
  with transformers 5.5.x (`ReasoningEffort` ImportError).
- `Blaizzy/mlx-vlm#943` (`==0.4.4`, **critical**) — Gemma 4
  26b-a4b-it-4bit vision NaN corruption (model-scoped).
- `ml-explore/mlx#3384` (`<=0.31.1`, **critical**) — SDPA numerical
  divergence / token repetition on 4-bit quantised models. Silent
  quality regression in a hot path.
- `ml-explore/mlx-lm#897` (`>=0.31.0,<=0.31.2`, high) —
  `mlx_lm.server` chat completions crash with transformers ≥ 5.0.
- `Blaizzy/mlx-vlm#999` (`==0.4.4`, high) — server clears Metal
  cache after every request, destroying KV prefix cache.
- `ml-explore/mlx#3350` (`<=0.31.2`, high) — Metal allocator buffer
  pool unbounded growth on monotonic-size allocations. Maintainer
  closed won't-fix; mitigation pushed to callers
  (`mx.set_cache_limit` + `mx.clear_cache` on growth thresholds).
- `ml-explore/mlx#3390` / `#3317` / `#3224` (`<=0.31.2`, high) —
  **the 7th kernel-panic root-cause class.** `eval.cpp::check_error`
  throws from Metal completion handlers running on
  `com.Metal.CompletionQueueDispatch` (libdispatch/GCD); blocks on
  that queue are not exception-safe, so `std::terminate` → `abort()`
  → uncatchable SIGABRT. PR #3318 proposed `check_error_deferred`
  and was closed without merge — upstream stance is that process
  state is undefined post-throw and the fix belongs in general
  thread-safety work. metal-guard can only partially mitigate:
  `AGX_RELAX_CDM_CTXSTORE_TIMEOUT=1` (auto-set at module import
  since v0.5.0) reduces the GPU watchdog false-positives that most
  commonly trigger this abort, and subprocess isolation via
  `MLXSubprocessRunner` keeps the parent and sibling workers alive.

### Added (subprocess runner hardening — H7)

`MLXSubprocessRunner.__init__` now calls `acquire_mlx_lock(
f"mlx_subprocess_runner:{model_id}")` before spawning the worker,
and releases the lock through `_cleanup()` on both graceful
shutdown and forced kill. This closes the last gap in cross-process
MLX exclusion: `bench_scoped_load()` and `call_model()` already
acquired the lock, but `MLXSubprocessRunner` did not — any
concurrent MLX acquirer (pytest running `bench_scoped_load`, a
second bench CLI, an acceptance test) could legally overwrite the
lock mid-run while the subprocess was still holding Metal buffers.
This was the root cause of a real kernel panic on 2026-04-15 where
a pytest run inadvertently stole the lock from a running bench.

The worker also now sets `METALGUARD_SUBPROCESS_WORKER=1` in its
own environment so `detect_process_mode()` inside the child returns
`"subprocess_worker"` and picks up the `skip_process_lock=True`
default, preventing the child from trying to re-acquire the lock
the parent already owns.

### Fixed

- `test_returns_empty_for_clean_environment` updated to use
  `mlx-lm 0.31.3` / `mlx 0.32.0` / `mlx-vlm 0.5.0` as its clean-version
  sentinel — hypothetical future releases past every existing
  advisory range.

### Test results

126 passed (82 baseline + 44 new covering advisories, `audit_wired_limit`
with mocked sysctl, prefill estimate/require_fit/recommend_chunk_size,
KVGrowthTracker concurrency and ceiling breach, process mode detection
across argv variants, Apple Feedback formatter section presence and
breadcrumb truncation, and MLXSubprocessRunner lock acquire / release /
refuse).

### Not included (deferred)

- Inflight breadcrumb wrap around `mx.eval()` / `generate()` for
  forensic tagging of which request was in-flight at a completion-handler
  abort. The current subprocess reap path gives `(pid, model_id)`; a
  richer `(pid, model_id, prompt_hash, token_idx, wall_clock)`
  breadcrumb is planned but not yet implemented — await the 7th-class
  panic to become frequent enough to justify the additional write path.
- Stress test reproducing `mlx-lm#883` / `#854` server-mode behaviour.
  R6 provides defaults; actually launching a server to stress-test is
  separate work.

## [0.6.0] — 2026-04-14

### Changed (behaviour)

- **`acquire_mlx_lock(force=True)` is now a hardened reclaim.** Before
  v0.6.0, `force=True` unconditionally overwrote the existing lock file
  and left the previous holder running. In production that routinely
  left two MLX processes loading into the same GPU — the exact
  kernel-panic path
  (`IOGPUMemory.cpp:492 completeMemory() prepare count underflow`
  reachable in seconds). Starting v0.6.0, `force=True`:

  1. Sends `SIGTERM` to the current holder.
  2. Polls up to `MLX_FORCE_WAIT_SEC` (default 30 s) for the holder to
     exit. Zombie-aware: processes in state `Z` are treated as dead
     because Metal buffer release is tied to process exit, not to the
     parent's `wait()` reap.
  3. Unlinks the lock only after confirmed exit.
  4. Sleeps `MLX_RECLAIM_COOLDOWN_SEC` (default 8 s) to let the kernel's
     Metal buffer GC catch up before returning.

  If the holder refuses to exit, `MLXLockConflict` is raised with
  `holder["force_timeout"] = True` and the lock is **deliberately left
  intact** — this is the anti-panic invariant. Unlinking while a live
  peer still holds Metal buffers is exactly what the prior behaviour
  did wrong.

  If `SIGTERM` raises `PermissionError` (for example pid 1, or a peer
  owned by a different user), `MLXLockConflict(force_permission_denied=True)`
  is raised and again the lock is left intact — we cannot guarantee
  the peer released its Metal buffers, so refusing to acquire is the
  safe choice.

### Added

- **`_is_pid_alive` is now zombie-aware.** Helper `_is_zombie(pid)`
  parses `ps -p <pid> -o state=`; a first character of `Z` counts as
  dead. This closes an otherwise-silent livelock in the FORCE wait
  loop: the old check (`os.kill(pid, 0)`) returns success for zombies
  until the parent reaps them, which could be minutes under a busy
  launchd supervisor.

- **`MLXLockConflict.holder` typed failure fields** — callers can now
  distinguish:
  - `holder["force_timeout"]` — SIGTERM delivered, peer did not exit.
  - `holder["force_permission_denied"]` — SIGTERM denied by the OS.
  Both cases leave the lock intact.

- **New env vars** for tuning the FORCE path:
  - `MLX_FORCE_WAIT_SEC` (default 30) — seconds to wait after SIGTERM.
  - `MLX_RECLAIM_COOLDOWN_SEC` (default 8) — post-reclaim Metal buffer
    GC sleep. Set to 0 in tests / tight CI.

- **`check_version_advisories()`** — returns a list of active advisories
  for the `(mlx, mlx-lm, mlx-vlm)` versions installed in the current
  environment, mapped to upstream issue numbers + severity. Purely
  informational; intended for dashboards and startup logs. Initial
  advisories target mlx-lm 0.31.2 regressions:

  - [mlx-lm#1128](https://github.com/ml-explore/mlx-lm/issues/1128) —
    `TokenizerWrapper.think_start_id` crashes when `_think_start_tokens`
    is `None` (`TypeError: object of type 'NoneType' has no len()`).
  - [mlx-lm#1139](https://github.com/ml-explore/mlx-lm/issues/1139) —
    broadcast errors after the second voting round; reproducible
    regression vs 0.31.1.
  - [mlx-lm#1081](https://github.com/ml-explore/mlx-lm/issues/1081) —
    `ArraysCache.is_trimmable()` returns `True` but `trim()` does not
    exist (speculative decoding MTP cache-hit path only).
  - [mlx#3348](https://github.com/ml-explore/mlx/pull/3348) — merged
    2026-04-01 but not yet shipped in a PyPI release; observer-mode
    gate still blocked.

  ```python
  from metal_guard import check_version_advisories
  for a in check_version_advisories():
      print(f"[{a['severity']}] {a['package']} {a['installed_version']} — {a['issue']}")
  ```

- **`install_upstream_defensive_patches()`** — opt-in, version-gated
  monkey-patches for known upstream bugs. Each patch is idempotent,
  logs a WARNING when applied, and auto-skips when the installed
  package version is outside the affected range — so once upstream
  ships a fix this becomes a no-op without any caller change.

  Inaugural patch: `mlx_lm_1128_think_start_id` replaces
  `TokenizerWrapper.think_start_id` with an accessor that returns
  `None` when `_think_start_tokens is None` instead of raising
  `TypeError`. Scoped to mlx-lm `==0.31.2`.

  ```python
  from metal_guard import install_upstream_defensive_patches
  install_upstream_defensive_patches()
  # WARNING metal_guard: installed defensive patch for mlx-lm#1128 …
  ```

### Fixed

- **Version drift between `metal_guard.py::__version__` and
  `pyproject.toml::version`.** v0.5.0 shipped with `pyproject.toml`
  still pinned at 0.4.0, which made `importlib.metadata.version("metal-guard")`
  report the wrong value. Both now read 0.6.0.

### Tests

- 11 new tests (7 FORCE hardening, 4 zombie-aware liveness, 5 version
  advisories, 7 defensive-patches) — total 82/82 passing.

### Upgrade note

`acquire_mlx_lock(force=True)` is behaviourally different from prior
versions: it can now raise `MLXLockConflict` instead of silently
succeeding. Callers that relied on the old "always succeeds" semantics
need to catch `MLXLockConflict` and decide how to handle a stubborn
peer. This is intentional — the old behaviour was the kernel-panic
path.

---

## [0.5.0] — 2026-04-13

### Added

- **Layer 5: `bench_scoped_load`** — context manager for safe sequential
  model loading in long-running benchmark harnesses. Acquires the
  cross-process lock, loads via `mlx_lm.load` / `mlx_vlm.load`, runs
  `safe_cleanup` + 8s cooldown + post-unload memory verification on exit.

  Closes the gap where benchmark loops that bypass this library (calling
  `mlx_lm.load` + `mlx_lm.generate` directly) drift above the working-set
  limit on 64 GB Apple Silicon after 6+ large models and trigger the
  `IOGPUMemory.cpp:492 completeMemory() prepare count underflow` kernel
  panic.

  ```python
  from metal_guard import bench_scoped_load

  for model_id in candidate_models:  # 8+ large models
      with bench_scoped_load(model_id) as (model, tokenizer):
          score = run_eval(model, tokenizer, items)
          save_checkpoint(model_id, score)
  ```

- **Layer 6: Dual-mode switcher** — `current_mode()`, `is_defensive()`,
  `is_observer()`, `describe_mode()` driven by the `METALGUARD_MODE`
  env var. Defensive (default) actively blocks dangerous operations;
  Observer (opt-in) monitors and logs, permitting parallel dispatch.
  Intended for use after [mlx#3348](https://github.com/ml-explore/mlx/pull/3348)
  (CommandEncoder thread-local) ships in a release tag.

  ```bash
  export METALGUARD_MODE=defensive  # default, current behaviour
  export METALGUARD_MODE=observer   # opt-in after #3348 release
  ```

- **Layer 7: Subprocess isolation** — `MLXSubprocessRunner` + auto-managed
  `call_model_isolated()` pool for crash-safe MLX inference. Each model
  runs in its own worker subprocess; if the worker crashes via Metal
  SIGABRT the parent detects the broken pipe and spawns a replacement,
  leaving the main Python/Metal state intact.

  Addresses the class of `mlx::core::gpu::check_error` C++ exceptions
  thrown from Metal's GCD `CompletionQueueDispatch` queue — these
  cannot be caught by Python (they trigger `std::terminate → abort()`),
  so subprocess isolation is the only safe mitigation.

  ```python
  from metal_guard import MLXSubprocessRunner

  runner = MLXSubprocessRunner("mlx-community/Mistral-Small-3.2-24B-8bit")
  for prompt in prompts:
      result = runner.generate(prompt, max_tokens=4096)
  runner.shutdown()
  ```

  Worker includes chat template fallbacks for Mistral / Gemma / Phi
  families when `tokenizer.chat_template` is unset (observed on some
  mlx-community quantized uploads).

- **`MLX_LOCK_PATH` env var** — L8 process lock path is now overridable
  via the `MLX_LOCK_PATH` environment variable. Default unchanged
  (`~/.metal-guard/locks/mlx_exclusive.lock`).

### Summary — complete L1-L8 layered defense

| Layer | Concern | Mechanism |
|---|---|---|
| L1-L4 | Thread races, OOM, stale buffers | `MetalGuard` singleton (in-process) |
| L5 | Sequential big-model load drift | `bench_scoped_load` context manager |
| L6 | Mode switch between defensive/observer | `METALGUARD_MODE` env var |
| L7 | Metal C++ crashes | `MLXSubprocessRunner` + `call_model_isolated` |
| L8 | Cross-process contention | `mlx_exclusive_lock` / `acquire_mlx_lock` |

## [0.4.0] — 2026-04-13

### Added

- **Hardware-aware auto-configuration** — `detect_hardware()` identifies the
  Apple Silicon chip, total GPU memory, and tier (low/mid/high).
  `recommended_config()` returns safe defaults for watchdog thresholds,
  KV cache headroom, cooldown, and max concurrent models — tuned per tier:
  - **low** (8–16 GB, MBA/base MBP): conservative thresholds (warn 60%, critical 75%)
  - **mid** (32–64 GB, Mac Studio/MBP Max): balanced (warn 67%, critical 82%)
  - **high** (96–512 GB, Ultra/Max Pro): relaxed (warn 70%, critical 85%)

  ```python
  config = MetalGuard.recommended_config()
  print(f"{config['chip']} ({config['gpu_memory_gb']}GB) → tier {config['tier']}")
  metal_guard.start_watchdog(
      warn_pct=config["watchdog_warn_pct"],
      critical_pct=config["watchdog_critical_pct"],
  )
  ```

- **KV cache growth monitor** — `start_kv_cache_monitor()` tracks memory
  growth rate over a sliding 5-minute window. Fires `on_pressure` callback
  when available headroom drops below threshold or growth rate exceeds a
  limit (GB/min). Designed for long-running `mlx_lm.server` instances
  where KV cache grows unbounded across conversations.
  Addresses [mlx-lm#1047](https://github.com/ml-explore/mlx-lm/issues/1047)
  (KV cache OOM crash on 512 GB Mac Studio).

  ```python
  metal_guard.start_kv_cache_monitor(
      headroom_gb=8.0,
      growth_rate_warn_gb_per_min=2.0,
      on_pressure=lambda avail, rate: kv_cache.clear(),
  )
  ```

- **Cross-process mutual exclusion (Layer 8)** — `acquire_mlx_lock()`,
  `release_mlx_lock()`, `read_mlx_lock()`, and `mlx_exclusive_lock()` context
  manager. File-based lock at `~/.metal-guard/locks/mlx_exclusive.lock`
  prevents concurrent MLX workloads across process boundaries — the root
  cause of IOGPUMemory kernel panics when `mlx_lm.server`, benchmarks, or
  direct `mlx_lm.generate` calls run simultaneously with other MLX processes.
  Stale locks from crashed processes are self-healing (pid liveness check).
  New `MLXLockConflict` exception raised when a live process already holds
  the lock.

- **GPU watchdog detection** — `is_metal_oom()` now detects
  `kIOGPUCommandBufferCallbackErrorImpactingInteractivity`, the macOS GPU
  watchdog kill that terminates MLX training/inference when command buffers
  block WindowServer display compositing on MacBook.
  Addresses [mlx#3267](https://github.com/ml-explore/mlx/issues/3267).

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
