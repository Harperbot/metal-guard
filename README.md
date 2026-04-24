# MetalGuard

**English** | [繁體中文](README.zh-TW.md) | [日本語](README.ja.md)

GPU safety layer for [MLX](https://github.com/ml-explore/mlx) on Apple Silicon.

Prevents kernel panics and OOM crashes caused by Metal driver bugs when running MLX inference — especially multi-model pipelines, long-running servers, and agent frameworks with heavy tool calling.

**Current version:** v0.9.0 — see [CHANGELOG.md](CHANGELOG.md) for release history and per-feature rationale. v0.9.0 adds `subprocess_inference_guard` (B1), cross-model cadence + gemma-4 90-second floor (C5), `gemma4_generation_flush` (C7), and the `KNOWN_PANIC_MODELS` advisory registry.

## Landed here searching for one of these? You're in the right place.

If your Mac is panicking / rebooting / crashing while running MLX and you searched for any of the strings below, metal-guard is designed for you:

- `IOGPUMemory.cpp:492 completeMemory() prepare count underflow`
- `IOGPUMemory.cpp:550` kernel panic on Apple Silicon under MLX
- `kIOGPUCommandBufferCallbackErrorOutOfMemory`
- `mlx::core::gpu::check_error` → `std::terminate` → `abort` (SIGABRT)
- `mlx::core::metal::GPUMemoryAllocator` / `fPendingMemorySet`
- `IOGPUGroupMemory.cpp:219` pending memory set panic
- `mlx_lm.generate` crashes mid-inference, parent Python process dies
- `mlx_lm.server` OOM kernel panic / Mac reboot under sustained load
- `mlx_vlm` TurboQuant decode T=1 silent corruption (`mlx-vlm#967`)
- `com.apple.iokit.IOGPUFamily` (104.x / 129.x) referenced in a panic report
- `AGX_RELAX_CDM_CTXSTORE_TIMEOUT` mentioned by a maintainer
- `ImpactingInteractivity` / GPU watchdog killing MLX on MacBook
- Gemma 4 / Mistral-Small / Pixtral / Llama 4-bit produces garbage output
- M1 / M2 / M3 / M4 (Max / Ultra / Pro) Mac Studio / MacBook Pro kernel panic
- Long-context (≥ 65 k) prefill in MLX triggers reboot
- `transformers` 5.0 / 5.5 import errors from `mlx_vlm.load`
- Back-to-back MLX model loads cause IOGPU underflow panic

Related upstream tracking: `ml-explore/mlx#3186` / `#3346` / `#3348` / `#3350` / `#3384` / `#3390`, `ml-explore/mlx-lm#883` / `#854` / `#897` / `#1015` / `#1047`, `Blaizzy/mlx-vlm#943` / `#967` / `#999` / `#1011` / `#1016`. metal-guard watches these via `check_version_advisories()` and warns at startup if the installed versions are affected.

## The Problem

Apple's Metal GPU driver on Apple Silicon has a bug: when GPU memory management fails, **the kernel panics the entire machine** instead of gracefully killing the process.

```
panic(cpu 4 caller 0xfffffe0032a550f8):
  "completeMemory() prepare count underflow" @IOGPUMemory.cpp:492
```

This affects any workflow that loads and unloads multiple MLX models in sequence — the Metal driver's internal reference count can underflow, causing an unrecoverable kernel panic that reboots the machine. **This is not your code's fault.** It's a driver-level bug with no fix timeline. See [ml-explore/mlx-lm#883](https://github.com/ml-explore/mlx-lm/issues/883).

### Who is affected

| Workload | Risk | Why |
|----------|------|-----|
| Single-model server (LM Studio) | Low | One model, no switching |
| Multi-model pipeline | **High** | Every load/unload transition can panic |
| Long-running server (`mlx_lm.server`) | **High** | KV cache grows unbounded, Metal buffers accumulate |
| Agent framework + tool calling | **High** | 50–100 short generate() calls per conversation |
| TurboQuant KV cache compression | **High** | Pushes memory closer to the limit |
| 24/7 daemon | **Critical** | Memory drift over days, no natural cleanup point |

## Installation

```bash
pip install metal-guard
```

Single-file drop-in also works — `metal_guard.py` has zero dependencies beyond the Python standard library and optional `mlx`.

## Quick Start

```python
from metal_guard import metal_guard, require_cadence_clear, CircuitBreaker

# 1. Refuse back-to-back loads (L9)
require_cadence_clear("mlx-community/gemma-4-26b-a4b-it-4bit")

# 2. Refuse new workers after repeated panics (L9)
CircuitBreaker().check()

# 3. Register GPU-bound threads
import threading
thread = threading.Thread(target=run_mlx_generate, daemon=True)
thread.start()
metal_guard.register_thread(thread)
thread.join(timeout=120)

# 4. Safe model unloading (L1 + L2)
metal_guard.wait_for_threads()
metal_guard.safe_cleanup()            # gc + flush GPU + cooldown

# 5. OOM-protected inference (L3)
result = metal_guard.oom_protected(generate, model, tokenizer, prompt=p)

# 6. Pre-load pressure check (L4)
metal_guard.ensure_headroom(model_name="my-model-8bit")

# 7. Breadcrumbs for post-mortem forensics
metal_guard.breadcrumb("LOAD: my-model-8bit START")
```

Hardware-aware defaults in one line:

```python
config = MetalGuard.recommended_config()
metal_guard.start_watchdog(
    warn_pct=config["watchdog_warn_pct"],
    critical_pct=config["watchdog_critical_pct"],
)
metal_guard.start_kv_cache_monitor(headroom_gb=config["kv_headroom_gb"])
```

## Features

MetalGuard is organised as **defence layers (L1–L9)** plus a set of
**preventive helpers (R-series)**. Every feature is available from the
single `metal_guard` module. See [CHANGELOG.md](CHANGELOG.md) for when
each one landed and the incident that motivated it.

### L1 — Thread tracking

Register any thread that touches Metal so cleanup can wait for GPU work to finish before calling `mx.clear_cache()`.

| API | What it does |
|---|---|
| `metal_guard.register_thread(thread)` | Add a GPU-bound thread to the registry |
| `metal_guard.wait_for_threads(timeout=None) -> int` | Block until registered threads finish; returns count still alive |

### L2 — Safe cleanup

Ordered cleanup sequence that avoids the “main thread freed while worker thread still generating” race that was the original panic root cause.

| API | What it does |
|---|---|
| `metal_guard.flush_gpu()` | `mx.eval(sync) + mx.clear_cache()` — only safe after `wait_for_threads()` |
| `metal_guard.safe_cleanup()` | Full sequence: wait → `gc.collect` → flush → cooldown |
| `metal_guard.guarded_cleanup()` | Context manager that runs `safe_cleanup()` on exit |
| `kv_cache_clear_on_pressure(available_gb, growth_rate_gb_per_min)` | Ready-made `on_pressure` callback for the KV monitor |

### L3 — OOM recovery

Turn the raw C++ Metal OOM into a catchable Python exception with automatic cleanup and optional retry.

| API | What it does |
|---|---|
| `metal_guard.oom_protected(fn, *args, max_retries=1, **kwargs)` | Run with OOM catch → cleanup → retry |
| `metal_guard.oom_protected_context()` | Context-manager variant |
| `metal_guard.is_metal_oom(exc) -> bool` | Classify an arbitrary exception |
| `MetalOOMError` | Catchable exception, carries `MemoryStats` |

### L4 — Pre-load memory check

Refuse loads that will not fit, with model-size estimation from the HF model ID.

| API | What it does |
|---|---|
| `metal_guard.can_fit(model_size_gb, overhead_gb=2.0) -> bool` | Non-raising check |
| `metal_guard.require_fit(model_size_gb, model_name, overhead_gb=2.0)` | Clean up then raise `MemoryError` if it still won't fit |
| `MetalGuard.estimate_model_size_from_name(name)` *(static)* | Parse param count + quantisation → GB estimate |

### L5 — Long-running process safety

For `mlx_lm.server`, agent frameworks, and 24/7 daemons.

| API | What it does |
|---|---|
| `metal_guard.memory_stats() -> MemoryStats` | Snapshot (active / peak / limit / available / pct) |
| `metal_guard.is_pressure_high(threshold_pct=67.0) -> bool` | Quick pressure check |
| `metal_guard.ensure_headroom(model_name, threshold_pct=67.0)` | Clean up if pressure high, no-op otherwise |
| `metal_guard.log_memory(label, model_name)` | Log without cleanup |
| `metal_guard.start_periodic_flush(interval_secs=300)` | Background timer flush |
| `metal_guard.start_watchdog(interval_secs, warn_pct, critical_pct, on_critical)` | Drift watchdog with escalating response |
| `metal_guard.start_kv_cache_monitor(interval_secs, headroom_gb, growth_rate_warn, on_pressure)` | KV growth monitor, fire before OOM |
| `bench_scoped_load(model_id, ...)` | Context manager for sequential benchmark runs — guarantees unload before next load |

### L6 — Dual-mode switcher

Runtime-selectable defensive vs observer posture so you can A/B upstream mitigations without changing code.

| API | What it does |
|---|---|
| `current_mode() -> str` | `"defensive"` (default) or `"observer"` |
| `is_defensive() / is_observer() -> bool` | Convenience predicates |
| `describe_mode() -> dict` | Mode name, description, env var |

### L7 — Subprocess isolation

Run MLX in a fresh `multiprocessing` child so a kernel-level abort cannot kill the parent.

| API | What it does |
|---|---|
| `MLXSubprocessRunner(model_id, ...)` | Persistent worker subprocess, respawns on crash |
| `call_model_isolated(model_id, prompt, ...)` | One-shot helper: spawn → generate → shut down |
| `shutdown_all_workers()` | Force-terminate any runners tracked at exit |
| `SubprocessCrashError / SubprocessTimeoutError` | Typed failures for callers |

### L8 — Cross-process mutual exclusion

File lock under `MLX_LOCK_PATH` so bench / server / pipeline never initialise Metal on the same box simultaneously.

| API | What it does |
|---|---|
| `acquire_mlx_lock(label, force=False)` | Raise `MLXLockConflict` if held; `force=True` SIGTERMs the holder with timeout + cooldown |
| `release_mlx_lock() -> bool` | Release if this process holds it |
| `read_mlx_lock() -> dict \| None` | Non-blocking inspect; self-heals stale + zombie holders |
| `mlx_exclusive_lock(label)` | Context manager: acquire on enter, release on exit |

### L9 — Cadence, panic ingest, and circuit breaker *(v0.8.0)*

Last line of defence after the first eight layers. Written in response to a kernel panic that lived *below* the SIGABRT layer — by the time Python saw anything, the machine had already rebooted. The only fix was to avoid the panic trigger in the first place.

| API | What it does |
|---|---|
| `CadenceGuard(path=None, *, min_interval_sec=180)` | Persisted per-model load-timestamp store |
| `CadenceGuard.check(model_id)` / `.mark_load(model_id)` | Raise `CadenceViolation` if another load happened too recently |
| `require_cadence_clear(model_id, *, min_interval_sec=180)` | Atomic check + mark helper |
| `parse_panic_reports(directory=None, *, since_ts=None)` | Scan `/Library/Logs/DiagnosticReports/*.panic` and classify |
| `ingest_panics_jsonl(*, report_dir=None, jsonl_path=None) -> int` | Dedupe-append to `~/.cache/metal-guard/panics.jsonl` |
| `CircuitBreaker(*, window_sec=3600, panic_threshold=2, cooldown_sec=3600)` | Refuse new workers after a panic cluster |
| `CircuitBreaker.check() / .status() / .clear()` | Gate, dashboard, operator override |
| `detect_panic_signature(text) -> (name, explanation)` | Classify a panic log into `prepare_count_underflow` / `pending_memory_set` / `ctxstore_timeout` / `metal_oom` |

### Hardware awareness

| API | What it does |
|---|---|
| `MetalGuard.detect_hardware() -> dict` *(static)* | Chip, GPU memory, recommended working set, tier, IOGPUFamily kext version |
| `MetalGuard.recommended_config() -> dict` *(classmethod)* | Safe defaults for every L-layer on the detected hardware |

### Version advisories & upstream patches

| API | What it does |
|---|---|
| `check_version_advisories(packages=None) -> list[dict]` | Warn if installed `(mlx, mlx-lm, mlx-vlm, transformers)` versions trip a known advisory |
| `install_upstream_defensive_patches(force=False) -> dict[str, bool]` | Idempotent, version-gated monkey-patches for known upstream regressions |

### System audits

| API | What it does |
|---|---|
| `audit_wired_limit() -> dict` | Flag dangerous `iogpu.wired_limit_mb` overrides (mlx-lm#1047) |
| `read_gpu_driver_version() -> str \| None` | IOGPUFamily kext version (mlx#3186) |
| `log_system_audit_at_startup() -> dict` | Convenience wrapper for CLI / FastAPI lifespan |

### R-series preventive helpers

| API | What it does |
|---|---|
| `ModelDims`, `lookup_dims(model_id)`, `KNOWN_MODELS` | GQA-aware dimension lookup for curated models |
| `estimate_prefill_peak_alloc_gb(context_tokens, dims)` | Conservative per-layer + full-KV upper bound |
| `require_prefill_fit(context_tokens, dims, available_gb, ...)` | Raise `MetalOOMError` before any 30 GB single-alloc panic |
| `recommend_chunk_size(context_tokens, dims, ...)` | Binary-search advisory chunk size (purely advisory) |
| `describe_prefill_plan(context_tokens, model_id_or_dims, available_gb)` | Dashboard-safe null-tolerant summary |
| `KVGrowthTracker(...).start / add_bytes / finalize / snapshot` | Per-request cumulative KV guard — catches a single runaway request that the global pressure monitor misses |
| `detect_process_mode() -> ProcessMode` | `"server" / "embedded" / "notebook" / "cli" / "subprocess_worker"` |
| `apply_mode_defaults(mode=None) -> dict` | Mode-appropriate timeouts and ceilings |
| `describe_process_mode() -> dict` | Dashboard summary |
| `format_panic_for_apple_feedback(forensics, ...)` | Ready-to-paste Apple Feedback Assistant report |

### Forensics

| API | What it does |
|---|---|
| `metal_guard.breadcrumb(msg)` | Write an fsync'd line to the breadcrumb log (default `logs/metal_breadcrumb.log`) |

## Path defaults

All L9 artifacts use `~/.cache/metal-guard/`:

| File | Purpose | Overridable via |
|---|---|---|
| `~/.cache/metal-guard/cadence.json` | CadenceGuard timestamps | `CadenceGuard(path=...)` |
| `~/.cache/metal-guard/panics.jsonl` | Panic archive | `ingest_panics_jsonl(jsonl_path=...)` / `CircuitBreaker(jsonl_path=...)` |
| `~/.cache/metal-guard/breaker.json` | CircuitBreaker state | `CircuitBreaker(state_path=...)` |

The breadcrumb log defaults to `logs/metal_breadcrumb.log` (relative); override via `MetalGuard(breadcrumb_path=...)`.

## Architecture

```
┌─────────────────────────────────────────────────┐
│            Your Application Code                │
│  Agent loop / Server / Pipeline / Daemon        │
└──────────────────┬──────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────┐
│              MetalGuard                         │
│                                                 │
│  L9 CadenceGuard ──── refuse back-to-back loads │
│  L9 CircuitBreaker ── refuse after panic cluster│
│  L8 Process Lock ──── cross-process exclusion   │
│  L7 Subprocess ────── panic-isolated workers    │
│  L6 Dual mode ─────── defensive / observer      │
│  L5 Watchdogs ─────── memory + KV drift alerts  │
│  L4 Pre-load check ── can_fit / require_fit     │
│  L3 OOM recovery ──── catch + cleanup + retry   │
│  L2 Safe cleanup ──── gc + flush + cooldown     │
│  L1 Thread registry ─ wait before cleanup       │
│  R4 Prefill guard ─── refuse > ceiling prefills │
│  R5 KV tracker ────── per-request KV guard      │
│  R8 Apple Feedback ── forensics formatter       │
└──────────────────┬──────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────┐
│           MLX + Metal Driver                    │
│  ⚠️  Driver bug: panics instead of OOM          │
└─────────────────────────────────────────────────┘
```

## Tested on

- Mac Studio M1 Ultra (64 GB) — 9 kernel panics before MetalGuard, 24 h panic-free after L9 landed
- 10-person batch pipeline: ~90 model load/unload cycles, 994 s, zero crashes
- Models: Mistral-Small-3.2-24B, Phi-4-mini, Gemma-4-26B / 31B, Pixtral-12B, LFM2-VL-3B (4-bit and 8-bit)

## Known affected models (v0.9.0, 2026-04)

Some models have a race window wide enough that MetalGuard narrows it but does not close it. When that happens we record the model here so you can make an informed decision before loading it in production.

### `mlx-community/gemma-4-31b-it-8bit` — repeat offender

Two production kernel panics on Harper's Mac Studio, **24 hours apart, same pipeline, same model**, same panic signature `IOGPUMemory.cpp:492 "completeMemory() prepare count underflow"`:

| #   | Local time       | PID   | Spawn → panic | Context                                            |
|-----|------------------|-------|--------------:|----------------------------------------------------|
|  7  | 2026-04-23 03:14 | 67840 |        ~6 min | rezivot pipeline, no cross-model cadence wired yet |
| 11  | 2026-04-24 03:14 | 26608 |      ~1.5 min | same pipeline as #7; ~1.5 min worker-ready to panic despite classic L9 defences in place |

Community corroboration (all 2026-04):

- [Hannecke — "MLX Crashed My Mac" (Medium)](https://medium.com/@michael.hannecke/how-my-local-coding-agent-crashed-my-mac-and-what-i-learned-about-mlx-memory-management-e0cbad01553c) — M4 Max 64 GB, same signature; pivoted to `Qwen3-Coder-30B-A3B` MoE.
- [`lmstudio-ai/lmstudio-bug-tracker#1740`](https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/1740) "Gemma-4 31b KV excessive KV cache footprint" — confirms the hybrid-attention KV explosion: 26 GB VRAM for 8192 context; hybrid attention (50 sliding + 10 global) KV cache plus 8-bit weights (~34 GB) plus full-context KV pushes a 64 GB Mac over the edge.
- [`ml-explore/mlx-lm#883`](https://github.com/ml-explore/mlx-lm/issues/883) — M3 Ultra 96 GB, same signature.
- [`ml-explore/mlx#3186` (comment, 2026-04-24)](https://github.com/ml-explore/mlx/issues/3186#issuecomment-4314204974) — independent third-party data point: Mac mini M4 base 32 GB, macOS 26.4.1 (`25E253`), mlx 0.31.2, `mlx-community/Qwen3.6-35B-A3B-4bit`. Panic 8 min 16 s after `mlx_lm.server` start; `--prompt-cache-bytes 8 GiB` did not prevent it; the reporter adopted `llama.cpp` for production serving. Explicitly references this project's "two-trigger-path hypothesis."

**Bottom line.** macOS 26.4.x has not fixed the bug. macOS 26.5 beta has not fixed the bug. Adding RAM to 96 GB did not prevent it. MetalGuard v0.9.0 narrows the race windows (cross-model cadence, gemma-4 90-second floor, first-generate flush, subprocess inference guard) but does not eliminate panic on this model in Harper's workload.

You can query the advisory programmatically:

```python
from metal_guard import check_known_panic_model, warn_if_known_panic_model

advisory = check_known_panic_model(model_id)
if advisory is not None:
    # Decide: refuse load, switch backend, or proceed with explicit ack.
    ...

# Or fire-and-forget: idempotent, emits one log.warning per process per model.
warn_if_known_panic_model(model_id)
```

## When MetalGuard is not enough

If you engage every v0.9.0 defence (B1 + C5 + C7 + CircuitBreaker) and still observe repeat panics on the same model, that is a signal that the race window is wider than a userspace layer can narrow. Two escape hatches, in order of ROI:

1. **Switch backend.** [Ollama](https://ollama.com/) and [`llama.cpp`](https://github.com/ggml-org/llama.cpp) both use Metal MPS under the hood but run a persistent worker architecture that sidesteps the subprocess teardown race entirely. Harper's `harper-finance` project migrated to Ollama on 2026-04-23 and has run zero-panic since. The independent M4-base reporter on [`mlx#3186`](https://github.com/ml-explore/mlx/issues/3186#issuecomment-4314204974) made the same call for production serving. You lose some raw throughput (MLX was measured 30–55 % faster on prefill in that report); you gain "doesn't panic the machine."

2. **Pivot to a different model family.** Mixture-of-Experts (MoE) variants — e.g. [`mlx-community/gemma-4-26b-a4b-it-4bit`](https://huggingface.co/mlx-community/gemma-4-26b-a4b-it-4bit), `Qwen3-Coder-30B-A3B` — have a much smaller active-parameter footprint per forward pass and a narrower KV growth trajectory. Community reports (Hannecke, lmstudio#1740) converge on MoE as the most reliable same-ecosystem workaround.

MetalGuard is **complementary to both escape hatches** — `subprocess_inference_guard` is useful even under Ollama if you spawn per-request subprocess workers, and `CadenceGuard` still helps regardless of backend when you hot-swap models.

### One hard-learned SOP note

Panic #10 in our timeline (see [CHANGELOG](CHANGELOG.md)) was triggered by an *interactive* `python -c "import sentence_transformers"` on the host terminal — a version-verification command, not any production MLX workload. Anything that imports `torch`, `mlx`, `mlx_lm`, `mlx_vlm`, `sentence_transformers`, `transformers`, `diffusers`, or `accelerate` initialises the Metal MPS backend and can walk into the same kernel bug at process exit. During an active panic cooldown, prefer:

- `pip show <pkg>` for version info, or
- `python -c "import importlib.metadata as m; print(m.version('<pkg>'))"` which does not cascade-import the package.

**Never** run `python -c "import <ml-package>; print(<ml-package>.__version__)"` while a cooldown is active.

## Limitations — this is a workaround, not a fix

MetalGuard is a **userspace defensive layer**. The root bug lives inside Apple's IOGPUFamily kext ([mlx#3186](https://github.com/ml-explore/mlx/issues/3186)) and cannot be patched from Python. What MetalGuard actually does:

1. **Lowers the trigger rate** — L1–L5 and L9 CadenceGuard avoid the known trigger paths (back-to-back loads, thread-race cleanup, unbounded KV growth, prefill > single-alloc ceiling).
2. **Contains the blast radius** — L7 runs MLX in a subprocess so a catchable abort kills only the child. A *kernel* panic still reboots the whole machine; the subprocess isolation just means you know which model was holding the GPU when it happened.
3. **Prevents post-reboot cascades** — L9 CircuitBreaker refuses new worker spawns after ≥ 2 panics in a rolling hour, so the machine doesn't immediately reload the same model and replay the panic.

Panics are still possible (especially [mlx#3390](https://github.com/ml-explore/mlx/issues/3390) — the uncatchable completion-handler abort that dispatches on `com.Metal.CompletionQueueDispatch` before any Python signal handler can fire). Harper's box went from ~1.4 panics/day to zero in a 24 h window after L9 landed, but that is risk-reduction, not elimination. Until Apple ships a fixed kext, this is the upper bound of what a Python-side layer can do.

## Related upstream issues

| Issue | Problem | Feature |
|---|---|---|
| [mlx#3186](https://github.com/ml-explore/mlx/issues/3186) | IOGPUFamily kernel panic (canonical) | L1/L2/L8/L9 + `read_gpu_driver_version` |
| [mlx#3346](https://github.com/ml-explore/mlx/issues/3346) | `fPendingMemorySet` second signature | `detect_panic_signature` + L9 |
| [mlx#3348](https://github.com/ml-explore/mlx/pull/3348) | CommandEncoder thread-local | Advisory-gated observer mode |
| [mlx#3350](https://github.com/ml-explore/mlx/issues/3350) | MetalAllocator buffer-pool growth | Advisory + `mx.set_cache_limit` guidance |
| [mlx#3384](https://github.com/ml-explore/mlx/issues/3384) | 4-bit SDPA numerical divergence | `check_version_advisories` |
| [mlx#3390](https://github.com/ml-explore/mlx/issues/3390) | Uncatchable completion-handler abort | L7 subprocess isolation + `AGX_RELAX_CDM_CTXSTORE_TIMEOUT` |
| [mlx-lm#883](https://github.com/ml-explore/mlx-lm/issues/883) / [#1015](https://github.com/ml-explore/mlx-lm/issues/1015) | Kernel panic from KV cache growth | L1 thread + L2 safe cleanup |
| [mlx-lm#854](https://github.com/ml-explore/mlx-lm/issues/854) | Server OOM crash | L3 `oom_protected` + L5 periodic flush |
| [mlx-lm#897](https://github.com/ml-explore/mlx-lm/issues/897) | `mlx_lm.server` crash with transformers ≥ 5.0 | `check_version_advisories` |
| [mlx-lm#1047](https://github.com/ml-explore/mlx-lm/issues/1047) | `wired_limit` correlation with panics | `audit_wired_limit` |
| [mlx-lm#1128](https://github.com/ml-explore/mlx-lm/issues/1128) | `TokenizerWrapper.think_start_id` crash | `install_upstream_defensive_patches` |
| [mlx-vlm#943](https://github.com/Blaizzy/mlx-vlm/issues/943) / [#967](https://github.com/Blaizzy/mlx-vlm/pull/967) / [#999](https://github.com/Blaizzy/mlx-vlm/issues/999) | TurboQuant / cache-thrash / Gemma4 garbage | `check_version_advisories` |

## License

MIT
