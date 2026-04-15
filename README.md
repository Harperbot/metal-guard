# MetalGuard

**English** | [繁體中文](README.zh-TW.md) | [日本語](README.ja.md)

GPU safety layer for [MLX](https://github.com/ml-explore/mlx) on Apple Silicon.

Prevents kernel panics and OOM crashes caused by Metal driver bugs when running MLX inference — especially multi-model pipelines, long-running servers, and agent frameworks with heavy tool calling.

**Current version:** v0.7.0 — see [CHANGELOG.md](CHANGELOG.md) for the full release history.

## Landed here searching for one of these? You're in the right place.

If your Mac is panicking / rebooting / crashing while running MLX and you
searched for any of the strings below, metal-guard is designed for you:

- `IOGPUMemory.cpp:492 completeMemory() prepare count underflow`
- `IOGPUMemory.cpp:550` kernel panic on Apple Silicon under MLX
- `kIOGPUCommandBufferCallbackErrorOutOfMemory`
- `mlx::core::gpu::check_error` → `std::terminate` → `abort` (SIGABRT)
- `mlx::core::metal::GPUMemoryAllocator` / `fPendingMemorySet`
- `mlx_lm.generate` crashes mid-inference, parent Python process dies
- `mlx_lm.server` OOM kernel panic / Mac reboot under sustained load
- `mlx_vlm` TurboQuant decode T=1 silent corruption (`mlx-vlm#967`)
- `com.apple.iokit.IOGPUFamily` (104.x / 129.x) referenced in a panic report
- `AGX_RELAX_CDM_CTXSTORE_TIMEOUT` mentioned by a maintainer
- Gemma 4 / Mistral-Small / Pixtral / Llama 4-bit produces garbage output
- M1 / M2 / M3 / M4 (Max / Ultra / Pro) Mac Studio / MacBook Pro kernel panic
- Long-context (≥ 65 k) prefill in MLX triggers reboot
- `transformers` 5.0 / 5.5 import errors from `mlx_vlm.load`

Related upstream tracking issues: `ml-explore/mlx#3186` / `#3346` / `#3390` /
`#3348`, `ml-explore/mlx-lm#883` / `#854` / `#1047` / `#1015`,
`Blaizzy/mlx-vlm#967` / `#943` / `#1011` / `#1016`. metal-guard watches
these via `check_version_advisories()` and warns at startup if the
versions installed in the environment are affected.

## The Problem

Apple's Metal GPU driver on Apple Silicon has a bug: when GPU memory management fails, **the kernel panics the entire machine** instead of gracefully killing the process.

```
panic(cpu 4 caller 0xfffffe0032a550f8):
  "completeMemory() prepare count underflow" @IOGPUMemory.cpp:492
```

This affects any workflow that loads and unloads multiple MLX models in sequence — the Metal driver's internal reference count can underflow, causing an unrecoverable kernel panic that reboots the machine.

**This is not your code's fault.** It's a driver-level bug with no fix timeline. See [ml-explore/mlx-lm#883](https://github.com/ml-explore/mlx-lm/issues/883).

### Root Causes

We identified two trigger paths through crash forensics (breadcrumb logs + 9 kernel panics on M1 Ultra):

1. **Daemon thread race condition** — `mlx_lm.generate()` runs in a daemon thread holding GPU buffers. If `mx.clear_cache()` is called before the thread finishes, Metal tries to free buffers that are still in use → reference count underflow → kernel panic.

2. **Unconditional Metal initialization** — Calling `mx.eval()` or `mx.clear_cache()` when no models are loaded still initializes the Metal driver. If the driver is in an unstable state from prior crashes, this alone can trigger a panic.

### Who Is Affected

| Workload | Risk | Why |
|----------|------|-----|
| Single-model server (LM Studio) | Low | One model, no switching |
| Multi-model pipeline | **High** | Load → unload → load → unload, each transition can panic |
| Long-running server (mlx_lm.server) | **High** | KV cache grows unbounded, Metal buffers accumulate over hours |
| Agent framework + tool calling | **High** | 50-100 short generate() calls per conversation, fragmented Metal buffers accumulate |
| TurboQuant KV cache compression | **High** | Pushes memory closer to limits (50K-200K tokens), OOM more likely |
| 24/7 daemon (OpenClaw-style) | **Critical** | Memory drift over days, no natural cleanup point |

## Installation

```bash
pip install metal-guard
```

Or copy `metal_guard.py` into your project — it's a single file with no dependencies beyond the Python standard library.

## Quick Start

```python
from metal_guard import metal_guard

# 1. Register GPU-bound threads
import threading

thread = threading.Thread(target=run_mlx_generate, daemon=True)
thread.start()
metal_guard.register_thread(thread)  # Track it
thread.join(timeout=120)

# 2. Safe model unloading
def unload_model(cache):
    metal_guard.wait_for_threads()  # Wait for GPU work to finish
    had_models = bool(cache)
    cache.clear()
    if had_models:
        metal_guard.safe_cleanup()  # gc + flush GPU + cooldown

# 3. Check memory pressure before loading
metal_guard.ensure_headroom(model_name="my-model-8bit")
model, tokenizer = mlx_lm.load("my-model-8bit")

# 4. Breadcrumbs for crash forensics
metal_guard.breadcrumb("LOAD: my-model-8bit START")
```

## v0.7.0 Features

### The 7th kernel-panic root-cause class — completion-handler abort

Harper's 2026-04-16 survey of every MLX / mlx-lm / mlx-vlm issue filed
since mlx-lm 0.31.2 (~250 issues + 80 PRs) surfaced a root cause that
metal-guard could not previously name: `eval.cpp::check_error` throws
from `addCompletedHandler(...)` callbacks running on Apple's
`com.Metal.CompletionQueueDispatch` (GCD) queue. libdispatch blocks
are not exception-safe — `__cxa_throw` → `std::terminate` → `abort()`
→ uncatchable SIGABRT. Python's `try/except` around `mx.eval()` never
fires. Duplicate reports: `mlx#3224` (M3 Ultra 6 hr), `mlx#3317`
(M2 Ultra asyncio race). Umbrella: `mlx#2670`. PR #3318
(`check_error_deferred`) was closed without merge — upstream stance
is that process state is undefined post-throw.

metal-guard can only **partially mitigate** this class:

- `AGX_RELAX_CDM_CTXSTORE_TIMEOUT=1` is auto-set at module import
  (since v0.5.0), which reduces the GPU watchdog false-positives that
  most commonly trigger the abort.
- Subprocess isolation via `MLXSubprocessRunner` keeps the parent and
  sibling workers alive; only the in-flight child dies. The in-flight
  request itself is lost.

This is now called out explicitly in `_VERSION_ADVISORIES` with
severity `high` and full mitigation notes.

### R4 — Prefill allocation guard

```python
from metal_guard import (
    ModelDims, KNOWN_MODELS, require_prefill_fit, recommend_chunk_size,
)

dims = KNOWN_MODELS["Mistral-Small-3.2-24B"]
require_prefill_fit(
    context_tokens=131_072, dims=dims, available_gb=60.0,
)
# MetalOOMError: Prefill peak alloc 30.1 GB > single-alloc ceiling
# 5.0 GB. IOGPUFamily state corruption risk (mlx#3186).
```

Attention score tensors scale quadratically in context. Mistral-Small
24B × 131 k estimates ≈ 30 GB in a single Metal dispatch — well past
the empirical ~5 GB single-allocation ceiling where IOGPUFamily
starts corrupting state even when the device has 60+ GB free. A real
2026-04-15 kernel panic had exactly this shape; R4 refuses the load
before `mlx_lm.load` runs.

`recommend_chunk_size(...)` binary-searches the largest chunk that
fits — advisory only, metal-guard does not auto-chunk.

### R5 — Per-request KV cumulative tracker

```python
from metal_guard import kv_tracker

kv_tracker.start(request_id, ceiling_gb=10.0)
try:
    for tok in generate(...):
        kv_tracker.add_bytes(request_id, bytes_this_step)
        yield tok
finally:
    kv_tracker.finalize(request_id)
```

`MetalGuard.start_kv_cache_monitor` watches *global* Metal pressure
— a long-running request that steadily grows its KV cache can push
the device past the IOGPUFamily threshold while the global metric
still looks fine. Per-request tracking catches that specific request
early. Opt-in; untracked requests are no-ops.

### R6 — Process-mode detection

```python
from metal_guard import detect_process_mode, apply_mode_defaults

mode = detect_process_mode()  # "server" | "cli" | "subprocess_worker" | ...
cfg = apply_mode_defaults(mode)
# {'mode': 'server', 'generate_timeout_sec': 60.0, 'kv_ceiling_gb': 10.0, ...}
```

`server` mode is stricter (60 s generate timeout, 10 GB KV ceiling,
4 GB prefill ceiling) than `notebook` (600 s / 30 GB / 5 GB).
mlx-lm `#883` / `#854` clustered panic reports around long-lived
server loops that never flushed between concurrent requests — the
stricter defaults address that class.

### R8 — Apple Feedback Assistant formatter

```python
from metal_guard import format_panic_for_apple_feedback

report = format_panic_for_apple_feedback(forensics_dict)
# Paste directly into Feedback Assistant.
```

Mirrors the `mlx#3186` (FB22091885) template. Null-tolerant for
missing fields; optional breadcrumb suppression when forensics carry
prompt-sensitive content.

### H7 — MLXSubprocessRunner now acquires the MLX lock

`MLXSubprocessRunner.__init__` calls `acquire_mlx_lock(...)` before
spawning the worker and releases it on both graceful shutdown and
forced kill. This closes the last gap in cross-process MLX exclusion:
`bench_scoped_load()` and `call_model()` already acquired the lock,
but the subprocess runner did not — any concurrent MLX acquirer
could legally overwrite the lock mid-run while the worker was still
holding Metal buffers. Real kernel panic on 2026-04-15 had this
exact shape: a pytest run stole the lock from a running bench.

The worker also now sets `METALGUARD_SUBPROCESS_WORKER=1` in its own
environment so `detect_process_mode()` returns `"subprocess_worker"`
inside the child, preventing double lock acquisition.

### System-level audits (R2 / R3) — now public API

- `audit_wired_limit()` — `sysctl iogpu.wired_limit_mb` > 85 % triggers
  an advisory per `mlx-lm#1047`.
- `read_gpu_driver_version()` — `IOGPUFamily` kext bundle version for
  forensic correlation with `mlx#3186`.
- `log_system_audit_at_startup()` — convenience entry point.

---

## v0.6.0 Features

### Hardened `acquire_mlx_lock(force=True)` — incident-driven

Before v0.6.0, `force=True` unconditionally overwrote the lock file and left the previous holder running. In production that regularly left two MLX processes loading into the same GPU — the exact kernel-panic path (`IOGPUMemory.cpp:492 completeMemory() prepare count underflow` reachable in seconds). v0.6.0 hardens the reclaim:

```python
from metal_guard import acquire_mlx_lock, release_mlx_lock, MLXLockConflict

try:
    acquire_mlx_lock("rescuer", force=True)
    # → SIGTERM the holder, poll up to MLX_FORCE_WAIT_SEC (default 30s)
    #   for exit (zombie-aware), then sleep MLX_RECLAIM_COOLDOWN_SEC
    #   (default 8s) to let Metal buffer GC finish.
except MLXLockConflict as e:
    if e.holder.get("force_timeout"):
        print("Peer refuses to exit — lock deliberately left intact.")
    elif e.holder.get("force_permission_denied"):
        print("SIGTERM denied (e.g. different user) — cannot guarantee buffer release.")
    # In both cases the lock file is NOT removed — that is the anti-panic invariant.
finally:
    release_mlx_lock()
```

- **Zombie-aware liveness**: `_is_pid_alive` parses `ps -p <pid> -o state=` and treats `Z` (zombie) as dead — zombies have already released Metal buffers, otherwise the FORCE wait loop would livelock until the parent reaper caught up.
- **New env vars**: `MLX_FORCE_WAIT_SEC` (default 30), `MLX_RECLAIM_COOLDOWN_SEC` (default 8). Set to 0 in tests / tight CI.
- **Typed failure fields** on `MLXLockConflict.holder`: `force_timeout` and `force_permission_denied` so callers can branch without parsing error strings.

**Upgrade note**: callers that relied on the old "always succeeds" semantics must now catch `MLXLockConflict`. This is intentional — the old behaviour was the kernel-panic path.

### Version Advisory System

`check_version_advisories()` returns a list of active advisories for the `(mlx, mlx-lm, mlx-vlm)` versions installed in the current environment, mapped to upstream issue numbers + severity. Purely informational; intended for dashboards and startup logs.

```python
from metal_guard import check_version_advisories

for a in check_version_advisories():
    print(f"[{a['severity']}] {a['package']} {a['installed_version']} — {a['title']}")
    print(f"    {a['url']}")
```

Initial coverage targets mlx-lm 0.31.2 regressions and the #3348 gate:

| Issue | Affected | Severity | Symptom |
|-------|----------|----------|---------|
| [mlx-lm#1128](https://github.com/ml-explore/mlx-lm/issues/1128) | mlx-lm `==0.31.2` | high | `TokenizerWrapper.think_start_id` `len(None)` crash |
| [mlx-lm#1139](https://github.com/ml-explore/mlx-lm/issues/1139) | mlx-lm `==0.31.2` | high | Broadcast errors after second voting round |
| [mlx-lm#1081](https://github.com/ml-explore/mlx-lm/issues/1081) | mlx-lm `==0.31.2` | medium | `ArraysCache.is_trimmable()` but `trim()` missing (speculative decoding only) |
| [mlx#3348](https://github.com/ml-explore/mlx/pull/3348) | mlx `<0.31.2` | info | CommandEncoder thread-local fix merged 2026-04-01, not yet in a PyPI release — observer-mode gate still blocked |

### Upstream Defensive Patches

`install_upstream_defensive_patches()` installs narrow, version-gated, idempotent monkey-patches for known upstream bugs. Each patch logs a WARNING when applied and auto-skips when the installed package version is outside the affected range — once upstream ships a fix, this call becomes a no-op without any caller change.

```python
from metal_guard import install_upstream_defensive_patches

status = install_upstream_defensive_patches()
# → {'mlx_lm_1128_think_start_id': True}   if mlx-lm 0.31.2 is installed
# → {'mlx_lm_1128_think_start_id': False}  on other versions (nothing to patch)
```

Inaugural patch: **`mlx_lm_1128_think_start_id`** replaces `TokenizerWrapper.think_start_id` with an accessor that returns `None` when `_think_start_tokens is None` instead of raising `TypeError`. Scoped to mlx-lm `==0.31.2`.

## v0.5.0 Features

### Layer 5: `bench_scoped_load` — Sequential Benchmark Guard

Context manager for safely loading many large MLX models in a single Python process. Closes the gap where benchmark harnesses that bypass MetalGuard (calling `mlx_lm.load` + `mlx_lm.generate` directly in a loop) drift above the working-set limit on 64 GB Apple Silicon after 6+ large models and trigger an `IOGPUMemory.cpp:492 completeMemory() prepare count underflow` kernel panic.

```python
from metal_guard import bench_scoped_load

for model_id in candidate_models:  # 8+ large models
    with bench_scoped_load(model_id) as (model, tokenizer):
        score = run_eval(model, tokenizer, items)
        save_checkpoint(model_id, score)
```

Every entry acquires the cross-process lock and loads fresh via `mlx_lm.load` / `mlx_vlm.load`. Every exit runs `safe_cleanup` + 8s cooldown + post-unload memory verification. Metal's lazy page reclaimer only returns pages when the NEXT load() allocates — `bench_scoped_load` routes every iteration through the defensive stack.

### Layer 6: Dual-Mode Switcher

Switch between `defensive` (default) and `observer` modes via the `METALGUARD_MODE` environment variable. Intended for use after [mlx#3348](https://github.com/ml-explore/mlx/pull/3348) (CommandEncoder thread-local) ships in a release tag.

```bash
export METALGUARD_MODE=defensive  # default, actively blocks dangerous operations
export METALGUARD_MODE=observer   # opt-in after #3348; monitors + logs only
```

```python
from metal_guard import current_mode, is_observer, describe_mode

if is_observer():
    # parallel dispatch is permitted
    pass

print(describe_mode())
# {'mode': 'defensive', 'description': '...', 'env_var': '...'}
```

The five long-term primitives (`safe_cleanup`, thread registry, `oom_protected_context`, breadcrumb logging, `memory_stats`) remain active in BOTH modes — they address concerns orthogonal to thread safety.

### Layer 7: Subprocess Isolation

`MLXSubprocessRunner` + auto-managed `call_model_isolated()` pool for crash-safe MLX inference. Addresses `mlx::core::gpu::check_error` C++ exceptions thrown from Metal's GCD `CompletionQueueDispatch` queue — these cannot be caught by Python (they trigger `std::terminate → abort()`), so subprocess isolation is the only safe mitigation.

```python
from metal_guard import MLXSubprocessRunner

runner = MLXSubprocessRunner("mlx-community/Mistral-Small-3.2-24B-8bit")
for prompt in prompts:
    result = runner.generate(prompt, max_tokens=4096)
runner.shutdown()
```

Or drop-in replace a direct call with the auto-managed pool:

```python
from metal_guard import call_model_isolated

# Automatically creates + reuses a worker per model; respawns on crash
result = call_model_isolated(prompt, model="mlx-community/Phi-4-mini-4bit")
```

Worker includes chat template fallbacks for Mistral (`[INST]`), Gemma (`<start_of_turn>`), and Phi families when `tokenizer.chat_template` is unset (observed on some mlx-community quantized uploads).

### `MLX_LOCK_PATH` Configurability

The L8 cross-process lock file path is now overridable via the `MLX_LOCK_PATH` environment variable (default `~/.metal-guard/locks/mlx_exclusive.lock`).

## v0.4.0 Features

### Hardware-Aware Auto-Configuration

Different Apple Silicon machines need different safety thresholds. An 8GB MacBook Air can't use the same settings as a 512GB Mac Studio. MetalGuard now detects your hardware and recommends appropriate values.

```python
from metal_guard import MetalGuard

config = MetalGuard.recommended_config()
print(f"{config['chip']} ({config['gpu_memory_gb']}GB) → tier: {config['tier']}")
# Apple M1 Ultra (64.0GB) → tier: mid

# Use recommended values directly
metal_guard.start_watchdog(
    warn_pct=config["watchdog_warn_pct"],
    critical_pct=config["watchdog_critical_pct"],
)
```

| Tier | Memory | Warn | Critical | Max Models |
|------|--------|------|----------|------------|
| low | 8–16 GB | 60% | 75% | 1 |
| mid | 32–64 GB | 67% | 82% | 2 |
| high | 96–512 GB | 70% | 85% | 3 |

### KV Cache Growth Monitor

For long-running servers where KV cache grows unbounded across conversations. Tracks memory growth rate over a sliding window and fires a callback before OOM. Addresses [mlx-lm#1047](https://github.com/ml-explore/mlx-lm/issues/1047).

```python
def handle_pressure(available_gb, growth_rate):
    log.warning("KV pressure: %.1fGB free, growing %.1fGB/min", available_gb, growth_rate)
    kv_cache.clear()

metal_guard.start_kv_cache_monitor(
    interval_secs=30,
    headroom_gb=8.0,
    growth_rate_warn_gb_per_min=2.0,
    on_pressure=handle_pressure,
)
```

### TurboQuant / Mixed-Precision Estimator Support

`estimate_model_size_from_name()` now correctly handles TurboQuant (TQ3/TQ4) and Unsloth UD-MLX model naming conventions:

| Pattern | Example | Estimate |
|---------|---------|----------|
| TQ4 | `gemma-4-31b-it-TQ4-MLX` | 15.5 GB |
| TQ3 | `gemma-4-31b-it-TQ3-MLX` | 11.6 GB |
| UD-MLX-4bit | `gemma-4-31b-it-UD-MLX-4bit` | 15.5 GB |

Note: TQ models compress KV cache, allowing much longer contexts (50K-200K tokens). The estimator reports *model weight* footprint only — actual runtime memory depends on context length.

### Cross-Process Mutual Exclusion (Layer 8)

File-based lock that prevents concurrent MLX workloads across process boundaries. This is the root cause of kernel panics when `mlx_lm.server`, benchmarks, or direct `mlx_lm.generate` calls run simultaneously with other MLX processes.

```python
from metal_guard import mlx_exclusive_lock, acquire_mlx_lock, release_mlx_lock

# Context manager (preferred)
with mlx_exclusive_lock("my_script"):
    model, tokenizer = mlx_lm.load("mlx-community/gemma-4-31b-it-8bit")
    result = mlx_lm.generate(model, tokenizer, prompt="Hello")

# Explicit acquire/release
acquire_mlx_lock("my_server")
try:
    serve_forever()
finally:
    release_mlx_lock()

# Check without blocking
from metal_guard import read_mlx_lock
info = read_mlx_lock()  # None if free, dict with pid/label/cmdline if held
```

**Self-healing:** Stale locks from crashed processes are automatically cleaned up via pid liveness checks. No manual cleanup needed after crashes.

**Raises `MLXLockConflict`** with the holder's pid, label, and cmdline when another live process holds the lock — so you get a clear error message instead of a kernel panic.

## v0.3.0 Features

### Pre-generate Metal Health Probe

Verify the Metal command queue is alive before starting a long `generate()` call. If the GPU is in a bad state from a prior crash, this fails at a controlled point (~1ms) instead of mid-inference.

```python
metal_guard.probe_metal_health()  # crash here, not mid-generate
result = generate(model, tokenizer, prompt=prompt)
```

### SIGABRT Signal Handler (Crash Forensics)

MLX's C++ runtime can throw exceptions from Metal's GCD CompletionQueueDispatch queue — a code path Python cannot catch. When this happens, `std::terminate` calls `abort()` and the process dies. This handler writes a final breadcrumb before the crash for post-mortem analysis.

```python
metal_guard.install_abort_handler()  # call once at startup
# ... later, if Metal SIGABRT occurs:
# → breadcrumb written: "SIGABRT: Metal command buffer error detected..."
# → log.critical written
# → process terminates with proper crash report
```

### 6-bit / 3-bit / mxfp4 Estimator Fix

`estimate_model_size_from_name()` now supports mixed-precision and emerging quantization formats:

| Format | Multiplier | Example |
|---|---|---|
| `6bit` | 0.75 | `LFM2-24B-A2B-MLX-6bit` → 18 GB (was incorrectly 48 GB) |
| `3bit` / `int3` | 0.375 | TurboQuant 3-bit KV cache models |
| `mxfp4` | 0.5 | Metal FP4 mixed-precision format |

## v0.2.3 Features

### Escalated Retry in `require_fit` (v0.2.3)

A two-tier retry strategy for tight-memory ensemble workloads. Fixes the
observed OOM path where the standard `safe_cleanup` leaves enough stale
GPU buffers that a large follow-up model still can't fit — particularly
common on M1 Ultra running multi-debater ensembles where each KOL sees
the full cycle mistral-24B → phi-4-mini → gemma-4-26B and the next batch
tries to load mistral-24B again before Metal has returned pages to the OS.

```python
from metal_guard import metal_guard

# Standard call (backward compatible — no escalation):
metal_guard.require_fit(24.0, model_name="Mistral-24B-8bit")

# Escalated retry: drop Python-side references, second cleanup,
# extra cooldown, re-check. Opt-in via escalated_cooldown_sec > 0:
metal_guard.require_fit(
    24.0,
    model_name="Mistral-24B-8bit",
    cache_clear_cb=my_model_cache.clear,
    escalated_cooldown_sec=5.0,
)
```

**How the two tiers work:**

| Tier | Action | When |
|---|---|---|
| 1. Standard | `safe_cleanup()` (wait threads + gc + flush + internal cooldown) | `can_fit` fails on first check |
| 2. Escalated | `cache_clear_cb()` → `safe_cleanup()` → `mlx.reset_peak_memory()` → `sleep(escalated_cooldown_sec)` → re-check | Tier 1 still insufficient AND caller opted in |

The escalation is **opt-in** because MetalGuard has no knowledge of the
caller's model-cache implementation. You pass in a `cache_clear_cb`
(typically `your_cache_dict.clear`) and a cooldown long enough for Metal
to actually return pages to the OS. 5 seconds is empirically sufficient
for a 24GB model on M1 Ultra.

Errors in `cache_clear_cb` are logged but non-fatal — escalation
continues with its own `safe_cleanup` so a bad cache-clear callback
can't poison the recovery path.

If escalation still can't free enough memory, `MemoryError` is raised
with the word `escalated cleanup` in the message (useful for grep in
production logs) and suggests reducing max loaded models or switching
to a smaller quant.

### Model Size Estimator (v0.2.2)

Parse a model's Metal footprint directly from its name. Designed to feed
into `require_fit` so multi-model ensemble workloads can proactively evict
cached models before hitting the Metal working-set limit.

```python
from metal_guard import MetalGuard, metal_guard

# Static method — no instance required
size = MetalGuard.estimate_model_size_from_name(
    "mlx-community/Mistral-Small-24B-8bit"
)
# → 24.0 GB  (24B params × 1.0 bytes/param for 8-bit)

size = MetalGuard.estimate_model_size_from_name(
    "mlx-community/Phi-4-mini-instruct-4bit"
)
# → 2.0 GB  (mini-class fallback: 4B × 0.5 for 4-bit)

# Pair with require_fit as a pre-load gate
name = "mlx-community/gemma-4-31b-8bit"
size = MetalGuard.estimate_model_size_from_name(name)
if size is not None:
    metal_guard.require_fit(size, model_name=name)
model = load(name)  # refused before load if Metal can't fit
```

**Why this exists:** A multi-model batch that loaded mistral-24B-8bit
→ phi-4-mini-4bit → gemma-4-26B-8bit in sequence was accumulating
cached models until the Metal working-set limit (~51 GB on M1 Ultra)
was exceeded. The Metal completion queue then threw an uncaught
`std::runtime_error` propagating as `EXC_CRASH (SIGABRT)`. With the
estimator, callers get a clean `MemoryError` refusal before touching
Metal, rather than crashing the process mid-generate.

**Supported patterns:**

| Pattern | Example | Result |
|---|---|---|
| `<N>B` + bits | `Mistral-24B-8bit` | 24 × 1.0 = 24 GB |
| `<N>M` + bits | `tiny-350m-4bit` | 0.350 × 0.5 = 0.175 GB |
| Size class + bits | `phi-4-mini-4bit` | 4 × 0.5 = 2 GB (mini class) |
| Size class + default | `foo-small` | 7 × 2.0 = 14 GB (fp16 default) |
| Unparseable | `mystery-model` | `None` → caller falls back |

Quantization multipliers: `16bit/fp16/bf16` → 2.0, `8bit/int8` → 1.0,
`4bit/int4/q4` → 0.5, `2bit/int2` → 0.25. Default when unspecified: 2.0
(conservative fp16 upper bound).

Size-class fallbacks: `mini` → 4B, `small` → 7B, `medium` → 13B,
`large` → 70B, `xl` → 13B.

Returns `None` when no size hint is parseable so callers can fall back
to the threshold-based `ensure_headroom` path.

### AGX Driver Workaround (v0.2.2)

Sets `AGX_RELAX_CDM_CTXSTORE_TIMEOUT=1` at import time if not already
set. Suggested by @zcbenz (MLX maintainer) in
[mlx#3267](https://github.com/ml-explore/mlx/issues/3267) — relaxes
the IOGPUFamily command buffer context store timeout to reduce kernel
panics on long-running GPU workloads. Zero-cost, safe to set
unconditionally.

### Additional OOM Pattern (v0.2.2)

`is_metal_oom` now detects the `fPendingMemorySet` panic signature
reported in [mlx#3346](https://github.com/ml-explore/mlx/issues/3346)
by @yoyaku155, alongside the existing `Insufficient Memory` and
`kIOGPUCommandBufferCallbackErrorOutOfMemory` patterns.

## v0.2 Features

### OOM Recovery

Catches Metal OOM errors and converts them to recoverable `MetalOOMError` instead of crashing the process. Addresses [mlx-lm#1015](https://github.com/ml-explore/mlx-lm/issues/1015) and [#854](https://github.com/ml-explore/mlx-lm/issues/854).

```python
from metal_guard import metal_guard, MetalOOMError

# Function wrapper — catches OOM, cleans up, retries once
result = metal_guard.oom_protected(
    generate, model, tokenizer, prompt=prompt, max_tokens=4096
)

# Context manager
with metal_guard.oom_protected_context():
    result = generate(model, tokenizer, prompt=prompt)

# For servers — return 503 instead of crashing
try:
    result = metal_guard.oom_protected(generate, model, tokenizer, prompt=prompt)
except MetalOOMError as e:
    return Response(status_code=503, body=f"GPU memory exhausted: {e.stats}")
```

### Pre-Load Memory Check

Prevents loading models that won't fit, avoiding the crash entirely. Addresses [mlx-lm#427](https://github.com/ml-explore/mlx-lm/issues/427) and [#1047](https://github.com/ml-explore/mlx-lm/issues/1047).

```python
# Check before loading
if not metal_guard.can_fit(model_size_gb=24.0):
    print("Not enough memory for 24GB model")

# Or let MetalGuard handle it (cleans up first, raises if still won't fit)
metal_guard.require_fit(24.0, model_name="Mistral-24B-8bit")
model = load("Mistral-24B-8bit")
```

### Periodic Flush

Background timer that prevents memory accumulation in long-running processes. Addresses [mlx-lm#854](https://github.com/ml-explore/mlx-lm/issues/854) and [mlx-examples#1124](https://github.com/ml-explore/mlx-examples/issues/1124).

```python
# Flush GPU cache every 5 minutes (skips if GPU threads are active)
metal_guard.start_periodic_flush(interval_secs=300)

# Stop when done
metal_guard.stop_periodic_flush()
```

### Memory Drift Watchdog

For 24/7 daemons and agent frameworks where memory drifts upward over hours. Escalating response: warn → flush → critical cleanup → app callback.

```python
def on_critical():
    """App-level response when memory is critical."""
    kv_cache.clear()
    log.error("Memory critical — KV cache dropped")

metal_guard.start_watchdog(
    interval_secs=60,       # Check every minute
    warn_pct=70.0,          # Flush at 70%
    critical_pct=85.0,      # Full cleanup + callback at 85%
    on_critical=on_critical,
)
```

**Why agent frameworks need this:** Each tool call / function call in an agent loop runs a separate `generate()`, accumulating fragmented Metal buffers. A single conversation with 50-100 tool calls can drift memory by several GB without any visible leak. The watchdog catches this drift before it reaches OOM.

## Full Examples

### Model Cache with Safety

```python
from metal_guard import metal_guard, MetalOOMError

class ModelCache:
    def __init__(self):
        self._models = {}

    def load(self, name, size_gb=None):
        if name in self._models:
            return self._models[name]
        if size_gb:
            metal_guard.require_fit(size_gb, model_name=name)
        else:
            metal_guard.ensure_headroom(model_name=name)
        metal_guard.breadcrumb(f"LOAD: {name} START")
        model = mlx_lm.load(name)
        self._models[name] = model
        metal_guard.breadcrumb(f"LOAD: {name} DONE")
        return model

    def unload_all(self):
        metal_guard.wait_for_threads()
        had_models = bool(self._models)
        self._models.clear()
        if had_models:
            metal_guard.safe_cleanup()

    def generate_safe(self, name, prompt, **kwargs):
        model, tokenizer = self.load(name)
        return metal_guard.oom_protected(
            mlx_lm.generate, model, tokenizer, prompt=prompt, **kwargs
        )
```

### Long-Running Agent Server

```python
from metal_guard import metal_guard

# Start watchdog for 24/7 operation
metal_guard.start_watchdog(
    interval_secs=120,
    warn_pct=65.0,
    critical_pct=80.0,
    on_critical=lambda: server.drop_oldest_session(),
)

# Each request uses OOM protection
@app.post("/v1/chat/completions")
async def chat(request):
    try:
        result = metal_guard.oom_protected(
            generate, model, tokenizer,
            prompt=request.prompt,
            max_tokens=request.max_tokens,
        )
        return {"choices": [{"message": {"content": result}}]}
    except MetalOOMError:
        return JSONResponse(status_code=503, content={"error": "GPU memory exhausted"})
```

### Testing (Prevent Metal in Unit Tests)

```python
# conftest.py
from unittest.mock import MagicMock, patch
import pytest

@pytest.fixture(autouse=True)
def _block_metal_gpu(request):
    """Prevent Metal GPU initialization in unit tests."""
    if "integration" in [m.name for m in request.node.iter_markers()]:
        yield
        return
    mock_mx = MagicMock()
    mock_mx.device_info.return_value = {"max_recommended_working_set_size": 48e9}
    mock_mx.get_active_memory.return_value = 0
    mock_mx.get_peak_memory.return_value = 0
    mock_mx.zeros.return_value = mock_mx
    with patch.dict("sys.modules", {"mlx.core": mock_mx}):
        yield
```

## API Reference

### Constructor

```python
MetalGuard(cooldown_secs=2.0, thread_timeout_secs=30.0, breadcrumb_path="logs/metal_breadcrumb.log")
```

A module-level singleton `metal_guard` is provided.

### Cross-Process Lock (v0.3.1, hardened in v0.6.0)

| Method | Description |
|--------|-------------|
| `acquire_mlx_lock(label, force=False)` | Acquire exclusive cross-process lock. Raises `MLXLockConflict` if held. `force=True` **since v0.6.0**: SIGTERMs the holder, waits up to `MLX_FORCE_WAIT_SEC`, sleeps `MLX_RECLAIM_COOLDOWN_SEC` post-reclaim; raises `MLXLockConflict(force_timeout=True)` if the holder refuses to exit (never unlinks while peer is alive) |
| `release_mlx_lock() -> bool` | Release lock if this process holds it |
| `read_mlx_lock() -> dict \| None` | Inspect lock without blocking. Self-heals stale + zombie locks |
| `mlx_exclusive_lock(label)` | Context manager: acquire on enter, release on exit |

### Version Advisories + Upstream Patches (v0.6.0)

| Method | Description |
|--------|-------------|
| `check_version_advisories(packages=None) -> list[dict]` | Active advisories for installed `(mlx, mlx-lm, mlx-vlm)` versions. Pass `packages={name: version}` to bypass `importlib.metadata` lookup (tests) |
| `install_upstream_defensive_patches(force=False) -> dict[str, bool]` | Opt-in, version-gated, idempotent monkey-patches. Returns `{patch_name: was_applied}`. `force=True` bypasses the version gate (tests only) |

### Thread Tracking

| Method | Description |
|--------|-------------|
| `register_thread(thread)` | Track a thread holding GPU buffers |
| `wait_for_threads(timeout=None) -> int` | Block until GPU threads finish. Returns count still alive |

### GPU Cleanup

| Method | Description |
|--------|-------------|
| `flush_gpu()` | `mx.eval(sync)` + `mx.clear_cache()`. Only after `wait_for_threads()` |
| `safe_cleanup()` | Full sequence: wait → gc.collect → flush → cooldown |
| `guarded_cleanup()` | Context manager that runs `safe_cleanup()` on exit |

### OOM Recovery (v0.2)

| Method | Description |
|--------|-------------|
| `oom_protected(fn, *args, max_retries=1, **kwargs)` | Run function with OOM catch + cleanup + retry |
| `oom_protected_context()` | Context manager version |
| `is_metal_oom(exc) -> bool` | Check if exception is Metal OOM |

### Pre-Load Check (v0.2)

| Method | Description |
|--------|-------------|
| `can_fit(model_size_gb, overhead_gb=2.0) -> bool` | Check if model fits in available memory |
| `require_fit(model_size_gb, model_name, overhead_gb=2.0)` | Clean up + raise MemoryError if won't fit |
| `estimate_model_size_from_name(name) -> float \| None` *(v0.2.2, static)* | Parse param count + quantization from model name → estimated GB |

### Memory Pressure

| Method | Description |
|--------|-------------|
| `memory_stats() -> MemoryStats` | Current GPU memory snapshot (active, peak, limit, available) |
| `is_pressure_high(threshold_pct=67.0) -> bool` | Check if peak memory exceeds threshold |
| `ensure_headroom(model_name, threshold_pct=67.0)` | Clean up if pressure high, no-op otherwise |
| `log_memory(label, model_name)` | Log memory state without cleanup |

### Long-Running Process Safety (v0.2)

| Method | Description |
|--------|-------------|
| `start_periodic_flush(interval_secs=300)` | Background timer to flush GPU cache |
| `stop_periodic_flush()` | Stop periodic flush |
| `start_watchdog(interval_secs, warn_pct, critical_pct, on_critical)` | Memory drift watchdog with escalating response |

### Hardware Detection (v0.4.0)

| Method | Description |
|--------|-------------|
| `detect_hardware() -> dict` *(static)* | Detect chip, memory, tier |
| `recommended_config() -> dict` *(classmethod)* | Hardware-appropriate thresholds for all features |

### KV Cache Monitor (v0.4.0)

| Method | Description |
|--------|-------------|
| `start_kv_cache_monitor(interval_secs, headroom_gb, growth_rate_warn, on_pressure)` | Track KV cache growth rate, fire callback before OOM |
| `stop_kv_cache_monitor()` | Stop the KV cache monitor |

### Forensics

| Method | Description |
|--------|-------------|
| `breadcrumb(msg)` | Write fsync'd line to breadcrumb log |

## Architecture

```
┌─────────────────────────────────────────────────┐
│            Your Application Code                │
│                                                 │
│  Agent loop / Server / Pipeline / Daemon        │
│  model = load("model-a")                        │
│  result = metal_guard.oom_protected(generate, …) │
│  unload_all()                                   │
└──────────────────┬──────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────┐
│              MetalGuard                         │
│                                                 │
│  Process Lock ─────── cross-process exclusion   │
│  Thread Registry ──── wait before cleanup       │
│  Safe Cleanup ─────── gc + flush + cooldown     │
│  OOM Recovery ─────── catch + cleanup + retry   │
│  Pre-Load Check ───── can_fit / require_fit     │
│  Pressure Monitor ─── headroom before load      │
│  Periodic Flush ───── background cache clear    │
│  Memory Watchdog ──── drift detection + alerts  │
│  Breadcrumb Log ───── crash forensics           │
└──────────────────┬──────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────┐
│           MLX + Metal Driver                    │
│                                                 │
│  mx.eval()  mx.clear_cache()  mx.zeros()       │
│  ⚠️  Driver bug: panics instead of OOM         │
└─────────────────────────────────────────────────┘
```

## Tested On

- Mac Studio M1 Ultra (64GB) — 9 kernel panics before MetalGuard, 0 after
- 10-person batch pipeline: ~90 model load/unload cycles, 994 seconds, zero crashes
- Models: Mistral-Small-3.2-24B, Phi-4-mini, Gemma-4-26B/31B, Pixtral-12B, LFM2-VL-3B (8-bit and 4-bit)

## Related Issues

Issues that MetalGuard addresses:

| Issue | Problem | MetalGuard Feature |
|-------|---------|-------------------|
| [mlx-lm#883](https://github.com/ml-explore/mlx-lm/issues/883) | Kernel panic from KV cache growth | Thread tracking + safe cleanup |
| [mlx-lm#1015](https://github.com/ml-explore/mlx-lm/issues/1015) | generate() OOM crashes process | `oom_protected()` |
| [mlx-lm#854](https://github.com/ml-explore/mlx-lm/issues/854) | Server OOM crash, no HTTP error | `oom_protected()` + `periodic_flush` |
| [mlx-lm#427](https://github.com/ml-explore/mlx-lm/issues/427) | M1 MBA crash on model load | `can_fit()` / `require_fit()` |
| [mlx-lm#1047](https://github.com/ml-explore/mlx-lm/issues/1047) | KV cache OOM on large model | `can_fit()` + `ensure_headroom()` |
| [mlx-examples#1124](https://github.com/ml-explore/mlx-examples/issues/1124) | Server memory leak → reboot | `periodic_flush` + `watchdog` |
| [mlx#2133](https://github.com/ml-explore/mlx/issues/2133) | Thread safety ongoing | `register_thread()` + `wait_for_threads()` |
| [mlx-lm#1128](https://github.com/ml-explore/mlx-lm/issues/1128) | `TokenizerWrapper.think_start_id` `len(None)` crash on mlx-lm 0.31.2 | `install_upstream_defensive_patches()` |
| [mlx-lm#1139](https://github.com/ml-explore/mlx-lm/issues/1139) | mlx-lm 0.31.2 broadcast-error regression | `check_version_advisories()` — warning only, downgrade to 0.31.1 |
| [mlx-lm#1081](https://github.com/ml-explore/mlx-lm/issues/1081) | `ArraysCache.is_trimmable()` but `trim()` missing | `check_version_advisories()` — callers guard with `hasattr(c, 'trim')` |
| [mlx#3348](https://github.com/ml-explore/mlx/pull/3348) | CommandEncoder thread-local (merged 2026-04-01, not yet in PyPI) | `check_version_advisories()` tracks observer-mode gate |

## License

MIT
