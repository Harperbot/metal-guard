# MetalGuard

**English** | [繁體中文](README.zh-TW.md) | [日本語](README.ja.md)

GPU safety layer for [MLX](https://github.com/ml-explore/mlx) on Apple Silicon.

Prevents kernel panics and OOM crashes caused by Metal driver bugs when running MLX inference — especially multi-model pipelines, long-running servers, and agent frameworks with heavy tool calling.

**Current version:** v0.3.1 — see [CHANGELOG.md](CHANGELOG.md) for the full release history.

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

## v0.3.1 Features

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

### Cross-Process Lock (v0.3.1)

| Method | Description |
|--------|-------------|
| `acquire_mlx_lock(label, force=False)` | Acquire exclusive cross-process lock. Raises `MLXLockConflict` if held |
| `release_mlx_lock() -> bool` | Release lock if this process holds it |
| `read_mlx_lock() -> dict \| None` | Inspect lock without blocking. Self-heals stale locks |
| `mlx_exclusive_lock(label)` | Context manager: acquire on enter, release on exit |

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

## License

MIT
