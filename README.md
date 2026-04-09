# MetalGuard

**English** | [繁體中文](README.zh-TW.md) | [日本語](README.ja.md)

GPU safety layer for [MLX](https://github.com/ml-explore/mlx) on Apple Silicon.

Prevents kernel panics caused by the IOGPUFamily Metal driver bug when repeatedly loading and unloading MLX models.

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

- **Single-model servers** (mlx_lm.server, LM Studio): Low risk. One model loaded for the lifetime of the process.
- **Multi-model pipelines**: **High risk.** Loading model A → unload → load model B → unload → repeat. Each transition is a potential panic trigger.

If your workload loads and unloads 3+ different MLX models per session, you need MetalGuard.

## Installation

```bash
pip install metal-guard
```

Or copy `metal_guard.py` into your project — it's a single file with no dependencies beyond the Python standard library.

## Usage

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
# ... if kernel panic happens here, breadcrumb log shows last operation
```

### Full Model Cache Example

```python
from metal_guard import metal_guard

class ModelCache:
    def __init__(self):
        self._models = {}

    def load(self, name):
        if name in self._models:
            return self._models[name]

        # Check pressure and clean up if needed
        metal_guard.ensure_headroom(model_name=name)

        metal_guard.breadcrumb(f"LOAD: {name} START")
        model = mlx_lm.load(name)
        self._models[name] = model
        metal_guard.breadcrumb(f"LOAD: {name} DONE")
        return model

    def unload_all(self):
        metal_guard.wait_for_threads()      # Never free while GPU is busy
        had_models = bool(self._models)
        self._models.clear()
        if had_models:
            metal_guard.safe_cleanup()       # gc + flush + cooldown
        # If cache was empty, skip Metal entirely (prevents panic trigger #2)

    def generate(self, name, prompt, **kwargs):
        model, tokenizer = self.load(name)

        def _run():
            return mlx_lm.generate(model, tokenizer, prompt=prompt, **kwargs)

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        metal_guard.register_thread(thread)  # Track GPU thread
        thread.join(timeout=120)
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

### `MetalGuard(cooldown_secs=2.0, thread_timeout_secs=30.0, breadcrumb_path="logs/metal_breadcrumb.log")`

Create a MetalGuard instance. A module-level singleton `metal_guard` is provided.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `cooldown_secs` | 2.0 | Sleep after GPU flush for Metal driver buffer reclamation |
| `thread_timeout_secs` | 30.0 | Max wait time for GPU threads |
| `breadcrumb_path` | `"logs/metal_breadcrumb.log"` | Path for crash forensics log. `None` to disable |

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

### Memory Pressure

| Method | Description |
|--------|-------------|
| `memory_stats() -> MemoryStats` | Current GPU memory snapshot |
| `is_pressure_high(threshold_pct=67.0) -> bool` | Check if peak memory exceeds threshold |
| `ensure_headroom(model_name, threshold_pct=67.0)` | Clean up if pressure high, no-op otherwise |
| `log_memory(label, model_name)` | Log memory state without cleanup |

### Forensics

| Method | Description |
|--------|-------------|
| `breadcrumb(msg)` | Write fsync'd line to breadcrumb log |

## Architecture

```
┌─────────────────────────────────────────────┐
│           Your Application Code             │
│                                             │
│  model = load("model-a")                    │
│  result = generate(model, prompt)           │
│  unload_all()  ← DON'T call mx.clear_cache │
│                   directly, use MetalGuard  │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│             MetalGuard                      │
│                                             │
│  Thread Registry ─── wait before cleanup    │
│  Safe Cleanup ────── gc + flush + cooldown  │
│  Pressure Monitor ── headroom before load   │
│  Breadcrumb Log ──── crash forensics        │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│          MLX + Metal Driver                 │
│                                             │
│  mx.eval()  mx.clear_cache()  mx.zeros()   │
│  ⚠️  Driver bug: panics instead of OOM     │
└─────────────────────────────────────────────┘
```

## Tested On

- Mac Studio M1 Ultra (64GB) — 9 kernel panics before MetalGuard, 0 after
- 10-person batch pipeline: ~90 model load/unload cycles, 994 seconds, zero crashes
- Models: Mistral-Small-3.2-24B, Phi-4-mini, Gemma-4-26B/31B, Pixtral-12B, LFM2-VL-3B (8-bit and 4-bit)

## Related Issues

- [ml-explore/mlx-lm#883](https://github.com/ml-explore/mlx-lm/issues/883) — Kernel panic from unbounded KV cache
- [ml-explore/mlx#2133](https://github.com/ml-explore/mlx/issues/2133) — Thread safety ongoing
- [ml-explore/mlx#3126](https://github.com/ml-explore/mlx/issues/3126) — Sub-thread exit crash
- [ml-explore/mlx#3078](https://github.com/ml-explore/mlx/issues/3078) — Concurrent inference unsupported

## License

MIT
