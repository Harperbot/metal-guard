# MetalGuard

**English** | [繁體中文](README.zh-TW.md) | [日本語](README.ja.md)

GPU safety layer for [MLX](https://github.com/ml-explore/mlx) on Apple Silicon.

Prevents kernel panics and OOM crashes caused by Metal driver bugs when running MLX inference — especially multi-model pipelines, long-running servers, and agent frameworks with heavy tool calling.

**Current version: v0.11.3** — see [CHANGELOG.md](CHANGELOG.md) for release history and per-feature rationale.

### What's in v0.11

Built on the 2026-04-27 community sweep (mlx-lm#1185 / #1206 / mlx-vlm#1064 / omlx#578 / #862 / #902):

- **`error_classifier`** — central regex table for **6 distinct error severities**: `kernel_panic` / `process_abort` / `command_buffer_oom` / `gpu_hang` / `gpu_page_fault` / `descriptor_leak`. `SubprocessCrashError` now exposes `.error_class` + `.recovery_hint` for caller routing.
- **L10b — process-abort scanner** — `scan_recent_aborts(24h)` sibling to `scan_recent_panics(72h)`. Aborts are non-rebooting failures; counted separately so they don't trip the kernel-panic lockout. `CooldownVerdict.abort_count_24h` exposed for dashboards.
- **L13b — Apple GPU family detection** — `apple_gpu_family()` reads `mx.device_info()` and maps `applegpu_g13`/`g14`/`g15`/`g16`/`g17` → `M1`/`M2`/`M3`/`M4`/`M5`. Surfaces `resource_limit` (mlx-lm#1185 descriptor cap, 499000 on M1 Ultra).
- **L14 — descriptor-leak heuristic** — `ResourceTracker(cold_restart_after=4000)` tracks inferences-since-cold-restart so callers can pre-emptively `shutdown()` + spawn new subprocess before hitting the descriptor limit. `mx.clear_cache()` doesn't release descriptor handles; only subprocess respawn does.
- **`breadcrumb_with_meta(tag, payload, **meta)`** — structured breadcrumb format `[ts] TAG: payload | k=v k=v` for richer postmortem forensics. L11 orphan parser updated to lazy regex (backward-compat with legacy `breadcrumb()`).
- **`KNOWN_PANIC_MODELS` schema upgrade** — adds `tier` (panic / abort / degradation), `error_classes[]` (multiple modes per model + per-GPU-family confirmation), `verified_safe_alternative`. New helpers: `check_known_panic_model_for_gpu(model, gpu_family="M5")` / `models_by_tier()` / `models_affecting_gpu_family()`.
- **4 new registry entries** covering Qwen3.5/Qwen3.6/Qwen3-VL family across M4 / M5 hardware.
- **Hotfix**: PEP 639 license-classifier conflict in `pyproject.toml` that blocked every `pip install` since v0.9.0.

### What's in v0.10

metal-guard now covers the **full lifecycle** of an Apple-Silicon kernel panic — before, during, and after:

| Phase | Layer | What it does |
|---|---|---|
| Before | L1–L9 (v0.1–v0.9) | Thread tracking / cleanup / OOM recovery / pre-load checks / long-run safety / dual-mode / subprocess isolation / cross-process lock / cadence + circuit breaker |
| **After reboot (new)** | **L10 panic cooldown gate** | Refuses to re-launch MLX work for 2h–72h after a panic, preventing immediate auto-re-panic when launchd respawns plists |
| **Pre-panic warning (new)** | **L11 subprocess orphan monitor** | Detects `SUBPROC_PRE` without matching `SUBPROC_POST` after 90s — SIGKILL the worker before the kernel does |
| **After reboot (new)** | **L12 postmortem auto-collect** | Bundles `panic-full-*.panic` + breadcrumb tail + `mx.metal` stats + `index.md` summary into a single directory |
| **Cross-process state (new)** | **L13 status snapshot** | Versioned JSON for menu bar / dashboard / ssh inspection consumers — no `import metal_guard` needed downstream |
| All layers | **`KNOWN_PANIC_MODELS` registry** | Community-curated `(model, hardware, panic signature, workload, workaround)` data — see [Community Panic Registry](#-community-panic-registry--known_panic_models) below |

New CLI surface ships with v0.10:

```bash
metal-guard panic-gate            # L10: rc=0 proceed / rc=2 cooldown / rc≥3 broken
metal-guard postmortem ./bundle   # L12: collect after reboot
metal-guard status-write --once   # L13: write JSON snapshot
metal-guard orphan-scan           # L11: pre-panic stuck-worker detection
metal-guard ack                   # L10: clear lockout (require explicit user touch)
mlx-safe-python -c "import torch" # interactive shell guard — refuses ad-hoc imports during cooldown
```

v0.10 promotes four defensive layers from Harper's private fork after two weeks of production validation across 11 panic incidents. The honest caveat from earlier releases still holds: metal-guard narrows race windows around the Apple IOGPU driver bug — it does not fix the bug. v0.10 extends the defence surface from "during run" to "after reboot" and "before kernel kill".

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

## 📋 Community Panic Registry — `KNOWN_PANIC_MODELS`

**A user-curated list of MLX models that kernel-panic Apple Silicon Macs in production, with hardware contexts, root-cause hypotheses, and verified workarounds.**

Apple's IOGPUFamily driver bug has no fix timeline. While the bug is upstream, **which models trigger it under which workloads is a community-knowable thing** — but it's currently scattered across GitHub issue threads, lmstudio bug reports, Discord screenshots, and individual `panic-full-*.panic` files nobody publishes.

metal-guard provides a structured home for this knowledge:

```python
from metal_guard import check_known_panic_model, warn_if_known_panic_model

# Check before loading
advisory = check_known_panic_model("mlx-community/gemma-4-31b-it-8bit")
if advisory is not None:
    print(advisory["recommendation"])
    # → "metal-guard v0.9.0 narrows the race window... but does NOT eliminate
    #    panic on this model. Switch backend (Ollama / llama.cpp) or pivot
    #    to MoE variant (e.g. mlx-community/gemma-4-26b-a4b-it-4bit)."

# Or fire-and-forget warning at load time (per-process dedup)
warn_if_known_panic_model(model_id)
```

Each entry carries:
- **`panic_signature`** — the exact `IOGPUMemory.cpp:NNN` line + keyword to match against your `panic-full-*.panic` log
- **`reproductions`** — production data points (hardware, RAM, time-to-panic, workload)
- **`community`** — cross-references to GitHub issues / lmstudio bugs / forum threads where others hit the same panic
- **`recommendation`** — actionable workaround (backend switch / model pivot / cadence config)
- **`upstream`** — links to the GitHub issues tracking the underlying driver bug

### How to contribute

If you've hit a kernel panic on a specific MLX model **with metal-guard's defensive layers fully engaged**, your data point is valuable. Open a [Known Panic Model report](https://github.com/Harperbot/metal-guard/issues/new?template=known-panic-report.yml) — the template walks you through the schema (model ID / hardware / panic signature / workload / time-to-panic / verified workaround). Schema docs in [CONTRIBUTING.md](CONTRIBUTING.md#known-panic-models-schema).

The registry is intentionally conservative — entries require either a confirmed production reproduction or a clear upstream issue with reproducible signature. We don't want false positives blacklisting models that work fine for most users.

**Why not just read mlx#3186 comments?** Because that thread mixes hardware reports, hypotheses, attempted fixes, and unrelated discussion. The registry distils it into structured advisory data your code can `check_known_panic_model()` against — and your panic report doesn't disappear into a 50-comment thread.

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

> **PyPI status (2026-04-27)**: metal-guard is **not yet on PyPI**. Use one of the three options below until v0.10.x ships there. Tracking issue: open an issue if you need PyPI urgently.

### Option A — pip from GitHub (recommended, 1 line)

Installs from a tagged release — gives you the `metal-guard` and `mlx-safe-python` console scripts plus the `metal_guard` Python module:

```bash
pip install "git+https://github.com/Harperbot/metal-guard.git@v0.11.3"
```

After install:

```bash
metal-guard --version          # → metal-guard 0.11.3
metal-guard panic-gate         # L10 cooldown verdict
metal-guard status             # full snapshot
mlx-safe-python -c "import torch"   # interactive shell guard
```

To upgrade to a future release: `pip install --upgrade "git+https://github.com/Harperbot/metal-guard.git@vX.Y.Z"`.

### Option B — Single-file drop-in (zero install, no pip)

`metal_guard.py` has **zero dependencies** beyond the Python standard library (and optional `mlx` for memory introspection). Download once, import directly:

```bash
mkdir -p ~/lib/metal-guard
curl -L -o ~/lib/metal-guard/metal_guard.py \
  https://raw.githubusercontent.com/Harperbot/metal-guard/v0.11.3/metal_guard.py
```

Then in your code:

```python
import sys; sys.path.insert(0, "/Users/<you>/lib/metal-guard")
import metal_guard as mg
verdict = mg.evaluate_panic_cooldown()
print(verdict.exit_code, verdict.reason)
```

This path is the right choice for launchd plist wrappers, panic-recovery scripts, and CI runners that must work even when the rest of the Python install is wedged.

### Option C — Local clone (for development / running tests)

```bash
git clone https://github.com/Harperbot/metal-guard.git
cd metal-guard
pip install -e ".[test]"
pytest -q
```

Editable install picks up your local edits without re-installing. The `[test]` extra pulls in `pytest>=7.0`.

### Verifying the install

After Option A or C, the gate should self-test:

```bash
$ metal-guard panic-gate
🟢 PROCEED  no recent IOGPU panics
  24h=0 72h=0
$ metal-guard status
metal-guard 0.11.3  🟢 OK
  mode        defensive — defensive mode (default)
  panics      0 in last 72h
  ...
```

If `metal-guard` is not on `PATH` after pip install, your `pip --user` bin dir is probably missing — `python3 -m metal_guard_cli panic-gate` works as a fallback.

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

MetalGuard is organised as **defence layers (L1–L13)** plus a set of
**preventive helpers (R-series)** and the **`KNOWN_PANIC_MODELS` registry**.
Every feature is available from the single `metal_guard` module — install via
`pip install "git+https://github.com/Harperbot/metal-guard.git@v0.11.3"` or
drop `metal_guard.py` in your `PYTHONPATH` (see [Installation](#installation) above). See [CHANGELOG.md](CHANGELOG.md) for when
each layer landed and the incident that motivated it.

Layer ordering is a defence-in-depth onion: L1–L8 narrow race windows during a
run, L9 + L11 short-circuit just before a kernel-level abort, L10 + L12 handle
recovery after a panic + reboot, and L13 surfaces all of the above as a JSON
snapshot for cross-process consumers.

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

### L10 — Panic cooldown gate *(v0.10.0)*

After a kernel panic + macOS reboot, launchd auto-respawns plists ~14 minutes later. Without a gate, the next MLX workload can immediately re-trigger the same driver bug. L10 reads `/Library/Logs/DiagnosticReports/` for AND-pattern IOGPU panics and applies a staircase cooldown (1 panic → 2h, ≥2 in 24h or ≥3 in 72h → lockout requiring `~/.metal-guard-ack` touch).

| API | What it does |
|---|---|
| `evaluate_panic_cooldown() -> CooldownVerdict` | Stdlib-only evaluation; `verdict.exit_code` ∈ {0=proceed, 2=cooldown, ≥3=gate broken} |
| `scan_recent_panics(hours=72.0) -> list[PanicRecord]` | AND-pattern (`prepare_count_underflow` + `IOGPUMemory.cpp:NNN`) scan |
| `mark_panic_sentinel_cooldown(duration_hours)` | Extend cooldown beyond DiagnosticReports rotation lag (called by L12) |
| `ack_panic_lockout()` | Atomic touch `~/.metal-guard-ack` to clear an active lockout |
| `clear_panic_ack()` / `clear_panic_sentinel()` | Operator overrides |
| `metal-guard panic-gate` | CLI wrapper for plist scripts (mirrors verdict exit codes) |
| `metal-guard ack` | CLI wrapper for ack |

Env: `METALGUARD_PANIC_COOLDOWN_STAGE1_H` / `_LOCKOUT_24H_N` / `_LOCKOUT_72H_N` / `_LOCKOUT_MAX_H` / `_GATE_DISABLED=1`.

### L11 — Subprocess orphan monitor *(v0.10.0)*

Pre-panic signal: a `SUBPROC_PRE: <model>` breadcrumb without a matching `SUBPROC_POST` after 90 seconds strongly suggests Metal is stuck. Caller can SIGKILL the worker pid before the kernel does (saves a reboot).

| API | What it does |
|---|---|
| `scan_orphan_subproc_pre(threshold_sec=90.0) -> list[OrphanPre]` | FIFO-paired PRE↔POST scan over breadcrumb tail |
| `metal-guard orphan-scan [--threshold-sec N]` | CLI wrapper |

Disabled by `METALGUARD_SUBPROC_ORPHAN_WATCH_DISABLED=1`. Threshold via `METALGUARD_SUBPROC_ORPHAN_THRESHOLD_SEC`.

### L12 — Postmortem auto-collect *(v0.10.0)*

After a panic + reboot, this collects the diagnostic bundle into a single directory: panic-full-*.panic files (capped 5 files / 5MB each), last 500 lines of `metal_breadcrumb.log`, panics.jsonl history, mx.metal stats, and an `index.md` summary. When a panic is found, also writes a sentinel cooldown so L10 defers further runs even if DiagnosticReports rotates.

| API | What it does |
|---|---|
| `run_postmortem(output_dir) -> dict` | Full orchestration; returns paths + panic count |
| `metal-guard postmortem <output_dir>` | CLI wrapper |

Kill-switch: `METALGUARD_POSTMORTEM_DISABLED=1`. Designed to be called from a launchd wrapper after reboot; pair with [Telegram alerts in pure bash](#) for ops integration.

### L13 — Status snapshot *(v0.10.0)*

Versioned JSON snapshot for cross-process consumers (menu bar apps, dashboards, ssh inspection scripts) that should not import `metal_guard` directly. Schema is append-only across minor versions.

| API | What it does |
|---|---|
| `get_status_snapshot(*, include_panics=True, breadcrumb_lines=20) -> dict` | Aggregate memory / KV monitor / panics / lock holder / mode / L10 verdict |
| `write_status_snapshot(out_path=None)` | Atomic write to `~/.cache/metal-guard/status.json` |
| `metal-guard status-write [--once \| --interval 30]` | CLI / daemon wrapper |
| `STATUS_SNAPSHOT_SCHEMA_VERSION` | Bumped on breaking changes |

Run as a 30s-interval daemon under launchd to feed your menu bar app:

```xml
<plist version="1.0"><dict>
  <key>Label</key><string>com.metal-guard.status-writer</string>
  <key>ProgramArguments</key>
  <array>
    <string>/usr/bin/env</string><string>metal-guard</string>
    <string>status-write</string><string>--interval</string><string>30</string>
  </array>
  <key>KeepAlive</key><true/>
</dict></plist>
```

### Interactive shell guard

`scripts/mlx-safe-python` (bash, single file, stdlib only) — drop into PATH to refuse ad-hoc `python -c "import torch/mlx"` while a cooldown is active. Lets `pip` / `build` / `venv` pass through (they don't import Metal). Exit codes: 0 ran / 10 blocked / 11 fail-open.

```bash
mlx-safe-python -c "import torch; print(torch.__version__)"   # blocked in cooldown
mlx-safe-python -m pip show torch                              # passes — no Metal import
MLX_SAFE_PYTHON_FORCE=1 mlx-safe-python -c "..."               # explicit override + WARN
```

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
