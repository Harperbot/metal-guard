# Changelog

All notable changes to **metal-guard** are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.9.0] — 2026-04-25

Minor release consolidating **panic #7–#11 findings** from Harper's
production timeline (2026-04-16 → 2026-04-24) into the open-source
distribution. Brings three new defences (B1 subprocess guard, C5
cross-model cadence, C7 gemma-4 first-generate flush) and one new
piece of advisory data: `KNOWN_PANIC_MODELS`.

**The honest caveat upfront.** metal-guard v0.9.0 narrows multiple race
windows around the Apple IOGPU driver bug. It reduces panic frequency
on every workload we've exercised. It **does not eliminate panic**
on every model — specifically, `mlx-community/gemma-4-31b-it-8bit`
still panicked on a production pipeline at Harper after every defence
in this release was engaged (panic #11, 2026-04-24). When metal-guard
is engaged and a model continues to panic in production, the right
operational answer is to **switch backend** (Ollama / llama.cpp) or
**pivot to a different model family** — see "When metal-guard is not
enough" below.

### Added

- **`subprocess_inference_guard(model_id)` (B1).** Module-level
  contextmanager that wraps every `gen_fn(...)` call inside an MLX
  subprocess worker. Performs `mx.clear_cache()` PRE,
  `mx.synchronize()` POST, `mx.clear_cache()` POST, and emits
  `SUBPROC_PRE` / `SUBPROC_POST` breadcrumbs. Harper recorded 6
  consecutive subprocess-path kernel panics in 3 days (2026-04-20 →
  2026-04-23) before this guard; the streak ended on the first
  `gen_fn` invocation after it was wired in.

- **Cross-model cadence in `CadenceGuard` (C5).** Reject back-to-back
  loads of *different* models within a configurable window.
  `CadenceGuard(cross_model_interval_sec=…)` opts in; the default is
  `0.0` (disabled) to preserve v0.8.0 semantics. Env var
  `METALGUARD_CROSS_MODEL_INTERVAL` sets a process-wide default
  (resolver helper: `_resolve_cross_model_interval`). Violations raise
  `CrossModelCadenceViolation`, a subclass of `CadenceViolation` so
  existing `except CadenceViolation` keeps working.

- **Gemma-4 90-second floor (C5).** `mlx-community/gemma-4-*`,
  `unsloth/gemma-4-*`, and `mlx-models/gemma-4-*` always enforce a
  minimum 90 s cross-model cadence regardless of configured base.
  Constant: `GEMMA4_MIN_CROSS_MODEL_INTERVAL_SEC = 90.0`. All 8/8
  kernel panics in Harper's 2026-04 timeline with an identifiable
  at-panic model were in the gemma-4 family; panic #6 landed 66 s
  after prior unload, so 90 s = 66 s + ~36 % safety margin.

- **`gemma4_generation_flush(model_id, generate_call_count)` (C7).**
  First-generate settle window: `mx.synchronize()` +
  `mx.clear_cache()` + `time.sleep(3.0)` before the *first*
  `generate()` on a freshly-loaded gemma-4 worker. No-op on
  subsequent calls and on non-gemma-4 models. Env overrides:
  `METALGUARD_GEMMA4_FIRSTGEN_DISABLED=1`,
  `METALGUARD_GEMMA4_FIRSTGEN_SLEEP_SEC=<seconds>`. Harper's empirical
  breakdown: 7 of 8 gemma-4 panics landed on the first `generate()`
  within 7–66 s of worker-ready; the pre-existing flush barriers
  caught none of them.

  **Renamed from the internal `gemma4_firstgen_guard`** — the name
  "guard" incorrectly suggested a block. This function is a flush +
  settle window, not a gate. If you need to gate, use
  `CircuitBreaker` or `require_cadence_clear`.

- **`KNOWN_PANIC_MODELS` advisory registry.** Module-level dict mapping
  model IDs to structured advisories (`panic_signature`,
  `reproductions`, `community`, `recommendation`, `upstream`).
  Companion helpers:
    - `check_known_panic_model(model_id) -> dict | None`
    - `warn_if_known_panic_model(model_id) -> bool` (idempotent; emits
      one `log.warning` per process per model)

  Policy is the caller's — metal-guard does not refuse loads on its
  own. v0.9.0 ships with one entry: `mlx-community/gemma-4-31b-it-8bit`.

- **`require_cadence_clear(..., cross_model_interval_sec=...)`.** New
  keyword argument mirrors the `CadenceGuard` constructor param.
  `None` (default) delegates to `_resolve_cross_model_interval` which
  reads `METALGUARD_CROSS_MODEL_INTERVAL` or falls back to the C5
  default of 60 s. `0.0` disables. When `guard=` is supplied, the
  guard's own `cross_model_interval_sec` wins.

### Known affected models

#### `mlx-community/gemma-4-31b-it-8bit` — repeat offender

Two production kernel panics on Harper's box, 24 hours apart, same
model, same pipeline, same panic signature:

| #   | Date/time (local)  | PID   | Spawn → panic | Context                                  |
|-----|--------------------|-------|--------------:|-------------------------------------------|
| 7   | 2026-04-23 03:14   | 67840 |        ~6 min | rezivot pipeline, pre-C5                   |
| 11  | 2026-04-24 03:14   | 26608 |      ~1.5 min | same pipeline as #7; post-C5 but ~1.5 min to panic anyway |

Signature for both: `IOGPUMemory.cpp:492 "completeMemory() prepare
count underflow"`, Gen-4 hybrid attention, no concurrent generates.

**Community corroboration (all 2026-04):**

- [Hannecke — "MLX Crashed My Mac"](https://medium.com/@michael.hannecke/how-my-local-coding-agent-crashed-my-mac-and-what-i-learned-about-mlx-memory-management-e0cbad01553c)
  — M4 Max 64 GB, same panic signature, pivoted to
  `Qwen3-Coder-30B-A3B` MoE as workaround.
- [`lmstudio-ai/lmstudio-bug-tracker#1740`](https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/1740)
  "Gemma-4 31b KV excessive KV cache footprint" — corroborates hybrid
  attention (50 sliding + 10 global) KV cache + 8-bit weights (~34 GB)
  + full context KV (20 GB+) > 54 GB memory pressure on the 64 GB
  class. The thread documents 26 GB VRAM for a mere 8192 context.
- [`ml-explore/mlx-lm#883`](https://github.com/ml-explore/mlx-lm/issues/883)
  — M3 Ultra 96 GB reports the same panic signature on the same model
  family.
- [`ml-explore/mlx#3186` (comment 2026-04-24)](https://github.com/ml-explore/mlx/issues/3186#issuecomment-4314204974)
  — independent third-party data point: Mac mini M4 base 32 GB,
  macOS 26.4.1 (`25E253`), mlx 0.31.2, mlx-lm 0.31.3, model
  `mlx-community/Qwen3.6-35B-A3B-4bit`. Panic 8 min 16 s after
  `mlx_lm.server` start; `--prompt-cache-bytes 8 GiB` did not
  prevent it; reporter adopted `llama.cpp` for production serving.

**What this means in practice.** macOS 26.4.x has not fixed the bug.
26.5 beta has not fixed the bug. `--prompt-cache-bytes` does not
prevent it. Adding RAM to 96 GB does not prevent it. metal-guard
v0.9.0 narrows the race windows but does not eliminate panic on this
specific model in Harper's workload.

### When metal-guard is not enough

If you engage every defence in v0.9.0 (B1 + C5 + C7 + CircuitBreaker)
and still observe repeat panics on the same model, that is a signal
that the race window on that model is wider than metal-guard can
narrow. Two escape hatches, in order of ROI:

1. **Switch backend.** Ollama and `llama.cpp` both use Metal MPS
   under the hood but run a persistent worker architecture that
   avoids the subprocess teardown race entirely. Harper's
   `harper-finance` project migrated to Ollama 2026-04-23 and has
   run zero-panic since. The independent `mlx#3186` M4-base
   reporter adopted `llama.cpp` for the same reason. You lose some
   raw throughput (MLX was measured 30–55 % faster on prefill in
   that report); you gain "doesn't panic the machine."

2. **Pivot to a different model family.** Mixture-of-Experts
   variants (e.g. `mlx-community/gemma-4-26b-a4b-it-4bit`,
   `Qwen3-Coder-30B-A3B`) have a much smaller active-parameter
   footprint per forward pass and a narrower KV growth trajectory.
   Community reports (Hannecke, lmstudio#1740) converge on MoE as
   the most reliable same-ecosystem workaround.

metal-guard is complementary to both — `subprocess_inference_guard`
is useful even under Ollama if you spawn per-request subprocess
workers, and `CadenceGuard` still helps regardless of backend when
you hot-swap models.

### Panic timeline (Harper internal, 2026-04)

For calibration on what "engaged every defence" means in practice:

| #   | Date (local) | Signature                                 | Trigger                                                  | Defence landed       |
|-----|--------------|-------------------------------------------|----------------------------------------------------------|----------------------|
|  7  | 2026-04-23 03:14 | `IOGPUMemory.cpp:492 prepare_count_underflow` | cross-model cadence not wired; gemma-4-31b-8bit | C5 phase-1 shipped |
|  8  | 2026-04-23 14:07 | `IOGPUMemory.cpp:492`                     | GC / Metal async race across subprocess teardown          | 0.8.6 hotfix 4-barrier |
|  9  | 2026-04-23 17:40 | `IOGPUMemory.cpp:492`                     | Phase-2 +833-line regression                             | reverted via `git stash` |
| 10  | 2026-04-23 19:46 | `IOGPUMemory.cpp:492`                     | interactive `python -c "import sentence_transformers"` for version verification → `torch` MPS backend init → process exit race → same Apple kernel bug | ad-hoc import SOP + `mlx-safe-python` wrapper |
| 11  | 2026-04-24 03:14 | `IOGPUMemory.cpp:492`                     | same pipeline as #7; panic ~1.5 min after worker ready despite classic L9 in place | C7 flush + gemma-4 floor (this release) |

Panic #10 is worth calling out: it was triggered by a *verification*
command on the host terminal, not by any MLX workload in production.
Anything that imports `torch`, `mlx`, `mlx_lm`, `mlx_vlm`,
`sentence_transformers`, `transformers`, `diffusers`, or
`accelerate` initialises the Metal MPS backend and can walk into the
same kernel bug at process exit. If your team uses metal-guard, the
operational lesson is: during an active cooldown, verify package
versions with `pip show <pkg>` or
`python -c "import importlib.metadata as m; print(m.version('<pkg>'))"`
— *never* `python -c "import <ml-pkg>; print(<ml-pkg>.__version__)"`.

### Fixed

- `prepare_count_underflow` panics on subprocess-isolated MLX workers
  (via B1 `subprocess_inference_guard`) — see v0.8.1 history above.
- `prepare_count_underflow` panics on back-to-back loads of *different*
  models within seconds of each other — classic `CadenceViolation`
  only caught same-model patterns; `CrossModelCadenceViolation`
  extends coverage to the cross-model axis that Harper's panic #7
  exposed.
- First-generate race window on gemma-4 family — `gemma4_generation_flush`
  inserts a mandatory synchronize + clear + sleep before the first
  forward pass, extending the settle window the four pre-existing
  flush barriers failed to cover.

### Changed

- `CadenceGuard.__init__` accepts a new keyword-only argument
  `cross_model_interval_sec` (default `0.0`, backwards-compat).
  Property `cadence_guard.cross_model_interval_sec` exposes the
  configured value.
- `require_cadence_clear` accepts a new keyword-only argument
  `cross_model_interval_sec`. Behaviour for existing calls is
  unchanged unless the env var `METALGUARD_CROSS_MODEL_INTERVAL` is
  set.

### Migration

- **Pure additions on the public API surface.** Existing v0.8.0 code
  continues to work without changes.
- **Env-var opt-in.** If you want cross-model cadence without code
  changes, set `METALGUARD_CROSS_MODEL_INTERVAL=60` in your
  environment and call `require_cadence_clear()` as before.
- **Gemma-4 users get the 90-second floor automatically** the moment
  `cross_model_interval_sec > 0.0` (or the env var is set). This
  cannot be opted out of for the gemma-4 family by setting the base
  to 0 — the floor fires regardless. This is the one intentional
  asymmetry in the release and reflects the empirical panic data.

### Tests

- 38 new tests in `tests/test_v090_cross_model_cadence.py` covering
  `_is_gemma4_family` (13 parametrised cases),
  `_resolve_cross_model_interval` (6 cases: default / env / explicit
  / invalid-fallback / negative-clamp / zero-disable),
  `CrossModelCadenceViolation` inheritance + fields,
  `CadenceGuard` cross-model check (same-model priority, zero-disabled
  fallthrough, zero-still-floors-gemma4, pass-after-interval),
  `require_cadence_clear` param plumbing,
  `gemma4_generation_flush` (non-gemma / count>0 / env-disabled /
  sleep-env-override / invalid-sleep-env-fallback),
  `KNOWN_PANIC_MODELS` hit/miss + `warn_if_known_panic_model`
  idempotence. Full suite: **204 passed** (166 pre-existing + 38
  new).

### Upstream references

Open upstream issues consistent with the v0.9.0 advisory content, all
open at release time:

- [`ml-explore/mlx#3186`](https://github.com/ml-explore/mlx/issues/3186)
  — canonical subprocess isolation / `prepare_count_underflow` thread.
  Third-party corroboration (2026-04-24, M4 base 32 GB, Qwen3.6-35B-A3B):
  [`comment 4314204974`](https://github.com/ml-explore/mlx/issues/3186#issuecomment-4314204974).
- [`ml-explore/mlx#3346`](https://github.com/ml-explore/mlx/issues/3346)
  — kernel panic reproducer catalogue.
- [`ml-explore/mlx-lm#883`](https://github.com/ml-explore/mlx-lm/issues/883)
  — subprocess worker panic report (Hannecke, M4 Max 64 GB).
- [`ml-explore/mlx-lm#1047`](https://github.com/ml-explore/mlx-lm/issues/1047)
  — Kimi K2.5 KV cache OOM on M3 Ultra.
- [`ml-explore/mlx/#3267`](https://github.com/ml-explore/mlx/issues/3267)
  — GPU watchdog kills MLX when display is active (wontfix). Used to
  justify the `AGX_RELAX_CDM_CTXSTORE_TIMEOUT=1` import-time workaround
  kept from v0.2.x.

---

## [0.8.0] — 2026-04-17

Minor release porting **Layer 9 (L9)** from Harper's internal fork
into the open-source distribution. L9 was written in response to a
production kernel panic on 2026-04-16 23:33:27 — `IOGPUMemory.cpp:492
"completeMemory() prepare count underflow"` — that fired during the
*first* `generate()` after a freshly-loaded subprocess worker. The
L6 SIGABRT handler could not catch it: the panic lived at kernel
level, before any user-space signal. The only defence left was to
stop doing back-to-back loads in the first place.

After 24 hours of real-world exposure, the baseline panic rate on
Harper's box dropped from ~1.4/day to zero — the first
panic-free 24 h window in the post-panic sample. L9 is now stable
enough to ship outside the private fork.

### Added

- **`CadenceGuard`** (L9 primary defence). Per-model load-timestamp
  store with configurable minimum interval (default **180 s**).
  Persisted to `~/.cache/metal-guard/cadence.json` via atomic
  write (`os.replace`), so the mark survives subprocess spawn,
  process kill, and kernel panic. A worker can check-and-mark in
  one call via the `require_cadence_clear(model_id)` helper; on
  violation it raises `CadenceViolation` with `.model_id`,
  `.last_ts`, `.min_interval`, and `.delta` for structured
  handling. Stale entries (> 4 h) are GC'd on next mark.

- **`CadenceViolation`** exception. Callers inside the worker
  subprocess are expected to `sys.exit(2)` on catch so the parent
  runner propagates it as a normal worker error rather than a
  mysterious crash.

- **Panic ingest** — `parse_panic_reports(directory, *,
  since_ts=None)` scans `/Library/Logs/DiagnosticReports/` (and
  its `Retired/` subdir) for `*.panic` files, classifies each via
  `detect_panic_signature()`, and returns a list of `dict`
  records with `ts` / `signature` / `explanation` / `pid` /
  `source_file`. Timestamp comes from the embedded `Calendar:
  0x<sec> 0x<usec>` field; falls back to `os.path.getmtime` when
  absent. **Never raises** — returns `[]` if the directory is
  unreadable (modern macOS requires admin privileges for
  `/Library/Logs/DiagnosticReports/`).

- **`ingest_panics_jsonl(*, report_dir=None, jsonl_path=None)`**
  appends new panic records into a dedupe'd JSONL archive
  (default `~/.cache/metal-guard/panics.jsonl`). Dedupes by both
  `source_file` and `(ts_bucket, pid)` event key to handle the
  macOS quirk where a single panic produces both
  `.contents.panic` and `panic-full-...` copies. Idempotent:
  returns the number of new records (0 once caught up).
  Best-effort writes — panic archival must never itself crash the
  caller.

- **`CircuitBreaker`** (L9 secondary defence). Reads the JSONL
  archive and refuses new workers when **≥2 panics within the
  trailing 1 h** (both thresholds configurable). A trip persists
  via `~/.cache/metal-guard/breaker.json` so the cooldown survives
  process restart. `check()` raises `MLXCooldownActive` which
  carries `panic_count`, `window_sec`, `cooldown_until`, and
  `remaining_sec` so the caller can surface an HTTP 503 / task
  decline / CLI exit rather than retry-and-panic. `status()`
  returns a dashboard-safe snapshot; `clear()` is the operator
  override.

- **`detect_panic_signature(text)`** classifies a kernel-panic log
  snippet into one of four signatures:
  `prepare_count_underflow` (IOGPUMemory.cpp:492 — mlx-lm#883 /
  #1015), `pending_memory_set` (IOGPUGroupMemory.cpp:219 —
  mlx#3346), `ctxstore_timeout` (mlx#3267), and a `metal_oom`
  fallback. Returns `(None, None)` for panics that do not match
  any known MLX-related signature — callers can route those to a
  generic bucket.

- **`kv_cache_clear_on_pressure(available_gb,
  growth_rate_gb_per_min)`** — ready-made callback for
  `MetalGuard.start_kv_cache_monitor(on_pressure=...)`. Calls
  `mx.clear_cache()` and logs the trigger. No-op when MLX is not
  importable.

- **`MetalGuard.detect_hardware()`** now also returns
  `gpu_driver_version` (the IOGPUFamily kext bundle version read
  via `kextstat`/`ioreg`). Panic reports on mlx#3186 pin the
  fault to this specific kext, so recording the driver revision
  at startup adds forensic context for future crash correlation.
  Value is `None` if the kext reader fails.

### Tests

- 31 new tests under `tests/test_l9_*.py` covering CadenceGuard
  (8), CircuitBreaker (9), panic ingest (8), and signature
  detection (6). Full suite: **157 passed** (126 existing + 31
  new) on Python 3.14.

### Rationale

v0.7.1 closed the prefill-allocation gap (R4 auto-wired into
`_worker_main`). But a workload that stayed under the R4 ceiling
could still panic if it loaded a new model while the previous
one's IOGPU accounting hadn't fully drained. The Harper panic on
2026-04-16 reproduced exactly that path: two 4-bit loads within
~40 s. L9 enforces ≥180 s cadence per-model and trips a 1 h
cooldown after any 2-panic cluster in a rolling hour. Empirically
this window catches catastrophic clusters without punishing the
steady-state background rate.

### Path defaults

Open-source defaults use `~/.cache/metal-guard/` for all L9
artifacts:

- `~/.cache/metal-guard/cadence.json` — CadenceGuard timestamps
- `~/.cache/metal-guard/panics.jsonl` — panic archive
- `~/.cache/metal-guard/breaker.json` — CircuitBreaker state

All three are overridable via constructor / keyword arguments.

## [0.7.1] — 2026-04-16

Patch release wiring R4 (`require_prefill_fit`) into the actual call
paths so prevention is **always-on** instead of opt-in. v0.7.0 shipped
R4 as a public helper but callers had to remember to invoke it; a
workload calling `call_model_isolated()` directly would still hit an
IOGPU panic on an over-large prefill.

### Changed

- `_worker_main` (spawned by `MLXSubprocessRunner`) now calls
  `require_prefill_fit()` between prompt formatting and the actual
  `gen_fn(...)` dispatch. Tokenises the formatted prompt through the
  already-loaded tokenizer (falls back to `len(prompt) // 4` if
  tokenisation fails), adds `max_tokens`, looks up dims in
  `KNOWN_MODELS`, and queries `memory_stats().available_gb`.
  `MetalOOMError` raised by the guard is caught by the worker's
  existing `except Exception` handler and returned to the parent as
  a normal error reply — no crash, no SIGABRT.
- `bench_scoped_load()` logs a `describe_prefill_plan(context=131072)`
  advisory at scope entry when the model is in `KNOWN_MODELS`. Gives
  the bench harness ahead-of-time visibility into which cells would
  be refused before it wastes time loading the model.

### Rationale

Harper's 2026-04-15 kernel panic (Mistral-Small-3.2-24B × 131 k ×
8-bit, estimate ≈ 30 GB single allocation) had R4 **available** in
the library for ~17 hours before the actual panic — but nothing was
calling it. The guard's value is zero if the prevention path is not
reached on the happy path. This release closes the gap: any workload
going through `MLXSubprocessRunner` (the subprocess-isolated call
path, recommended for production) now gets R4 protection
automatically.

Unknown models continue to skip silently (debug log only); adding a
model to `KNOWN_MODELS` is the one-line opt-in for any additional
shape.

### Tests

126 passed, no regressions on existing suite. The prevention path
itself is exercised by the Harper-local fork's test_prefill_guard_wire
suite (5 cases covering known-model-at-131k raise, unknown-model
skip, small-context pass, unpopulated stats skip, and tokenizer
failure fallback).

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
