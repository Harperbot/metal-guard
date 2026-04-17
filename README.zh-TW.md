# MetalGuard

[English](README.md) | **繁體中文** | [日本語](README.ja.md)

Apple Silicon 上 [MLX](https://github.com/ml-explore/mlx) 的 GPU 安全層。

防止 MLX 推論時 Metal 驅動程式 bug 造成的 kernel panic 與 OOM 崩潰 —— 尤其是多模型 pipeline、長時間運行的 server、以及大量 tool calling 的 agent 框架。

**當前版本：** v0.8.0 — 發行歷史與每個功能的背景見 [CHANGELOG.md](CHANGELOG.md)。

## 搜這些字串跑來的？你來對地方了。

如果你的 Mac 在跑 MLX 時 panic / 重開 / 崩潰，搜到以下任一字串，metal-guard 就是為你寫的：

- `IOGPUMemory.cpp:492 completeMemory() prepare count underflow`
- `IOGPUMemory.cpp:550` Apple Silicon 在 MLX 下 kernel panic
- `kIOGPUCommandBufferCallbackErrorOutOfMemory`
- `mlx::core::gpu::check_error` → `std::terminate` → `abort`（SIGABRT）
- `mlx::core::metal::GPUMemoryAllocator` / `fPendingMemorySet`
- `IOGPUGroupMemory.cpp:219` pending memory set panic
- `mlx_lm.generate` 推論中途崩掉，連父 Python 程序也死
- `mlx_lm.server` 長時間壓力下 OOM kernel panic / Mac 重開
- `mlx_vlm` TurboQuant decode T=1 靜默腐蝕（`mlx-vlm#967`）
- 崩潰報告內看到 `com.apple.iokit.IOGPUFamily`（104.x / 129.x）
- Maintainer 提過 `AGX_RELAX_CDM_CTXSTORE_TIMEOUT`
- `ImpactingInteractivity` / GPU watchdog 在 MacBook 殺掉 MLX
- Gemma 4 / Mistral-Small / Pixtral / Llama 4-bit 輸出亂碼
- M1 / M2 / M3 / M4（Max / Ultra / Pro）Mac Studio / MacBook Pro kernel panic
- 長 context（≥ 65 k）prefill 觸發重開
- `transformers` 5.0 / 5.5 讓 `mlx_vlm.load` import 錯誤
- 連續載入 MLX 模型導致 IOGPU underflow panic

相關上游追蹤：`ml-explore/mlx#3186` / `#3346` / `#3348` / `#3350` / `#3384` / `#3390`、`ml-explore/mlx-lm#883` / `#854` / `#897` / `#1015` / `#1047`、`Blaizzy/mlx-vlm#943` / `#967` / `#999` / `#1011` / `#1016`。metal-guard 透過 `check_version_advisories()` 監看這些 issue，若安裝的版本受影響會在啟動時警告。

## 問題

Apple Silicon 的 Metal GPU 驅動有一個 bug：GPU 記憶體管理失敗時，**kernel 會 panic 整台機器**，而不是乾淨地殺掉 process。

```
panic(cpu 4 caller 0xfffffe0032a550f8):
  "completeMemory() prepare count underflow" @IOGPUMemory.cpp:492
```

任何依序載入、卸載多個 MLX 模型的流程都會中招 —— Metal 驅動內部的 reference count 會 underflow，造成無法回復的 kernel panic 把機器重開。**不是你的程式碼有問題**，是驅動程式層級的 bug，沒有修復時程。見 [ml-explore/mlx-lm#883](https://github.com/ml-explore/mlx-lm/issues/883)。

### 誰會中

| Workload | 風險 | 原因 |
|----------|------|------|
| 單一模型 server（LM Studio） | 低 | 只有一個模型，不切換 |
| 多模型 pipeline | **高** | 每次 load/unload 切換都可能 panic |
| 長時間運行 server（`mlx_lm.server`） | **高** | KV cache 無邊界成長，Metal buffer 累積 |
| Agent 框架 + tool calling | **高** | 每段對話 50–100 次短 generate() |
| TurboQuant KV cache 壓縮 | **高** | 記憶體逼近上限 |
| 24/7 daemon | **嚴重** | 記憶體漂移跨日，沒有自然清理點 |

## 安裝

```bash
pip install metal-guard
```

也可以直接把 `metal_guard.py` 丟進專案 —— 除 Python 標準庫與可選 `mlx` 外沒有依賴。

## 快速開始

```python
from metal_guard import metal_guard, require_cadence_clear, CircuitBreaker

# 1. 拒絕連續載入（L9）
require_cadence_clear("mlx-community/gemma-4-26b-a4b-it-4bit")

# 2. panic 密集出現後拒絕新 worker（L9）
CircuitBreaker().check()

# 3. 註冊 GPU 綁定 thread
import threading
thread = threading.Thread(target=run_mlx_generate, daemon=True)
thread.start()
metal_guard.register_thread(thread)
thread.join(timeout=120)

# 4. 安全卸載模型（L1 + L2）
metal_guard.wait_for_threads()
metal_guard.safe_cleanup()            # gc + flush GPU + cooldown

# 5. OOM 保護推論（L3）
result = metal_guard.oom_protected(generate, model, tokenizer, prompt=p)

# 6. 載入前記憶體壓力檢查（L4）
metal_guard.ensure_headroom(model_name="my-model-8bit")

# 7. 崩潰後期鑑識用的 breadcrumb
metal_guard.breadcrumb("LOAD: my-model-8bit START")
```

依硬體自動取安全預設：

```python
config = MetalGuard.recommended_config()
metal_guard.start_watchdog(
    warn_pct=config["watchdog_warn_pct"],
    critical_pct=config["watchdog_critical_pct"],
)
metal_guard.start_kv_cache_monitor(headroom_gb=config["kv_headroom_gb"])
```

## 功能

MetalGuard 依**防禦層（L1–L9）**組織，加上一組**預防性 helper（R 系列）**。所有功能都從同一個 `metal_guard` 模組取用。每項功能何時落地、對應哪次事件，見 [CHANGELOG.md](CHANGELOG.md)。

### L1 — Thread 追蹤

註冊任何會碰 Metal 的 thread，cleanup 才能等 GPU 工作做完才呼叫 `mx.clear_cache()`。

| API | 作用 |
|---|---|
| `metal_guard.register_thread(thread)` | 把 GPU 綁定 thread 加入註冊表 |
| `metal_guard.wait_for_threads(timeout=None) -> int` | 阻塞到註冊的 thread 結束；回傳仍存活的數量 |

### L2 — 安全清理

依序的清理流程，避免「主 thread 已釋放，worker thread 還在 generate」的競態 —— 原始 panic 根因。

| API | 作用 |
|---|---|
| `metal_guard.flush_gpu()` | `mx.eval(sync) + mx.clear_cache()` —— 只能在 `wait_for_threads()` 之後 |
| `metal_guard.safe_cleanup()` | 完整流程：wait → `gc.collect` → flush → cooldown |
| `metal_guard.guarded_cleanup()` | Context manager，退出時跑 `safe_cleanup()` |
| `kv_cache_clear_on_pressure(available_gb, growth_rate_gb_per_min)` | KV 監控器的現成 `on_pressure` callback |

### L3 — OOM 回復

把 C++ 端的 Metal OOM 轉成可 catch 的 Python 例外，自動清理並可選重試。

| API | 作用 |
|---|---|
| `metal_guard.oom_protected(fn, *args, max_retries=1, **kwargs)` | 執行時 OOM catch → 清理 → 重試 |
| `metal_guard.oom_protected_context()` | Context manager 版本 |
| `metal_guard.is_metal_oom(exc) -> bool` | 分類任意例外 |
| `MetalOOMError` | 可 catch 的例外，附 `MemoryStats` |

### L4 — 載入前記憶體檢查

拒絕裝不下的載入，並可從 HF model ID 估算模型大小。

| API | 作用 |
|---|---|
| `metal_guard.can_fit(model_size_gb, overhead_gb=2.0) -> bool` | 不拋例外的檢查 |
| `metal_guard.require_fit(model_size_gb, model_name, overhead_gb=2.0)` | 清理後仍裝不下就拋 `MemoryError` |
| `MetalGuard.estimate_model_size_from_name(name)`（static） | 從名字解出參數量 + 量化 → GB 估值 |

### L5 — 長時間運行安全

給 `mlx_lm.server`、agent 框架、24/7 daemon 用。

| API | 作用 |
|---|---|
| `metal_guard.memory_stats() -> MemoryStats` | 快照（active / peak / limit / available / pct） |
| `metal_guard.is_pressure_high(threshold_pct=67.0) -> bool` | 快速壓力檢查 |
| `metal_guard.ensure_headroom(model_name, threshold_pct=67.0)` | 壓力高才清理，否則 no-op |
| `metal_guard.log_memory(label, model_name)` | 只記錄不清理 |
| `metal_guard.start_periodic_flush(interval_secs=300)` | 背景定時 flush |
| `metal_guard.start_watchdog(interval_secs, warn_pct, critical_pct, on_critical)` | 記憶體漂移監控，逐級反應 |
| `metal_guard.start_kv_cache_monitor(interval_secs, headroom_gb, growth_rate_warn, on_pressure)` | KV 成長監控，OOM 之前就觸發 |
| `bench_scoped_load(model_id, ...)` | 連續 benchmark 的 context manager，保證下次載入前已卸載 |

### L6 — 雙模式切換

執行時可切換 defensive / observer 姿態，讓你能 A/B 上游 mitigation 而不必改程式碼。

| API | 作用 |
|---|---|
| `current_mode() -> str` | `"defensive"`（預設）或 `"observer"` |
| `is_defensive() / is_observer() -> bool` | 方便判斷 |
| `describe_mode() -> dict` | 模式名稱、說明、env var |

### L7 — Subprocess 隔離

在新的 `multiprocessing` 子程序跑 MLX，kernel 層級 abort 就殺不到父程序。

| API | 作用 |
|---|---|
| `MLXSubprocessRunner(model_id, ...)` | 持續性的 worker 子程序，崩潰自動重生 |
| `call_model_isolated(model_id, prompt, ...)` | 一次性 helper：spawn → generate → 收尾 |
| `shutdown_all_workers()` | 結束時強制停掉所有 runner |
| `SubprocessCrashError / SubprocessTimeoutError` | Typed 失敗給 caller 處理 |

### L8 — 跨程序互斥鎖

`MLX_LOCK_PATH` 下的檔案鎖，避免 bench / server / pipeline 同時在同一台機器初始化 Metal。

| API | 作用 |
|---|---|
| `acquire_mlx_lock(label, force=False)` | 被佔用時拋 `MLXLockConflict`；`force=True` 會 SIGTERM 持有者並帶 timeout + cooldown |
| `release_mlx_lock() -> bool` | 本程序持有時釋放 |
| `read_mlx_lock() -> dict \| None` | 不阻塞的查詢；自動處理 stale + zombie |
| `mlx_exclusive_lock(label)` | Context manager：進入取得、離開釋放 |

### L9 — Cadence、panic ingest、circuit breaker *(v0.8.0)*

前八層都過了之後的最後一道防線。源自一次 **SIGABRT 層級也攔不到**的 kernel panic —— Python 看到任何東西之前，機器已經重開了。唯一的防法是一開始就不讓那個 trigger 發生。

| API | 作用 |
|---|---|
| `CadenceGuard(path=None, *, min_interval_sec=180)` | 永續化的 per-model 載入時間戳記 |
| `CadenceGuard.check(model_id)` / `.mark_load(model_id)` | 太快再載就拋 `CadenceViolation` |
| `require_cadence_clear(model_id, *, min_interval_sec=180)` | check + mark 的原子 helper |
| `parse_panic_reports(directory=None, *, since_ts=None)` | 掃描 `/Library/Logs/DiagnosticReports/*.panic` 並分類 |
| `ingest_panics_jsonl(*, report_dir=None, jsonl_path=None) -> int` | 去重附加到 `~/.cache/metal-guard/panics.jsonl` |
| `CircuitBreaker(*, window_sec=3600, panic_threshold=2, cooldown_sec=3600)` | panic 密集後拒絕新 worker |
| `CircuitBreaker.check() / .status() / .clear()` | 守門、儀表板、操作員 override |
| `detect_panic_signature(text) -> (name, explanation)` | 把 panic log 分類成 `prepare_count_underflow` / `pending_memory_set` / `ctxstore_timeout` / `metal_oom` |

### 硬體感知

| API | 作用 |
|---|---|
| `MetalGuard.detect_hardware() -> dict`（static） | 晶片、GPU 記憶體、建議 working set、tier、IOGPUFamily kext 版本 |
| `MetalGuard.recommended_config() -> dict`（classmethod） | 在偵測到的硬體上，每個 L 層對應的安全預設 |

### 版本 advisory 與上游 patch

| API | 作用 |
|---|---|
| `check_version_advisories(packages=None) -> list[dict]` | 安裝的 `(mlx, mlx-lm, mlx-vlm, transformers)` 版本踩到已知 advisory 就警告 |
| `install_upstream_defensive_patches(force=False) -> dict[str, bool]` | Idempotent、版本門控的 monkey-patch |

### 系統稽核

| API | 作用 |
|---|---|
| `audit_wired_limit() -> dict` | 警告危險的 `iogpu.wired_limit_mb` override（mlx-lm#1047） |
| `read_gpu_driver_version() -> str \| None` | IOGPUFamily kext 版本（mlx#3186） |
| `log_system_audit_at_startup() -> dict` | CLI / FastAPI lifespan 用的便利包裝 |

### R 系列預防 helper

| API | 作用 |
|---|---|
| `ModelDims`、`lookup_dims(model_id)`、`KNOWN_MODELS` | GQA 感知的 curated 模型維度查表 |
| `estimate_prefill_peak_alloc_gb(context_tokens, dims)` | 保守的 per-layer + 全 KV 上界估算 |
| `require_prefill_fit(context_tokens, dims, available_gb, ...)` | 在任何 30 GB 單次分配 panic 之前就拋 `MetalOOMError` |
| `recommend_chunk_size(context_tokens, dims, ...)` | 二分搜尋建議 chunk 大小（純建議） |
| `describe_prefill_plan(context_tokens, model_id_or_dims, available_gb)` | Dashboard 安全、容忍 null 的摘要 |
| `KVGrowthTracker(...).start / add_bytes / finalize / snapshot` | Per-request 累計 KV 守門 —— 抓全域壓力監控錯過的暴走請求 |
| `detect_process_mode() -> ProcessMode` | `"server" / "embedded" / "notebook" / "cli" / "subprocess_worker"` |
| `apply_mode_defaults(mode=None) -> dict` | 依模式給 timeout 與 ceiling |
| `describe_process_mode() -> dict` | Dashboard 摘要 |
| `format_panic_for_apple_feedback(forensics, ...)` | 可貼進 Apple Feedback Assistant 的報告 |

### 崩潰鑑識

| API | 作用 |
|---|---|
| `metal_guard.breadcrumb(msg)` | 寫入 fsync 過的 breadcrumb 記錄（預設 `logs/metal_breadcrumb.log`） |

## 預設路徑

L9 所有產出檔案都在 `~/.cache/metal-guard/`：

| 檔案 | 用途 | 可經由此蓋掉 |
|---|---|---|
| `~/.cache/metal-guard/cadence.json` | CadenceGuard 時間戳 | `CadenceGuard(path=...)` |
| `~/.cache/metal-guard/panics.jsonl` | Panic 歸檔 | `ingest_panics_jsonl(jsonl_path=...)` / `CircuitBreaker(jsonl_path=...)` |
| `~/.cache/metal-guard/breaker.json` | CircuitBreaker 狀態 | `CircuitBreaker(state_path=...)` |

Breadcrumb 記錄預設為相對路徑 `logs/metal_breadcrumb.log`，以 `MetalGuard(breadcrumb_path=...)` 蓋掉。

## 架構

```
┌─────────────────────────────────────────────────┐
│              你的應用程式碼                     │
│  Agent loop / Server / Pipeline / Daemon        │
└──────────────────┬──────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────┐
│              MetalGuard                         │
│                                                 │
│  L9 CadenceGuard ──── 拒絕連續載入              │
│  L9 CircuitBreaker ── panic 密集後拒絕          │
│  L8 Process Lock ──── 跨程序互斥                │
│  L7 Subprocess ────── panic 隔離的 worker       │
│  L6 Dual mode ─────── defensive / observer      │
│  L5 Watchdog ──────── 記憶體 + KV 漂移告警      │
│  L4 Pre-load check ── can_fit / require_fit     │
│  L3 OOM recovery ──── catch + cleanup + retry   │
│  L2 Safe cleanup ──── gc + flush + cooldown     │
│  L1 Thread registry ─ cleanup 前先等            │
│  R4 Prefill guard ─── 超過 ceiling 的 prefill 拒絕 │
│  R5 KV tracker ────── per-request KV 守門       │
│  R8 Apple Feedback ── 鑑識 formatter            │
└──────────────────┬──────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────┐
│           MLX + Metal Driver                    │
│  ⚠️  驅動 bug：panic 而非 OOM                   │
└─────────────────────────────────────────────────┘
```

## 實測

- Mac Studio M1 Ultra（64 GB）—— 裝 MetalGuard 前 9 次 kernel panic，L9 上線後 24 h 無 panic
- 10 人批次 pipeline：約 90 次模型 load/unload 循環、994 秒、零崩潰
- 模型：Mistral-Small-3.2-24B、Phi-4-mini、Gemma-4-26B / 31B、Pixtral-12B、LFM2-VL-3B（4-bit 與 8-bit）

## 限制 —— 這是硬撐，不是修好

MetalGuard 是 **userspace 的防禦層**。根本 bug 在 Apple 的 IOGPUFamily kext（[mlx#3186](https://github.com/ml-explore/mlx/issues/3186)），Python 端根本碰不到。實際做的是：

1. **降低觸發率** —— L1–L5 與 L9 CadenceGuard 避開已知觸發路徑（連續載入、thread 競態 cleanup、KV 無限成長、prefill 超過單次分配上限）。
2. **縮小爆炸半徑** —— L7 把 MLX 放到子程序，可 catch 的 abort 只會殺子程序。但 *kernel* panic 還是會整台機器重開；subprocess 隔離只是讓你知道當時哪個模型握著 GPU。
3. **避免重開後雪崩** —— L9 CircuitBreaker 在一小時內 ≥ 2 次 panic 後拒絕新 worker，讓機器不會重開後立刻又載入同樣模型再 panic 一次。

**panic 還是可能發生**（尤其 [mlx#3390](https://github.com/ml-explore/mlx/issues/3390) —— 不可 catch 的 completion-handler abort 是在 `com.Metal.CompletionQueueDispatch` 上派發的，任何 Python signal handler 都來不及接）。Harper 的機器從每天約 1.4 次 panic 降到 L9 上線後 24 小時零 panic，這是**降低風險**，不是**消除風險**。Apple 修好 kext 之前，這就是 Python-side 防禦層能做到的上限。

## 相關上游 issue

| Issue | 問題 | 對應功能 |
|---|---|---|
| [mlx#3186](https://github.com/ml-explore/mlx/issues/3186) | IOGPUFamily kernel panic（標準案） | L1/L2/L8/L9 + `read_gpu_driver_version` |
| [mlx#3346](https://github.com/ml-explore/mlx/issues/3346) | `fPendingMemorySet` 第二簽名 | `detect_panic_signature` + L9 |
| [mlx#3348](https://github.com/ml-explore/mlx/pull/3348) | CommandEncoder thread-local | Advisory 門控的 observer mode |
| [mlx#3350](https://github.com/ml-explore/mlx/issues/3350) | MetalAllocator buffer pool 成長 | Advisory + `mx.set_cache_limit` 指引 |
| [mlx#3384](https://github.com/ml-explore/mlx/issues/3384) | 4-bit SDPA 數值偏移 | `check_version_advisories` |
| [mlx#3390](https://github.com/ml-explore/mlx/issues/3390) | 不可 catch 的 completion-handler abort | L7 subprocess 隔離 + `AGX_RELAX_CDM_CTXSTORE_TIMEOUT` |
| [mlx-lm#883](https://github.com/ml-explore/mlx-lm/issues/883) / [#1015](https://github.com/ml-explore/mlx-lm/issues/1015) | KV cache 成長造成 kernel panic | L1 thread + L2 safe cleanup |
| [mlx-lm#854](https://github.com/ml-explore/mlx-lm/issues/854) | Server OOM crash | L3 `oom_protected` + L5 periodic flush |
| [mlx-lm#897](https://github.com/ml-explore/mlx-lm/issues/897) | `mlx_lm.server` 與 transformers ≥ 5.0 衝突 | `check_version_advisories` |
| [mlx-lm#1047](https://github.com/ml-explore/mlx-lm/issues/1047) | `wired_limit` 與 panic 的關聯 | `audit_wired_limit` |
| [mlx-lm#1128](https://github.com/ml-explore/mlx-lm/issues/1128) | `TokenizerWrapper.think_start_id` 崩潰 | `install_upstream_defensive_patches` |
| [mlx-vlm#943](https://github.com/Blaizzy/mlx-vlm/issues/943) / [#967](https://github.com/Blaizzy/mlx-vlm/pull/967) / [#999](https://github.com/Blaizzy/mlx-vlm/issues/999) | TurboQuant / cache-thrash / Gemma4 輸出亂碼 | `check_version_advisories` |

## License

MIT
