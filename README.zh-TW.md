# MetalGuard

[English](README.md) | **繁體中文** | [日本語](README.ja.md)

Apple Silicon 上 [MLX](https://github.com/ml-explore/mlx) 的 GPU 安全層。

防止 MLX 推論時 Metal 驅動程式 bug 造成的 kernel panic 與 OOM 崩潰 —— 尤其是多模型 pipeline、長時間運行的 server、以及大量 tool calling 的 agent 框架。

**當前版本：v0.11.7** — 發行歷史與每個功能的背景見 [CHANGELOG.md](CHANGELOG.md)。

### v0.10 帶來什麼

metal-guard 現在涵蓋 Apple Silicon kernel panic 的**完整生命週期** —— panic 之前、之中、之後：

| 階段 | Layer | 作用 |
|---|---|---|
| 之前 | L1–L9（v0.1–v0.9） | thread 追蹤 / cleanup / OOM 復原 / 載入前檢查 / 長跑安全 / dual-mode / subprocess 隔離 / cross-process lock / cadence + circuit breaker |
| **重開後（新）** | **L10 panic cooldown gate** | panic 後 2h–72h 拒絕重啟 MLX 工作 — launchd 自動 respawn plist 時不會立刻再 panic |
| **panic 前警告（新）** | **L11 subprocess orphan monitor** | 偵測 `SUBPROC_PRE` 沒有對應 `SUBPROC_POST` 超過 90s — 在 kernel 殺 worker 之前先 SIGKILL |
| **重開後（新）** | **L12 postmortem auto-collect** | 把 `panic-full-*.panic` + breadcrumb 尾段 + `mx.metal` 統計 + `index.md` 摘要打包進一個目錄 |
| **跨程序狀態（新）** | **L13 status snapshot** | versioned JSON 給 menu bar / dashboard / ssh 巡檢消費 —— 下游不需 `import metal_guard` |
| 全 layer | **`KNOWN_PANIC_MODELS` 登記表** | 社群共筆 `(model, 硬體, panic 簽名, workload, workaround)` 資料 —— 見下方 [社群 panic 模型登記表](#-社群-panic-模型登記表--known_panic_models) |

v0.10 帶來的新 CLI：

```bash
metal-guard panic-gate            # L10: rc=0 通過 / rc=2 cooldown / rc≥3 gate 壞
metal-guard postmortem ./bundle   # L12: 重開後採集
metal-guard status-write --once   # L13: 寫 JSON snapshot
metal-guard orphan-scan           # L11: pre-panic stuck-worker 偵測
metal-guard ack                   # L10: 清 lockout（必須 user 親手）
mlx-safe-python -c "import torch" # 互動 shell 守衛 — cooldown 中拒 ad-hoc import
```

v0.10 把 Harper 私用 fork 內 4 個防線在 production 驗證兩週、跨 11 次 panic 後 promote 到公開版。從早期版本就有的誠實 caveat 仍適用：metal-guard 縮的是 Apple IOGPU driver bug 周圍的 race window —— 它**沒修** bug 本身。v0.10 把防禦面從「執行中」擴到「重開後」+「kernel kill 之前」。

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

## 📋 社群 panic 模型登記表 — `KNOWN_PANIC_MODELS`

**user 共筆整理「在 Apple Silicon Mac 上會 kernel panic 的 MLX 模型清單」，含硬體脈絡、根因假說、與經驗證的 workaround。**

Apple IOGPUFamily driver bug 沒有修復時程。雖然 bug 在 upstream，但**哪些模型在哪些 workload 下會踩雷是社群可知的事** —— 只是目前散落在 GitHub issue 串、lmstudio bug 報告、Discord 截圖、跟個人沒上傳的 `panic-full-*.panic` 檔裡。

metal-guard 提供結構化的整理空間：

```python
from metal_guard import check_known_panic_model, warn_if_known_panic_model

# 載入前檢查
advisory = check_known_panic_model("mlx-community/gemma-4-31b-it-8bit")
if advisory is not None:
    print(advisory["recommendation"])

# 或載入時 fire-and-forget 警告（每 process 每 model_id 只警告一次）
warn_if_known_panic_model(model_id)
```

每筆登記項含：
- **`panic_signature`** — 跟你 `panic-full-*.panic` log 比對的精確 `IOGPUMemory.cpp:NNN` 行號 + 關鍵字
- **`reproductions`** — production 數據點（硬體 / RAM / panic 距載入時間 / workload）
- **`community`** — 其他踩同雷的 GitHub issue / lmstudio bug / 論壇 thread 交叉引用
- **`recommendation`** — 可行 workaround（換 backend / 改 model / cadence 設定）
- **`upstream`** — 追蹤底層 driver bug 的 GitHub issue 連結

### 怎麼貢獻

如果你在某個 MLX 模型上踩過 kernel panic **且 metal-guard 防線都已啟用**，你的數據點有價值。開一個 [Known Panic Model report](https://github.com/Harperbot/metal-guard/issues/new?template=known-panic-report.yml) — issue template 會引導你填 schema（model ID / 硬體 / panic 簽名 / workload / panic 距載入時間 / 經驗證的 workaround）。Schema 文件見 [CONTRIBUTING.md](CONTRIBUTING.md#known-panic-models-schema)。

登記表設計上保守 — 入庫條件是「production 確實重現」或「upstream issue 有清楚簽名」。我們不希望 false positive 把多數 user 都正常跑的模型黑掉。

**為什麼不直接讀 mlx#3186 留言？** 因為那條 thread 混了硬體報告、假說、嘗試修法、跟無關討論。Registry 把它蒸餾成 code 可 `check_known_panic_model()` 的結構化 advisory — 而且你的 panic 報告不會消失在 50 條留言裡。

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

> **PyPI 狀態（2026-04-27）**：metal-guard **尚未上 PyPI**。在 v0.10.x 上 PyPI 之前，請從以下三條路擇一。急需 PyPI 請開 issue。

### 選項 A — pip 從 GitHub 裝（推薦，一行）

從 tag release 裝 —— 會拿到 `metal-guard` 跟 `mlx-safe-python` 兩個 console scripts 跟 `metal_guard` Python module：

```bash
pip install "git+https://github.com/Harperbot/metal-guard.git@v0.11.7"
```

裝完：

```bash
metal-guard --version          # → metal-guard 0.11.7
metal-guard panic-gate         # L10 cooldown 判斷
metal-guard status             # 完整 snapshot
mlx-safe-python -c "import torch"   # 互動 shell 守衛
```

升級新版本：`pip install --upgrade "git+https://github.com/Harperbot/metal-guard.git@vX.Y.Z"`。

### 選項 B — 單檔丟入（零安裝、不用 pip）

`metal_guard.py` **零依賴**（除 Python 標準庫 + 可選 `mlx`）。下載一次，直接 import：

```bash
mkdir -p ~/lib/metal-guard
curl -L -o ~/lib/metal-guard/metal_guard.py \
  https://raw.githubusercontent.com/Harperbot/metal-guard/v0.11.7/metal_guard.py
```

程式裡：

```python
import sys; sys.path.insert(0, "/Users/<你>/lib/metal-guard")
import metal_guard as mg
verdict = mg.evaluate_panic_cooldown()
print(verdict.exit_code, verdict.reason)
```

這條路適合 launchd plist wrapper、panic recovery script、CI runner —— 即使 Python install 其他部分壞了也能跑。

### 選項 C — 本地 clone（開發 / 跑 tests）

```bash
git clone https://github.com/Harperbot/metal-guard.git
cd metal-guard
pip install -e ".[test]"
pytest -q
```

editable install 會即時反映本地改動。`[test]` extra 會拉 `pytest>=7.0`。

### 驗證安裝

選項 A 或 C 後 gate 該自檢通過：

```bash
$ metal-guard panic-gate
🟢 PROCEED  no recent IOGPU panics
  24h=0 72h=0
$ metal-guard status
metal-guard 0.11.7  🟢 OK
  mode        defensive — defensive mode (default)
  panics      0 in last 72h
  ...
```

若 `metal-guard` 不在 `PATH`，可能 `pip --user` bin dir 沒加 — 用 `python3 -m metal_guard_cli panic-gate` 替代。

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

## 已知受影響模型（v0.9.0, 2026-04）

有些模型的 race window 夠寬，MetalGuard 能收窄但收不掉。碰到時我們記在這邊，讓你在 production 載入前可以先做判斷。

### `mlx-community/gemma-4-31b-it-8bit` —— 重複犯案

Harper Mac Studio 上 **相隔 24 小時、同一條 pipeline、同一個 model**，兩次 production kernel panic，panic 簽名一致：`IOGPUMemory.cpp:492 "completeMemory() prepare count underflow"`。

| #   | 本地時間         | PID   | Spawn → panic | 情境                                                    |
|-----|------------------|-------|--------------:|---------------------------------------------------------|
|  7  | 2026-04-23 03:14 | 67840 |        約 6 分 | rezivot pipeline，當時還沒 wire 跨模型 cadence           |
| 11  | 2026-04-24 03:14 | 26608 |      約 1.5 分 | 同 #7 pipeline；classic L9 防線都在的情況下，worker ready 後 ~1.5 分仍 panic |

社群交叉佐證（全部 2026-04）：

- [Hannecke —「MLX Crashed My Mac」（Medium）](https://medium.com/@michael.hannecke/how-my-local-coding-agent-crashed-my-mac-and-what-i-learned-about-mlx-memory-management-e0cbad01553c) —— M4 Max 64 GB、同 panic 簽名；pivot 去 `Qwen3-Coder-30B-A3B` MoE。
- [`lmstudio-ai/lmstudio-bug-tracker#1740`](https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/1740)「Gemma-4 31b KV excessive KV cache footprint」—— 同一家族的 KV 膨脹問題：8192 context 就吃 26 GB VRAM；hybrid attention（50 sliding + 10 global）KV cache + 8-bit 權重（約 34 GB）+ full-context KV 把 64 GB Mac 逼過 unified memory 邊緣。
- [`ml-explore/mlx-lm#883`](https://github.com/ml-explore/mlx-lm/issues/883) —— M3 Ultra 96 GB，同 panic 簽名。
- [`ml-explore/mlx#3186`（2026-04-24 留言）](https://github.com/ml-explore/mlx/issues/3186#issuecomment-4314204974) —— 獨立第三方觀測：Mac mini M4 base 32 GB、macOS 26.4.1（`25E253`）、mlx 0.31.2、`mlx-community/Qwen3.6-35B-A3B-4bit`。`mlx_lm.server` 啟動 8 分 16 秒後 panic；`--prompt-cache-bytes 8 GiB` 擋不住；reporter 因此 production serving 改走 `llama.cpp`。留言中明確引用本專案的「two-trigger-path hypothesis」。

**結論。** macOS 26.4.x 沒修這個 bug。macOS 26.5 beta 沒修這個 bug。RAM 加到 96 GB 沒擋住。MetalGuard v0.9.0 收窄了多個 race window（跨模型 cadence、gemma-4 90 秒下限、首次 generate flush、subprocess inference guard），但 Harper 實際 workload 上對這個模型仍無法完全消除 panic。

程式內可查詢 advisory：

```python
from metal_guard import check_known_panic_model, warn_if_known_panic_model

advisory = check_known_panic_model(model_id)
if advisory is not None:
    # 自行決策：拒絕載入、換 backend、或明示 ack 後續載
    ...

# 或 fire-and-forget：同一個 model_id 每個 process 只 log.warning 一次
warn_if_known_panic_model(model_id)
```

## 當 MetalGuard 不夠的時候

如果 v0.9.0 所有防線都上了（B1 + C5 + C7 + CircuitBreaker），同一個模型還是重複 panic，那代表 race window 寬到 userspace 層收不下來。兩個換路方案，依投資報酬率排序：

1. **換 backend。** [Ollama](https://ollama.com/) 與 [`llama.cpp`](https://github.com/ggml-org/llama.cpp) 底層一樣是 Metal MPS，但走 persistent worker 架構，整條 subprocess teardown race 直接繞開。Harper 的 `harper-finance` 專案 2026-04-23 切 Ollama 之後零 panic。[`mlx#3186` 的 M4-base reporter](https://github.com/ml-explore/mlx/issues/3186#issuecomment-4314204974) production serving 因同樣理由改走 `llama.cpp`。代價是原始吞吐（該 report 量到 MLX 在 prefill 快 30–55 %），換到的是「不會把整台機器打翻」。

2. **換模型家族。** Mixture-of-Experts（MoE）變體 —— 例如 [`mlx-community/gemma-4-26b-a4b-it-4bit`](https://huggingface.co/mlx-community/gemma-4-26b-a4b-it-4bit)、`Qwen3-Coder-30B-A3B` —— 每次 forward 的 active-parameter 少得多，KV 成長曲線窄很多。社群紀錄（Hannecke、lmstudio#1740）都收斂到「同生態裡最穩的方案就是換 MoE」。

MetalGuard 跟兩個換路方案是**互補的** —— 就算你改用 Ollama，每個 request spawn 一次 subprocess worker 的話，`subprocess_inference_guard` 依然有用；就算你整套換 backend，只要會 hot-swap 模型，`CadenceGuard` 依然幫得上忙。

### 一條學費教訓的 SOP

我們時間線上的 panic #10（見 [CHANGELOG](CHANGELOG.md)）是 host terminal 上一條 *互動式* `python -c "import sentence_transformers"` 觸發的 —— 一個查版本的指令，跟 production MLX workload 完全無關。任何會 import `torch`、`mlx`、`mlx_lm`、`mlx_vlm`、`sentence_transformers`、`transformers`、`diffusers`、`accelerate` 的動作都會把 Metal MPS backend 初始化起來，然後在 process exit 階段踩到同一個 kernel bug。panic cooldown 作用中時，優先用：

- `pip show <pkg>` 查版本，或
- `python -c "import importlib.metadata as m; print(m.version('<pkg>'))"`（不會 cascade import 整個套件）。

**絕對不要**在 cooldown 作用中跑 `python -c "import <ml-package>; print(<ml-package>.__version__)"`。

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
