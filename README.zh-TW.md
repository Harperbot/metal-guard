# MetalGuard

[English](README.md) | **繁體中文** | [日本語](README.ja.md)

為 Apple Silicon 上的 [MLX](https://github.com/ml-explore/mlx) 提供 GPU 安全防護層。

防止因 Metal 驅動程式 bug 導致的 kernel panic 和 OOM crash——特別是多模型管線、長時間運行的伺服器，以及大量 tool calling 的 agent 框架。

**目前版本：** v0.7.0 — 完整發佈歷史見 [CHANGELOG.md](CHANGELOG.md)。

## 從以下任一關鍵字搜尋進來的？你找對地方了

Mac 跑 MLX 的時候整機 panic / 重開 / 當掉，又剛好從下列字串搜進來的話——
metal-guard 就是為你寫的：

- `IOGPUMemory.cpp:492 completeMemory() prepare count underflow`
- `IOGPUMemory.cpp:550` kernel panic（Apple Silicon + MLX）
- `kIOGPUCommandBufferCallbackErrorOutOfMemory`
- `mlx::core::gpu::check_error` → `std::terminate` → `abort`（SIGABRT）
- `mlx::core::metal::GPUMemoryAllocator` / `fPendingMemorySet`
- `mlx_lm.generate` 生成中途 crash，parent Python process 跟著死
- `mlx_lm.server` OOM kernel panic / Mac 重開（長時間跑 server）
- `mlx_vlm` TurboQuant decode T=1 silent 資料腐蝕（`mlx-vlm#967`）
- panic report 提到 `com.apple.iokit.IOGPUFamily`（104.x / 129.x）
- Maintainer 建議設 `AGX_RELAX_CDM_CTXSTORE_TIMEOUT`
- Gemma 4 / Mistral-Small / Pixtral / Llama 4-bit 輸出變亂碼
- M1 / M2 / M3 / M4（Max / Ultra / Pro）Mac Studio / MacBook Pro kernel panic
- 長 context（≥ 65k）prefill 觸發整機重開
- `mlx_vlm.load` 丟 `transformers` 5.0 / 5.5 ImportError

相關上游 tracking issue：`ml-explore/mlx#3186` / `#3346` / `#3390` /
`#3348`、`ml-explore/mlx-lm#883` / `#854` / `#1047` / `#1015`、
`Blaizzy/mlx-vlm#967` / `#943` / `#1011` / `#1016`。metal-guard 透過
`check_version_advisories()` 監控這些 issue，並在啟動時對受影響版本發
WARNING log。

## 問題是什麼

Apple Silicon 的 Metal GPU 驅動程式有一個 bug：**當 GPU 記憶體管理失敗時，它不會優雅地殺掉 process，而是直接讓整台電腦 kernel panic 重開機。**

```
panic(cpu 4 caller 0xfffffe0032a550f8):
  "completeMemory() prepare count underflow" @IOGPUMemory.cpp:492
```

**這不是你的程式的問題。** 這是驅動程式層級的 bug，目前沒有修復時程。參見 [ml-explore/mlx-lm#883](https://github.com/ml-explore/mlx-lm/issues/883)。

### 誰會受影響

| 使用場景 | 風險 | 原因 |
|---------|------|------|
| 單模型伺服器（LM Studio） | 低 | 一個模型，不切換 |
| 多模型管線 | **高** | 載入→卸載→載入→卸載，每次切換都可能 panic |
| 長時間伺服器（mlx_lm.server） | **高** | KV cache 無限增長，Metal buffer 累積 |
| Agent 框架 + tool calling | **高** | 每個對話 50-100 次短推論，碎片化 Metal buffer 累積 |
| TurboQuant KV cache 壓縮 | **高** | 記憶體更接近上限（50K-200K tokens），OOM 更容易 |
| 24/7 常駐程式（OpenClaw 類型） | **極高** | 記憶體隨天數漂移，沒有自然清理時機 |

## 安裝

```bash
pip install metal-guard
```

或直接把 `metal_guard.py` 複製到專案中——單檔，零依賴。

## 快速開始

```python
from metal_guard import metal_guard

# 1. 追蹤 GPU thread
thread = threading.Thread(target=run_mlx_generate, daemon=True)
thread.start()
metal_guard.register_thread(thread)

# 2. 安全卸載模型
metal_guard.wait_for_threads()
cache.clear()
metal_guard.safe_cleanup()

# 3. 載入前檢查記憶體
metal_guard.ensure_headroom(model_name="my-model-8bit")
```

## v0.7.0 新功能

### 第 7 條 kernel panic root cause — completion-handler abort

Harper 2026-04-16 掃完整個 MLX 生態自 mlx-lm 0.31.2 後的 issue（約 250 條 issue + 80 條 PR），發現一條 metal-guard 原本沒點名的 root cause：`eval.cpp::check_error` 在 `addCompletedHandler(...)` 的 callback 裡拋例外，這些 callback 跑在 Apple 的 `com.Metal.CompletionQueueDispatch`（GCD）queue 上。libdispatch block 不是 exception-safe——`__cxa_throw` → `std::terminate` → `abort()` → 無法攔截的 SIGABRT。Python 在 `mx.eval()` 外的 `try/except` 永遠不會觸發。重複報告：`mlx#3224`（M3 Ultra 跑 6 小時）、`mlx#3317`（M2 Ultra asyncio race）。Umbrella：`mlx#2670`。PR #3318（`check_error_deferred` 模式）被 closed 不 merge——上游認定「throw 後 process state 已 undefined」。

metal-guard 對這條只能**部分緩解**：

- `AGX_RELAX_CDM_CTXSTORE_TIMEOUT=1` 自 v0.5.0 起 module import 時自動設定，降低最常觸發 abort 的 GPU watchdog false-positive 頻率
- 透過 `MLXSubprocessRunner` 的 subprocess 隔離讓 parent 和兄弟 worker 存活；in-flight child 會死、in-flight request 會丟

`_VERSION_ADVISORIES` 已加入此 issue（severity `high` + 完整 mitigation）。

### R4 — Prefill 配置防線

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

Attention score tensor 跟 context 平方成長。Mistral-Small 24B × 131k 估算單次 Metal dispatch 要 ~30 GB——遠超實測約 5 GB 的 single-allocation 天花板（即使 device 上還有 60+ GB 空閒，IOGPUFamily 也會開始 corrupt state）。Harper 2026-04-15 踩到這條，整機 reboot；R4 在 `mlx_lm.load` 跑之前就會 refuse。

`recommend_chunk_size(...)` 用 binary search 找最大可容納的 chunk——純 advisory，metal-guard 不會替 caller 自動分 chunk。

### R5 — Per-request KV 累積 tracker

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

`MetalGuard.start_kv_cache_monitor` 看的是**全 device** Metal pressure——單一長 request 慢慢長 KV cache 可能在 global metric 看起來還正常的時候，已經推過 IOGPUFamily 閾值。per-request tracking 早一步在那個 request 身上攔截。opt-in；未啟用的 request 不會影響 generate loop。

### R6 — Process mode 偵測

```python
from metal_guard import detect_process_mode, apply_mode_defaults

mode = detect_process_mode()  # "server" | "cli" | "subprocess_worker" | ...
cfg = apply_mode_defaults(mode)
# {'mode': 'server', 'generate_timeout_sec': 60.0, 'kv_ceiling_gb': 10.0, ...}
```

`server` 模式嚴格（60 秒 generate timeout、10 GB KV ceiling、4 GB prefill ceiling），`notebook` 寬鬆（600 / 30 / 5）。mlx-lm `#883` / `#854` 把多起 panic 集中在「server 長時間跑、concurrent request 之間沒 flush」的情境——嚴格預設值正是為了守這類 case。

### R8 — Apple Feedback Assistant 格式器

```python
from metal_guard import format_panic_for_apple_feedback

report = format_panic_for_apple_feedback(forensics_dict)
# 直接貼進 Feedback Assistant
```

對齊 `mlx#3186`（FB22091885）的 template。欄位缺失時容錯渲染為 `unknown`；forensic 內含敏感 prompt 時可透過 flag 關掉 breadcrumb 段。

### H7 — MLXSubprocessRunner 現在會取 MLX lock

`MLXSubprocessRunner.__init__` 在 spawn worker 前呼叫 `acquire_mlx_lock(...)`，shutdown / kill 兩個路徑都會 release。這補上了跨 process 互斥的最後一個洞：`bench_scoped_load()` 跟 `call_model()` 早就會取 lock，但 subprocess runner 原本沒有——任何 concurrent MLX acquirer（跑 `bench_scoped_load` 的 pytest、第二個 bench CLI、acceptance test）可以在 worker 還抱著 Metal buffer 的時候合法地覆蓋 lock file。2026-04-15 的真實 kernel panic 就是這個 shape：pytest 從 bench 手上搶到 lock。

worker 本身現在也會把 `METALGUARD_SUBPROCESS_WORKER=1` 寫進 env，讓 child 內部的 `detect_process_mode()` 回傳 `"subprocess_worker"`——避免重複取已經被 parent 持有的 lock。

### 系統層 audit（R2 / R3）現在是 public API

- `audit_wired_limit()` — `sysctl iogpu.wired_limit_mb` 超過 85% 觸發 advisory（依 `mlx-lm#1047`）
- `read_gpu_driver_version()` — 讀 `IOGPUFamily` kext bundle 版本，為未來 `mlx#3186` 類型的 crash 建 forensic 對應基準
- `log_system_audit_at_startup()` — 一次跑完以上兩項的 convenience entry point

---

## v0.6.0 新功能

### `acquire_mlx_lock(force=True)` 強化 — incident-driven

v0.6.0 之前，`force=True` 會**無條件覆蓋** lock file，讓前一個持有者繼續跑。在實務上這常造成兩個 MLX process 同時把模型載入同一顆 GPU — 這就是 kernel panic 路徑（`IOGPUMemory.cpp:492 completeMemory() prepare count underflow`，幾秒內就會炸）。v0.6.0 把 FORCE 重寫成「真正搶鎖」：

```python
from metal_guard import acquire_mlx_lock, release_mlx_lock, MLXLockConflict

try:
    acquire_mlx_lock("rescuer", force=True)
    # → SIGTERM 舊持有者 → 輪詢最多 MLX_FORCE_WAIT_SEC（預設 30 秒）
    #   確認退出（zombie-aware）→ 休眠 MLX_RECLAIM_COOLDOWN_SEC
    #   （預設 8 秒）讓 Metal buffer GC 完成。
except MLXLockConflict as e:
    if e.holder.get("force_timeout"):
        print("Peer 拒絕退出 — lock 刻意保留不 unlink。")
    elif e.holder.get("force_permission_denied"):
        print("SIGTERM 被拒（例如不同 user）— 無法保證 buffer 已釋放。")
    # 兩種情況都**不會 unlink** lock file — 這是反 panic 不變式。
finally:
    release_mlx_lock()
```

- **Zombie-aware liveness**：`_is_pid_alive` 會 parse `ps -p <pid> -o state=`，把 `Z`（zombie）視為死亡 — zombie 已釋放 Metal buffer，否則 FORCE wait loop 會等到 parent reaper 才解開。
- **新 env vars**：`MLX_FORCE_WAIT_SEC`（預設 30）、`MLX_RECLAIM_COOLDOWN_SEC`（預設 8）。測試 / tight CI 可設為 0。
- **`MLXLockConflict.holder` 新 typed fields**：`force_timeout` 和 `force_permission_denied`，caller 不用解析錯誤字串就能分辨失敗模式。

**升級注意**：以前倚賴「`force=True` 一定成功」的 caller 現在必須 catch `MLXLockConflict`。這是刻意為之 — 舊行為就是 kernel panic 路徑。

### 版本 Advisory 系統

`check_version_advisories()` 回傳當前環境已安裝的 `(mlx, mlx-lm, mlx-vlm)` 版本對應的 active advisories，映射到 upstream issue 編號與 severity。純資訊性，用於 dashboard 和啟動 log。

```python
from metal_guard import check_version_advisories

for a in check_version_advisories():
    print(f"[{a['severity']}] {a['package']} {a['installed_version']} — {a['title']}")
    print(f"    {a['url']}")
```

初始覆蓋範圍聚焦在 mlx-lm 0.31.2 regressions 與 #3348 gate：

| Issue | 影響版本 | Severity | 症狀 |
|-------|---------|----------|------|
| [mlx-lm#1128](https://github.com/ml-explore/mlx-lm/issues/1128) | mlx-lm `==0.31.2` | high | `TokenizerWrapper.think_start_id` `len(None)` crash |
| [mlx-lm#1139](https://github.com/ml-explore/mlx-lm/issues/1139) | mlx-lm `==0.31.2` | high | 第二輪投票後 broadcast error |
| [mlx-lm#1081](https://github.com/ml-explore/mlx-lm/issues/1081) | mlx-lm `==0.31.2` | medium | `ArraysCache.is_trimmable()` 但沒有 `trim()`（只影響 speculative decoding） |
| [mlx#3348](https://github.com/ml-explore/mlx/pull/3348) | mlx `<0.31.2` | info | CommandEncoder thread-local 已於 2026-04-01 merged，但還沒 ship 到 PyPI — observer mode 切換 gate 仍 blocked |

### Upstream Defensive Patches

`install_upstream_defensive_patches()` 安裝精確範圍、version-gated、冪等的 monkey-patch 修補已知 upstream bug。每個 patch 套用時會 log WARNING，並在偵測到已安裝版本不在 affected 範圍時自動 skip — 上游一旦 ship fix，這個呼叫就自動變 no-op，caller 不用改 code。

```python
from metal_guard import install_upstream_defensive_patches

status = install_upstream_defensive_patches()
# → {'mlx_lm_1128_think_start_id': True}   若裝的是 mlx-lm 0.31.2
# → {'mlx_lm_1128_think_start_id': False}  其他版本（沒東西要 patch）
```

首款 patch：**`mlx_lm_1128_think_start_id`** 將 `TokenizerWrapper.think_start_id` 換成一個安全 accessor，在 `_think_start_tokens is None` 時回傳 `None` 而不是 raise `TypeError`。僅套用於 mlx-lm `==0.31.2`。

## v0.5.0 新功能

### Layer 5：`bench_scoped_load` — 序列化 benchmark 防護

Context manager，用於在單一 Python process 內安全載入多個大型 MLX 模型。關閉 benchmark harness 繞過 MetalGuard（直接 for loop 呼叫 `mlx_lm.load` + `mlx_lm.generate`）在 64 GB Apple Silicon 載入 6+ 大模型後 working-set 漂移超限、觸發 `IOGPUMemory.cpp:492 completeMemory() prepare count underflow` kernel panic 的漏洞。

```python
from metal_guard import bench_scoped_load

for model_id in candidate_models:  # 8+ 大模型
    with bench_scoped_load(model_id) as (model, tokenizer):
        score = run_eval(model, tokenizer, items)
        save_checkpoint(model_id, score)
```

每次進入 context：取得 cross-process lock、透過 `mlx_lm.load` / `mlx_vlm.load` 直接載入。每次離開：`safe_cleanup` + 8 秒冷卻 + post-unload 記憶體驗證。Metal 的 lazy page reclaimer 只在**下一次 load() 分配**時才歸還 stale pages —— `bench_scoped_load` 讓每個迭代都走完整防護棧。

### Layer 6：Dual-Mode 切換器

透過 `METALGUARD_MODE` 環境變數在 `defensive`（預設）和 `observer` 模式之間切換。為 [mlx#3348](https://github.com/ml-explore/mlx/pull/3348)（CommandEncoder thread-local）發佈後預先準備。

```bash
export METALGUARD_MODE=defensive  # 預設，主動阻擋危險操作
export METALGUARD_MODE=observer   # #3348 release 後 opt-in；僅監控 + 記錄
```

```python
from metal_guard import current_mode, is_observer, describe_mode

if is_observer():
    # 允許 parallel dispatch
    pass

print(describe_mode())
# {'mode': 'defensive', 'description': '...', 'env_var': '...'}
```

五個長期 active primitives（`safe_cleanup`、thread registry、`oom_protected_context`、breadcrumb logging、`memory_stats`）在**兩種模式都保持啟用** —— 它們處理的是跟 thread safety 正交的問題。

### Layer 7：Subprocess 隔離

`MLXSubprocessRunner` + 自動管理的 `call_model_isolated()` pool，提供 crash-safe MLX 推論。處理從 Metal GCD `CompletionQueueDispatch` queue 丟出的 `mlx::core::gpu::check_error` C++ exceptions —— Python 無法 catch（直接觸發 `std::terminate → abort()`），所以 subprocess 隔離是唯一安全的緩解方式。

```python
from metal_guard import MLXSubprocessRunner

runner = MLXSubprocessRunner("mlx-community/Mistral-Small-3.2-24B-8bit")
for prompt in prompts:
    result = runner.generate(prompt, max_tokens=4096)
runner.shutdown()
```

或用 drop-in 替代直接呼叫的自動管理 pool：

```python
from metal_guard import call_model_isolated

# 自動建立 + 重用每個模型的 worker；crash 時自動 respawn
result = call_model_isolated(prompt, model="mlx-community/Phi-4-mini-4bit")
```

Worker 針對 Mistral（`[INST]`）、Gemma（`<start_of_turn>`）、Phi 系列內建 chat template fallback，處理 `tokenizer.chat_template` 未設定的情況（某些 mlx-community 量化版會 strip 掉 template）。

### `MLX_LOCK_PATH` 可配置化

L8 cross-process lock 檔案路徑現在可透過 `MLX_LOCK_PATH` 環境變數覆蓋（預設 `~/.metal-guard/locks/mlx_exclusive.lock`）。

## v0.4.0 新功能

### 硬體感知自動配置

不同的 Apple Silicon 機器需要不同的安全閾值。8GB MacBook Air 不能用跟 512GB Mac Studio 一樣的設定。MetalGuard 現在會偵測硬體並建議適當的數值。

```python
from metal_guard import MetalGuard

config = MetalGuard.recommended_config()
print(f"{config['chip']} ({config['gpu_memory_gb']}GB) → 等級: {config['tier']}")
# Apple M1 Ultra (64.0GB) → 等級: mid

# 直接使用建議值
metal_guard.start_watchdog(
    warn_pct=config["watchdog_warn_pct"],
    critical_pct=config["watchdog_critical_pct"],
)
```

| 等級 | 記憶體 | 警告 | 臨界 | 最大同時模型數 |
|------|--------|------|------|--------------|
| low | 8–16 GB | 60% | 75% | 1 |
| mid | 32–64 GB | 67% | 82% | 2 |
| high | 96–512 GB | 70% | 85% | 3 |

### KV Cache 增長監控

給長時間運行的伺服器用——KV cache 在多輪對話中無限增長。追蹤記憶體增長速率，在 OOM 前觸發 callback。對應 [mlx-lm#1047](https://github.com/ml-explore/mlx-lm/issues/1047)。

```python
def handle_pressure(available_gb, growth_rate):
    log.warning("KV 壓力: %.1fGB 可用, 增長 %.1fGB/min", available_gb, growth_rate)
    kv_cache.clear()

metal_guard.start_kv_cache_monitor(
    interval_secs=30,
    headroom_gb=8.0,
    growth_rate_warn_gb_per_min=2.0,
    on_pressure=handle_pressure,
)
```

### TurboQuant / 混合精度估算支援

`estimate_model_size_from_name()` 現在正確處理 TurboQuant（TQ3/TQ4）和 Unsloth UD-MLX 模型命名：

| 格式 | 範例 | 估算 |
|------|------|------|
| TQ4 | `gemma-4-31b-it-TQ4-MLX` | 15.5 GB |
| TQ3 | `gemma-4-31b-it-TQ3-MLX` | 11.6 GB |
| UD-MLX-4bit | `gemma-4-31b-it-UD-MLX-4bit` | 15.5 GB |

注意：TQ 模型壓縮 KV cache，可以支撐更長的 context（50K-200K tokens）。估算器只報告*模型權重*佔用——實際運行記憶體取決於 context 長度。

### 跨 Process 互斥鎖（Layer 8）

基於檔案的鎖機制，防止跨 process 的 MLX 工作負載同時執行。這是 `mlx_lm.server`、benchmark 或直接 `mlx_lm.generate` 呼叫與其他 MLX process 同時運行時導致 kernel panic 的根本原因。

```python
from metal_guard import mlx_exclusive_lock, acquire_mlx_lock, release_mlx_lock

# Context manager（推薦）
with mlx_exclusive_lock("my_script"):
    model, tokenizer = mlx_lm.load("mlx-community/gemma-4-31b-it-8bit")
    result = mlx_lm.generate(model, tokenizer, prompt="Hello")

# 明確的 acquire/release
acquire_mlx_lock("my_server")
try:
    serve_forever()
finally:
    release_mlx_lock()

# 檢查但不阻擋
from metal_guard import read_mlx_lock
info = read_mlx_lock()  # 空閒回傳 None，有人持有回傳 dict（含 pid/label/cmdline）
```

**自我修復：** 從 crash process 留下的過期 lock 會透過 pid 存活檢查自動清理。crash 後不需要手動清理。

**衝突時拋出 `MLXLockConflict`**，包含持有者的 pid、label 和 cmdline，讓你看到清楚的錯誤訊息而不是 kernel panic。

## v0.3.0 新功能

### Pre-generate Metal 健康探測

在開始長時間 `generate()` 前驗證 Metal command queue 是否正常。如果 GPU 因先前 crash 處於異常狀態，會在受控時機（~1ms）失敗，而不是在推論途中掛掉。

```python
metal_guard.probe_metal_health()  # 在這裡掛，不在 generate 中間
result = generate(model, tokenizer, prompt=prompt)
```

### SIGABRT 信號處理（crash forensics）

MLX 的 C++ runtime 會從 Metal 的 GCD CompletionQueueDispatch queue 拋出 exception——Python 無法攔截。此 handler 在 process 死亡前寫入最後的 breadcrumb，供事後分析。

```python
metal_guard.install_abort_handler()  # 啟動時呼叫一次
```

### 6-bit / 3-bit / mxfp4 模型大小估算修正

`estimate_model_size_from_name()` 新增混合精度和新量化格式支援：

| 格式 | 乘數 | 範例 |
|---|---|---|
| `6bit` | 0.75 | `LFM2-24B-A2B-MLX-6bit` → 18 GB（之前錯誤估為 48 GB）|
| `3bit` / `int3` | 0.375 | TurboQuant 3-bit KV cache 模型 |
| `mxfp4` | 0.5 | Metal FP4 混合精度格式 |

## v0.2.3 新功能

### `require_fit` 升級重試機制 (v0.2.3)

針對記憶體吃緊的 ensemble 工作負載加入二層重試策略。解決已觀察到的
OOM 路徑：標準 `safe_cleanup` 結束後還留有足夠的殘存 GPU buffer，
導致下一個大模型仍然塞不下 — 特別常見於 M1 Ultra 跑多辯手 ensemble，
每個 KOL 都要經歷 mistral-24B → phi-4-mini → gemma-4-26B 的完整輪轉，
而下一批的 mistral-24B 在 Metal 還沒把 pages 還給 OS 之前就要載入。

```python
from metal_guard import metal_guard

# 標準呼叫（向後相容 — 不觸發 escalation）:
metal_guard.require_fit(24.0, model_name="Mistral-24B-8bit")

# 升級重試：丟棄 Python 端 cache 參考、再清理一次、額外冷卻、重新檢查
# opt-in 方式是傳 escalated_cooldown_sec > 0:
metal_guard.require_fit(
    24.0,
    model_name="Mistral-24B-8bit",
    cache_clear_cb=my_model_cache.clear,
    escalated_cooldown_sec=5.0,
)
```

**兩層的運作方式：**

| 層級 | 動作 | 觸發條件 |
|---|---|---|
| 1. 標準 | `safe_cleanup()`（等 thread + gc + flush + 內部冷卻） | `can_fit` 第一次檢查失敗 |
| 2. 升級 | `cache_clear_cb()` → `safe_cleanup()` → `mlx.reset_peak_memory()` → `sleep(escalated_cooldown_sec)` → 再檢查 | 第 1 層仍然不夠 **且** 呼叫方有 opt-in |

升級路徑**預設關閉**，因為 MetalGuard 不知道呼叫方的 model cache 是
怎麼實作的。你傳進一個 `cache_clear_cb`（通常是 `your_cache_dict.clear`）
以及一個夠長的冷卻時間，讓 Metal 真的把 pages 還給 OS。5 秒在 M1 Ultra
上跑 24GB 模型經驗上足夠。

`cache_clear_cb` 裡面的例外**會記錄但不致命** — 升級路徑會繼續走自己的
`safe_cleanup`，所以壞掉的 cache-clear callback 不會毒化整個恢復路徑。

如果升級後還是放不下，會丟出 `MemoryError` 並在訊息裡包含
`escalated cleanup` 字樣（production log grep 用），同時建議減少同時
載入的模型數或切換到更小的量化版本。

### 模型大小估算器 (v0.2.2)

直接從模型名稱解析 Metal 記憶體佔用量。設計用來搭配 `require_fit` 作為
多模型輪流工作負載的預測性閘門 — 在撞到 Metal 工作區上限之前就主動
卸載快取模型。

```python
from metal_guard import MetalGuard, metal_guard

# 靜態方法，不需要 instance
size = MetalGuard.estimate_model_size_from_name(
    "mlx-community/Mistral-Small-24B-8bit"
)
# → 24.0 GB  (24B 參數 × 1.0 bytes/param for 8-bit)

size = MetalGuard.estimate_model_size_from_name(
    "mlx-community/Phi-4-mini-instruct-4bit"
)
# → 2.0 GB  (mini-class fallback: 4B × 0.5 for 4-bit)

# 搭配 require_fit 作為載入前閘門
name = "mlx-community/gemma-4-31b-8bit"
size = MetalGuard.estimate_model_size_from_name(name)
if size is not None:
    metal_guard.require_fit(size, model_name=name)
model = load(name)  # 如果 Metal 裝不下，載入前就被拒絕
```

**為什麼需要這個**：一個依序載入 mistral-24B-8bit → phi-4-mini-4bit
→ gemma-4-26B-8bit 的多模型工作負載會讓 ModelCache 持續累積，直到
超過 Metal 工作區上限（M1 Ultra 約 51 GB）。Metal completion queue
會丟出未捕捉的 `std::runtime_error`，最終變成 `EXC_CRASH (SIGABRT)`
殺掉整個 process。有了這個估算器之後，呼叫方可以在真正碰到 Metal 前
就拿到乾淨的 `MemoryError`，而不是 generate 到一半才 crash。

**支援的格式**：

| 格式 | 範例 | 結果 |
|---|---|---|
| `<N>B` + 位元 | `Mistral-24B-8bit` | 24 × 1.0 = 24 GB |
| `<N>M` + 位元 | `tiny-350m-4bit` | 0.350 × 0.5 = 0.175 GB |
| 大小類別 + 位元 | `phi-4-mini-4bit` | 4 × 0.5 = 2 GB（mini 類別） |
| 大小類別 + 預設 | `foo-small` | 7 × 2.0 = 14 GB（fp16 預設） |
| 無法解析 | `mystery-model` | `None` → 呼叫方走舊路徑 |

量化倍率：`16bit/fp16/bf16` → 2.0、`8bit/int8` → 1.0、
`4bit/int4/q4` → 0.5、`2bit/int2` → 0.25。未指定時預設為 2.0
（保守的 fp16 上限）。

大小類別 fallback：`mini` → 4B、`small` → 7B、`medium` → 13B、
`large` → 70B、`xl` → 13B。

無法從名稱解析時回傳 `None`，呼叫方可以 fallback 到原本門檻式的
`ensure_headroom` 路徑。

### AGX 驅動程式繞過 (v0.2.2)

import 時自動設定 `AGX_RELAX_CDM_CTXSTORE_TIMEOUT=1`（若尚未設定）。
這是 MLX 維護者 @zcbenz 在
[mlx#3267](https://github.com/ml-explore/mlx/issues/3267) 的建議 —
放寬 IOGPUFamily 的 command buffer context store timeout，可減少
長時間 GPU 工作負載的 kernel panic。零成本，無條件設定安全。

### 新增 OOM 偵測模式 (v0.2.2)

`is_metal_oom` 現在會偵測 `fPendingMemorySet` 這個 panic 訊號 —
由 @yoyaku155 在
[mlx#3346](https://github.com/ml-explore/mlx/issues/3346) 回報 —
跟原本的 `Insufficient Memory` 和
`kIOGPUCommandBufferCallbackErrorOutOfMemory` 並列。

## v0.2 新功能

### OOM 復原

捕捉 Metal OOM 錯誤，轉成可恢復的 `MetalOOMError`，不會 crash process。

```python
from metal_guard import metal_guard, MetalOOMError

# 自動 catch OOM → 清理 → 重試一次
result = metal_guard.oom_protected(generate, model, tokenizer, prompt=prompt)

# 伺服器用法——回傳 503 而不是 crash
try:
    result = metal_guard.oom_protected(generate, model, tokenizer, prompt=prompt)
except MetalOOMError as e:
    return Response(status_code=503, body=f"GPU 記憶體不足: {e.stats}")
```

### 載入前記憶體檢查

避免載入放不下的模型。

```python
if not metal_guard.can_fit(model_size_gb=24.0):
    print("記憶體不足以載入 24GB 模型")

# 或讓 MetalGuard 處理（先清理，放不下就報錯）
metal_guard.require_fit(24.0, model_name="Mistral-24B-8bit")
```

### 定時清理

長時間運行的 process 背景定時清理。

```python
metal_guard.start_periodic_flush(interval_secs=300)  # 每 5 分鐘
```

### 記憶體漂移看門狗

給 24/7 常駐程式和 agent 框架用——記憶體慢慢漲的時候，分級應對。

```python
def on_critical():
    kv_cache.clear()
    log.error("記憶體到達臨界值——KV cache 已清除")

metal_guard.start_watchdog(
    interval_secs=60,       # 每分鐘檢查
    warn_pct=70.0,          # 70% 時 flush
    critical_pct=85.0,      # 85% 時完整清理 + callback
    on_critical=on_critical,
)
```

**為什麼 agent 框架需要這個：** 每次 tool call / function call 都跑一次 `generate()`，累積碎片化的 Metal buffer。一個對話 50-100 次 tool call，記憶體可以漂移好幾 GB，完全看不出有 leak。看門狗在漂移到 OOM 之前就會攔住。

## 完整範例

### 帶安全防護的 Model Cache

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
        model = mlx_lm.load(name)
        self._models[name] = model
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

### 長時間運行的 Agent 伺服器

```python
# 啟動看門狗
metal_guard.start_watchdog(
    interval_secs=120,
    warn_pct=65.0,
    critical_pct=80.0,
    on_critical=lambda: server.drop_oldest_session(),
)

# 每個請求用 OOM 防護
@app.post("/v1/chat/completions")
async def chat(request):
    try:
        result = metal_guard.oom_protected(generate, model, tokenizer, prompt=request.prompt)
        return {"choices": [{"message": {"content": result}}]}
    except MetalOOMError:
        return JSONResponse(status_code=503, content={"error": "GPU 記憶體不足"})
```

## API 參考

### 跨 Process 鎖 (v0.3.1)

| 方法 | 說明 |
|------|------|
| `acquire_mlx_lock(label, force=False)` | 取得跨 process 獨佔鎖。有人持有時拋出 `MLXLockConflict` |
| `release_mlx_lock() -> bool` | 釋放鎖（僅當本 process 持有時） |
| `read_mlx_lock() -> dict \| None` | 檢查鎖狀態，不阻擋。自動清理過期鎖 |
| `mlx_exclusive_lock(label)` | Context manager：進入時取得，離開時釋放 |

### Thread 追蹤

| 方法 | 說明 |
|------|------|
| `register_thread(thread)` | 追蹤持有 GPU buffer 的 thread |
| `wait_for_threads(timeout) -> int` | 等待 GPU thread 結束 |

### GPU 清理

| 方法 | 說明 |
|------|------|
| `flush_gpu()` | `mx.eval(sync)` + `mx.clear_cache()` |
| `safe_cleanup()` | 完整流程：wait → gc → flush → cooldown |
| `guarded_cleanup()` | Context manager |

### OOM 復原 (v0.2)

| 方法 | 說明 |
|------|------|
| `oom_protected(fn, *args, max_retries=1)` | 捕捉 OOM → 清理 → 重試 |
| `oom_protected_context()` | Context manager 版本 |
| `is_metal_oom(exc) -> bool` | 判斷是否為 Metal OOM |

### 載入前檢查 (v0.2)

| 方法 | 說明 |
|------|------|
| `can_fit(model_size_gb, overhead_gb=2.0) -> bool` | 模型能否放進可用記憶體 |
| `require_fit(model_size_gb, model_name)` | 放不下就清理，還不行就報錯 |
| `estimate_model_size_from_name(name) -> float \| None` *(v0.2.2, 靜態)* | 從模型名稱解析參數量 + 量化等級 → 估算 GB |

### 記憶體壓力

| 方法 | 說明 |
|------|------|
| `memory_stats() -> MemoryStats` | 目前 GPU 記憶體快照 |
| `is_pressure_high(threshold_pct) -> bool` | 峰值是否超過閾值 |
| `ensure_headroom(model_name, threshold_pct)` | 壓力高時清理，否則不動 |

### 長時間運行安全 (v0.2)

| 方法 | 說明 |
|------|------|
| `start_periodic_flush(interval_secs=300)` | 背景定時清理 |
| `start_watchdog(interval, warn_pct, critical_pct, on_critical)` | 記憶體漂移看門狗 |

### 硬體偵測 (v0.4.0)

| 方法 | 說明 |
|------|------|
| `detect_hardware() -> dict` *(靜態)* | 偵測晶片、記憶體、等級 |
| `recommended_config() -> dict` *(類別方法)* | 針對硬體的建議閾值 |

### KV Cache 監控 (v0.4.0)

| 方法 | 說明 |
|------|------|
| `start_kv_cache_monitor(interval, headroom_gb, growth_rate_warn, on_pressure)` | 追蹤 KV cache 增長，OOM 前觸發 callback |
| `stop_kv_cache_monitor()` | 停止 KV cache 監控 |

### 鑑識

| 方法 | 說明 |
|------|------|
| `breadcrumb(msg)` | 寫入 fsync 的 crash 鑑識日誌 |

## 對應的社群問題

| Issue | 問題 | MetalGuard 功能 |
|-------|------|----------------|
| [mlx-lm#883](https://github.com/ml-explore/mlx-lm/issues/883) | KV cache 增長 → kernel panic | Thread 追蹤 + 安全清理 |
| [mlx-lm#1015](https://github.com/ml-explore/mlx-lm/issues/1015) | generate() OOM 殺 process | `oom_protected()` |
| [mlx-lm#854](https://github.com/ml-explore/mlx-lm/issues/854) | 伺服器 OOM，沒有 HTTP 錯誤 | `oom_protected()` + `periodic_flush` |
| [mlx-lm#427](https://github.com/ml-explore/mlx-lm/issues/427) | M1 MBA 載入模型就 crash | `can_fit()` / `require_fit()` |
| [mlx-lm#1047](https://github.com/ml-explore/mlx-lm/issues/1047) | 大模型 KV cache OOM | `can_fit()` + `ensure_headroom()` |
| [mlx-examples#1124](https://github.com/ml-explore/mlx-examples/issues/1124) | 伺服器記憶體 leak → 重開機 | `periodic_flush` + `watchdog` |

## 實測結果

- Mac Studio M1 Ultra (64GB) — 之前 9 次 kernel panic，之後 0 次
- 10 人批次管線：約 90 次模型載卸，994 秒，零 crash
- 測試模型：Mistral-Small-3.2-24B、Phi-4-mini、Gemma-4-26B/31B、Pixtral-12B、LFM2-VL-3B

## 授權

MIT
