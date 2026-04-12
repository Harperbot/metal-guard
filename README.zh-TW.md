# MetalGuard

[English](README.md) | **繁體中文** | [日本語](README.ja.md)

為 Apple Silicon 上的 [MLX](https://github.com/ml-explore/mlx) 提供 GPU 安全防護層。

防止因 Metal 驅動程式 bug 導致的 kernel panic 和 OOM crash——特別是多模型管線、長時間運行的伺服器，以及大量 tool calling 的 agent 框架。

**目前版本：** v0.4.0 — 完整發佈歷史見 [CHANGELOG.md](CHANGELOG.md)。

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
