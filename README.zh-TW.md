# MetalGuard

[English](README.md) | **繁體中文** | [日本語](README.ja.md)

為 Apple Silicon 上的 [MLX](https://github.com/ml-explore/mlx) 提供 GPU 安全防護層。

防止因 Metal 驅動程式 bug 導致的 kernel panic，特別是在反覆載入/卸載 MLX 模型時。

## 問題是什麼

Apple Silicon 的 Metal GPU 驅動程式有一個 bug：**當 GPU 記憶體管理失敗時，它不會優雅地殺掉 process，而是直接讓整台電腦 kernel panic 重開機。**

```
panic(cpu 4 caller 0xfffffe0032a550f8):
  "completeMemory() prepare count underflow" @IOGPUMemory.cpp:492
```

任何連續載入和卸載多個 MLX 模型的工作流程都會受影響——Metal 驅動程式的內部 reference count 可能 underflow，導致無法恢復的 kernel panic。

**這不是你的程式的問題。** 這是驅動程式層級的 bug，目前沒有修復時程。參見 [ml-explore/mlx-lm#883](https://github.com/ml-explore/mlx-lm/issues/883)。

### 根因

我們透過 crash 鑑識（breadcrumb log + M1 Ultra 上 9 次 kernel panic）辨識出兩個觸發路徑：

1. **Daemon thread 競爭條件** — `mlx_lm.generate()` 在 daemon thread 中執行並持有 GPU buffer。如果在 thread 結束前呼叫 `mx.clear_cache()`，Metal 會試圖釋放仍在使用的 buffer → reference count underflow → kernel panic。

2. **無條件的 Metal 初始化** — 即使沒有模型載入，呼叫 `mx.eval()` 或 `mx.clear_cache()` 仍會初始化 Metal 驅動程式。如果驅動程式因先前的 crash 而處於不穩定狀態，這本身就能觸發 panic。

### 誰會受影響

- **單模型伺服器**（mlx_lm.server、LM Studio）：低風險。整個 process 生命週期只載入一個模型。
- **多模型管線**：**高風險。** 載入模型 A → 卸載 → 載入模型 B → 卸載 → 反覆進行。每次切換都是潛在的 panic 觸發點。

如果你的工作流程每次 session 會載入和卸載 3 個以上不同的 MLX 模型，你需要 MetalGuard。

## 安裝

```bash
pip install metal-guard
```

或者直接把 `metal_guard.py` 複製到你的專案中——它是一個單獨的檔案，除了 Python 標準函式庫外沒有其他依賴。

## 使用方式

```python
from metal_guard import metal_guard

# 1. 追蹤 GPU thread
import threading

thread = threading.Thread(target=run_mlx_generate, daemon=True)
thread.start()
metal_guard.register_thread(thread)  # 追蹤它
thread.join(timeout=120)

# 2. 安全卸載模型
def unload_model(cache):
    metal_guard.wait_for_threads()  # 等 GPU 工作完成
    had_models = bool(cache)
    cache.clear()
    if had_models:
        metal_guard.safe_cleanup()  # gc + flush GPU + cooldown

# 3. 載入前檢查記憶體壓力
metal_guard.ensure_headroom(model_name="my-model-8bit")
model, tokenizer = mlx_lm.load("my-model-8bit")

# 4. Crash 鑑識用 breadcrumb
metal_guard.breadcrumb("LOAD: my-model-8bit START")
# ...如果 kernel panic 發生在這裡，breadcrumb log 會顯示最後的操作
```

### 完整 Model Cache 範例

```python
from metal_guard import metal_guard

class ModelCache:
    def __init__(self):
        self._models = {}

    def load(self, name):
        if name in self._models:
            return self._models[name]
        metal_guard.ensure_headroom(model_name=name)
        metal_guard.breadcrumb(f"LOAD: {name} START")
        model = mlx_lm.load(name)
        self._models[name] = model
        metal_guard.breadcrumb(f"LOAD: {name} DONE")
        return model

    def unload_all(self):
        metal_guard.wait_for_threads()      # GPU 忙碌時絕不釋放
        had_models = bool(self._models)
        self._models.clear()
        if had_models:
            metal_guard.safe_cleanup()       # gc + flush + cooldown
        # 如果 cache 是空的，完全跳過 Metal（防止觸發路徑 #2）
```

### 測試（阻止 Metal 進入單元測試）

```python
# conftest.py
from unittest.mock import MagicMock, patch
import pytest

@pytest.fixture(autouse=True)
def _block_metal_gpu(request):
    """阻止單元測試中的 Metal GPU 初始化。"""
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

## API 參考

### `MetalGuard(cooldown_secs=2.0, thread_timeout_secs=30.0, breadcrumb_path="logs/metal_breadcrumb.log")`

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `cooldown_secs` | 2.0 | GPU flush 後的等待時間，讓 Metal 驅動程式回收 buffer |
| `thread_timeout_secs` | 30.0 | 等待 GPU thread 的最大時間 |
| `breadcrumb_path` | `"logs/metal_breadcrumb.log"` | crash 鑑識日誌路徑。設 `None` 停用 |

### Thread 追蹤

| 方法 | 說明 |
|------|------|
| `register_thread(thread)` | 追蹤持有 GPU buffer 的 thread |
| `wait_for_threads(timeout) -> int` | 等待 GPU thread 結束。回傳仍存活的數量 |

### GPU 清理

| 方法 | 說明 |
|------|------|
| `flush_gpu()` | `mx.eval(sync)` + `mx.clear_cache()`。必須在 `wait_for_threads()` 之後 |
| `safe_cleanup()` | 完整流程：wait → gc.collect → flush → cooldown |
| `guarded_cleanup()` | Context manager，離開時執行 `safe_cleanup()` |

### 記憶體壓力

| 方法 | 說明 |
|------|------|
| `memory_stats() -> MemoryStats` | 目前 GPU 記憶體快照 |
| `is_pressure_high(threshold_pct) -> bool` | 峰值記憶體是否超過閾值 |
| `ensure_headroom(model_name, threshold_pct)` | 壓力高時清理，否則不動（零開銷） |
| `log_memory(label, model_name)` | 記錄記憶體狀態，不清理 |

### 鑑識

| 方法 | 說明 |
|------|------|
| `breadcrumb(msg)` | 寫入 fsync 的 breadcrumb 日誌 |

## 實測結果

- Mac Studio M1 Ultra (64GB) — MetalGuard 之前 9 次 kernel panic，之後 0 次
- 10 人批次管線：約 90 次模型載入/卸載，994 秒，零 crash
- 測試模型：Mistral-Small-3.2-24B、Phi-4-mini、Gemma-4-26B/31B、Pixtral-12B、LFM2-VL-3B（8-bit 和 4-bit）

## 相關 Issue

- [ml-explore/mlx-lm#883](https://github.com/ml-explore/mlx-lm/issues/883) — KV cache 無限增長導致 kernel panic
- [ml-explore/mlx#2133](https://github.com/ml-explore/mlx/issues/2133) — Thread safety 持續問題
- [ml-explore/mlx#3126](https://github.com/ml-explore/mlx/issues/3126) — Sub-thread exit crash
- [ml-explore/mlx#3078](https://github.com/ml-explore/mlx/issues/3078) — 不支援 concurrent inference

## 授權

MIT
