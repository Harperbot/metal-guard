# MetalGuard

[English](README.md) | [繁體中文](README.zh-TW.md) | **日本語**

Apple Silicon 上の [MLX](https://github.com/ml-explore/mlx) 向け GPU 安全レイヤー。

MLX モデルの繰り返しロード・アンロード時に Metal ドライバーのバグが引き起こすカーネルパニックを防止します。

## 問題の概要

Apple Silicon の Metal GPU ドライバーにはバグがあり、**GPU メモリ管理に失敗した場合、プロセスを正常に終了させるのではなく、マシン全体がカーネルパニックで再起動します。**

```
panic(cpu 4 caller 0xfffffe0032a550f8):
  "completeMemory() prepare count underflow" @IOGPUMemory.cpp:492
```

複数の MLX モデルを順次ロード・アンロードするワークフローすべてに影響します。Metal ドライバーの内部参照カウントがアンダーフローし、回復不能なカーネルパニックを引き起こす可能性があります。

**これはあなたのコードのバグではありません。** ドライバーレベルのバグであり、修正時期は未定です。参照：[ml-explore/mlx-lm#883](https://github.com/ml-explore/mlx-lm/issues/883)

### 根本原因

クラッシュフォレンジック（breadcrumb ログ + M1 Ultra での 9 回のカーネルパニック）を通じて、2 つのトリガーパスを特定しました：

1. **デーモンスレッドの競合状態** — `mlx_lm.generate()` がデーモンスレッドで実行され、GPU バッファを保持。スレッド終了前に `mx.clear_cache()` が呼ばれると、Metal は使用中のバッファを解放しようとし → 参照カウントアンダーフロー → カーネルパニック。

2. **無条件の Metal 初期化** — モデルが未ロードでも `mx.eval()` や `mx.clear_cache()` を呼ぶと Metal ドライバーが初期化される。以前のクラッシュでドライバーが不安定な状態にある場合、これだけでパニックが発生する可能性がある。

### 影響を受ける対象

- **シングルモデルサーバー**（mlx_lm.server、LM Studio）：低リスク。プロセスの全期間で 1 つのモデルのみ。
- **マルチモデルパイプライン**：**高リスク。** モデル A ロード → アンロード → モデル B ロード → アンロード → 繰り返し。切り替えのたびにパニックの可能性。

セッションごとに 3 つ以上の異なる MLX モデルをロード・アンロードする場合、MetalGuard が必要です。

## インストール

```bash
pip install metal-guard
```

または `metal_guard.py` をプロジェクトにコピーしてください。Python 標準ライブラリ以外の依存関係はありません。

## 使い方

```python
from metal_guard import metal_guard

# 1. GPU スレッドの追跡
import threading

thread = threading.Thread(target=run_mlx_generate, daemon=True)
thread.start()
metal_guard.register_thread(thread)  # 追跡する
thread.join(timeout=120)

# 2. 安全なモデルアンロード
def unload_model(cache):
    metal_guard.wait_for_threads()  # GPU の処理完了を待つ
    had_models = bool(cache)
    cache.clear()
    if had_models:
        metal_guard.safe_cleanup()  # gc + GPU フラッシュ + クールダウン

# 3. ロード前のメモリ圧力チェック
metal_guard.ensure_headroom(model_name="my-model-8bit")
model, tokenizer = mlx_lm.load("my-model-8bit")

# 4. クラッシュフォレンジック用 breadcrumb
metal_guard.breadcrumb("LOAD: my-model-8bit START")
# ...ここでカーネルパニックが発生した場合、breadcrumb ログに最後の操作が記録される
```

### 完全な ModelCache の例

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
        metal_guard.wait_for_threads()      # GPU 使用中は解放しない
        had_models = bool(self._models)
        self._models.clear()
        if had_models:
            metal_guard.safe_cleanup()       # gc + flush + cooldown
        # キャッシュが空なら Metal を完全にスキップ（トリガーパス #2 を防止）
```

### テスト（ユニットテストで Metal を防止）

```python
# conftest.py
from unittest.mock import MagicMock, patch
import pytest

@pytest.fixture(autouse=True)
def _block_metal_gpu(request):
    """ユニットテストでの Metal GPU 初期化を防止。"""
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

## API リファレンス

### `MetalGuard(cooldown_secs=2.0, thread_timeout_secs=30.0, breadcrumb_path="logs/metal_breadcrumb.log")`

| パラメータ | デフォルト | 説明 |
|-----------|----------|------|
| `cooldown_secs` | 2.0 | GPU フラッシュ後の待機時間（Metal ドライバーのバッファ回収用） |
| `thread_timeout_secs` | 30.0 | GPU スレッドの最大待機時間 |
| `breadcrumb_path` | `"logs/metal_breadcrumb.log"` | クラッシュフォレンジックのログパス。`None` で無効化 |

### スレッド追跡

| メソッド | 説明 |
|---------|------|
| `register_thread(thread)` | GPU バッファを保持するスレッドを追跡 |
| `wait_for_threads(timeout) -> int` | GPU スレッド完了まで待機。まだ生存中の数を返す |

### GPU クリーンアップ

| メソッド | 説明 |
|---------|------|
| `flush_gpu()` | `mx.eval(sync)` + `mx.clear_cache()`。`wait_for_threads()` の後にのみ使用 |
| `safe_cleanup()` | 完全シーケンス：wait → gc.collect → flush → cooldown |
| `guarded_cleanup()` | コンテキストマネージャ。終了時に `safe_cleanup()` を実行 |

### メモリ圧力

| メソッド | 説明 |
|---------|------|
| `memory_stats() -> MemoryStats` | 現在の GPU メモリスナップショット |
| `is_pressure_high(threshold_pct) -> bool` | ピークメモリが閾値を超えているか |
| `ensure_headroom(model_name, threshold_pct)` | 圧力が高ければクリーンアップ、そうでなければ何もしない |
| `log_memory(label, model_name)` | メモリ状態をログ出力（クリーンアップなし） |

### フォレンジック

| メソッド | 説明 |
|---------|------|
| `breadcrumb(msg)` | fsync された breadcrumb ログに書き込み |

## テスト実績

- Mac Studio M1 Ultra (64GB) — MetalGuard 導入前 9 回のカーネルパニック、導入後 0 回
- 10 ユーザーバッチパイプライン：約 90 回のモデルロード/アンロード、994 秒、クラッシュゼロ
- テストモデル：Mistral-Small-3.2-24B、Phi-4-mini、Gemma-4-26B/31B、Pixtral-12B、LFM2-VL-3B（8-bit / 4-bit）

## 関連 Issue

- [ml-explore/mlx-lm#883](https://github.com/ml-explore/mlx-lm/issues/883) — KV キャッシュ無制限成長によるカーネルパニック
- [ml-explore/mlx#2133](https://github.com/ml-explore/mlx/issues/2133) — スレッドセーフティの継続的問題
- [ml-explore/mlx#3126](https://github.com/ml-explore/mlx/issues/3126) — サブスレッドの終了時クラッシュ
- [ml-explore/mlx#3078](https://github.com/ml-explore/mlx/issues/3078) — concurrent inference 未サポート

## ライセンス

MIT
