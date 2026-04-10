# MetalGuard

[English](README.md) | [繁體中文](README.zh-TW.md) | **日本語**

Apple Silicon 上の [MLX](https://github.com/ml-explore/mlx) 向け GPU 安全レイヤー。

Metal ドライバーのバグによるカーネルパニックや OOM クラッシュを防止します。特にマルチモデルパイプライン、長時間稼働サーバー、ツールコールが多いエージェントフレームワーク向け。

## 問題の概要

Apple Silicon の Metal GPU ドライバーにはバグがあり、**GPU メモリ管理に失敗した場合、プロセスを正常終了させるのではなく、マシン全体がカーネルパニックで再起動します。**

```
panic(cpu 4 caller 0xfffffe0032a550f8):
  "completeMemory() prepare count underflow" @IOGPUMemory.cpp:492
```

**これはあなたのコードのバグではありません。** ドライバーレベルのバグであり、修正時期は未定です。参照：[ml-explore/mlx-lm#883](https://github.com/ml-explore/mlx-lm/issues/883)

### 影響を受ける対象

| ワークロード | リスク | 理由 |
|-------------|--------|------|
| シングルモデルサーバー（LM Studio） | 低 | モデル切り替えなし |
| マルチモデルパイプライン | **高** | ロード→アンロード→ロード、切り替えごとにパニックの可能性 |
| 長時間稼働サーバー（mlx_lm.server） | **高** | KV キャッシュ無制限成長、Metal バッファ蓄積 |
| エージェントフレームワーク + ツールコール | **高** | 会話ごとに50-100回の短い推論、断片化した Metal バッファの蓄積 |
| TurboQuant KV キャッシュ圧縮 | **高** | メモリ上限に近づく（50K-200Kトークン）、OOM 発生しやすい |
| 24時間365日デーモン（OpenClaw型） | **極高** | 日数とともにメモリドリフト、自然なクリーンアップポイントなし |

## インストール

```bash
pip install metal-guard
```

または `metal_guard.py` をプロジェクトにコピー。単一ファイル、依存関係なし。

## クイックスタート

```python
from metal_guard import metal_guard

# 1. GPU スレッド追跡
thread = threading.Thread(target=run_mlx_generate, daemon=True)
thread.start()
metal_guard.register_thread(thread)

# 2. 安全なモデルアンロード
metal_guard.wait_for_threads()
cache.clear()
metal_guard.safe_cleanup()

# 3. ロード前のメモリチェック
metal_guard.ensure_headroom(model_name="my-model-8bit")
```

## v0.2.2 新機能

### モデルサイズ推定器 (v0.2.2)

モデル名から Metal メモリフットプリントを直接パースします。複数モデルの
アンサンブルワークロードで ModelCache にキャッシュされた古いモデルを
Metal のワーキングセット上限に到達する前に事前退避するための `require_fit`
ゲートとして設計されています。

```python
from metal_guard import MetalGuard, metal_guard

# 静的メソッド — インスタンス不要
size = MetalGuard.estimate_model_size_from_name(
    "mlx-community/Mistral-Small-24B-8bit"
)
# → 24.0 GB  (24B パラメータ × 1.0 bytes/param for 8-bit)

size = MetalGuard.estimate_model_size_from_name(
    "mlx-community/Phi-4-mini-instruct-4bit"
)
# → 2.0 GB  (mini クラスのフォールバック: 4B × 0.5 for 4-bit)

# require_fit とペアでロード前ゲートとして使用
name = "mlx-community/gemma-4-31b-8bit"
size = MetalGuard.estimate_model_size_from_name(name)
if size is not None:
    metal_guard.require_fit(size, model_name=name)
model = load(name)  # Metal に収まらない場合、ロード前に拒否される
```

**なぜ必要か**: `mistral-24B-8bit → phi-4-mini-4bit → gemma-4-26B-8bit`
を順次ロードする複数モデルのバッチ処理では、ModelCache がすべての
モデルを保持し続け、Metal のワーキングセット上限（M1 Ultra で約 51 GB）を
超えてしまいます。その結果、Metal 補完キューから捕捉されていない
`std::runtime_error` が投げられ、最終的に `EXC_CRASH (SIGABRT)` として
プロセスが死亡します。この推定器があれば、呼び出し側は Metal に
触れる前にクリーンな `MemoryError` 拒否を受け取れるため、generate の
途中でクラッシュすることがなくなります。

**対応パターン**:

| パターン | 例 | 結果 |
|---|---|---|
| `<N>B` + ビット数 | `Mistral-24B-8bit` | 24 × 1.0 = 24 GB |
| `<N>M` + ビット数 | `tiny-350m-4bit` | 0.350 × 0.5 = 0.175 GB |
| サイズクラス + ビット数 | `phi-4-mini-4bit` | 4 × 0.5 = 2 GB (mini クラス) |
| サイズクラス + デフォルト | `foo-small` | 7 × 2.0 = 14 GB (fp16 デフォルト) |
| 解析不可 | `mystery-model` | `None` → 呼び出し側がフォールバック |

量子化乗数: `16bit/fp16/bf16` → 2.0、`8bit/int8` → 1.0、
`4bit/int4/q4` → 0.5、`2bit/int2` → 0.25。指定されない場合のデフォルトは
2.0（fp16 の保守的な上限）。

サイズクラスのフォールバック: `mini` → 4B、`small` → 7B、
`medium` → 13B、`large` → 70B、`xl` → 13B。

名前からサイズヒントが解析できない場合は `None` を返すため、呼び出し側は
従来のしきい値ベースの `ensure_headroom` パスにフォールバックできます。

### AGX ドライバ回避策 (v0.2.2)

インポート時に `AGX_RELAX_CDM_CTXSTORE_TIMEOUT=1` を設定します
（既に設定されていない場合）。MLX メンテナ @zcbenz 氏による
[mlx#3267](https://github.com/ml-explore/mlx/issues/3267) での提案で、
IOGPUFamily のコマンドバッファコンテキストストアタイムアウトを
緩和し、長時間実行される GPU ワークロードでの kernel panic を
減らします。ゼロコスト、無条件設定で安全です。

### OOM パターン追加検出 (v0.2.2)

`is_metal_oom` が `fPendingMemorySet` panic シグネチャを検出するように
なりました — @yoyaku155 氏が
[mlx#3346](https://github.com/ml-explore/mlx/issues/3346) で報告した
もので、既存の `Insufficient Memory` および
`kIOGPUCommandBufferCallbackErrorOutOfMemory` パターンと並列で扱われます。

## v0.2 新機能

### OOM リカバリー

Metal OOM エラーをキャッチし、回復可能な `MetalOOMError` に変換。プロセスをクラッシュさせません。

```python
from metal_guard import metal_guard, MetalOOMError

# OOM キャッチ → クリーンアップ → リトライ
result = metal_guard.oom_protected(generate, model, tokenizer, prompt=prompt)

# サーバー向け — クラッシュではなく503を返す
try:
    result = metal_guard.oom_protected(generate, model, tokenizer, prompt=prompt)
except MetalOOMError as e:
    return Response(status_code=503, body=f"GPU メモリ不足: {e.stats}")
```

### ロード前メモリチェック

収まらないモデルのロードを防止。

```python
if not metal_guard.can_fit(model_size_gb=24.0):
    print("24GBモデルに十分なメモリがありません")

metal_guard.require_fit(24.0, model_name="Mistral-24B-8bit")
```

### 定期フラッシュ

長時間プロセスのバックグラウンド定期クリーンアップ。

```python
metal_guard.start_periodic_flush(interval_secs=300)  # 5分ごと
```

### メモリドリフトウォッチドッグ

24時間稼働デーモンとエージェントフレームワーク向け。段階的対応：警告 → フラッシュ → 危険時クリーンアップ → コールバック。

```python
def on_critical():
    kv_cache.clear()
    log.error("メモリ危険 — KV キャッシュをクリア")

metal_guard.start_watchdog(
    interval_secs=60,       # 毎分チェック
    warn_pct=70.0,          # 70%でフラッシュ
    critical_pct=85.0,      # 85%で完全クリーンアップ + コールバック
    on_critical=on_critical,
)
```

**エージェントフレームワークにこれが必要な理由：** ツールコール / ファンクションコールのたびに `generate()` が実行され、断片化した Metal バッファが蓄積します。50-100回のツールコールで数GBのメモリドリフトが発生しますが、明らかなリークはありません。ウォッチドッグはOOMに達する前にこのドリフトを検出します。

## 完全な例

### 安全機能付き ModelCache

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

### 長時間稼働エージェントサーバー

```python
metal_guard.start_watchdog(
    interval_secs=120,
    warn_pct=65.0,
    critical_pct=80.0,
    on_critical=lambda: server.drop_oldest_session(),
)

@app.post("/v1/chat/completions")
async def chat(request):
    try:
        result = metal_guard.oom_protected(generate, model, tokenizer, prompt=request.prompt)
        return {"choices": [{"message": {"content": result}}]}
    except MetalOOMError:
        return JSONResponse(status_code=503, content={"error": "GPU メモリ不足"})
```

## API リファレンス

### スレッド追跡

| メソッド | 説明 |
|---------|------|
| `register_thread(thread)` | GPU バッファ保持スレッドを追跡 |
| `wait_for_threads(timeout) -> int` | GPU スレッド完了まで待機 |

### GPU クリーンアップ

| メソッド | 説明 |
|---------|------|
| `flush_gpu()` | `mx.eval(sync)` + `mx.clear_cache()` |
| `safe_cleanup()` | 完全シーケンス：wait → gc → flush → cooldown |
| `guarded_cleanup()` | コンテキストマネージャ |

### OOM リカバリー (v0.2)

| メソッド | 説明 |
|---------|------|
| `oom_protected(fn, *args, max_retries=1)` | OOM キャッチ → クリーンアップ → リトライ |
| `oom_protected_context()` | コンテキストマネージャ版 |
| `is_metal_oom(exc) -> bool` | Metal OOM かどうか判定 |

### ロード前チェック (v0.2)

| メソッド | 説明 |
|---------|------|
| `can_fit(model_size_gb, overhead_gb=2.0) -> bool` | 利用可能なメモリにモデルが収まるか |
| `require_fit(model_size_gb, model_name)` | 収まらなければクリーンアップ、それでもダメならエラー |
| `estimate_model_size_from_name(name) -> float \| None` *(v0.2.2, 静的)* | モデル名からパラメータ数 + 量子化を解析 → 推定 GB |

### メモリ圧力

| メソッド | 説明 |
|---------|------|
| `memory_stats() -> MemoryStats` | 現在の GPU メモリスナップショット |
| `is_pressure_high(threshold_pct) -> bool` | ピークが閾値を超えているか |
| `ensure_headroom(model_name, threshold_pct)` | 圧力が高ければクリーンアップ |

### 長時間稼働安全 (v0.2)

| メソッド | 説明 |
|---------|------|
| `start_periodic_flush(interval_secs=300)` | バックグラウンド定期フラッシュ |
| `start_watchdog(interval, warn_pct, critical_pct, on_critical)` | メモリドリフトウォッチドッグ |

### フォレンジック

| メソッド | 説明 |
|---------|------|
| `breadcrumb(msg)` | fsync されたクラッシュフォレンジックログ |

## 対応するコミュニティの問題

| Issue | 問題 | MetalGuard 機能 |
|-------|------|----------------|
| [mlx-lm#883](https://github.com/ml-explore/mlx-lm/issues/883) | KV キャッシュ成長 → カーネルパニック | スレッド追跡 + 安全クリーンアップ |
| [mlx-lm#1015](https://github.com/ml-explore/mlx-lm/issues/1015) | generate() OOM でプロセス終了 | `oom_protected()` |
| [mlx-lm#854](https://github.com/ml-explore/mlx-lm/issues/854) | サーバー OOM、HTTP エラーなし | `oom_protected()` + `periodic_flush` |
| [mlx-lm#427](https://github.com/ml-explore/mlx-lm/issues/427) | M1 MBA でモデルロード時クラッシュ | `can_fit()` / `require_fit()` |
| [mlx-lm#1047](https://github.com/ml-explore/mlx-lm/issues/1047) | 大型モデル KV キャッシュ OOM | `can_fit()` + `ensure_headroom()` |
| [mlx-examples#1124](https://github.com/ml-explore/mlx-examples/issues/1124) | サーバーメモリリーク → 再起動 | `periodic_flush` + `watchdog` |

## テスト実績

- Mac Studio M1 Ultra (64GB) — MetalGuard 前 9回カーネルパニック、導入後 0回
- 10ユーザーバッチ：約90回モデルロード/アンロード、994秒、クラッシュゼロ
- テストモデル：Mistral-Small-3.2-24B、Phi-4-mini、Gemma-4-26B/31B、Pixtral-12B、LFM2-VL-3B

## ライセンス

MIT
