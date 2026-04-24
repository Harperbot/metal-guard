# MetalGuard

[English](README.md) | [繁體中文](README.zh-TW.md) | **日本語**

Apple Silicon 上で [MLX](https://github.com/ml-explore/mlx) を動かすための GPU セーフティレイヤー。

MLX 推論中に Metal ドライバーのバグが引き起こすカーネルパニックや OOM クラッシュを防止します —— 特にマルチモデルパイプライン、長時間稼働サーバー、ツール呼び出しが多いエージェントフレームワークを想定しています。

**現在のバージョン：** v0.9.0 — リリース履歴と各機能の背景は [CHANGELOG.md](CHANGELOG.md) を参照してください。v0.9.0 では `subprocess_inference_guard`（B1）、cross-model cadence + gemma-4 90 秒フロア（C5）、`gemma4_generation_flush`（C7）、および `KNOWN_PANIC_MODELS` アドバイザリレジストリを追加しました。

## 以下のキーワードで辿り着いた方、ここが答えです。

MLX を動かしている Mac が panic / 再起動 / クラッシュし、以下のいずれかのキーワードを検索した方のために metal-guard は設計されました：

- `IOGPUMemory.cpp:492 completeMemory() prepare count underflow`
- `IOGPUMemory.cpp:550` Apple Silicon + MLX のカーネルパニック
- `kIOGPUCommandBufferCallbackErrorOutOfMemory`
- `mlx::core::gpu::check_error` → `std::terminate` → `abort`（SIGABRT）
- `mlx::core::metal::GPUMemoryAllocator` / `fPendingMemorySet`
- `IOGPUGroupMemory.cpp:219` pending memory set の panic
- `mlx_lm.generate` が推論中にクラッシュ、親 Python プロセスも死亡
- `mlx_lm.server` の持続負荷下での OOM カーネルパニック / Mac 再起動
- `mlx_vlm` TurboQuant decode T=1 のサイレント破損（`mlx-vlm#967`）
- panic レポートに `com.apple.iokit.IOGPUFamily`（104.x / 129.x）
- メンテナーが `AGX_RELAX_CDM_CTXSTORE_TIMEOUT` に言及
- `ImpactingInteractivity` / MacBook 上で GPU watchdog が MLX を kill
- Gemma 4 / Mistral-Small / Pixtral / Llama 4-bit の出力が壊れる
- M1 / M2 / M3 / M4（Max / Ultra / Pro）Mac Studio / MacBook Pro のカーネルパニック
- 長コンテキスト（≥ 65 k）の prefill で再起動
- `transformers` 5.0 / 5.5 で `mlx_vlm.load` が import エラー
- MLX モデルの連続ロードで IOGPU underflow panic

関連する upstream のトラッキング：`ml-explore/mlx#3186` / `#3346` / `#3348` / `#3350` / `#3384` / `#3390`、`ml-explore/mlx-lm#883` / `#854` / `#897` / `#1015` / `#1047`、`Blaizzy/mlx-vlm#943` / `#967` / `#999` / `#1011` / `#1016`。metal-guard は `check_version_advisories()` でこれらを監視し、インストール済みバージョンが影響を受ける場合は起動時に警告します。

## 問題

Apple Silicon の Metal GPU ドライバーにはバグがあり、GPU メモリ管理が失敗したときに **プロセスをきれいに落とす代わりにカーネルが panic してマシンごと落ちます**。

```
panic(cpu 4 caller 0xfffffe0032a550f8):
  "completeMemory() prepare count underflow" @IOGPUMemory.cpp:492
```

複数の MLX モデルをシーケンシャルにロード／アンロードするあらゆるワークフローで発生し得ます —— Metal ドライバー内部の参照カウントがアンダーフローし、回復不能なカーネルパニックでマシンが再起動します。**あなたのコードの問題ではありません。** ドライバーレベルのバグで、修正の見通しもありません。[ml-explore/mlx-lm#883](https://github.com/ml-explore/mlx-lm/issues/883) を参照。

### 影響を受けるワークロード

| ワークロード | リスク | 理由 |
|----------|------|------|
| 単一モデルサーバー（LM Studio） | 低 | 一つのモデルしか使わない |
| マルチモデルパイプライン | **高** | load/unload の遷移ごとに panic 可能性 |
| 長時間サーバー（`mlx_lm.server`） | **高** | KV cache が無制限に膨張、Metal buffer 蓄積 |
| エージェント + tool calling | **高** | 対話ごとに 50–100 回の短い generate() |
| TurboQuant KV cache 圧縮 | **高** | メモリ上限に近づく |
| 24/7 daemon | **重大** | 日をまたいでメモリドリフト、自然なクリーンアップ地点がない |

## インストール

```bash
pip install metal-guard
```

`metal_guard.py` をプロジェクトにコピーしても OK —— Python 標準ライブラリと任意の `mlx` 以外に依存はありません。

## クイックスタート

```python
from metal_guard import metal_guard, require_cadence_clear, CircuitBreaker

# 1. 連続ロードを拒否（L9）
require_cadence_clear("mlx-community/gemma-4-26b-a4b-it-4bit")

# 2. panic が多発した後の新規 worker を拒否（L9）
CircuitBreaker().check()

# 3. GPU バウンドな thread を登録
import threading
thread = threading.Thread(target=run_mlx_generate, daemon=True)
thread.start()
metal_guard.register_thread(thread)
thread.join(timeout=120)

# 4. 安全なモデルアンロード（L1 + L2）
metal_guard.wait_for_threads()
metal_guard.safe_cleanup()            # gc + flush GPU + cooldown

# 5. OOM 保護された推論（L3）
result = metal_guard.oom_protected(generate, model, tokenizer, prompt=p)

# 6. ロード前のメモリ圧検査（L4）
metal_guard.ensure_headroom(model_name="my-model-8bit")

# 7. クラッシュ時の事後解析用 breadcrumb
metal_guard.breadcrumb("LOAD: my-model-8bit START")
```

ハードウェア対応のデフォルトを一行で：

```python
config = MetalGuard.recommended_config()
metal_guard.start_watchdog(
    warn_pct=config["watchdog_warn_pct"],
    critical_pct=config["watchdog_critical_pct"],
)
metal_guard.start_kv_cache_monitor(headroom_gb=config["kv_headroom_gb"])
```

## 機能

MetalGuard は**防御レイヤー（L1–L9）**と、補完する**予防的ヘルパー（R シリーズ）**で構成されます。すべての機能は単一の `metal_guard` モジュールから利用できます。各機能がいつ導入され、どの事例が動機だったかは [CHANGELOG.md](CHANGELOG.md) を参照してください。

### L1 — Thread 追跡

Metal を触る thread を登録しておき、cleanup が `mx.clear_cache()` を呼ぶ前に GPU 作業の完了を待てるようにします。

| API | 役割 |
|---|---|
| `metal_guard.register_thread(thread)` | GPU バウンド thread をレジストリに追加 |
| `metal_guard.wait_for_threads(timeout=None) -> int` | 登録済み thread が終わるまでブロック；残存数を返す |

### L2 — 安全なクリーンアップ

「メインスレッドが解放した buffer をワーカースレッドがまだ使っている」という元祖 panic 根因の競合を避ける、順序付きクリーンアップシーケンス。

| API | 役割 |
|---|---|
| `metal_guard.flush_gpu()` | `mx.eval(sync) + mx.clear_cache()` —— `wait_for_threads()` の後でのみ安全 |
| `metal_guard.safe_cleanup()` | 完全シーケンス：wait → `gc.collect` → flush → cooldown |
| `metal_guard.guarded_cleanup()` | 終了時に `safe_cleanup()` を走らせるコンテキストマネージャ |
| `kv_cache_clear_on_pressure(available_gb, growth_rate_gb_per_min)` | KV モニタに差し込める既製の `on_pressure` コールバック |

### L3 — OOM 回復

生の C++ Metal OOM を、自動クリーンアップと任意のリトライ付きで catch 可能な Python 例外に変換。

| API | 役割 |
|---|---|
| `metal_guard.oom_protected(fn, *args, max_retries=1, **kwargs)` | OOM catch → cleanup → retry で実行 |
| `metal_guard.oom_protected_context()` | コンテキストマネージャ版 |
| `metal_guard.is_metal_oom(exc) -> bool` | 任意の例外を分類 |
| `MetalOOMError` | `MemoryStats` 付きの catch 可能例外 |

### L4 — ロード前メモリ検査

収まらないロードを拒否し、HF model ID からモデルサイズを推定。

| API | 役割 |
|---|---|
| `metal_guard.can_fit(model_size_gb, overhead_gb=2.0) -> bool` | 例外を投げない検査 |
| `metal_guard.require_fit(model_size_gb, model_name, overhead_gb=2.0)` | クリーンアップしてもまだ入らなければ `MemoryError` |
| `MetalGuard.estimate_model_size_from_name(name)`（static） | 名前からパラメータ数 + 量子化を解析して GB 推定 |

### L5 — 長時間稼働の安全装置

`mlx_lm.server`、エージェントフレームワーク、24/7 daemon 向け。

| API | 役割 |
|---|---|
| `metal_guard.memory_stats() -> MemoryStats` | スナップショット（active / peak / limit / available / pct） |
| `metal_guard.is_pressure_high(threshold_pct=67.0) -> bool` | 簡易圧検査 |
| `metal_guard.ensure_headroom(model_name, threshold_pct=67.0)` | 圧が高ければクリーンアップ、そうでなければ no-op |
| `metal_guard.log_memory(label, model_name)` | クリーンアップせずログのみ |
| `metal_guard.start_periodic_flush(interval_secs=300)` | バックグラウンド定期 flush |
| `metal_guard.start_watchdog(interval_secs, warn_pct, critical_pct, on_critical)` | メモリドリフトのエスカレーション監視 |
| `metal_guard.start_kv_cache_monitor(interval_secs, headroom_gb, growth_rate_warn, on_pressure)` | KV 成長監視、OOM の前に発火 |
| `bench_scoped_load(model_id, ...)` | 連続ベンチマーク用のコンテキストマネージャ、次ロード前に必ずアンロード |

### L6 — デュアルモードスイッチャー

上流の mitigation を A/B するためにコード変更なしで defensive / observer を切替可能。

| API | 役割 |
|---|---|
| `current_mode() -> str` | `"defensive"`（デフォルト）または `"observer"` |
| `is_defensive() / is_observer() -> bool` | 便利な判定 |
| `describe_mode() -> dict` | モード名、説明、環境変数 |

### L7 — サブプロセス隔離

新しい `multiprocessing` 子プロセスで MLX を動かし、カーネルレベルの abort が親プロセスを殺さないようにします。

| API | 役割 |
|---|---|
| `MLXSubprocessRunner(model_id, ...)` | 永続 worker サブプロセス、クラッシュ時は再生成 |
| `call_model_isolated(model_id, prompt, ...)` | 単発ヘルパー：spawn → generate → シャットダウン |
| `shutdown_all_workers()` | 終了時に全 runner を強制停止 |
| `SubprocessCrashError / SubprocessTimeoutError` | 呼び出し側向けの型付き失敗 |

### L8 — プロセス間相互排他

`MLX_LOCK_PATH` 下のファイルロックで、同一マシン上で bench / server / pipeline が同時に Metal を初期化しないようにします。

| API | 役割 |
|---|---|
| `acquire_mlx_lock(label, force=False)` | 保有中なら `MLXLockConflict`；`force=True` は保有者に SIGTERM + タイムアウト + cooldown |
| `release_mlx_lock() -> bool` | このプロセスが保有していれば解放 |
| `read_mlx_lock() -> dict \| None` | 非ブロック検査；stale + zombie を自動修復 |
| `mlx_exclusive_lock(label)` | コンテキストマネージャ：enter で取得、exit で解放 |

### L9 — Cadence、panic ingest、circuit breaker *(v0.8.0)*

前 8 レイヤーを全て通り抜けた後の最終防衛線。**SIGABRT 層でも捕まえられなかった**カーネルパニックへの回答 —— Python が何かを目にする前にマシンはすでに再起動していました。唯一の対策は最初から panic trigger を避けることです。

| API | 役割 |
|---|---|
| `CadenceGuard(path=None, *, min_interval_sec=180)` | 永続化されたモデル別ロードタイムスタンプ |
| `CadenceGuard.check(model_id)` / `.mark_load(model_id)` | 近すぎるロードがあれば `CadenceViolation` |
| `require_cadence_clear(model_id, *, min_interval_sec=180)` | check + mark のアトミックヘルパー |
| `parse_panic_reports(directory=None, *, since_ts=None)` | `/Library/Logs/DiagnosticReports/*.panic` を走査して分類 |
| `ingest_panics_jsonl(*, report_dir=None, jsonl_path=None) -> int` | 重複排除で `~/.cache/metal-guard/panics.jsonl` に追記 |
| `CircuitBreaker(*, window_sec=3600, panic_threshold=2, cooldown_sec=3600)` | panic クラスター後に新規 worker を拒否 |
| `CircuitBreaker.check() / .status() / .clear()` | ゲート、ダッシュボード、オペレーター上書き |
| `detect_panic_signature(text) -> (name, explanation)` | panic ログを `prepare_count_underflow` / `pending_memory_set` / `ctxstore_timeout` / `metal_oom` に分類 |

### ハードウェア認識

| API | 役割 |
|---|---|
| `MetalGuard.detect_hardware() -> dict`（static） | チップ、GPU メモリ、推奨ワーキングセット、tier、IOGPUFamily kext バージョン |
| `MetalGuard.recommended_config() -> dict`（classmethod） | 検出されたハードウェアに対する各 L レイヤーの安全デフォルト |

### バージョン advisory と upstream パッチ

| API | 役割 |
|---|---|
| `check_version_advisories(packages=None) -> list[dict]` | インストール済み `(mlx, mlx-lm, mlx-vlm, transformers)` が既知の advisory に該当すれば警告 |
| `install_upstream_defensive_patches(force=False) -> dict[str, bool]` | Idempotent、バージョンゲート付きの monkey-patch |

### システム監査

| API | 役割 |
|---|---|
| `audit_wired_limit() -> dict` | 危険な `iogpu.wired_limit_mb` オーバーライドを警告（mlx-lm#1047） |
| `read_gpu_driver_version() -> str \| None` | IOGPUFamily kext のバージョン（mlx#3186） |
| `log_system_audit_at_startup() -> dict` | CLI / FastAPI lifespan 向けのラッパー |

### R シリーズ予防ヘルパー

| API | 役割 |
|---|---|
| `ModelDims`、`lookup_dims(model_id)`、`KNOWN_MODELS` | GQA 対応のキュレーション済みモデル次元ルックアップ |
| `estimate_prefill_peak_alloc_gb(context_tokens, dims)` | 保守的な per-layer + 全 KV の上限推定 |
| `require_prefill_fit(context_tokens, dims, available_gb, ...)` | 30 GB 単発確保 panic の前に `MetalOOMError` |
| `recommend_chunk_size(context_tokens, dims, ...)` | 二分探索による推奨チャンクサイズ（助言のみ） |
| `describe_prefill_plan(context_tokens, model_id_or_dims, available_gb)` | ダッシュボード安全、null 耐性のあるサマリ |
| `KVGrowthTracker(...).start / add_bytes / finalize / snapshot` | リクエスト別累積 KV ガード —— グローバル圧監視が見逃す暴走リクエストを捕捉 |
| `detect_process_mode() -> ProcessMode` | `"server" / "embedded" / "notebook" / "cli" / "subprocess_worker"` |
| `apply_mode_defaults(mode=None) -> dict` | モード別のタイムアウトと上限 |
| `describe_process_mode() -> dict` | ダッシュボードサマリ |
| `format_panic_for_apple_feedback(forensics, ...)` | そのまま Apple Feedback Assistant に貼り付けられるレポート |

### フォレンジクス

| API | 役割 |
|---|---|
| `metal_guard.breadcrumb(msg)` | fsync 済みの breadcrumb ログに書き込み（デフォルト `logs/metal_breadcrumb.log`） |

## デフォルトパス

L9 のすべての成果物は `~/.cache/metal-guard/` 配下：

| ファイル | 用途 | 上書き方法 |
|---|---|---|
| `~/.cache/metal-guard/cadence.json` | CadenceGuard タイムスタンプ | `CadenceGuard(path=...)` |
| `~/.cache/metal-guard/panics.jsonl` | panic アーカイブ | `ingest_panics_jsonl(jsonl_path=...)` / `CircuitBreaker(jsonl_path=...)` |
| `~/.cache/metal-guard/breaker.json` | CircuitBreaker 状態 | `CircuitBreaker(state_path=...)` |

breadcrumb ログのデフォルトは相対パス `logs/metal_breadcrumb.log`。`MetalGuard(breadcrumb_path=...)` で上書き可能。

## アーキテクチャ

```
┌─────────────────────────────────────────────────┐
│            アプリケーションコード               │
│  Agent loop / Server / Pipeline / Daemon        │
└──────────────────┬──────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────┐
│              MetalGuard                         │
│                                                 │
│  L9 CadenceGuard ──── 連続ロードを拒否          │
│  L9 CircuitBreaker ── panic クラスター後に拒否  │
│  L8 Process Lock ──── プロセス間排他            │
│  L7 Subprocess ────── panic 隔離された worker   │
│  L6 Dual mode ─────── defensive / observer      │
│  L5 Watchdog ──────── メモリ + KV ドリフト警告  │
│  L4 Pre-load check ── can_fit / require_fit     │
│  L3 OOM recovery ──── catch + cleanup + retry   │
│  L2 Safe cleanup ──── gc + flush + cooldown     │
│  L1 Thread registry ─ cleanup 前に待機          │
│  R4 Prefill guard ─── ceiling 超えの prefill 拒否 │
│  R5 KV tracker ────── per-request KV ガード     │
│  R8 Apple Feedback ── フォレンジクス formatter  │
└──────────────────┬──────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────┐
│           MLX + Metal Driver                    │
│  ⚠️  ドライバーバグ：OOM 代わりに panic        │
└─────────────────────────────────────────────────┘
```

## 実測

- Mac Studio M1 Ultra（64 GB）—— MetalGuard 導入前 9 回のカーネルパニック、L9 投入後 24 時間 panic ゼロ
- 10 人バッチパイプライン：約 90 回のモデル load/unload サイクル、994 秒、クラッシュゼロ
- モデル：Mistral-Small-3.2-24B、Phi-4-mini、Gemma-4-26B / 31B、Pixtral-12B、LFM2-VL-3B（4-bit と 8-bit）

## 既知の影響を受けるモデル（v0.9.0, 2026-04）

一部のモデルは race window が広く、MetalGuard は狭めることはできても閉じることはできません。その場合はここに記録し、プロダクションで読み込む前に判断材料にできるようにしています。

### `mlx-community/gemma-4-31b-it-8bit` —— 繰り返し発生

Harper の Mac Studio で **24 時間を空けて、同じパイプライン、同じモデル** の 2 回のプロダクション kernel panic。panic シグネチャは同一で `IOGPUMemory.cpp:492 "completeMemory() prepare count underflow"`。

| #   | ローカル時刻     | PID   | Spawn → panic | 状況                                                         |
|-----|------------------|-------|--------------:|--------------------------------------------------------------|
|  7  | 2026-04-23 03:14 | 67840 |        約 6 分 | rezivot パイプライン、cross-model cadence 未配線             |
| 11  | 2026-04-24 03:14 | 26608 |      約 1.5 分 | #7 と同じパイプライン；classic L9 防御があっても worker ready 後 ~1.5 分で panic |

コミュニティでの裏付け（すべて 2026-04）：

- [Hannecke —「MLX Crashed My Mac」（Medium）](https://medium.com/@michael.hannecke/how-my-local-coding-agent-crashed-my-mac-and-what-i-learned-about-mlx-memory-management-e0cbad01553c) —— M4 Max 64 GB、同じシグネチャ；`Qwen3-Coder-30B-A3B` MoE に pivot。
- [`lmstudio-ai/lmstudio-bug-tracker#1740`](https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/1740)「Gemma-4 31b KV excessive KV cache footprint」—— 同系統の KV 膨張を裏付け：8192 context で 26 GB VRAM；hybrid attention（sliding 50 + global 10）KV cache + 8-bit 重み（約 34 GB）+ full-context KV が 64 GB Mac の unified memory を超えていく。
- [`ml-explore/mlx-lm#883`](https://github.com/ml-explore/mlx-lm/issues/883) —— M3 Ultra 96 GB、同じシグネチャ。
- [`ml-explore/mlx#3186`（2026-04-24 コメント）](https://github.com/ml-explore/mlx/issues/3186#issuecomment-4314204974) —— 独立した第三者観測：Mac mini M4 base 32 GB、macOS 26.4.1（`25E253`）、mlx 0.31.2、`mlx-community/Qwen3.6-35B-A3B-4bit`。`mlx_lm.server` 起動から 8 分 16 秒で panic；`--prompt-cache-bytes 8 GiB` でも防げず；投稿者はプロダクションサービングを `llama.cpp` に切替。本プロジェクトの「two-trigger-path hypothesis」を明示的に引用しています。

**結論。** macOS 26.4.x はこのバグを修正していません。macOS 26.5 beta も修正していません。RAM を 96 GB に増やしても防げません。MetalGuard v0.9.0 は複数の race window を狭めますが（cross-model cadence、gemma-4 90 秒フロア、first-generate flush、subprocess inference guard）、Harper の実ワークロードではこのモデルの panic を完全には排除できません。

プログラムからアドバイザリを照会できます：

```python
from metal_guard import check_known_panic_model, warn_if_known_panic_model

advisory = check_known_panic_model(model_id)
if advisory is not None:
    # 判断：ロード拒否、backend 切替、または明示的 ack 後に続行
    ...

# あるいは fire-and-forget：同一 model_id につきプロセスごとに log.warning 1 回のみ
warn_if_known_panic_model(model_id)
```

## MetalGuard で足りないとき

v0.9.0 のすべての防御（B1 + C5 + C7 + CircuitBreaker）を有効にしても同じモデルで panic が繰り返す場合、それは race window が userspace 層で閉じられないほど広いというシグナルです。投資対効果順に 2 つのエスケープハッチがあります：

1. **backend を切り替える。** [Ollama](https://ollama.com/) と [`llama.cpp`](https://github.com/ggml-org/llama.cpp) はどちらも Metal MPS を内部で使いますが、persistent worker アーキテクチャなのでサブプロセス teardown race を丸ごと回避します。Harper の `harper-finance` プロジェクトは 2026-04-23 に Ollama に移行し、以降 panic ゼロで稼働中。[`mlx#3186` の M4-base 報告者](https://github.com/ml-explore/mlx/issues/3186#issuecomment-4314204974) も同じ理由でプロダクションサービングを `llama.cpp` に切替えています。失うのは生スループット（同レポートでは MLX が prefill で 30–55 % 速いと計測）、得るのは「マシンを落とさない」。

2. **別のモデルファミリーに pivot する。** Mixture-of-Experts（MoE）系統 —— 例：[`mlx-community/gemma-4-26b-a4b-it-4bit`](https://huggingface.co/mlx-community/gemma-4-26b-a4b-it-4bit)、`Qwen3-Coder-30B-A3B` —— は 1 回の forward あたり active-parameter が格段に小さく、KV 成長曲線も狭めです。コミュニティの報告（Hannecke、lmstudio#1740）は「同じエコシステム内で最も確実な回避策は MoE」で一致しています。

MetalGuard は **両方のエスケープハッチと補完的です** —— Ollama 下でもリクエストごとにサブプロセス worker を spawn する構成なら `subprocess_inference_guard` は有効ですし、モデルを hot-swap する限り backend によらず `CadenceGuard` は依然として役立ちます。

### 授業料で学んだ SOP

タイムラインの panic #10（[CHANGELOG](CHANGELOG.md) 参照）は、ホスト端末での *対話的* な `python -c "import sentence_transformers"` が引き金でした —— バージョン確認のためのコマンドで、プロダクション MLX ワークロードとは無関係です。`torch`、`mlx`、`mlx_lm`、`mlx_vlm`、`sentence_transformers`、`transformers`、`diffusers`、`accelerate` のいずれかを import すると Metal MPS backend が初期化され、プロセス exit 時に同じ kernel バグを踏みます。panic cooldown 中は次を優先してください：

- バージョン確認は `pip show <pkg>`、または
- `python -c "import importlib.metadata as m; print(m.version('<pkg>'))"`（パッケージを cascade import しない）。

panic cooldown 中は **絶対に** `python -c "import <ml-package>; print(<ml-package>.__version__)"` を実行しないでください。

## 制限事項 —— これは回避策であり、修正ではありません

MetalGuard は **userspace の防御レイヤー**です。根本バグは Apple の IOGPUFamily kext（[mlx#3186](https://github.com/ml-explore/mlx/issues/3186)）内部にあり、Python からは触れません。MetalGuard が実際に行うのは：

1. **トリガー率の低減** —— L1–L5 と L9 CadenceGuard で既知のトリガー経路（連続ロード、thread 競合の cleanup、KV 無制限成長、単発確保上限を超える prefill）を回避。
2. **爆発半径の封じ込め** —— L7 は MLX をサブプロセスで走らせるので、catch 可能な abort は子プロセスだけを殺します。ただし *カーネル* panic はマシンごと再起動します；サブプロセス隔離は「panic 発生時にどのモデルが GPU を握っていたか」を知るための手段です。
3. **再起動後の連鎖を防止** —— L9 CircuitBreaker は直近 1 時間で ≥ 2 回の panic 後に新規 worker を拒否し、再起動直後に同じモデルをロードして panic を再演しないようにします。

panic は依然として発生し得ます（特に [mlx#3390](https://github.com/ml-explore/mlx/issues/3390) —— `com.Metal.CompletionQueueDispatch` 上でディスパッチされる catch 不能な completion-handler abort。Python signal handler が発火する前にプロセスが終わります）。Harper のマシンは 1 日平均 1.4 回の panic から、L9 投入後 24 時間 panic ゼロになりました。ただしこれは**リスク低減**であり、**リスク排除**ではありません。Apple が kext を修正するまで、これが Python-side 層で到達できる上限です。

## 関連する upstream issue

| Issue | 問題 | 対応機能 |
|---|---|---|
| [mlx#3186](https://github.com/ml-explore/mlx/issues/3186) | IOGPUFamily カーネルパニック（正規ケース） | L1/L2/L8/L9 + `read_gpu_driver_version` |
| [mlx#3346](https://github.com/ml-explore/mlx/issues/3346) | `fPendingMemorySet` 第二シグネチャ | `detect_panic_signature` + L9 |
| [mlx#3348](https://github.com/ml-explore/mlx/pull/3348) | CommandEncoder thread-local | advisory ゲートによる observer モード |
| [mlx#3350](https://github.com/ml-explore/mlx/issues/3350) | MetalAllocator バッファプールの膨張 | advisory + `mx.set_cache_limit` ガイダンス |
| [mlx#3384](https://github.com/ml-explore/mlx/issues/3384) | 4-bit SDPA の数値ずれ | `check_version_advisories` |
| [mlx#3390](https://github.com/ml-explore/mlx/issues/3390) | catch 不能な completion-handler abort | L7 サブプロセス隔離 + `AGX_RELAX_CDM_CTXSTORE_TIMEOUT` |
| [mlx-lm#883](https://github.com/ml-explore/mlx-lm/issues/883) / [#1015](https://github.com/ml-explore/mlx-lm/issues/1015) | KV cache 膨張によるカーネルパニック | L1 thread + L2 safe cleanup |
| [mlx-lm#854](https://github.com/ml-explore/mlx-lm/issues/854) | サーバー OOM クラッシュ | L3 `oom_protected` + L5 periodic flush |
| [mlx-lm#897](https://github.com/ml-explore/mlx-lm/issues/897) | `mlx_lm.server` と transformers ≥ 5.0 の衝突 | `check_version_advisories` |
| [mlx-lm#1047](https://github.com/ml-explore/mlx-lm/issues/1047) | `wired_limit` と panic の相関 | `audit_wired_limit` |
| [mlx-lm#1128](https://github.com/ml-explore/mlx-lm/issues/1128) | `TokenizerWrapper.think_start_id` クラッシュ | `install_upstream_defensive_patches` |
| [mlx-vlm#943](https://github.com/Blaizzy/mlx-vlm/issues/943) / [#967](https://github.com/Blaizzy/mlx-vlm/pull/967) / [#999](https://github.com/Blaizzy/mlx-vlm/issues/999) | TurboQuant / cache-thrash / Gemma4 出力破損 | `check_version_advisories` |

## ライセンス

MIT
