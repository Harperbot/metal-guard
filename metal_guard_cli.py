"""metal-guard CLI — standalone status / panics / breadcrumb / mode 查詢工具。

Phase 4 spec section 6 第 2 項：「metal-guard 加 standalone CLI status（PyPI
用戶用）」。給未透過 Harper 自家 CLI 的下游用戶一鍵看 metal-guard 健康狀態。

用法：
  metal-guard                 # 同 `status`，full snapshot
  metal-guard status          # full snapshot（panic gate + MLX lock + mode + bread）
  metal-guard panics [--since-hours N] [--json]
  metal-guard breadcrumb [-n N] [--json]
  metal-guard mode [--json]
  metal-guard --version
  metal-guard --help

Exit codes:
  0  healthy（無 panics、無 stale lock）
  1  warning（24h/72h 有 panics，或 lock 被 stale process 持有）
  2  critical（>5 panics in 72h）
  64 usage error
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import metal_guard as _mg


__all__ = ["main", "safe_python_main"]

_DEFAULT_SINCE_HOURS = 72
_BREADCRUMB_DEFAULT_LINES = 20
_PANIC_WARN_72H = 1
_PANIC_CRITICAL_72H = 5

_VERSION = getattr(_mg, "__version__", None) or "0.9.0"


def _ansi(stream) -> dict[str, str]:
    if stream.isatty():
        return {
            "red": "\033[31m", "yellow": "\033[33m", "green": "\033[32m",
            "dim": "\033[2m", "bold": "\033[1m", "reset": "\033[0m",
        }
    return {k: "" for k in ("red", "yellow", "green", "dim", "bold", "reset")}


# ── data collection ─────────────────────────────────────────────────


def _warn(msg: str) -> None:
    """Surface non-fatal collector failure to stderr — critic R1 P2 修：
    避免「靜默吞掉錯誤」（python/coding-style.md error handling 紅線）。
    `--json` 不該污染 stdout 但 stderr 可印（caller 能 redirect 過濾）。"""
    sys.stderr.write(f"metal-guard: {msg}\n")


def _collect_panics(since_hours: float = _DEFAULT_SINCE_HOURS) -> list[dict[str, Any]]:
    since_ts = time.time() - since_hours * 3600.0
    try:
        return _mg.parse_panic_reports(since_ts=since_ts)
    except AttributeError as e:
        _warn(f"parse_panic_reports unavailable in this metal-guard version: {e}")
        return []
    except OSError as e:
        _warn(f"panic report scan IO error: {e}")
        return []


def _collect_lock() -> dict[str, Any] | None:
    try:
        return _mg.read_mlx_lock()
    except AttributeError as e:
        _warn(f"read_mlx_lock unavailable: {e}")
        return None
    except OSError as e:
        _warn(f"mlx lock read IO error: {e}")
        return None


def _collect_mode() -> dict[str, str]:
    try:
        return _mg.describe_mode()
    except AttributeError as e:
        _warn(f"describe_mode unavailable (metal-guard version too old): {e}")
        return {"mode": "unknown",
                "description": "metal-guard mode API unavailable",
                "env": "METALGUARD_MODE"}


def _collect_memory() -> dict[str, Any] | None:
    """嘗試取 MetalGuard memory stats — 需要 mlx 已安裝且 Apple Silicon。

    CLI process 沒 import mlx 屬正常情形，回 None 不 warn（避免 noise）。
    其他例外（API 改變等）才 warn。
    """
    try:
        stats = _mg.metal_guard.memory_stats()
        return {
            "active_gb": round(stats.active_gb, 2),
            "peak_gb": round(stats.peak_gb, 2),
            "available_gb": round(stats.available_gb, 2),
            "limit_gb": round(stats.limit_gb, 2),
            "active_pct": round(stats.active_pct, 1),
            "peak_pct": round(stats.peak_pct, 1),
        }
    except ImportError:
        return None  # mlx 未裝是常態，不 warn
    except (AttributeError, RuntimeError) as e:
        _warn(f"memory_stats unavailable: {e}")
        return None


def _collect_breadcrumb_tail(n_lines: int = _BREADCRUMB_DEFAULT_LINES) -> dict[str, Any]:
    """讀 breadcrumb log 最後 N 行。優先用 metal_guard 的 logs/，fallback 到
    `~/.harper/logs/metal_breadcrumb.log`（Harper 慣用路徑）。"""
    candidates = [
        Path(getattr(_mg.metal_guard, "_breadcrumb_path", None) or "")
        if getattr(_mg, "metal_guard", None) else None,
        Path.home() / ".harper" / "logs" / "metal_breadcrumb.log",
        Path("logs/metal_breadcrumb.log"),
    ]
    for path in candidates:
        if path is None or not path:
            continue
        try:
            if not path.is_file():
                continue
            with path.open("r", encoding="utf-8", errors="replace") as fh:
                lines = fh.readlines()
            return {"path": str(path), "lines": [ln.rstrip() for ln in lines[-n_lines:]]}
        except OSError:
            continue
    return {"path": None, "lines": []}


def _classify_health(panics: list[dict[str, Any]],
                     lock: dict[str, Any] | None) -> tuple[str, int]:
    """回 (label, exit_code)。"""
    panic_count = len(panics)
    if panic_count >= _PANIC_CRITICAL_72H:
        return "critical", 2
    # Lock 被 stale process 持有 = warn（dead pid 可能 lock leak）
    if lock and lock.get("stale"):
        return "warn", 1
    if panic_count >= _PANIC_WARN_72H:
        return "warn", 1
    return "ok", 0


def _build_status_payload(*, since_hours: float = _DEFAULT_SINCE_HOURS,
                          breadcrumb_lines: int = _BREADCRUMB_DEFAULT_LINES) -> dict[str, Any]:
    panics = _collect_panics(since_hours)
    lock = _collect_lock()
    mode = _collect_mode()
    memory = _collect_memory()
    breadcrumb = _collect_breadcrumb_tail(breadcrumb_lines)
    health, exit_code = _classify_health(panics, lock)
    return {
        "version": _VERSION,
        "health": health,
        "exit_code": exit_code,
        "panics_window_hours": since_hours,
        "panics_count": len(panics),
        "panics": panics,
        "mlx_lock": lock,
        "mode": mode,
        "memory": memory,
        "breadcrumb": breadcrumb,
    }


# ── render ──────────────────────────────────────────────────────────


def _fmt_health_badge(health: str, color: dict[str, str]) -> str:
    if health == "ok":
        return f"{color['green']}🟢 OK{color['reset']}"
    if health == "warn":
        return f"{color['yellow']}🟡 WARN{color['reset']}"
    if health == "critical":
        return f"{color['red']}🔴 CRITICAL{color['reset']}"
    return f"{color['dim']}? {health}{color['reset']}"


def _render_status(p: dict[str, Any], stream) -> None:
    c = _ansi(stream)
    badge = _fmt_health_badge(p["health"], c)
    stream.write(
        f"{c['bold']}metal-guard {p['version']}{c['reset']}  {badge}\n"
    )
    # Mode
    mode = p["mode"]
    stream.write(
        f"  {c['dim']}mode{c['reset']}        "
        f"{mode.get('mode', '?')} — {c['dim']}{mode.get('description', '')}{c['reset']}\n"
    )
    # Panic gate
    pcount = p["panics_count"]
    pcolor = c['green'] if pcount == 0 else (c['red'] if pcount >= _PANIC_CRITICAL_72H else c['yellow'])
    stream.write(
        f"  {c['dim']}panics{c['reset']}      "
        f"{pcolor}{pcount}{c['reset']} in last {p['panics_window_hours']:.0f}h\n"
    )
    if pcount > 0:
        for ev in p["panics"][-5:]:  # 最後 5 個
            sig = ev.get("signature", "?")
            ts = ev.get("ts", 0)
            ts_str = time.strftime("%Y-%m-%d %H:%M", time.localtime(ts)) if ts else "?"
            stream.write(f"    {c['dim']}- {ts_str} {sig}{c['reset']}\n")

    # MLX lock
    lock = p["mlx_lock"]
    if lock:
        label = lock.get("label", "?")
        pid = lock.get("pid", "?")
        stale = " (STALE)" if lock.get("stale") else ""
        scolor = c['red'] if lock.get("stale") else c['yellow']
        stream.write(
            f"  {c['dim']}mlx-lock{c['reset']}    "
            f"{scolor}held by {label} pid={pid}{stale}{c['reset']}\n"
        )
    else:
        stream.write(f"  {c['dim']}mlx-lock{c['reset']}    {c['green']}none{c['reset']}\n")

    # Memory（可能 None）
    mem = p["memory"]
    if mem is not None:
        active_pct = mem["active_pct"]
        mcolor = c['red'] if active_pct >= 80 else (c['yellow'] if active_pct >= 60 else c['green'])
        stream.write(
            f"  {c['dim']}memory{c['reset']}      "
            f"{mcolor}{mem['active_gb']:.2f} / {mem['limit_gb']:.0f} GB"
            f" ({active_pct:.0f}%){c['reset']}"
            f" {c['dim']}peak {mem['peak_gb']:.2f} GB{c['reset']}\n"
        )
    else:
        stream.write(
            f"  {c['dim']}memory{c['reset']}      "
            f"{c['dim']}unavailable (mlx not loaded){c['reset']}\n"
        )

    # Breadcrumb
    bc = p["breadcrumb"]
    bc_path = bc.get("path")
    bc_lines = bc.get("lines", [])
    if bc_path:
        stream.write(
            f"  {c['dim']}breadcrumb{c['reset']}  {bc_path} "
            f"{c['dim']}({len(bc_lines)} recent lines){c['reset']}\n"
        )
        if bc_lines:
            for ln in bc_lines[-3:]:  # 概覽只印最後 3 行
                stream.write(f"    {c['dim']}{ln[:100]}{c['reset']}\n")
    else:
        stream.write(
            f"  {c['dim']}breadcrumb{c['reset']}  "
            f"{c['dim']}no log file found{c['reset']}\n"
        )


def _render_panics(p: dict[str, Any], stream) -> None:
    c = _ansi(stream)
    panics = p["panics"]
    if not panics:
        stream.write(
            f"{c['green']}No panics in last {p['panics_window_hours']:.0f}h.{c['reset']}\n"
        )
        return
    stream.write(
        f"{c['bold']}{len(panics)} panic(s) in last {p['panics_window_hours']:.0f}h:{c['reset']}\n"
    )
    for ev in panics:
        ts = ev.get("ts", 0)
        ts_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts)) if ts else "?"
        sig = ev.get("signature", "?")
        pid = ev.get("pid")
        explanation = ev.get("explanation") or ""
        src = ev.get("source_file", "")
        stream.write(
            f"  {c['red']}{ts_str}{c['reset']}  "
            f"sig={sig}  pid={pid}\n"
            f"    {c['dim']}{explanation}{c['reset']}\n"
            f"    {c['dim']}{src}{c['reset']}\n"
        )


def _render_breadcrumb(p: dict[str, Any], stream) -> None:
    c = _ansi(stream)
    bc = p["breadcrumb"]
    path = bc.get("path")
    lines = bc.get("lines", [])
    if not path:
        stream.write(
            f"{c['yellow']}No breadcrumb log found.{c['reset']}\n"
            f"  Searched paths:\n"
            f"    {c['dim']}- metal_guard.metal_guard._breadcrumb_path{c['reset']}\n"
            f"    {c['dim']}- ~/.harper/logs/metal_breadcrumb.log{c['reset']}\n"
            f"    {c['dim']}- ./logs/metal_breadcrumb.log{c['reset']}\n"
        )
        return
    stream.write(f"{c['bold']}{path}{c['reset']}  ({len(lines)} lines)\n")
    for ln in lines:
        stream.write(f"  {ln}\n")


def _render_mode(p: dict[str, Any], stream) -> None:
    c = _ansi(stream)
    mode = p["mode"]
    stream.write(
        f"{c['bold']}metal-guard mode:{c['reset']} "
        f"{mode.get('mode', '?')}\n"
        f"  {c['dim']}{mode.get('description', '')}{c['reset']}\n"
    )
    env = mode.get("env", "METALGUARD_MODE")
    val = os.environ.get(env, "")
    stream.write(f"  {c['dim']}env {env}={val or '(unset)'}{c['reset']}\n")


# ── main ────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    # `--json` 用 parent parser 共用，每個 subcmd 都繼承（critic R1 P1#1 修）
    # — 既支援 `metal-guard --json status` 也支援 `metal-guard status --json`。
    # `default=SUPPRESS` 防 subparser 預設值覆蓋 top-level 設定（argparse
    # 已知 gotcha：parents=[...] + subparser 各自有 `default=False` 時
    # subparser parse 階段會 reset 屬性）。
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--json", action="store_true", default=argparse.SUPPRESS,
                        help="Emit raw JSON instead of human-readable output")

    p = argparse.ArgumentParser(
        prog="metal-guard",
        description="metal-guard standalone CLI — panic gate / MLX lock / mode / breadcrumb",
        parents=[common],
    )
    p.add_argument("--version", action="version", version=f"metal-guard {_VERSION}")
    sub = p.add_subparsers(dest="cmd")

    sub.add_parser("status", help="Full snapshot (default)", parents=[common])

    pp = sub.add_parser("panics", help="List recent kernel panics", parents=[common])
    pp.add_argument("--since-hours", type=float, default=_DEFAULT_SINCE_HOURS,
                    help=f"Time window in hours (default {_DEFAULT_SINCE_HOURS})")

    pb = sub.add_parser("breadcrumb", help="Tail metal_breadcrumb.log",
                        parents=[common])
    pb.add_argument("-n", "--lines", type=int, default=_BREADCRUMB_DEFAULT_LINES,
                    help=f"Number of trailing lines (default {_BREADCRUMB_DEFAULT_LINES})")

    sub.add_parser("mode", help="Show current metal-guard mode", parents=[common])

    # v0.10 — L10/L11/L12/L13 wiring（Harper-private port）
    sub.add_parser(
        "panic-gate",
        help="L10 evaluate panic cooldown (rc=0/2/3 for plist wrappers)",
        parents=[common],
    )

    pm = sub.add_parser(
        "postmortem",
        help="L12 collect panic + breadcrumb bundle",
        parents=[common],
    )
    pm.add_argument(
        "output_dir",
        help="Bundle output directory (created if missing)",
    )

    sw = sub.add_parser(
        "status-write",
        help="L13 write JSON snapshot to file (--once or daemon)",
        parents=[common],
    )
    sw.add_argument(
        "--out",
        type=Path,
        default=Path.home() / ".cache" / "metal-guard" / "status.json",
        help="Output JSON path (default ~/.cache/metal-guard/status.json)",
    )
    sw.add_argument(
        "--interval",
        type=float,
        default=30.0,
        help="Daemon mode write interval in seconds (default 30s)",
    )
    sw.add_argument(
        "--once",
        action="store_true",
        help="Write once and exit (overrides --interval)",
    )

    osub = sub.add_parser(
        "orphan-scan",
        help="L11 scan breadcrumb for SUBPROC_PRE without matching POST",
        parents=[common],
    )
    osub.add_argument(
        "--threshold-sec",
        type=float,
        default=90.0,
        help="Age threshold (seconds) before a PRE counts as orphan (default 90s)",
    )

    sub.add_parser(
        "ack",
        help="L10 touch ~/.metal-guard-ack to clear lockout",
        parents=[common],
    )
    return p


def _cmd_panic_gate(args, stream) -> int:
    """L10 — evaluate cooldown, mirror Harper plist wrapper exit codes."""
    verdict = _mg.evaluate_panic_cooldown()
    payload = {
        "exit_code": verdict.exit_code,
        "reason": verdict.reason,
        "recent_panics_24h": verdict.recent_panics_24h,
        "recent_panics_72h": verdict.recent_panics_72h,
        "cooldown_until": (
            verdict.cooldown_until.isoformat(timespec="seconds")
            if verdict.cooldown_until else None
        ),
    }
    if getattr(args, "json", False):
        json.dump(payload, stream, indent=2)
        stream.write("\n")
    else:
        c = _ansi(stream)
        if verdict.exit_code == 0:
            color = c["green"]
            badge = "🟢 PROCEED"
        elif verdict.exit_code == 2:
            color = c["yellow"]
            badge = "🟡 COOLDOWN"
        else:
            color = c["red"]
            badge = "🔴 GATE BROKEN"
        stream.write(f"{color}{badge}{c['reset']}  {verdict.reason}\n")
        stream.write(
            f"  {c['dim']}24h={verdict.recent_panics_24h} "
            f"72h={verdict.recent_panics_72h}{c['reset']}\n"
        )
    return verdict.exit_code


def _cmd_postmortem(args, stream) -> int:
    """L12 — collect bundle into args.output_dir."""
    out_dir = Path(args.output_dir)
    result = _mg.run_postmortem(out_dir)
    if getattr(args, "json", False):
        json.dump(result, stream, indent=2, default=str)
        stream.write("\n")
    else:
        c = _ansi(stream)
        if result["status"] == "disabled":
            stream.write(f"{c['yellow']}postmortem disabled (env kill-switch){c['reset']}\n")
        else:
            stream.write(
                f"{c['bold']}postmortem bundle:{c['reset']} {result['output_dir']}\n"
                f"  {c['dim']}panic files copied:{c['reset']} {result['panic_count']}\n"
                f"  {c['dim']}index:{c['reset']} {result['index']}\n"
            )
            if result.get("sentinel"):
                stream.write(
                    f"  {c['dim']}sentinel cooldown written:{c['reset']} {result['sentinel']}\n"
                )
    return 0


def _cmd_status_write(args, stream) -> int:
    """L13 — atomic snapshot writer (one-shot or daemon mode)."""
    out = Path(args.out)
    if args.once:
        try:
            written = _mg.write_status_snapshot(out)
        except OSError as exc:
            _warn(f"snapshot write failed: {exc}")
            return 1
        if not getattr(args, "json", False):
            stream.write(f"wrote snapshot to {written}\n")
        else:
            json.dump({"out": str(written), "mode": "once"}, stream, indent=2)
            stream.write("\n")
        return 0

    # Daemon mode — interval loop with SIGTERM/SIGINT graceful shutdown
    import signal as _signal

    running = {"flag": True}

    def _stop(*_a):
        running["flag"] = False
        sys.stderr.write("metal-guard status-write: SIGTERM received\n")

    _signal.signal(_signal.SIGTERM, _stop)
    _signal.signal(_signal.SIGINT, _stop)

    sys.stderr.write(
        f"metal-guard status-write: out={out} interval={args.interval:.1f}s\n"
    )
    consecutive_errors = 0
    while running["flag"]:
        try:
            _mg.write_status_snapshot(out)
            consecutive_errors = 0
        except OSError as exc:
            consecutive_errors += 1
            _warn(f"snapshot write failed (#{consecutive_errors}): {exc}")
            if consecutive_errors >= 10:
                # Widen interval to avoid log spam under sustained failure
                time.sleep(args.interval * 5)
                continue
        # Sleep in 1s chunks so SIGTERM is responsive
        remaining = args.interval
        while remaining > 0 and running["flag"]:
            chunk = min(remaining, 1.0)
            time.sleep(chunk)
            remaining -= chunk
    return 0


def _cmd_orphan_scan(args, stream) -> int:
    """L11 — scan breadcrumb for orphan SUBPROC_PRE entries."""
    orphans = _mg.scan_orphan_subproc_pre(threshold_sec=args.threshold_sec)
    payload = [
        {
            "model_id": o.model_id,
            "pre_ts": o.pre_ts.isoformat(timespec="seconds"),
            "age_sec": round(o.age_sec, 1),
            "pid": o.pid,
        }
        for o in orphans
    ]
    if getattr(args, "json", False):
        json.dump({"threshold_sec": args.threshold_sec, "orphans": payload}, stream, indent=2)
        stream.write("\n")
    else:
        c = _ansi(stream)
        if not orphans:
            stream.write(
                f"{c['green']}No orphan SUBPROC_PRE entries "
                f"(threshold {args.threshold_sec:.0f}s).{c['reset']}\n"
            )
        else:
            stream.write(
                f"{c['red']}{len(orphans)} orphan SUBPROC_PRE entries "
                f"(threshold {args.threshold_sec:.0f}s):{c['reset']}\n"
            )
            for o in payload:
                stream.write(
                    f"  pid={o['pid']} model={o['model_id']} "
                    f"age={o['age_sec']:.0f}s pre_ts={o['pre_ts']}\n"
                )
    return 1 if orphans else 0


def _cmd_ack(args, stream) -> int:
    """L10 — touch ack file to clear active lockout."""
    path = _mg.ack_panic_lockout()
    if getattr(args, "json", False):
        json.dump({"ack_path": str(path), "status": "touched"}, stream, indent=2)
        stream.write("\n")
    else:
        c = _ansi(stream)
        stream.write(
            f"{c['green']}ack written:{c['reset']} {path}\n"
            f"  {c['dim']}lockout will clear if cooldown gate runs within 24h "
            f"and no newer panic occurs{c['reset']}\n"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    cmd = args.cmd or "status"

    # v0.10 subcommands take direct paths — don't run full status payload
    if cmd == "panic-gate":
        return _cmd_panic_gate(args, sys.stdout)
    if cmd == "postmortem":
        return _cmd_postmortem(args, sys.stdout)
    if cmd == "status-write":
        return _cmd_status_write(args, sys.stdout)
    if cmd == "orphan-scan":
        return _cmd_orphan_scan(args, sys.stdout)
    if cmd == "ack":
        return _cmd_ack(args, sys.stdout)

    since_hours = getattr(args, "since_hours", _DEFAULT_SINCE_HOURS)
    bc_lines = getattr(args, "lines", _BREADCRUMB_DEFAULT_LINES)
    payload = _build_status_payload(
        since_hours=since_hours, breadcrumb_lines=bc_lines,
    )

    if getattr(args, "json", False):
        json.dump(payload, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
        return payload["exit_code"]

    if cmd == "status":
        _render_status(payload, sys.stdout)
    elif cmd == "panics":
        _render_panics(payload, sys.stdout)
    elif cmd == "breadcrumb":
        _render_breadcrumb(payload, sys.stdout)
    elif cmd == "mode":
        _render_mode(payload, sys.stdout)
    else:
        sys.stderr.write(f"metal-guard: unknown command: {cmd}\n")
        return 64

    return payload["exit_code"]


def safe_python_main(argv: list[str] | None = None) -> int:
    """Pip-installable entry replacing the `scripts/mlx-safe-python` bash wrapper.

    `pyproject.toml` exposes this as the `mlx-safe-python` console script. Calls
    L10's `evaluate_panic_cooldown()` in-process (no subprocess fork → no module
    path bug — critic R1 P0-1) then exec's the user-supplied python invocation.

    Exit codes match the bash wrapper:
        0   ran python successfully (or cooldown absent)
        10  blocked by cooldown (set ``MLX_SAFE_PYTHON_FORCE=1`` to override)
        11  panic gate itself broken (fail-open: runs python anyway, WARN)
        127 python binary not found

    Env:
        MLX_SAFE_PYTHON_BIN              python binary (default `which python3`)
        MLX_SAFE_PYTHON_FORCE=1          run despite cooldown
        METALGUARD_PANIC_GATE_DISABLED=1 disables the L10 gate itself
    """
    import shutil as _shutil

    if argv is None:
        argv = sys.argv[1:]

    python_bin = os.environ.get("MLX_SAFE_PYTHON_BIN") or _shutil.which("python3")
    if not python_bin or not os.access(python_bin, os.X_OK):
        sys.stderr.write(
            "[mlx-safe-python] FATAL: no python3 on PATH "
            "(or MLX_SAFE_PYTHON_BIN not executable)\n"
        )
        return 127

    # Pip / build / venv tooling does not import torch / mlx at CLI level —
    # let these pass without gate-blocking. Strict allowlist match: only
    # bypass when argv[0]=="-m" AND argv[1] is a known-safe module name.
    _safe_modules = {"pip", "pip3", "build", "virtualenv", "venv",
                     "ensurepip", "tomllib"}
    if len(argv) >= 2 and argv[0] == "-m":
        mod_name = argv[1].split(".", 1)[0]  # `pip` from `pip.foo`
        # Match `pip3.14` etc.
        if mod_name in _safe_modules or any(
            argv[1].startswith(f"{m}3.") or argv[1].startswith(f"{m}.")
            for m in ("pip", "pip3")
        ):
            os.execv(python_bin, [python_bin, *argv])
            return 0  # never reached

    # Run L10 gate directly (no subprocess — see P0-1 fix)
    try:
        verdict = _mg.evaluate_panic_cooldown()
    except Exception as exc:  # noqa: BLE001 fail-open on gate breakage
        sys.stderr.write(
            f"[mlx-safe-python] WARN: panic-gate raised "
            f"({type(exc).__name__}: {exc}), proceeding fail-open\n"
        )
        os.environ["PYTHONNOUSERSITE"] = "1"
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
        os.execv(python_bin, [python_bin, *argv])
        return 11

    if verdict.exit_code == 0:
        os.execv(python_bin, [python_bin, *argv])
        return 0  # never reached

    if verdict.exit_code == 2:
        if os.environ.get("MLX_SAFE_PYTHON_FORCE") == "1":
            sys.stderr.write(
                f"[mlx-safe-python] WARN: cooldown active, proceeding due to "
                f"MLX_SAFE_PYTHON_FORCE=1\n"
                f"[mlx-safe-python] gate: {verdict.reason}\n"
            )
            os.environ["PYTHONNOUSERSITE"] = "1"
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
            os.execv(python_bin, [python_bin, *argv])
            return 0
        sys.stderr.write(
            f"[mlx-safe-python] BLOCKED: metal-guard panic cooldown is active.\n"
            f"[mlx-safe-python] gate: {verdict.reason}\n"
            f"[mlx-safe-python] To override for a single invocation:\n"
            f"    MLX_SAFE_PYTHON_FORCE=1 mlx-safe-python {' '.join(argv)}\n"
            f"[mlx-safe-python] To verify package versions WITHOUT importing, prefer:\n"
            f"    pip show <package>\n"
            f"    python -c 'import importlib.metadata; "
            f"print(importlib.metadata.version(\"<pkg>\"))'\n"
        )
        return 10

    # exit_code >= 3 — gate broken
    sys.stderr.write(
        f"[mlx-safe-python] WARN: panic-gate rc={verdict.exit_code}, "
        f"proceeding fail-open\n"
        f"[mlx-safe-python] gate: {verdict.reason}\n"
    )
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    os.execv(python_bin, [python_bin, *argv])
    return 11


if __name__ == "__main__":
    sys.exit(main())
