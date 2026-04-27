"""Tests for metal-guard standalone CLI (Phase 4 spec section 6 第 2 項)。"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import metal_guard_cli as cli  # noqa: E402


# ── _classify_health ──────────────────────────────────────────────


class TestClassifyHealth:
    def test_no_panics_no_lock_returns_ok(self):
        h, rc = cli._classify_health([], None)
        assert h == "ok"
        assert rc == 0

    def test_one_panic_returns_warn(self):
        h, rc = cli._classify_health([{"signature": "x"}], None)
        assert h == "warn"
        assert rc == 1

    def test_five_panics_returns_critical(self):
        panics = [{"signature": "x"} for _ in range(5)]
        h, rc = cli._classify_health(panics, None)
        assert h == "critical"
        assert rc == 2

    def test_stale_lock_returns_warn(self):
        h, rc = cli._classify_health([], {"label": "x", "stale": True})
        assert h == "warn"
        assert rc == 1

    def test_active_lock_returns_ok(self):
        """Live lock holder（非 stale）+ 無 panics → 仍 ok。"""
        h, rc = cli._classify_health([], {"label": "x", "stale": False})
        assert h == "ok"
        assert rc == 0


# ── _build_status_payload (mocked) ────────────────────────────────


class TestBuildStatusPayload:
    def test_envelope_keys(self, monkeypatch):
        monkeypatch.setattr(cli, "_collect_panics", lambda *a, **kw: [])
        monkeypatch.setattr(cli, "_collect_lock", lambda: None)
        monkeypatch.setattr(cli, "_collect_mode",
                            lambda: {"mode": "defensive", "description": "x", "env": "METALGUARD_MODE"})
        monkeypatch.setattr(cli, "_collect_memory", lambda: None)
        monkeypatch.setattr(cli, "_collect_breadcrumb_tail",
                            lambda *a, **kw: {"path": None, "lines": []})
        p = cli._build_status_payload()
        assert set(p.keys()) >= {
            "version", "health", "exit_code", "panics_count", "panics",
            "mlx_lock", "mode", "memory", "breadcrumb",
        }
        assert p["health"] == "ok"
        assert p["exit_code"] == 0


# ── main / cli wiring ─────────────────────────────────────────────


class TestMainCLI:
    @pytest.fixture(autouse=True)
    def _stub_collectors(self, monkeypatch):
        monkeypatch.setattr(cli, "_collect_panics", lambda *a, **kw: [])
        monkeypatch.setattr(cli, "_collect_lock", lambda: None)
        monkeypatch.setattr(cli, "_collect_mode", lambda: {
            "mode": "defensive", "description": "test", "env": "METALGUARD_MODE",
        })
        monkeypatch.setattr(cli, "_collect_memory", lambda: None)
        monkeypatch.setattr(cli, "_collect_breadcrumb_tail",
                            lambda *a, **kw: {"path": "/tmp/x", "lines": ["L1", "L2"]})

    def test_default_status_exit_zero_when_healthy(self, capsys):
        rc = cli.main([])
        assert rc == 0
        out = capsys.readouterr().out
        assert "metal-guard" in out
        assert "OK" in out

    def test_status_explicit(self, capsys):
        rc = cli.main(["status"])
        assert rc == 0
        assert "metal-guard" in capsys.readouterr().out

    def test_panics_subcommand(self, capsys):
        rc = cli.main(["panics"])
        assert rc == 0
        assert "No panics" in capsys.readouterr().out

    def test_breadcrumb_subcommand(self, capsys):
        rc = cli.main(["breadcrumb"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "/tmp/x" in out
        assert "L1" in out and "L2" in out

    def test_mode_subcommand(self, capsys):
        rc = cli.main(["mode"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "defensive" in out
        assert "METALGUARD_MODE" in out

    def test_json_status_returns_valid_json(self, capsys):
        rc = cli.main(["--json", "status"])
        assert rc == 0
        body = json.loads(capsys.readouterr().out)
        assert body["health"] == "ok"
        assert body["exit_code"] == 0
        assert "version" in body

    def test_json_flag_works_after_subcommand(self, capsys):
        """`metal-guard status --json` 順序也可（critic R1 P1#1 修）。"""
        rc = cli.main(["status", "--json"])
        assert rc == 0
        body = json.loads(capsys.readouterr().out)
        assert body["health"] == "ok"

    def test_json_flag_on_panics_subcommand(self, capsys):
        rc = cli.main(["panics", "--json"])
        assert rc == 0
        body = json.loads(capsys.readouterr().out)
        assert "panics" in body
        assert isinstance(body["panics"], list)

    def test_json_flag_on_breadcrumb_subcommand(self, capsys):
        rc = cli.main(["breadcrumb", "--json"])
        assert rc == 0
        body = json.loads(capsys.readouterr().out)
        assert "breadcrumb" in body

    def test_status_warn_with_panics(self, monkeypatch, capsys):
        monkeypatch.setattr(cli, "_collect_panics", lambda *a, **kw: [
            {"signature": "x", "ts": 1700000000, "pid": 1, "explanation": "y",
             "source_file": "/x.panic"},
        ])
        rc = cli.main([])
        assert rc == 1
        assert "WARN" in capsys.readouterr().out

    def test_status_critical_with_5_panics(self, monkeypatch, capsys):
        monkeypatch.setattr(cli, "_collect_panics", lambda *a, **kw: [
            {"signature": "x", "ts": 1700000000, "pid": 1, "explanation": "y",
             "source_file": "/x.panic"},
        ] * 5)
        rc = cli.main([])
        assert rc == 2
        assert "CRITICAL" in capsys.readouterr().out


# ── breadcrumb tail file lookup ───────────────────────────────────


class TestBreadcrumbTail:
    def test_returns_empty_when_no_log_found(self, tmp_path, monkeypatch):
        # 確保所有 fallback 路徑都不存在
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(cli._mg, "metal_guard", type("X", (), {"_breadcrumb_path": None})())
        result = cli._collect_breadcrumb_tail(5)
        assert result["path"] is None
        assert result["lines"] == []

    def test_warn_surfaces_to_stderr_on_attribute_error(self, monkeypatch, capsys):
        """API 缺失時走 stderr 不 silent (critic R1 P2 修)。"""
        def _bad():
            raise AttributeError("describe_mode not in this version")
        monkeypatch.setattr(cli._mg, "describe_mode", _bad)
        result = cli._collect_mode()
        err = capsys.readouterr().err
        assert "describe_mode unavailable" in err
        assert result["mode"] == "unknown"

    def test_warn_does_not_fire_for_missing_mlx(self, monkeypatch, capsys):
        """ImportError on memory_stats 是常態（CLI 未載 mlx），不該 noise。"""
        def _missing():
            raise ImportError("mlx not installed")
        # patch metal_guard 單例的 memory_stats
        class _Stub:
            def memory_stats(self):
                _missing()
        monkeypatch.setattr(cli._mg, "metal_guard", _Stub())
        result = cli._collect_memory()
        err = capsys.readouterr().err
        assert err == ""  # 不該印 warning
        assert result is None

    def test_reads_existing_log_tail(self, tmp_path, monkeypatch):
        log = tmp_path / "metal_breadcrumb.log"
        log.write_text("\n".join(f"line-{i}" for i in range(20)) + "\n", encoding="utf-8")
        # Force override breadcrumb_path candidate
        monkeypatch.setattr(cli._mg, "metal_guard",
                            type("X", (), {"_breadcrumb_path": str(log)})())
        result = cli._collect_breadcrumb_tail(5)
        assert result["path"] == str(log)
        assert len(result["lines"]) == 5
        assert result["lines"][-1] == "line-19"
