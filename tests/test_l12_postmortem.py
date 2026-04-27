"""L12 postmortem auto-collect tests (v0.10.0)。"""
from __future__ import annotations

import datetime
import os

import pytest

import metal_guard as mg


def _fake_panic_file(panic_dir, name, mtime_dt):
    path = panic_dir / name
    path.write_text(
        "panic: prepare_count_underflow\n"
        "at IOGPUMemory.cpp:492 completeMemory()\n"
    )
    ts = mtime_dt.timestamp()
    os.utime(path, (ts, ts))
    return path


@pytest.fixture
def isolated_state(tmp_path, monkeypatch):
    panic_dir = tmp_path / "panics"
    panic_dir.mkdir()
    monkeypatch.setattr(mg, "PANIC_REPORTS_GLOBS", (str(panic_dir / "*.panic"),))
    monkeypatch.setattr(
        mg, "_PANIC_SENTINEL_PATH", str(tmp_path / "sentinel.json"),
    )
    monkeypatch.setattr(
        mg, "_PANIC_LOCKOUT_ACK_PATH", str(tmp_path / "ack"),
    )
    return {"panic_dir": panic_dir, "tmp": tmp_path}


def test_no_panics_writes_index_only(isolated_state, tmp_path):
    out = tmp_path / "bundle"
    result = mg.run_postmortem(out)
    assert result["status"] == "collected"
    assert result["panic_count"] == 0
    assert result["sentinel"] is None
    assert (out / "index.md").exists()


def test_panic_within_window_is_copied(isolated_state, tmp_path):
    now = datetime.datetime.now()
    _fake_panic_file(
        isolated_state["panic_dir"], "panic-1.panic",
        now - datetime.timedelta(hours=2),
    )
    out = tmp_path / "bundle"
    result = mg.run_postmortem(out)
    assert result["panic_count"] == 1
    assert (out / "panic-1.panic").exists()
    # Sentinel written when panics found
    assert result["sentinel"] is not None


def test_panic_outside_window_not_copied(isolated_state, tmp_path):
    now = datetime.datetime.now()
    _fake_panic_file(
        isolated_state["panic_dir"], "old.panic",
        now - datetime.timedelta(hours=48),
    )
    out = tmp_path / "bundle"
    result = mg.run_postmortem(out)
    # within_hours default 24
    assert result["panic_count"] == 0


def test_kill_switch_disables(isolated_state, tmp_path, monkeypatch):
    monkeypatch.setenv("METALGUARD_POSTMORTEM_DISABLED", "1")
    out = tmp_path / "bundle"
    result = mg.run_postmortem(out)
    assert result["status"] == "disabled"


def test_index_md_lists_collected_files(isolated_state, tmp_path):
    now = datetime.datetime.now()
    _fake_panic_file(
        isolated_state["panic_dir"], "p1.panic",
        now - datetime.timedelta(hours=1),
    )
    out = tmp_path / "bundle"
    mg.run_postmortem(out)
    index_text = (out / "index.md").read_text()
    assert "Postmortem" in index_text
    assert "p1.panic" in index_text
    # Next-steps section includes the registry link
    assert "issues/new?template=known-panic-report.yml" in index_text
