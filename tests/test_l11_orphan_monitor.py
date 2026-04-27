"""L11 subprocess orphan monitor tests (v0.10.0)。

Verify FIFO-pairing + threshold semantics + kill-switch.
"""
from __future__ import annotations

import datetime

import pytest

import metal_guard as mg


def _make_breadcrumb(tmp_path, lines: list[str]):
    p = tmp_path / "metal_breadcrumb.log"
    p.write_text("\n".join(lines) + "\n")
    return p


def test_no_breadcrumb_returns_empty(tmp_path):
    missing = tmp_path / "nope.log"
    orphans = mg.scan_orphan_subproc_pre(breadcrumb_path=str(missing))
    assert orphans == []


def test_paired_pre_post_no_orphan(tmp_path):
    now = datetime.datetime.now()
    pre_ts = now - datetime.timedelta(seconds=200)
    post_ts = now - datetime.timedelta(seconds=180)
    log = _make_breadcrumb(tmp_path, [
        f"[{pre_ts.strftime('%Y-%m-%d %H:%M:%S')}] SUBPROC_PRE: model-a",
        f"[{post_ts.strftime('%Y-%m-%d %H:%M:%S')}] SUBPROC_POST: model-a",
    ])
    orphans = mg.scan_orphan_subproc_pre(
        threshold_sec=90.0, breadcrumb_path=str(log), now=now,
    )
    assert orphans == []


def test_unpaired_pre_old_enough_is_orphan(tmp_path):
    now = datetime.datetime.now()
    pre_ts = now - datetime.timedelta(seconds=200)
    log = _make_breadcrumb(tmp_path, [
        f"[{pre_ts.strftime('%Y-%m-%d %H:%M:%S')}] SUBPROC_PRE: model-stuck",
    ])
    orphans = mg.scan_orphan_subproc_pre(
        threshold_sec=90.0, breadcrumb_path=str(log), now=now,
    )
    assert len(orphans) == 1
    assert orphans[0].model_id == "model-stuck"
    assert orphans[0].age_sec >= 90.0


def test_unpaired_pre_under_threshold_not_orphan(tmp_path):
    now = datetime.datetime.now()
    pre_ts = now - datetime.timedelta(seconds=30)
    log = _make_breadcrumb(tmp_path, [
        f"[{pre_ts.strftime('%Y-%m-%d %H:%M:%S')}] SUBPROC_PRE: model-fast",
    ])
    orphans = mg.scan_orphan_subproc_pre(
        threshold_sec=90.0, breadcrumb_path=str(log), now=now,
    )
    assert orphans == []


def test_fifo_pairing_per_model(tmp_path):
    """Multiple PRE/POST per model must FIFO-pair, not LIFO."""
    now = datetime.datetime.now()
    t1 = now - datetime.timedelta(seconds=300)
    t2 = now - datetime.timedelta(seconds=250)
    t3 = now - datetime.timedelta(seconds=200)  # POST for first PRE
    t4 = now - datetime.timedelta(seconds=150)  # POST for second PRE
    log = _make_breadcrumb(tmp_path, [
        f"[{t1.strftime('%Y-%m-%d %H:%M:%S')}] SUBPROC_PRE: model-x",
        f"[{t2.strftime('%Y-%m-%d %H:%M:%S')}] SUBPROC_PRE: model-x",
        f"[{t3.strftime('%Y-%m-%d %H:%M:%S')}] SUBPROC_POST: model-x",
        f"[{t4.strftime('%Y-%m-%d %H:%M:%S')}] SUBPROC_POST: model-x",
    ])
    orphans = mg.scan_orphan_subproc_pre(
        threshold_sec=90.0, breadcrumb_path=str(log), now=now,
    )
    assert orphans == []  # both pairs matched


def test_pid_traced_from_worker_ready(tmp_path):
    now = datetime.datetime.now()
    pre_ts = now - datetime.timedelta(seconds=200)
    log = _make_breadcrumb(tmp_path, [
        f"[{pre_ts.strftime('%Y-%m-%d %H:%M:%S')}] SUBPROCESS_WORKER: model-x ready, pid=12345",
        f"[{pre_ts.strftime('%Y-%m-%d %H:%M:%S')}] SUBPROC_PRE: model-x",
    ])
    orphans = mg.scan_orphan_subproc_pre(
        threshold_sec=90.0, breadcrumb_path=str(log), now=now,
    )
    assert len(orphans) == 1
    assert orphans[0].pid == 12345


def test_kill_switch_disables(tmp_path, monkeypatch):
    now = datetime.datetime.now()
    pre_ts = now - datetime.timedelta(seconds=300)
    log = _make_breadcrumb(tmp_path, [
        f"[{pre_ts.strftime('%Y-%m-%d %H:%M:%S')}] SUBPROC_PRE: model-stuck",
    ])
    monkeypatch.setenv("METALGUARD_SUBPROC_ORPHAN_WATCH_DISABLED", "1")
    orphans = mg.scan_orphan_subproc_pre(
        threshold_sec=90.0, breadcrumb_path=str(log), now=now,
    )
    assert orphans == []


def test_orphans_sorted_oldest_first(tmp_path):
    now = datetime.datetime.now()
    log = _make_breadcrumb(tmp_path, [
        f"[{(now - datetime.timedelta(seconds=200)).strftime('%Y-%m-%d %H:%M:%S')}] SUBPROC_PRE: model-recent",
        f"[{(now - datetime.timedelta(seconds=400)).strftime('%Y-%m-%d %H:%M:%S')}] SUBPROC_PRE: model-old",
    ])
    orphans = mg.scan_orphan_subproc_pre(
        threshold_sec=90.0, breadcrumb_path=str(log), now=now,
    )
    assert len(orphans) == 2
    assert orphans[0].model_id == "model-old"  # oldest first
    assert orphans[0].age_sec > orphans[1].age_sec


def test_malformed_lines_skipped(tmp_path):
    now = datetime.datetime.now()
    pre_ts = now - datetime.timedelta(seconds=200)
    log = _make_breadcrumb(tmp_path, [
        "garbage line without timestamp",
        "[bad-date-format] SUBPROC_PRE: model-bad",
        f"[{pre_ts.strftime('%Y-%m-%d %H:%M:%S')}] SUBPROC_PRE: model-good",
        "",  # empty line
    ])
    orphans = mg.scan_orphan_subproc_pre(
        threshold_sec=90.0, breadcrumb_path=str(log), now=now,
    )
    assert len(orphans) == 1
    assert orphans[0].model_id == "model-good"
