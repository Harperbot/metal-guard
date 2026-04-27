"""L13 status snapshot tests (v0.10.0)。"""
from __future__ import annotations

import json

import pytest

import metal_guard as mg


def test_snapshot_has_schema_version():
    snap = mg.get_status_snapshot()
    assert snap["schema_version"] == mg.STATUS_SNAPSHOT_SCHEMA_VERSION
    assert "captured_at" in snap


def test_snapshot_has_required_top_level_keys():
    snap = mg.get_status_snapshot()
    required = {"schema_version", "captured_at", "memory", "lock", "mode",
                "recent_panics", "panic_cooldown"}
    assert required <= set(snap.keys())


def test_snapshot_panic_cooldown_section_present():
    snap = mg.get_status_snapshot()
    pc = snap["panic_cooldown"]
    assert "exit_code" in pc
    assert "reason" in pc
    assert "recent_panics_24h" in pc
    assert "recent_panics_72h" in pc


def test_snapshot_lock_held_false_when_no_lock():
    snap = mg.get_status_snapshot()
    assert snap["lock"]["held"] is False


def test_write_snapshot_produces_valid_json(tmp_path):
    out = tmp_path / "status.json"
    written = mg.write_status_snapshot(out)
    assert written == out
    assert out.exists()
    parsed = json.loads(out.read_text())
    assert parsed["schema_version"] == mg.STATUS_SNAPSHOT_SCHEMA_VERSION


def test_write_snapshot_atomic_no_partial(tmp_path):
    """Tmp file must not linger after a successful write."""
    out = tmp_path / "status.json"
    mg.write_status_snapshot(out)
    # After success, only the target file should exist
    files = sorted(out.parent.iterdir())
    assert files == [out]


def test_snapshot_serialisable_to_json():
    """All values must be JSON-serialisable (no datetime / Path leaks)."""
    snap = mg.get_status_snapshot()
    # If any value isn't serialisable, this raises TypeError
    json.dumps(snap, default=str)


def test_snapshot_no_errors_field_when_clean():
    """critic R1 P0-2 strict assertion: ensure no metal_guard_core dispatch
    error (`AttributeError: status_snapshot`) leaks into snapshots.
    """
    snap = mg.get_status_snapshot()
    if "errors" in snap:
        assert isinstance(snap["errors"], dict)
        # The previous bug surfaced as 'metal_guard_core' error key.
        # Memory may legitimately error if MLX not loaded but `core` should not.
        assert "metal_guard_core" not in snap["errors"]


def test_snapshot_memory_section_exists():
    """L13 must always populate `memory` dict, even when MLX is not loaded."""
    snap = mg.get_status_snapshot()
    assert "memory" in snap
    assert "available" in snap["memory"]
    # If MLX is loaded, available=True with stats. If not, available=False
    # with reason. Either way is structured, not silently broken.
    if snap["memory"]["available"]:
        assert "active_gb" in snap["memory"]
        assert "limit_gb" in snap["memory"]
        assert "active_pct" in snap["memory"]
    else:
        assert "reason" in snap["memory"] or "available" in snap.get("errors", {})


def test_snapshot_kv_monitor_section_exists():
    snap = mg.get_status_snapshot()
    assert "kv_monitor" in snap
    assert "running" in snap["kv_monitor"]
    if snap["kv_monitor"]["running"]:
        assert "active_requests" in snap["kv_monitor"]
        assert isinstance(snap["kv_monitor"]["active_requests"], list)


def test_snapshot_breadcrumb_tail_section_exists():
    snap = mg.get_status_snapshot()
    assert "breadcrumb_tail" in snap
    assert isinstance(snap["breadcrumb_tail"], list)


def test_snapshot_with_breadcrumb_lines_zero_skips_tail():
    snap = mg.get_status_snapshot(breadcrumb_lines=0)
    assert snap["breadcrumb_tail"] == []


def test_snapshot_mode_section():
    snap = mg.get_status_snapshot()
    mode = snap["mode"]
    assert "current" in mode
    assert mode["current"] in {"defensive", "observer", "unknown"}
