"""L10 panic cooldown gate tests (v0.10.0)。

Cover staircase policy + sentinel + ack mechanics. AND-pattern panic
detection is mocked at file-content level — no real `/Library/Logs/...`
dependency.
"""
from __future__ import annotations

import datetime
import json
import os

import pytest

import metal_guard as mg


# ── helpers ──────────────────────────────────────────────────────


def _fake_panic_file(tmp_path, name: str, mtime_dt: datetime.datetime,
                     core: bool = True, context: bool = True):
    path = tmp_path / name
    body_lines = []
    if core:
        body_lines.append("panic: prepare_count_underflow detected\n")
    if context:
        body_lines.append("at IOGPUMemory.cpp:492 completeMemory()\n")
    body_lines.append("...other content...\n")
    path.write_text("".join(body_lines))
    ts = mtime_dt.timestamp()
    os.utime(path, (ts, ts))
    return path


@pytest.fixture
def isolated_state(tmp_path, monkeypatch):
    """Redirect sentinel + ack + panic glob to tmp_path."""
    sentinel = tmp_path / "panic-sentinel.json"
    ack = tmp_path / "metal-guard-ack"
    monkeypatch.setattr(mg, "_PANIC_SENTINEL_PATH", str(sentinel))
    monkeypatch.setattr(mg, "_PANIC_LOCKOUT_ACK_PATH", str(ack))
    panic_dir = tmp_path / "panics"
    panic_dir.mkdir()
    monkeypatch.setattr(
        mg, "PANIC_REPORTS_GLOBS", (str(panic_dir / "*.panic"),),
    )
    return {
        "tmp": tmp_path,
        "sentinel": sentinel,
        "ack": ack,
        "panic_dir": panic_dir,
    }


# ── AND-pattern detection ─────────────────────────────────────────


class TestPanicSignatureMatch:
    def test_proceed_when_no_panic_files(self, isolated_state):
        v = mg.evaluate_panic_cooldown()
        assert v.exit_code == 0
        assert "no recent IOGPU" in v.reason

    def test_match_requires_both_core_and_context(self, isolated_state):
        # core only — should NOT match
        _fake_panic_file(
            isolated_state["panic_dir"], "panic-1.panic",
            datetime.datetime.now() - datetime.timedelta(minutes=10),
            core=True, context=False,
        )
        records = mg.scan_recent_panics(72.0)
        assert len(records) == 0

    def test_context_only_does_not_match(self, isolated_state):
        _fake_panic_file(
            isolated_state["panic_dir"], "panic-2.panic",
            datetime.datetime.now() - datetime.timedelta(minutes=10),
            core=False, context=True,
        )
        records = mg.scan_recent_panics(72.0)
        assert len(records) == 0

    def test_both_signatures_match(self, isolated_state):
        _fake_panic_file(
            isolated_state["panic_dir"], "panic-3.panic",
            datetime.datetime.now() - datetime.timedelta(minutes=10),
            core=True, context=True,
        )
        records = mg.scan_recent_panics(72.0)
        assert len(records) == 1


# ── Staircase policy ───────────────────────────────────────────────


class TestStaircaseCooldown:
    def test_one_panic_24h_triggers_2h_cooldown(self, isolated_state):
        _fake_panic_file(
            isolated_state["panic_dir"], "p1.panic",
            datetime.datetime.now() - datetime.timedelta(minutes=30),
        )
        v = mg.evaluate_panic_cooldown()
        assert v.exit_code == 2
        assert v.recent_panics_24h == 1
        assert "cooldown" in v.reason.lower()

    def test_one_panic_outside_2h_window_proceeds(self, isolated_state):
        _fake_panic_file(
            isolated_state["panic_dir"], "p1.panic",
            datetime.datetime.now() - datetime.timedelta(hours=3),
        )
        v = mg.evaluate_panic_cooldown()
        # 1 panic in 24h triggers stage 1 (2h) — now > 3h after panic → expired
        assert v.exit_code == 0
        assert "expired" in v.reason

    def test_two_panics_24h_triggers_lockout(self, isolated_state):
        now = datetime.datetime.now()
        _fake_panic_file(
            isolated_state["panic_dir"], "p1.panic",
            now - datetime.timedelta(hours=4),
        )
        _fake_panic_file(
            isolated_state["panic_dir"], "p2.panic",
            now - datetime.timedelta(hours=1),
        )
        v = mg.evaluate_panic_cooldown()
        assert v.exit_code == 2
        assert "lockout" in v.reason.lower()
        assert v.recent_panics_24h == 2

    def test_three_panics_72h_triggers_lockout(self, isolated_state):
        now = datetime.datetime.now()
        # 3 panics across 72h, none within 24h
        for i, hours_ago in enumerate([28, 40, 60]):
            _fake_panic_file(
                isolated_state["panic_dir"], f"p{i}.panic",
                now - datetime.timedelta(hours=hours_ago),
            )
        v = mg.evaluate_panic_cooldown()
        assert v.exit_code == 2
        assert "lockout" in v.reason.lower()


# ── Sentinel ──────────────────────────────────────────────────────


class TestSentinel:
    def test_sentinel_extends_cooldown(self, isolated_state):
        future = datetime.datetime.now() + datetime.timedelta(hours=1)
        isolated_state["sentinel"].write_text(json.dumps({
            "cooldown_until": future.isoformat(timespec="seconds"),
            "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        }))
        v = mg.evaluate_panic_cooldown()
        assert v.exit_code == 2
        assert "sentinel" in v.reason

    def test_expired_sentinel_ignored(self, isolated_state):
        past = datetime.datetime.now() - datetime.timedelta(minutes=5)
        isolated_state["sentinel"].write_text(json.dumps({
            "cooldown_until": past.isoformat(timespec="seconds"),
        }))
        v = mg.evaluate_panic_cooldown()
        # No panics + expired sentinel → proceed
        assert v.exit_code == 0

    def test_corrupt_sentinel_ignored(self, isolated_state):
        isolated_state["sentinel"].write_text("not valid json {")
        v = mg.evaluate_panic_cooldown()
        assert v.exit_code == 0

    def test_mark_panic_sentinel_atomic(self, isolated_state):
        path = mg.mark_panic_sentinel_cooldown(2.0)
        assert path.exists()
        data = json.loads(path.read_text())
        assert "cooldown_until" in data
        assert "created_at" in data


# ── Ack mechanics ─────────────────────────────────────────────────


class TestAck:
    def test_ack_clears_lockout(self, isolated_state):
        now = datetime.datetime.now()
        _fake_panic_file(
            isolated_state["panic_dir"], "p1.panic",
            now - datetime.timedelta(hours=2),
        )
        _fake_panic_file(
            isolated_state["panic_dir"], "p2.panic",
            now - datetime.timedelta(hours=1),
        )
        # Lockout active before ack
        assert mg.evaluate_panic_cooldown().exit_code == 2
        # Ack the lockout
        mg.ack_panic_lockout()
        v = mg.evaluate_panic_cooldown()
        assert v.exit_code == 0
        assert "ack" in v.reason

    def test_ack_invalidated_by_newer_panic(self, isolated_state):
        """Critical safety: ack at T, panic at T+1h → ack must be re-required."""
        now = datetime.datetime.now()
        # First, create old panics + ack the lockout
        _fake_panic_file(
            isolated_state["panic_dir"], "p1.panic",
            now - datetime.timedelta(hours=4),
        )
        _fake_panic_file(
            isolated_state["panic_dir"], "p2.panic",
            now - datetime.timedelta(hours=3),
        )
        mg.ack_panic_lockout()
        # Now a NEW panic occurs — mtime explicitly 5min after ack so the
        # comparison is unambiguous (real-world: ack at T, new panic at T+N)
        _fake_panic_file(
            isolated_state["panic_dir"], "p3.panic",
            now + datetime.timedelta(minutes=5),
        )
        v = mg.evaluate_panic_cooldown(
            now=now + datetime.timedelta(minutes=10),  # evaluate after the new panic
        )
        # Should NOT be cleared by ack — new panic invalidates it
        # (3 panics in 72h triggers lockout regardless)
        assert v.exit_code == 2
        assert "lockout" in v.reason.lower()

    def test_clear_ack_removes_file(self, isolated_state):
        mg.ack_panic_lockout()
        assert isolated_state["ack"].exists()
        assert mg.clear_panic_ack() is True
        assert not isolated_state["ack"].exists()

    def test_clear_ack_idempotent_when_absent(self, isolated_state):
        assert mg.clear_panic_ack() is False


# ── Kill switch ────────────────────────────────────────────────────


class TestKillSwitch:
    def test_kill_switch_short_circuits(self, isolated_state, monkeypatch):
        # 3 panics — would normally lockout
        now = datetime.datetime.now()
        for i in range(3):
            _fake_panic_file(
                isolated_state["panic_dir"], f"p{i}.panic",
                now - datetime.timedelta(hours=i + 1),
            )
        monkeypatch.setenv("METALGUARD_PANIC_GATE_DISABLED", "1")
        v = mg.evaluate_panic_cooldown()
        assert v.exit_code == 0
        assert "kill" in v.reason.lower()
