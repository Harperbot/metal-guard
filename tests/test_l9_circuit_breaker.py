"""L9 CircuitBreaker tests — rolling-window panic threshold (2026-04-16)."""
from __future__ import annotations

import json
import time

import pytest

from metal_guard import CircuitBreaker, MLXCooldownActive


@pytest.fixture
def paths(tmp_path):
    return {
        "jsonl": str(tmp_path / "panics.jsonl"),
        "state": str(tmp_path / "breaker.json"),
    }


def _write_panics(jsonl_path, ts_list):
    with open(jsonl_path, "w") as f:
        for ts in ts_list:
            f.write(json.dumps({
                "ts": ts,
                "signature": "prepare_count_underflow",
                "pid": 12345,
                "source_file": f"/fake/{ts}.panic",
            }))
            f.write("\n")


def test_no_panics_passes(paths):
    breaker = CircuitBreaker(
        jsonl_path=paths["jsonl"],
        state_path=paths["state"],
    )
    # No file at all — passes
    breaker.check()


def test_one_recent_panic_passes(paths):
    _write_panics(paths["jsonl"], [time.time() - 300])  # 5 min ago
    breaker = CircuitBreaker(
        jsonl_path=paths["jsonl"],
        state_path=paths["state"],
    )
    breaker.check()  # only 1, below threshold


def test_two_recent_panics_trips(paths):
    now = time.time()
    # Default window is 1h (3600s). Both panics must fall inside it.
    _write_panics(paths["jsonl"], [now - 1500, now - 600])  # 25min + 10min ago
    breaker = CircuitBreaker(
        jsonl_path=paths["jsonl"],
        state_path=paths["state"],
    )
    with pytest.raises(MLXCooldownActive) as excinfo:
        breaker.check()
    assert excinfo.value.panic_count == 2
    assert excinfo.value.remaining_sec > 0


def test_old_panics_outside_window_ignored(paths):
    now = time.time()
    # Default window 1h: 2h-ago + 3h-ago are out; 10min-ago is in
    _write_panics(
        paths["jsonl"],
        [now - 3 * 3600, now - 2 * 3600, now - 600],
    )
    breaker = CircuitBreaker(
        jsonl_path=paths["jsonl"],
        state_path=paths["state"],
    )
    # Only 1 panic in the window → passes
    breaker.check()


def test_cooldown_state_persists(paths):
    now = time.time()
    _write_panics(paths["jsonl"], [now - 1800, now - 900])  # 2 recent
    breaker = CircuitBreaker(
        jsonl_path=paths["jsonl"],
        state_path=paths["state"],
        cooldown_sec=60,
    )
    with pytest.raises(MLXCooldownActive):
        breaker.check()

    # Immediately re-check — should still be in cooldown even without new panics
    _write_panics(paths["jsonl"], [])
    with pytest.raises(MLXCooldownActive):
        breaker.check()


def test_cooldown_expires(paths):
    now = time.time()
    _write_panics(paths["jsonl"], [now - 1800, now - 900])
    breaker = CircuitBreaker(
        jsonl_path=paths["jsonl"],
        state_path=paths["state"],
        cooldown_sec=0.2,  # expires quickly for test
    )
    with pytest.raises(MLXCooldownActive):
        breaker.check()

    time.sleep(0.3)
    # Remove the panics so check() doesn't re-enter cooldown
    _write_panics(paths["jsonl"], [])
    breaker.check()  # passes


def test_clear_removes_cooldown(paths):
    now = time.time()
    _write_panics(paths["jsonl"], [now - 100, now - 50])
    breaker = CircuitBreaker(
        jsonl_path=paths["jsonl"],
        state_path=paths["state"],
    )
    with pytest.raises(MLXCooldownActive):
        breaker.check()

    breaker.clear()
    # Remove panics to avoid re-entering cooldown
    _write_panics(paths["jsonl"], [])
    breaker.check()


def test_status_reports_state(paths):
    now = time.time()
    _write_panics(paths["jsonl"], [now - 1800, now - 900])
    breaker = CircuitBreaker(
        jsonl_path=paths["jsonl"],
        state_path=paths["state"],
    )

    status_before = breaker.status()
    assert status_before["in_cooldown"] is False
    assert status_before["recent_panic_count"] == 2

    with pytest.raises(MLXCooldownActive):
        breaker.check()

    status_after = breaker.status()
    assert status_after["in_cooldown"] is True
    assert status_after["cooldown_remaining_sec"] > 0
    assert status_after["recent_panic_count"] == 2


def test_threshold_configurable(paths):
    now = time.time()
    _write_panics(paths["jsonl"], [now - 100])  # 1 panic
    breaker = CircuitBreaker(
        jsonl_path=paths["jsonl"],
        state_path=paths["state"],
        panic_threshold=1,
    )
    with pytest.raises(MLXCooldownActive):
        breaker.check()
