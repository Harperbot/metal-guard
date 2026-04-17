"""L9 CadenceGuard tests — back-to-back load protection (2026-04-16)."""
from __future__ import annotations

import json
import time

import pytest

from metal_guard import (
    CadenceGuard,
    CadenceViolation,
    require_cadence_clear,
)


@pytest.fixture
def cadence_path(tmp_path):
    return str(tmp_path / "cadence.json")


def test_fresh_load_allowed(cadence_path):
    guard = CadenceGuard(cadence_path, min_interval_sec=180)
    # No prior mark → check passes
    guard.check("mlx-community/gemma-4-26b-a4b-it-4bit")
    guard.mark_load("mlx-community/gemma-4-26b-a4b-it-4bit")
    # Next immediate check should fail
    with pytest.raises(CadenceViolation) as excinfo:
        guard.check("mlx-community/gemma-4-26b-a4b-it-4bit")
    assert excinfo.value.model_id == "mlx-community/gemma-4-26b-a4b-it-4bit"
    assert excinfo.value.delta < 2.0
    assert excinfo.value.min_interval == 180


def test_violation_after_mark_persists_across_instances(cadence_path):
    """Mark persists to disk so new process instances see the violation."""
    guard1 = CadenceGuard(cadence_path, min_interval_sec=180)
    guard1.mark_load("model-a")

    # Simulate a fresh process (new CadenceGuard reading same path)
    guard2 = CadenceGuard(cadence_path, min_interval_sec=180)
    with pytest.raises(CadenceViolation):
        guard2.check("model-a")


def test_recovery_after_interval_elapses(cadence_path):
    guard = CadenceGuard(cadence_path, min_interval_sec=0.5)
    guard.mark_load("model-b")
    with pytest.raises(CadenceViolation):
        guard.check("model-b")
    time.sleep(0.6)
    # After the interval elapses, check should pass
    guard.check("model-b")


def test_different_models_independent(cadence_path):
    guard = CadenceGuard(cadence_path, min_interval_sec=180)
    guard.mark_load("model-x")
    # Different model name is not blocked
    guard.check("model-y")


def test_require_cadence_clear_helper(cadence_path):
    guard = CadenceGuard(cadence_path, min_interval_sec=180)
    # First call passes and marks
    require_cadence_clear("modelz", guard=guard)
    # Second immediate call raises
    with pytest.raises(CadenceViolation):
        require_cadence_clear("modelz", guard=guard)


def test_stale_gc(cadence_path):
    """Entries older than 4h are pruned on next mark (keep the file small)."""
    guard = CadenceGuard(cadence_path, min_interval_sec=180)
    # Seed a very-old timestamp directly
    very_old = time.time() - 5 * 3600  # 5h ago
    with open(cadence_path, "w") as f:
        json.dump({"stale-model": very_old, "fresh-model": time.time() - 60}, f)

    guard.mark_load("new-model")

    with open(cadence_path) as f:
        data = json.load(f)

    assert "stale-model" not in data
    assert "fresh-model" in data
    assert "new-model" in data


def test_corrupt_file_treated_as_empty(cadence_path):
    # Write garbage
    with open(cadence_path, "w") as f:
        f.write("not json {{{")

    guard = CadenceGuard(cadence_path, min_interval_sec=180)
    # Should behave as if file is empty — check passes
    guard.check("anything")


def test_violation_message_contains_delta_and_min(cadence_path):
    guard = CadenceGuard(cadence_path, min_interval_sec=180)
    guard.mark_load("m")
    with pytest.raises(CadenceViolation) as excinfo:
        guard.check("m")
    message = str(excinfo.value)
    assert "'m'" in message
    assert "180" in message
    assert "ago" in message
