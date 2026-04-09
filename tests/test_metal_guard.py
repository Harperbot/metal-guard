"""Tests for MetalGuard — all Metal GPU operations are mocked."""

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from metal_guard import MetalGuard, MemoryStats


@pytest.fixture
def guard(tmp_path):
    """MetalGuard with breadcrumbs written to tmp dir."""
    return MetalGuard(
        cooldown_secs=0.01,  # Fast tests
        thread_timeout_secs=2.0,
        breadcrumb_path=str(tmp_path / "breadcrumb.log"),
    )


@pytest.fixture(autouse=True)
def _mock_mlx():
    """Prevent real Metal GPU initialization in tests."""
    mock_mx = MagicMock()
    mock_mx.device_info.return_value = {
        "max_recommended_working_set_size": 48_000_000_000,
    }
    mock_mx.get_active_memory.return_value = 0
    mock_mx.get_peak_memory.return_value = 0
    mock_mx.zeros.return_value = mock_mx
    with patch.dict("sys.modules", {"mlx.core": mock_mx}):
        yield mock_mx


# ── MemoryStats ──────────────────────────────────────────────────────────


class TestMemoryStats:
    def test_zero_stats(self):
        s = MemoryStats()
        assert s.active_gb == 0
        assert s.peak_pct == 0

    def test_percentages(self):
        s = MemoryStats(
            active_bytes=24_000_000_000,
            peak_bytes=36_000_000_000,
            limit_bytes=48_000_000_000,
        )
        assert s.active_pct == 50.0
        assert s.peak_pct == 75.0
        assert s.active_gb == 24.0

    def test_str(self):
        s = MemoryStats(
            active_bytes=1_000_000_000,
            peak_bytes=2_000_000_000,
            limit_bytes=48_000_000_000,
        )
        text = str(s)
        assert "1.0GB" in text
        assert "2.0GB" in text


# ── Thread tracking ──────────────────────────────────────────────────────


class TestThreadTracking:
    def test_register_and_wait_completed_thread(self, guard):
        t = threading.Thread(target=lambda: None)
        t.start()
        t.join()
        guard.register_thread(t)
        assert guard.wait_for_threads() == 0

    def test_wait_for_running_thread(self, guard):
        event = threading.Event()
        t = threading.Thread(target=event.wait, daemon=True)
        t.start()
        guard.register_thread(t)

        # Thread is alive
        assert t.is_alive()

        # Release it
        event.set()
        still_alive = guard.wait_for_threads(timeout=2.0)
        assert still_alive == 0

    def test_timeout_returns_alive_count(self, guard):
        event = threading.Event()
        t = threading.Thread(target=event.wait, daemon=True)
        t.start()
        guard.register_thread(t)

        still_alive = guard.wait_for_threads(timeout=0.01)
        assert still_alive == 1

        event.set()
        t.join(timeout=1)

    def test_dead_threads_pruned_on_register(self, guard):
        t1 = threading.Thread(target=lambda: None)
        t1.start()
        t1.join()
        guard.register_thread(t1)

        t2 = threading.Thread(target=lambda: None)
        t2.start()
        t2.join()
        guard.register_thread(t2)

        # Dead threads should be pruned
        assert guard.wait_for_threads() == 0


# ── GPU cleanup ──────────────────────────────────────────────────────────


class TestCleanup:
    def test_flush_gpu_calls_mlx(self, guard, _mock_mlx):
        guard.flush_gpu()
        _mock_mlx.eval.assert_called_once()
        _mock_mlx.clear_cache.assert_called_once()

    def test_safe_cleanup_full_sequence(self, guard, _mock_mlx):
        guard.safe_cleanup()
        _mock_mlx.eval.assert_called_once()
        _mock_mlx.clear_cache.assert_called_once()

    def test_guarded_cleanup_context_manager(self, guard, _mock_mlx):
        with guard.guarded_cleanup():
            pass  # Simulate work
        _mock_mlx.clear_cache.assert_called_once()

    def test_flush_noop_without_mlx(self, guard):
        with patch.dict("sys.modules", {"mlx.core": None}):
            # Should not raise
            guard.flush_gpu()


# ── Memory pressure ──────────────────────────────────────────────────────


class TestMemoryPressure:
    def test_no_pressure(self, guard, _mock_mlx):
        _mock_mlx.get_peak_memory.return_value = 10_000_000_000  # 10GB
        assert not guard.is_pressure_high(threshold_pct=67.0)

    def test_high_pressure(self, guard, _mock_mlx):
        _mock_mlx.get_peak_memory.return_value = 40_000_000_000  # 40GB / 48GB = 83%
        assert guard.is_pressure_high(threshold_pct=67.0)

    def test_ensure_headroom_cleans_when_needed(self, guard, _mock_mlx):
        _mock_mlx.get_peak_memory.return_value = 40_000_000_000
        guard.ensure_headroom(model_name="test-model")
        _mock_mlx.clear_cache.assert_called()
        _mock_mlx.reset_peak_memory.assert_called()

    def test_ensure_headroom_noop_when_fine(self, guard, _mock_mlx):
        _mock_mlx.get_peak_memory.return_value = 5_000_000_000
        guard.ensure_headroom(model_name="test-model")
        _mock_mlx.clear_cache.assert_not_called()

    def test_memory_stats(self, guard, _mock_mlx):
        _mock_mlx.get_active_memory.return_value = 1_000_000_000
        _mock_mlx.get_peak_memory.return_value = 2_000_000_000
        stats = guard.memory_stats()
        assert stats.active_bytes == 1_000_000_000
        assert stats.limit_bytes == 48_000_000_000


# ── Breadcrumb ───────────────────────────────────────────────────────────


class TestBreadcrumb:
    def test_writes_to_file(self, guard, tmp_path):
        guard.breadcrumb("TEST: hello")
        content = (tmp_path / "breadcrumb.log").read_text()
        assert "TEST: hello" in content

    def test_multiple_breadcrumbs(self, guard, tmp_path):
        guard.breadcrumb("STEP 1")
        guard.breadcrumb("STEP 2")
        lines = (tmp_path / "breadcrumb.log").read_text().strip().split("\n")
        assert len(lines) == 2

    def test_disabled_breadcrumbs(self, tmp_path):
        g = MetalGuard(breadcrumb_path=None)
        g.breadcrumb("should not crash")  # No-op
