"""Tests for MetalGuard — all Metal GPU operations are mocked."""

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from metal_guard import MetalGuard, MetalOOMError, MemoryStats


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
        assert s.available_gb == 0

    def test_percentages(self):
        s = MemoryStats(
            active_bytes=24_000_000_000,
            peak_bytes=36_000_000_000,
            limit_bytes=48_000_000_000,
        )
        assert s.active_pct == 50.0
        assert s.peak_pct == 75.0
        assert s.active_gb == 24.0
        assert s.available_gb == 24.0

    def test_str(self):
        s = MemoryStats(
            active_bytes=1_000_000_000,
            peak_bytes=2_000_000_000,
            limit_bytes=48_000_000_000,
        )
        text = str(s)
        assert "1.0GB" in text
        assert "2.0GB" in text
        assert "avail=" in text


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
        assert t.is_alive()
        event.set()
        assert guard.wait_for_threads(timeout=2.0) == 0

    def test_timeout_returns_alive_count(self, guard):
        event = threading.Event()
        t = threading.Thread(target=event.wait, daemon=True)
        t.start()
        guard.register_thread(t)
        assert guard.wait_for_threads(timeout=0.01) == 1
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
            pass
        _mock_mlx.clear_cache.assert_called_once()

    def test_flush_noop_without_mlx(self, guard):
        with patch.dict("sys.modules", {"mlx.core": None}):
            guard.flush_gpu()  # Should not raise


# ── OOM recovery ─────────────────────────────────────────────────────────


class TestOOMRecovery:
    def test_is_metal_oom_detects_insufficient_memory(self):
        exc = RuntimeError(
            "[METAL] Command buffer execution failed: Insufficient Memory "
            "(00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)"
        )
        assert MetalGuard.is_metal_oom(exc) is True

    def test_is_metal_oom_detects_error_code(self):
        exc = RuntimeError("kIOGPUCommandBufferCallbackErrorOutOfMemory")
        assert MetalGuard.is_metal_oom(exc) is True

    def test_is_metal_oom_ignores_other_errors(self):
        assert MetalGuard.is_metal_oom(RuntimeError("some other error")) is False
        assert MetalGuard.is_metal_oom(ValueError("not runtime")) is False

    def test_oom_protected_passes_through_on_success(self, guard):
        result = guard.oom_protected(lambda: "hello")
        assert result == "hello"

    def test_oom_protected_raises_non_oom_errors(self, guard):
        def bad():
            raise RuntimeError("not OOM")
        with pytest.raises(RuntimeError, match="not OOM"):
            guard.oom_protected(bad)

    def test_oom_protected_catches_oom_and_retries(self, guard, _mock_mlx):
        call_count = 0

        def flaky():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError(
                    "[METAL] Command buffer execution failed: Insufficient Memory"
                )
            return "recovered"

        result = guard.oom_protected(flaky, max_retries=1)
        assert result == "recovered"
        assert call_count == 2
        _mock_mlx.clear_cache.assert_called()  # Cleanup happened

    def test_oom_protected_raises_metal_oom_after_retries(self, guard):
        def always_oom():
            raise RuntimeError(
                "[METAL] Command buffer execution failed: Insufficient Memory"
            )

        with pytest.raises(MetalOOMError):
            guard.oom_protected(always_oom, max_retries=1)

    def test_oom_protected_context_manager(self, guard):
        with pytest.raises(MetalOOMError):
            with guard.oom_protected_context():
                raise RuntimeError(
                    "[METAL] Command buffer execution failed: Insufficient Memory"
                )

    def test_oom_protected_context_passes_non_oom(self, guard):
        with pytest.raises(RuntimeError, match="other"):
            with guard.oom_protected_context():
                raise RuntimeError("other error")

    def test_metal_oom_error_has_stats(self, guard):
        def always_oom():
            raise RuntimeError(
                "kIOGPUCommandBufferCallbackErrorOutOfMemory"
            )
        with pytest.raises(MetalOOMError) as exc_info:
            guard.oom_protected(always_oom, max_retries=0)
        assert exc_info.value.stats is not None


# ── Pre-load memory check ────────────────────────────────────────────────


class TestCanFit:
    def test_fits_when_enough_memory(self, guard, _mock_mlx):
        _mock_mlx.get_active_memory.return_value = 10_000_000_000  # 10GB active
        # 48GB limit - 10GB active = 38GB available, need 24+2=26GB
        assert guard.can_fit(model_size_gb=24.0) is True

    def test_does_not_fit(self, guard, _mock_mlx):
        _mock_mlx.get_active_memory.return_value = 30_000_000_000  # 30GB active
        # 48GB - 30GB = 18GB available, need 24+2=26GB
        assert guard.can_fit(model_size_gb=24.0) is False

    def test_custom_overhead(self, guard, _mock_mlx):
        _mock_mlx.get_active_memory.return_value = 20_000_000_000  # 20GB
        # 48 - 20 = 28GB available, need 24+5=29GB
        assert guard.can_fit(model_size_gb=24.0, overhead_gb=5.0) is False
        # need 24+2=26GB
        assert guard.can_fit(model_size_gb=24.0, overhead_gb=2.0) is True

    def test_returns_true_when_mlx_unavailable(self, guard):
        with patch.dict("sys.modules", {"mlx.core": None}):
            assert guard.can_fit(model_size_gb=100.0) is True

    def test_require_fit_passes(self, guard, _mock_mlx):
        _mock_mlx.get_active_memory.return_value = 0
        guard.require_fit(24.0, model_name="test-model")  # Should not raise

    def test_require_fit_cleans_and_retries(self, guard, _mock_mlx):
        # First can_fit: too full (40GB active). After safe_cleanup: drops to 5GB.
        # memory_stats is called by can_fit (2x) + safe_cleanup internals,
        # so we switch after enough calls to simulate cleanup freeing memory.
        call_count = 0

        def dynamic_active():
            nonlocal call_count
            call_count += 1
            # First 2 calls: full (initial can_fit check).
            # After that: cleanup happened, memory freed.
            return 40_000_000_000 if call_count <= 1 else 5_000_000_000

        _mock_mlx.get_active_memory.side_effect = dynamic_active
        guard.require_fit(24.0, model_name="test")
        _mock_mlx.clear_cache.assert_called()  # Cleanup was triggered

    def test_require_fit_raises_if_hopeless(self, guard, _mock_mlx):
        _mock_mlx.get_active_memory.return_value = 45_000_000_000  # 45GB of 48GB
        with pytest.raises(MemoryError, match="Cannot fit"):
            guard.require_fit(24.0, model_name="huge-model")


# ── Model size estimator ─────────────────────────────────────────────────


class TestEstimateModelSize:
    """Model-name heuristic sizing for require_fit pre-load gates."""

    def test_billion_params_with_8bit(self):
        """24B × 8bit (1.0 bytes/param) = 24 GB."""
        assert MetalGuard.estimate_model_size_from_name(
            "mlx-community/Mistral-Small-3.2-24B-Instruct-2506-8bit"
        ) == pytest.approx(24.0)

    def test_billion_params_with_4bit(self):
        """31B × 4bit (0.5 bytes/param) = 15.5 GB."""
        assert MetalGuard.estimate_model_size_from_name(
            "mlx-community/gemma-4-31b-4bit"
        ) == pytest.approx(15.5)

    def test_billion_params_with_2bit(self):
        """2B × 2bit (0.25 bytes/param) = 0.5 GB."""
        assert MetalGuard.estimate_model_size_from_name(
            "mlx-community/tiny-2b-2bit"
        ) == pytest.approx(0.5)

    def test_fp16_default_when_bits_missing(self):
        """7B with no quant hint assumes fp16 → 14 GB upper bound."""
        assert MetalGuard.estimate_model_size_from_name(
            "Llama-7B-Instruct"
        ) == pytest.approx(14.0)

    def test_million_params(self):
        """350m × fp16 (2.0) = 0.7 GB."""
        assert MetalGuard.estimate_model_size_from_name(
            "tiny-350m"
        ) == pytest.approx(0.7)

    def test_mini_size_class_fallback(self):
        """phi-4-mini-4bit → mini (4B) × 4bit (0.5) = 2.0 GB."""
        assert MetalGuard.estimate_model_size_from_name(
            "mlx-community/Phi-4-mini-instruct-4bit"
        ) == pytest.approx(2.0)

    def test_small_size_class_fallback(self):
        """foo-small-8bit → small (7B) × 8bit (1.0) = 7.0 GB."""
        assert MetalGuard.estimate_model_size_from_name(
            "foo-small-8bit"
        ) == pytest.approx(7.0)

    def test_large_size_class_fallback(self):
        """bar-large (no quant) → large (70B) × fp16 (2.0) = 140 GB."""
        assert MetalGuard.estimate_model_size_from_name(
            "bar-large"
        ) == pytest.approx(140.0)

    def test_returns_none_for_unparseable(self):
        """Names without any size hints must return None so callers can
        fall back to threshold-based checks."""
        assert MetalGuard.estimate_model_size_from_name("mystery-model") is None
        assert MetalGuard.estimate_model_size_from_name("random-name") is None
        assert MetalGuard.estimate_model_size_from_name("") is None

    def test_returns_none_for_none_input(self):
        assert MetalGuard.estimate_model_size_from_name(None) is None  # type: ignore[arg-type]

    def test_does_not_match_version_numbers(self):
        """Version strings like '2506' or '3.2' must not be parsed as
        parameter counts. Only explicit <N>B / <N>M suffixes."""
        # '2506-8bit' — regex sees '8b' but negative lookahead (?![a-z])
        # rejects 'bit', and '2506' has no trailing B/M → fallback to None
        # unless another hint matches. In this name, 'Small' triggers the
        # size-class fallback to 7B. × 8bit → 7.0 GB.
        assert MetalGuard.estimate_model_size_from_name(
            "Mistral-Small-3.2-Instruct-2506-8bit"
        ) == pytest.approx(7.0)

    def test_bare_quant_hint_alone_returns_none(self):
        """'8bit' alone without any param-count hint returns None — the
        function refuses to guess at a size from quantization alone."""
        assert MetalGuard.estimate_model_size_from_name("some-model-8bit") is None


# ── Memory pressure ──────────────────────────────────────────────────────


class TestMemoryPressure:
    def test_no_pressure(self, guard, _mock_mlx):
        _mock_mlx.get_peak_memory.return_value = 10_000_000_000
        assert not guard.is_pressure_high(threshold_pct=67.0)

    def test_high_pressure(self, guard, _mock_mlx):
        _mock_mlx.get_peak_memory.return_value = 40_000_000_000
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
        assert stats.available_gb == 47.0


# ── Periodic flush ───────────────────────────────────────────────────────


class TestPeriodicFlush:
    def test_start_and_stop(self, guard):
        guard.start_periodic_flush(interval_secs=0.05)
        assert guard._flush_timer is not None
        time.sleep(0.02)
        guard.stop_periodic_flush()
        assert guard._flush_timer is None

    def test_flush_executes(self, guard, _mock_mlx):
        _mock_mlx.get_active_memory.return_value = 10_000_000_000
        _mock_mlx.get_peak_memory.return_value = 30_000_000_000  # >50% to trigger
        guard.start_periodic_flush(interval_secs=0.05)
        time.sleep(0.15)  # Wait for at least one tick
        guard.stop_periodic_flush()
        # flush_gpu should have been called at least once
        assert _mock_mlx.clear_cache.call_count >= 1

    def test_flush_skips_when_threads_active(self, guard, _mock_mlx):
        event = threading.Event()
        t = threading.Thread(target=event.wait, daemon=True)
        t.start()
        guard.register_thread(t)

        guard.start_periodic_flush(interval_secs=0.05)
        time.sleep(0.15)
        guard.stop_periodic_flush()

        # Flush should have been skipped (thread alive)
        _mock_mlx.clear_cache.assert_not_called()

        event.set()
        t.join(timeout=1)

    def test_double_start_cancels_previous(self, guard):
        guard.start_periodic_flush(interval_secs=1.0)
        first_timer = guard._flush_timer
        guard.start_periodic_flush(interval_secs=2.0)
        assert guard._flush_timer is not first_timer
        guard.stop_periodic_flush()


# ── Memory drift watchdog ─────────────────────────────────────────────────


class TestWatchdog:
    def test_watchdog_starts(self, guard):
        guard.start_watchdog(interval_secs=0.05)
        assert guard._flush_timer is not None
        guard.stop_periodic_flush()

    def test_watchdog_warns_on_high_memory(self, guard, _mock_mlx):
        _mock_mlx.get_active_memory.return_value = 36_000_000_000  # 75% of 48GB
        _mock_mlx.get_peak_memory.return_value = 36_000_000_000
        guard.start_watchdog(interval_secs=0.05, warn_pct=70.0, critical_pct=85.0)
        time.sleep(0.15)
        guard.stop_periodic_flush()
        # Should have called flush (warn level)
        assert _mock_mlx.clear_cache.call_count >= 1

    def test_watchdog_critical_triggers_callback(self, guard, _mock_mlx):
        _mock_mlx.get_active_memory.return_value = 44_000_000_000  # 92% of 48GB
        _mock_mlx.get_peak_memory.return_value = 44_000_000_000
        callback_called = threading.Event()

        def on_critical():
            callback_called.set()

        guard.start_watchdog(
            interval_secs=0.05,
            critical_pct=85.0,
            on_critical=on_critical,
        )
        callback_called.wait(timeout=1.0)
        guard.stop_periodic_flush()
        assert callback_called.is_set()

    def test_watchdog_tracks_drift(self, guard, _mock_mlx):
        # Start at 10GB, drift to 40GB
        call_count = 0

        def drifting_active():
            nonlocal call_count
            call_count += 1
            return 10_000_000_000 if call_count <= 1 else 40_000_000_000

        _mock_mlx.get_active_memory.side_effect = drifting_active
        _mock_mlx.get_peak_memory.return_value = 40_000_000_000
        guard.start_watchdog(interval_secs=0.05, warn_pct=70.0, critical_pct=85.0)
        time.sleep(0.2)
        guard.stop_periodic_flush()
        # Baseline should have been set
        assert guard._watchdog_baseline is not None


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
