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

    def test_require_fit_escalated_retry_calls_cache_clear(self, guard, _mock_mlx):
        """Escalated retry must invoke cache_clear_cb + sleep + re-check.

        Added in 0.2.3 — fixes the 2026-04-10 kantocamera OOM where standard
        safe_cleanup left 26.8GB active blocking a 24GB mistral load.
        """
        state = {"cleared": False}

        def dynamic_active():
            # Before cache_clear_cb runs: memory still full.
            # After cb runs: memory freed (simulates escalated recovery).
            return 5_000_000_000 if state["cleared"] else 40_000_000_000

        _mock_mlx.get_active_memory.side_effect = dynamic_active

        def clear_cb():
            state["cleared"] = True

        guard.require_fit(
            24.0,
            model_name="test",
            cache_clear_cb=clear_cb,
            escalated_cooldown_sec=0.01,  # tiny for fast test
        )
        assert state["cleared"], "cache_clear_cb should have been invoked during escalation"

    def test_require_fit_escalated_propagates_cb_exception_non_fatally(self, guard, _mock_mlx):
        """A failing cache_clear_cb must not abort escalation — logged, continues."""
        state = {"cb_attempted": False}

        def dynamic_active():
            # Memory still full until cb "would have" cleared it — but cb raises,
            # so escalation keeps going with its own safe_cleanup and reset.
            # Simulate: the second safe_cleanup + cooldown does free the memory.
            return 5_000_000_000 if state["cb_attempted"] else 40_000_000_000

        _mock_mlx.get_active_memory.side_effect = dynamic_active

        def bad_cb():
            state["cb_attempted"] = True
            raise RuntimeError("cache clear exploded")

        # Must not propagate the cb exception — escalation path keeps going
        # and eventually the second can_fit returns True.
        guard.require_fit(
            24.0,
            model_name="test",
            cache_clear_cb=bad_cb,
            escalated_cooldown_sec=0.01,
        )
        assert state["cb_attempted"], "bad cb should have been invoked before raising"

    def test_require_fit_escalated_still_raises_if_hopeless(self, guard, _mock_mlx):
        """If escalation can't free enough memory, raise MemoryError with escalated cooldown in message."""
        _mock_mlx.get_active_memory.return_value = 45_000_000_000  # always full
        with pytest.raises(MemoryError, match="escalated cleanup"):
            guard.require_fit(
                24.0,
                model_name="huge-model",
                cache_clear_cb=lambda: None,
                escalated_cooldown_sec=0.01,
            )

    def test_require_fit_backward_compat_no_escalated_param(self, guard, _mock_mlx):
        """Calling require_fit without new params still works (backward compat)."""
        _mock_mlx.get_active_memory.return_value = 0
        guard.require_fit(24.0, model_name="test")  # old-style call, no kwargs


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


# ── Cross-process lock FORCE hardening (v0.6.0) ──────────────────────────
#
# Hardened ``acquire_mlx_lock(force=True)`` semantics: SIGTERM the
# previous holder, poll for exit (with zombie-aware liveness), sleep a
# Metal-buffer-GC cooldown, THEN reclaim. Never silently unlinks while
# the peer is still alive — that was the pre-v0.6.0 kernel-panic path.


import json
import os
import signal
import subprocess
import sys
import textwrap

import metal_guard as mg
from metal_guard import MLXLockConflict, acquire_mlx_lock, read_mlx_lock, release_mlx_lock


@pytest.fixture
def _isolated_lock(tmp_path, monkeypatch):
    """Redirect the cross-process lock path to a tmp file."""
    fake = tmp_path / "mlx_exclusive.lock"
    monkeypatch.setattr(mg, "_PROCESS_LOCK_PATH", fake)
    yield fake
    if fake.exists():
        fake.unlink()


_CHILD_NORMAL_SCRIPT = textwrap.dedent(
    """
    import json, os, signal, sys, time
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    lock_path, label, sleep_sec = sys.argv[1], sys.argv[2], float(sys.argv[3])
    info = {
        "pid": os.getpid(), "label": label,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cmdline": "child_normal",
    }
    from pathlib import Path
    p = Path(lock_path); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(info))
    time.sleep(sleep_sec)
    """
).strip()

_CHILD_IGNORE_SCRIPT = textwrap.dedent(
    """
    import json, os, signal, sys, time
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    lock_path, label = sys.argv[1], sys.argv[2]
    info = {
        "pid": os.getpid(), "label": label,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cmdline": "child_ignore",
    }
    from pathlib import Path
    p = Path(lock_path); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(info))
    while True:
        time.sleep(0.5)
    """
).strip()


def _spawn_child(script, *args):
    return subprocess.Popen(
        [sys.executable, "-c", script, *map(str, args)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def _wait_for_lock_pid(lock_path, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if lock_path.exists():
            try:
                info = json.loads(lock_path.read_text())
                if isinstance(info.get("pid"), int):
                    return info
            except json.JSONDecodeError:
                pass
        time.sleep(0.05)
    raise TimeoutError(f"child did not write lock within {timeout}s")


class TestForceOverrideHardening:
    def test_force_terminates_holder_and_replaces(self, _isolated_lock, monkeypatch):
        """FORCE SIGTERMs the holder, waits, then reclaims."""
        monkeypatch.setattr(mg, "_FORCE_WAIT_SEC", 5.0)
        monkeypatch.setattr(mg, "_RECLAIM_COOLDOWN_SEC", 0.0)

        child = _spawn_child(_CHILD_NORMAL_SCRIPT, _isolated_lock, "peer", 60)
        try:
            holder = _wait_for_lock_pid(_isolated_lock)
            assert holder["pid"] == child.pid
            info = acquire_mlx_lock("rescuer", force=True)
            assert info["label"] == "rescuer"
            assert info["pid"] == os.getpid()
            assert child.wait(timeout=2) is not None
            current = read_mlx_lock()
            assert current["label"] == "rescuer"
        finally:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=2)
            release_mlx_lock()

    def test_force_raises_if_peer_ignores_sigterm(self, _isolated_lock, monkeypatch):
        """Peer traps SIGTERM → MLXLockConflict(force_timeout=True), lock NOT unlinked.

        This is the anti-panic invariant: never unlink while the peer
        might still hold Metal buffers.
        """
        monkeypatch.setattr(mg, "_FORCE_WAIT_SEC", 2.0)
        monkeypatch.setattr(mg, "_RECLAIM_COOLDOWN_SEC", 0.0)

        child = _spawn_child(_CHILD_IGNORE_SCRIPT, _isolated_lock, "stubborn")
        try:
            holder = _wait_for_lock_pid(_isolated_lock)
            assert holder["pid"] == child.pid
            with pytest.raises(MLXLockConflict) as exc_info:
                acquire_mlx_lock("rescuer", force=True)
            assert exc_info.value.holder.get("force_timeout") is True
            assert child.poll() is None, "child must still be alive"
            current = read_mlx_lock()
            assert current["label"] == "stubborn"
            assert current["pid"] == child.pid
        finally:
            child.kill()
            child.wait(timeout=5)

    def test_force_raises_permission_denied(self, _isolated_lock, monkeypatch):
        """PermissionError on SIGTERM → MLXLockConflict(force_permission_denied=True)."""
        monkeypatch.setattr(mg, "_FORCE_WAIT_SEC", 1.0)
        monkeypatch.setattr(mg, "_RECLAIM_COOLDOWN_SEC", 0.0)

        foreign = {
            "pid": 1, "label": "init_like", "started_at": "2026-04-14T00:00:00Z",
            "cmdline": "init",
        }
        _isolated_lock.write_text(json.dumps(foreign))

        def _deny(pid, sig):
            raise PermissionError(f"simulated pid={pid}")

        monkeypatch.setattr(mg.os, "kill", _deny)

        with pytest.raises(MLXLockConflict) as exc_info:
            acquire_mlx_lock("rescuer", force=True)
        assert exc_info.value.holder.get("force_permission_denied") is True
        assert "init_like" in _isolated_lock.read_text()

    def test_force_reclaims_when_peer_dies_between_read_and_signal(
        self, _isolated_lock, monkeypatch,
    ):
        """ProcessLookupError race → cleanly reclaim without raising."""
        monkeypatch.setattr(mg, "_FORCE_WAIT_SEC", 1.0)
        monkeypatch.setattr(mg, "_RECLAIM_COOLDOWN_SEC", 0.0)

        foreign = {
            "pid": 1, "label": "racy", "started_at": "2026-04-14T00:00:00Z",
            "cmdline": "x",
        }
        _isolated_lock.write_text(json.dumps(foreign))

        real_kill = os.kill
        term_calls = {"n": 0}

        def _race(pid, sig):
            if sig == signal.SIGTERM:
                term_calls["n"] += 1
                raise ProcessLookupError("peer died (simulated)")
            return real_kill(pid, sig)

        monkeypatch.setattr(mg.os, "kill", _race)

        info = acquire_mlx_lock("rescuer", force=True)
        assert info["label"] == "rescuer"
        assert term_calls["n"] == 1
        release_mlx_lock()

    def test_force_applies_post_reclaim_cooldown(self, _isolated_lock, monkeypatch):
        """After SIGTERM+exit, acquire sleeps the cooldown."""
        monkeypatch.setattr(mg, "_FORCE_WAIT_SEC", 5.0)
        monkeypatch.setattr(mg, "_RECLAIM_COOLDOWN_SEC", 0.3)

        child = _spawn_child(_CHILD_NORMAL_SCRIPT, _isolated_lock, "peer_cd", 60)
        try:
            _wait_for_lock_pid(_isolated_lock)
            t0 = time.monotonic()
            info = acquire_mlx_lock("rescuer_cd", force=True)
            elapsed = time.monotonic() - t0
            assert info["label"] == "rescuer_cd"
            assert elapsed >= 0.25, f"cooldown not applied: {elapsed:.3f}s"
            assert elapsed < 2.0, f"suspiciously slow: {elapsed:.3f}s"
        finally:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=2)
            release_mlx_lock()

    def test_normal_acquire_does_not_apply_cooldown(self, _isolated_lock, monkeypatch):
        """No cooldown when the lock is free — zero-latency path."""
        monkeypatch.setattr(mg, "_RECLAIM_COOLDOWN_SEC", 5.0)

        t0 = time.monotonic()
        info = acquire_mlx_lock("solo")
        elapsed = time.monotonic() - t0
        assert info["label"] == "solo"
        assert elapsed < 0.5, f"normal acquire took {elapsed:.3f}s"
        release_mlx_lock()

    def test_force_reclaims_stale_pid_silently(self, _isolated_lock, monkeypatch):
        """Dead pid → SIGTERM raises ProcessLookupError → clean reclaim."""
        monkeypatch.setattr(mg, "_RECLAIM_COOLDOWN_SEC", 0.0)
        stale = {
            "pid": 2_147_483_646,  # almost certainly dead
            "label": "ghost", "started_at": "2026-04-14T00:00:00Z",
            "cmdline": "ghost",
        }
        _isolated_lock.write_text(json.dumps(stale))
        # read_mlx_lock will detect dead pid and unlink before FORCE branch
        info = acquire_mlx_lock("rescuer", force=True)
        assert info["label"] == "rescuer"
        release_mlx_lock()


class TestZombieAwareLiveness:
    """_is_pid_alive treats zombies as dead because Metal buffers are released."""

    def test_live_non_zombie_is_alive(self):
        # Our own pid is alive, not a zombie.
        assert mg._is_pid_alive(os.getpid()) is True

    def test_dead_pid_is_not_alive(self):
        assert mg._is_pid_alive(2_147_483_646) is False

    def test_zombie_is_treated_as_dead(self, monkeypatch):
        # Force _is_zombie to report True for an otherwise-alive pid.
        monkeypatch.setattr(mg, "_is_zombie", lambda pid: True)
        assert mg._is_pid_alive(os.getpid()) is False

    def test_is_zombie_handles_missing_ps(self, monkeypatch):
        # Simulate ps missing / unusable — fall back to "not zombie".
        def _raise(*a, **kw):
            raise FileNotFoundError("ps not found")
        monkeypatch.setattr("subprocess.run", _raise)
        assert mg._is_zombie(os.getpid()) is False


# ── Version advisory system (v0.6.0) ─────────────────────────────────────


class TestVersionAdvisories:
    def test_returns_empty_for_clean_environment(self):
        # Pretend only safe versions are installed. Hypothetical future
        # versions past every existing advisory range:
        # mlx-lm 0.31.3 clears the 0.31.0–0.31.2 cluster;
        # mlx 0.32.0 clears #3348, #3350, #3384, #3390;
        # mlx-vlm 0.5.0 clears the 0.4.4 cluster.
        active = mg.check_version_advisories(
            packages={"mlx-lm": "0.31.3", "mlx": "0.32.0", "mlx-vlm": "0.5.0"},
        )
        assert active == []

    def test_flags_mlx_lm_0_31_2_as_1128(self):
        active = mg.check_version_advisories(packages={"mlx-lm": "0.31.2"})
        issues = {a["issue"] for a in active}
        assert "ml-explore/mlx-lm#1128" in issues
        assert "ml-explore/mlx-lm#1139" in issues
        # Every advisory carries the installed version + severity.
        for a in active:
            assert a["installed_version"] == "0.31.2"
            assert a["severity"] in {"critical", "high", "medium", "info"}
            assert a["package"] == "mlx-lm"
            assert a["url"].startswith("https://github.com/")

    def test_skips_package_not_installed(self):
        # No mlx-lm → no mlx-lm advisories, but mlx advisory still evaluates.
        active = mg.check_version_advisories(packages={"mlx": "0.31.1"})
        packages = {a["package"] for a in active}
        assert "mlx-lm" not in packages

    def test_spec_matches_exact(self):
        assert mg._spec_matches("0.31.2", "==0.31.2")
        assert not mg._spec_matches("0.31.1", "==0.31.2")

    def test_spec_matches_range(self):
        assert mg._spec_matches("0.31.1", "<0.31.2")
        assert not mg._spec_matches("0.31.2", "<0.31.2")
        assert mg._spec_matches("0.30.0", "<0.31.2")


# ── Upstream defensive patches (v0.6.0) ──────────────────────────────────


class _FakeTokenizerWrapper:
    """Stand-in for mlx_lm.tokenizer_utils.TokenizerWrapper."""

    _think_start_tokens = None

    @property
    def think_start_id(self):
        if len(self._think_start_tokens) > 1:  # mimics the upstream bug
            raise ValueError("multi")
        return self._think_start_tokens[0]


class TestUpstreamDefensivePatches:
    def test_original_property_crashes_on_none(self):
        # Sanity: the simulated bug reproduces without the patch.
        tw = _FakeTokenizerWrapper()
        with pytest.raises(TypeError):
            _ = tw.think_start_id

    def test_patch_applies_on_0_31_2(self, monkeypatch):
        fake_module = type(sys)("mlx_lm.tokenizer_utils")
        fake_module.TokenizerWrapper = _FakeTokenizerWrapper

        monkeypatch.setitem(sys.modules, "mlx_lm.tokenizer_utils", fake_module)
        monkeypatch.setattr(mg, "_installed_version",
                            lambda pkg: "0.31.2" if pkg == "mlx-lm" else None)

        applied = mg.install_upstream_defensive_patches()
        assert applied["mlx_lm_1128_think_start_id"] is True

        tw = _FakeTokenizerWrapper()
        assert tw.think_start_id is None  # no crash now

    def test_patch_skipped_on_0_31_1(self, monkeypatch):
        fake_module = type(sys)("mlx_lm.tokenizer_utils")
        fake_module.TokenizerWrapper = _FakeTokenizerWrapper
        monkeypatch.setitem(sys.modules, "mlx_lm.tokenizer_utils", fake_module)
        monkeypatch.setattr(mg, "_installed_version",
                            lambda pkg: "0.31.1" if pkg == "mlx-lm" else None)

        applied = mg.install_upstream_defensive_patches()
        assert applied["mlx_lm_1128_think_start_id"] is False

    def test_patch_is_idempotent(self, monkeypatch):
        # Re-call on an already-patched class is a no-op.
        fake_module = type(sys)("mlx_lm.tokenizer_utils")
        # Build a fresh class so prior tests don't bleed.
        class TW:
            _think_start_tokens = None

            @property
            def think_start_id(self):
                return len(self._think_start_tokens)

        fake_module.TokenizerWrapper = TW
        monkeypatch.setitem(sys.modules, "mlx_lm.tokenizer_utils", fake_module)
        monkeypatch.setattr(mg, "_installed_version",
                            lambda pkg: "0.31.2" if pkg == "mlx-lm" else None)

        first = mg.install_upstream_defensive_patches()
        assert first["mlx_lm_1128_think_start_id"] is True
        second = mg.install_upstream_defensive_patches()
        assert second["mlx_lm_1128_think_start_id"] is False

    def test_patch_skipped_when_mlx_lm_missing(self, monkeypatch):
        monkeypatch.setattr(mg, "_installed_version", lambda pkg: None)
        applied = mg.install_upstream_defensive_patches()
        assert applied["mlx_lm_1128_think_start_id"] is False

    def test_force_bypasses_version_gate(self, monkeypatch):
        fake_module = type(sys)("mlx_lm.tokenizer_utils")
        class TW:
            _think_start_tokens = None

            @property
            def think_start_id(self):
                return len(self._think_start_tokens)

        fake_module.TokenizerWrapper = TW
        monkeypatch.setitem(sys.modules, "mlx_lm.tokenizer_utils", fake_module)
        monkeypatch.setattr(mg, "_installed_version",
                            lambda pkg: "0.99.0" if pkg == "mlx-lm" else None)

        assert mg.install_upstream_defensive_patches()["mlx_lm_1128_think_start_id"] is False
        assert mg.install_upstream_defensive_patches(force=True)["mlx_lm_1128_think_start_id"] is True


# ═══════════════════════════════════════════════════════════════════════════
# 2026-04-16 community survey + R4/R5/R6/R8 port tests
# ═══════════════════════════════════════════════════════════════════════════


class TestNewAdvisories:
    """9 new entries from 2026-04-15 (mlx-vlm) + 2026-04-16 (survey) cohorts."""

    def test_mlx_vlm_967_race_fires_below_0_4_5(self):
        active = mg.check_version_advisories(packages={"mlx-vlm": "0.4.4"})
        issues = {a["issue"] for a in active}
        assert "Blaizzy/mlx-vlm#967" in issues
        race = next(a for a in active if a["issue"] == "Blaizzy/mlx-vlm#967")
        assert race["severity"] == "critical"

    def test_mlx_vlm_0_4_4_fires_all_four_cluster_entries(self):
        active = mg.check_version_advisories(packages={"mlx-vlm": "0.4.4"})
        issues = {a["issue"] for a in active}
        for n in ("967", "1016", "1011", "943"):
            assert f"Blaizzy/mlx-vlm#{n}" in issues

    def test_mlx_vlm_0_4_5_clears_cluster(self):
        active = mg.check_version_advisories(packages={"mlx-vlm": "0.4.5"})
        issues = {a["issue"] for a in active}
        for n in ("967", "1016", "1011", "943"):
            assert f"Blaizzy/mlx-vlm#{n}" not in issues

    def test_mlx_3384_sdpa_4bit_divergence_is_critical(self):
        active = mg.check_version_advisories(packages={"mlx": "0.31.1"})
        entry = next(
            a for a in active if a["issue"] == "ml-explore/mlx#3384"
        )
        assert entry["severity"] == "critical"

    def test_mlx_lm_897_applies_across_range(self):
        for ver in ("0.31.0", "0.31.1", "0.31.2"):
            active = mg.check_version_advisories(packages={"mlx-lm": ver})
            issues = {a["issue"] for a in active}
            assert "ml-explore/mlx-lm#897" in issues
        active = mg.check_version_advisories(packages={"mlx-lm": "0.31.3"})
        assert "ml-explore/mlx-lm#897" not in {
            a["issue"] for a in active
        }

    def test_mlx_3350_fires_through_0_31_2(self):
        for ver in ("0.31.0", "0.31.1", "0.31.2"):
            active = mg.check_version_advisories(packages={"mlx": ver})
            assert "ml-explore/mlx#3350" in {a["issue"] for a in active}

    def test_mlx_3390_completion_abort_fires(self):
        active = mg.check_version_advisories(packages={"mlx": "0.31.1"})
        entry = next(
            a for a in active if a["issue"] == "ml-explore/mlx#3390"
        )
        assert entry["severity"] == "high"
        assert "AGX_RELAX_CDM_CTXSTORE_TIMEOUT" in entry["mitigation"]

    def test_mlx_vlm_999_server_cache_thrash(self):
        active = mg.check_version_advisories(packages={"mlx-vlm": "0.4.4"})
        entry = next(
            a for a in active if a["issue"] == "Blaizzy/mlx-vlm#999"
        )
        assert "server" in entry.get("scope", "").lower()


class TestSystemAudit:
    """R2 — wired_limit + IOGPUFamily kext version audits."""

    def test_audit_wired_limit_default_mode(self, monkeypatch):
        def fake_sysctl(name, *, timeout=2.0):
            if name == "iogpu.wired_limit_mb":
                return "0"
            if name == "hw.memsize":
                return str(128 * 1024 ** 3)
            return None
        monkeypatch.setattr(mg, "_sysctl", fake_sysctl)
        result = mg.audit_wired_limit()
        assert result["mode"] == "default"
        assert result["advisory"] is None
        assert result["total_gb"] == 128.0

    def test_audit_wired_limit_override_over_threshold(self, monkeypatch):
        def fake_sysctl(name, *, timeout=2.0):
            if name == "iogpu.wired_limit_mb":
                return str(120 * 1024)  # 120 GB
            if name == "hw.memsize":
                return str(128 * 1024 ** 3)
            return None
        monkeypatch.setattr(mg, "_sysctl", fake_sysctl)
        result = mg.audit_wired_limit()
        assert result["mode"] == "override"
        assert result["advisory"] is not None
        assert "mlx-lm#1047" in result["advisory"]

    def test_audit_wired_limit_override_within_threshold(self, monkeypatch):
        def fake_sysctl(name, *, timeout=2.0):
            if name == "iogpu.wired_limit_mb":
                return str(60 * 1024)
            if name == "hw.memsize":
                return str(128 * 1024 ** 3)
            return None
        monkeypatch.setattr(mg, "_sysctl", fake_sysctl)
        result = mg.audit_wired_limit()
        assert result["mode"] == "override"
        assert result["advisory"] is None

    def test_audit_wired_limit_unknown_on_sysctl_fail(self, monkeypatch):
        monkeypatch.setattr(mg, "_sysctl", lambda *a, **kw: None)
        result = mg.audit_wired_limit()
        assert result["mode"] == "unknown"
        assert result["advisory"] is None

    def test_read_gpu_driver_version_returns_none_on_failure(self, monkeypatch):
        import subprocess

        def fake_run(*args, **kwargs):
            raise subprocess.SubprocessError("mocked failure")
        monkeypatch.setattr(subprocess, "run", fake_run)
        # Both kextstat and ioreg paths fail → None
        assert mg.read_gpu_driver_version(timeout=0.1) is None


class TestPrefillGuard:
    """R4 + R7."""

    def test_estimate_131k_mistral_over_5gb(self):
        dims = mg.KNOWN_MODELS["Mistral-Small-3.2-24B"]
        peak = mg.estimate_prefill_peak_alloc_gb(
            context_tokens=131072, dims=dims,
        )
        assert peak > 25.0

    def test_lookup_dims_with_namespace_prefix(self):
        dims = mg.lookup_dims("mlx-community/gemma-4-26b-a4b-it-4bit")
        assert dims is not None
        assert dims.n_heads == 16

    def test_lookup_dims_unknown_returns_none(self):
        assert mg.lookup_dims("no-such-model") is None

    def test_require_fit_passes_for_small(self):
        dims = mg.ModelDims(n_layers=16, n_heads=8, n_kv_heads=8, head_dim=128)
        mg.require_prefill_fit(
            context_tokens=512, dims=dims, available_gb=40.0,
        )

    def test_require_fit_raises_on_ceiling(self):
        dims = mg.KNOWN_MODELS["Mistral-Small-3.2-24B"]
        with pytest.raises(MetalOOMError, match="single-alloc ceiling"):
            mg.require_prefill_fit(
                context_tokens=131072, dims=dims, available_gb=200.0,
            )

    def test_require_fit_raises_on_headroom(self):
        dims = mg.ModelDims(n_layers=40, n_heads=8, n_kv_heads=8, head_dim=128)
        with pytest.raises(MetalOOMError, match="headroom"):
            mg.require_prefill_fit(
                context_tokens=20_000, dims=dims,
                available_gb=4.5, single_alloc_ceiling_gb=10.0,
            )

    def test_recommend_chunk_full_fits(self):
        dims = mg.ModelDims(n_layers=16, n_heads=8, n_kv_heads=8, head_dim=128)
        assert mg.recommend_chunk_size(context_tokens=1024, dims=dims) == 1024

    def test_recommend_chunk_binary_search(self):
        dims = mg.KNOWN_MODELS["Mistral-Small-3.2-24B"]
        chunk = mg.recommend_chunk_size(
            context_tokens=131072, dims=dims, single_alloc_ceiling_gb=4.0,
        )
        assert chunk < 131072
        assert mg.estimate_prefill_peak_alloc_gb(
            context_tokens=chunk, dims=dims,
        ) <= 4.0

    def test_describe_plan_unknown_model(self):
        d = mg.describe_prefill_plan(
            context_tokens=4096, model_id="no-such-model",
        )
        assert d["dims_known"] is False
        assert d["peak_alloc_gb"] is None

    def test_describe_plan_131k_refused(self):
        d = mg.describe_prefill_plan(
            context_tokens=131072,
            model_id="Mistral-Small-3.2-24B",
            available_gb=60.0,
        )
        assert d["fits_ceiling"] is False
        assert d["recommended_chunk_size"] is not None


class TestKVTracker:
    """R5 — per-request cumulative KV growth."""

    def test_start_add_finalize(self):
        t = mg.KVGrowthTracker()
        t.start("r", ceiling_gb=1.0)
        assert t.add_bytes("r", 100_000_000) == 100_000_000
        summary = t.finalize("r")
        assert summary["cumulative_gb"] == 0.1
        assert summary["aborted"] is False

    def test_ceiling_breach_raises(self):
        t = mg.KVGrowthTracker()
        t.start("r", ceiling_gb=0.1)
        with pytest.raises(MetalOOMError, match="ceiling"):
            t.add_bytes("r", 200_000_000)

    def test_aborted_state_re_raises(self):
        t = mg.KVGrowthTracker()
        t.start("r", ceiling_gb=0.1)
        with pytest.raises(MetalOOMError):
            t.add_bytes("r", 200_000_000)
        with pytest.raises(MetalOOMError):
            t.add_bytes("r", 1)

    def test_untracked_request_is_noop(self):
        t = mg.KVGrowthTracker()
        assert t.add_bytes("never-started", 1) == 0

    def test_rejects_invalid_inputs(self):
        t = mg.KVGrowthTracker()
        with pytest.raises(ValueError):
            t.start("r", ceiling_gb=0.0)
        t.start("r", ceiling_gb=1.0)
        with pytest.raises(ValueError):
            t.add_bytes("r", -1)

    def test_snapshot(self):
        t = mg.KVGrowthTracker()
        t.start("a", ceiling_gb=10.0)
        t.start("b", ceiling_gb=10.0)
        t.add_bytes("a", 5_000_000)
        snap = {r["request_id"]: r["cumulative_gb"] for r in t.snapshot()}
        assert snap["a"] == 0.005
        assert snap["b"] == 0.0


class TestProcessMode:
    """R6 — process classification + mode-specific defaults."""

    def test_subprocess_worker_env_wins(self, monkeypatch):
        monkeypatch.setenv("METALGUARD_SUBPROCESS_WORKER", "1")
        assert mg.detect_process_mode() == "subprocess_worker"

    def test_mlx_server_argv(self, monkeypatch):
        monkeypatch.delenv("METALGUARD_SUBPROCESS_WORKER", raising=False)
        monkeypatch.setattr("sys.argv", ["python", "-m", "mlx_lm.server"])
        assert mg.detect_process_mode() == "server"

    def test_uvicorn_argv0(self, monkeypatch):
        monkeypatch.delenv("METALGUARD_SUBPROCESS_WORKER", raising=False)
        monkeypatch.setattr("sys.argv", ["/path/to/uvicorn", "app:app"])
        assert mg.detect_process_mode() == "server"

    def test_plain_script_is_cli(self, monkeypatch):
        import sys as _sys
        monkeypatch.delenv("METALGUARD_SUBPROCESS_WORKER", raising=False)
        monkeypatch.setattr("sys.argv", ["my_script.py"])
        monkeypatch.delitem(_sys.modules, "ipykernel", raising=False)
        assert mg.detect_process_mode() == "cli"

    def test_apply_defaults_includes_mode_key(self):
        d = mg.apply_mode_defaults("server")
        assert d["mode"] == "server"
        assert "generate_timeout_sec" in d

    def test_server_stricter_than_notebook(self):
        srv = mg.apply_mode_defaults("server")
        nb = mg.apply_mode_defaults("notebook")
        assert srv["generate_timeout_sec"] < nb["generate_timeout_sec"]

    def test_subprocess_worker_skips_lock(self):
        d = mg.apply_mode_defaults("subprocess_worker")
        assert d.get("skip_process_lock") is True

    def test_describe_has_required_keys(self):
        d = mg.describe_process_mode()
        for key in ("mode", "argv0", "has_ipykernel", "defaults"):
            assert key in d


class TestAppleFeedback:
    """R8 — Feedback Assistant formatter."""

    _SAMPLE = {
        "panic_string": "IOGPUMemory.cpp:492 underflow",
        "panic_time": "2026-04-15T23:30:45Z",
        "hardware": {"chip": "M1 Ultra", "gpu_memory_gb": 128},
        "gpu_driver": "104.6.3",
        "os_version": "Darwin 24.6.0",
        "kernel_version": "Darwin Kernel 24.6.0",
        "mlx_versions": {"mlx": "0.31.1"},
        "repro_steps": ["Load Mistral 131k", "Observe panic"],
        "breadcrumbs": ["[23:06] LOAD START", "[23:25] CELL_START 131072"],
        "advisories": [{
            "severity": "critical", "package": "mlx", "installed_version": "0.31.1",
            "title": "SDPA 4-bit divergence", "url": "https://example.com/1",
        }],
    }

    def test_full_forensics_has_sections(self):
        report = mg.format_panic_for_apple_feedback(self._SAMPLE)
        for section in ("## System", "## Reproducer", "## Active advisories",
                        "## Breadcrumb tail", "mlx#3186"):
            assert section in report

    def test_exclude_breadcrumb_option(self):
        report = mg.format_panic_for_apple_feedback(
            self._SAMPLE, include_breadcrumb=False,
        )
        assert "Breadcrumb tail" not in report
        assert "CELL_START 131072" not in report

    def test_empty_forensics_renders_unknown(self):
        report = mg.format_panic_for_apple_feedback({})
        assert "unknown" in report
        assert "PANIC STRING" in report

    def test_truncates_long_breadcrumbs(self):
        forensics = dict(self._SAMPLE)
        forensics["breadcrumbs"] = [f"line {i}" for i in range(200)]
        report = mg.format_panic_for_apple_feedback(
            forensics, max_breadcrumb_lines=10,
        )
        assert "line 190" in report
        assert "line 189" not in report


def _module_level_stub_worker(model_id, backend, recv_conn, send_conn):
    """Module-level stub worker so multiprocessing spawn can pickle it.

    macOS multiprocessing default is spawn; local / closure functions
    are not picklable, so MLXSubprocessRunner tests that inject a fake
    worker must reference an importable name.
    """
    import multiprocessing
    send_conn.send({
        "type": "ready",
        "model_id": model_id,
        "pid": multiprocessing.current_process().pid,
    })
    try:
        while True:
            if not recv_conn.poll(timeout=5):
                break
            msg = recv_conn.recv()
            if msg is None or msg.get("type") == "shutdown":
                break
    except (EOFError, OSError):
        pass


class TestSubprocessRunnerLock:
    """H7 — MLXSubprocessRunner acquires the cross-process MLX lock."""

    def test_runner_acquires_lock_on_init(self, _isolated_lock, monkeypatch):
        """Construction writes this process's pid to the lock file."""
        monkeypatch.setattr(mg, "_worker_main", _module_level_stub_worker)

        runner = mg.MLXSubprocessRunner("fake-model")
        try:
            holder = mg.read_mlx_lock()
            assert holder is not None
            assert "mlx_subprocess_runner" in holder["label"]
        finally:
            runner.shutdown()

    def test_runner_releases_on_shutdown(self, _isolated_lock, monkeypatch):
        monkeypatch.setattr(mg, "_worker_main", _module_level_stub_worker)

        runner = mg.MLXSubprocessRunner("fake-model")
        assert mg.read_mlx_lock() is not None
        runner.shutdown()
        assert mg.read_mlx_lock() is None

    def test_runner_refuses_if_other_live_process_holds_lock(
        self, _isolated_lock, monkeypatch,
    ):
        """Pre-write lock with os.getppid() (live, not us)."""
        import json
        import os as _os
        other_pid = _os.getppid()
        _isolated_lock.write_text(json.dumps({
            "pid": other_pid, "label": "simulated_peer",
            "started_at": "2026-04-16T00:00:00Z",
            "cmdline": "peer", "host": "test",
        }))
        with pytest.raises(MLXLockConflict):
            mg.MLXSubprocessRunner("fake-model")
