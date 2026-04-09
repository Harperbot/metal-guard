"""MetalGuard — GPU safety layer for MLX on Apple Silicon.

Centralized workaround for the IOGPUFamily driver bug that causes kernel
panics (completeMemory() prepare count underflow @IOGPUMemory.cpp:492)
instead of graceful process termination when Metal GPU memory management
fails on Apple Silicon.

This affects any workload that repeatedly loads and unloads MLX models —
the Metal driver's internal reference count can underflow when GPU buffers
are freed while inference threads are still running, or when cleanup
operations race with lazy evaluation.

Root causes identified:
  1. Daemon thread race condition — generate thread holds GPU buffers,
     main thread calls mx.clear_cache() before thread finishes
  2. Unconditional Metal initialization — calling mx.eval()/mx.clear_cache()
     when no models are loaded still initializes the Metal driver, which can
     panic if the driver is in an unstable state from prior crashes

Reference: https://github.com/ml-explore/mlx-lm/issues/883
Related:
  - https://github.com/ml-explore/mlx/issues/2133 (thread safety)
  - https://github.com/ml-explore/mlx/issues/3126 (sub-thread exit crash)
  - https://github.com/ml-explore/mlx/issues/3078 (concurrent inference)

Usage:
    from metal_guard import metal_guard

    # Track GPU-bound threads before they start inference
    thread = threading.Thread(target=run_inference, daemon=True)
    thread.start()
    metal_guard.register_thread(thread)

    # Before any cleanup that touches mx.clear_cache()
    metal_guard.wait_for_threads()

    # Full cleanup: wait -> gc -> flush GPU -> cooldown
    metal_guard.safe_cleanup()

    # Check memory pressure before loading a new model
    metal_guard.ensure_headroom(model_name="my-model-8bit")

    # Crash-safe breadcrumb for post-mortem forensics
    metal_guard.breadcrumb("LOAD: my-model-8bit START")

License: MIT
"""

from __future__ import annotations

import gc
import logging
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Generator

__version__ = "0.1.0"

log = logging.getLogger("metal_guard")


# ---------------------------------------------------------------------------
# Memory stats
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MemoryStats:
    """Snapshot of Metal GPU memory state."""

    active_bytes: int = 0
    peak_bytes: int = 0
    limit_bytes: int = 0

    @property
    def active_gb(self) -> float:
        return self.active_bytes / 1e9

    @property
    def peak_gb(self) -> float:
        return self.peak_bytes / 1e9

    @property
    def limit_gb(self) -> float:
        return self.limit_bytes / 1e9

    @property
    def active_pct(self) -> float:
        return (self.active_bytes / self.limit_bytes * 100) if self.limit_bytes else 0

    @property
    def peak_pct(self) -> float:
        return (self.peak_bytes / self.limit_bytes * 100) if self.limit_bytes else 0

    def __str__(self) -> str:
        return (
            f"active={self.active_gb:.1f}GB peak={self.peak_gb:.1f}GB "
            f"limit={self.limit_gb:.0f}GB "
            f"(active {self.active_pct:.0f}% / peak {self.peak_pct:.0f}%)"
        )


# ---------------------------------------------------------------------------
# MetalGuard
# ---------------------------------------------------------------------------

class MetalGuard:
    """Centralized Metal GPU safety for MLX on Apple Silicon.

    Three responsibilities:
      1. **Thread tracking** — ensure no GPU thread is alive before cleanup
      2. **Safe cleanup** — gc + GPU sync + cache clear + cooldown
      3. **Memory pressure** — monitor and preemptively clean before overload

    The breadcrumb log is fsync'd before every dangerous Metal operation.
    After a kernel panic, the last breadcrumb shows which op crashed.

    Parameters:
        cooldown_secs: Sleep duration after GPU flush to let Metal driver
            reclaim buffers.  2s is safe for M1/M2/M3/M4 Ultra.
        thread_timeout_secs: Max time to wait for GPU threads before
            proceeding with cleanup.
        breadcrumb_path: File path for crash-safe breadcrumb log.
            Set to None to disable breadcrumbs.
    """

    def __init__(
        self,
        cooldown_secs: float = 2.0,
        thread_timeout_secs: float = 30.0,
        breadcrumb_path: str | None = "logs/metal_breadcrumb.log",
    ) -> None:
        self._threads: list[threading.Thread] = []
        self._lock = threading.Lock()
        self._cooldown = cooldown_secs
        self._thread_timeout = thread_timeout_secs
        self._breadcrumb_path = breadcrumb_path

    # ── Thread tracking ──────────────────────────────────────────────

    def register_thread(self, thread: threading.Thread) -> None:
        """Register a daemon thread that holds Metal GPU buffers.

        Call this immediately after starting any thread that runs
        mlx_lm.generate() or mlx_vlm.generate().  MetalGuard will
        ensure the thread finishes before any cleanup operation.
        """
        with self._lock:
            self._threads = [t for t in self._threads if t.is_alive()]
            self._threads.append(thread)

    def wait_for_threads(self, timeout: float | None = None) -> int:
        """Block until all registered GPU threads finish.

        Returns the number of threads still alive after timeout (0 = clean).
        """
        timeout = timeout or self._thread_timeout
        with self._lock:
            alive = [t for t in self._threads if t.is_alive()]
        if not alive:
            return 0

        self.breadcrumb(f"WAIT_GPU: {len(alive)} thread(s) alive, waiting {timeout}s")
        for t in alive:
            t.join(timeout=timeout)

        with self._lock:
            still_alive = [t for t in self._threads if t.is_alive()]
            self._threads = still_alive

        if still_alive:
            self.breadcrumb(f"WAIT_GPU: WARNING — {len(still_alive)} still alive")
            log.warning(
                "GPU threads: %d still alive after %.0fs timeout",
                len(still_alive), timeout,
            )
        else:
            self.breadcrumb("WAIT_GPU: all threads finished")
        return len(still_alive)

    # ── GPU cleanup ──────────────────────────────────────────────────

    def flush_gpu(self) -> None:
        """Flush Metal GPU buffers.

        Performs: mx.eval(sync) -> mx.clear_cache().
        MUST only be called after wait_for_threads() — calling this while
        a generate thread is still running WILL trigger the kernel panic.

        No-op if MLX is not installed.
        """
        try:
            import mlx.core as mx
            self.breadcrumb("FLUSH: mx.eval sync")
            mx.eval(mx.zeros(1))
            self.breadcrumb("FLUSH: mx.clear_cache()")
            mx.clear_cache()
            self.breadcrumb("FLUSH: done")
        except (ImportError, AttributeError):
            pass

    def safe_cleanup(self) -> None:
        """Full cleanup: wait for threads -> GC -> flush GPU -> cooldown.

        This is the ONLY correct way to release Metal GPU memory.
        Never call mx.clear_cache() directly — always go through this.
        """
        self.breadcrumb("CLEANUP: START")
        self.wait_for_threads()
        gc.collect()
        self.flush_gpu()
        time.sleep(self._cooldown)
        self.breadcrumb("CLEANUP: DONE")

    @contextmanager
    def guarded_cleanup(self) -> Generator[None, None, None]:
        """Context manager that runs safe_cleanup on exit.

        Usage:
            with metal_guard.guarded_cleanup():
                model_cache.clear()
        """
        try:
            yield
        finally:
            self.safe_cleanup()

    # ── Memory pressure ──────────────────────────────────────────────

    def memory_stats(self) -> MemoryStats:
        """Get current Metal GPU memory stats.

        Returns a zero MemoryStats if MLX is not available.
        """
        try:
            import mlx.core as mx
            info = mx.device_info()
            return MemoryStats(
                active_bytes=mx.get_active_memory(),
                peak_bytes=mx.get_peak_memory(),
                limit_bytes=info.get("max_recommended_working_set_size", 0),
            )
        except (ImportError, AttributeError):
            return MemoryStats()

    def is_pressure_high(
        self,
        threshold_pct: float = 67.0,
        model_name: str = "",
    ) -> bool:
        """Check if Metal peak memory exceeds threshold."""
        stats = self.memory_stats()
        if not stats.limit_bytes:
            return False
        if stats.peak_pct > threshold_pct:
            log.warning(
                "Metal peak %.0f%% (%.1fGB/%.0fGB) — cleanup needed%s",
                stats.peak_pct, stats.peak_gb, stats.limit_gb,
                f" before {model_name}" if model_name else "",
            )
            return True
        return False

    def ensure_headroom(
        self,
        model_name: str = "",
        threshold_pct: float = 67.0,
    ) -> None:
        """Clean up if memory pressure is high, then reset peak counter.

        Call before loading a new model.  If pressure is below threshold,
        this is a no-op (zero overhead on the hot path).
        """
        if not self.is_pressure_high(threshold_pct, model_name):
            return
        self.safe_cleanup()
        try:
            import mlx.core as mx
            mx.reset_peak_memory()
        except (ImportError, AttributeError):
            pass

    def log_memory(self, label: str, model_name: str = "") -> None:
        """Log current Metal memory state.  Lightweight, no cleanup."""
        stats = self.memory_stats()
        if stats.limit_bytes:
            log.info("Metal memory %s %s: %s", label, model_name, stats)

    # ── Breadcrumb ───────────────────────────────────────────────────

    def breadcrumb(self, msg: str) -> None:
        """Write a crash-safe breadcrumb to disk.

        Uses fsync to ensure the line survives a kernel panic.
        After a crash, read the last line of the breadcrumb log to
        identify which Metal operation triggered the panic.
        """
        if not self._breadcrumb_path:
            return
        line = f"[{time.strftime('%H:%M:%S')}] {msg}\n"
        try:
            os.makedirs(os.path.dirname(self._breadcrumb_path), exist_ok=True)
            with open(self._breadcrumb_path, "a") as f:
                f.write(line)
                f.flush()
                os.fsync(f.fileno())
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

metal_guard = MetalGuard()
