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
  - https://github.com/ml-explore/mlx-lm/issues/1015 (generate() OOM crash)
  - https://github.com/ml-explore/mlx-lm/issues/854 (server OOM crash)
  - https://github.com/ml-explore/mlx-lm/issues/1047 (KV cache OOM)

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

    # OOM-protected inference (catches Metal OOM, cleans up, returns None)
    result = metal_guard.oom_protected(generate, model, tokenizer, prompt=p)

    # Pre-load memory check
    if not metal_guard.can_fit(model_size_gb=24.0):
        raise MemoryError("Not enough Metal headroom for 24GB model")

    # Periodic flush for long-running servers
    metal_guard.start_periodic_flush(interval_secs=300)

    # Crash-safe breadcrumb for post-mortem forensics
    metal_guard.breadcrumb("LOAD: my-model-8bit START")

License: MIT
"""

from __future__ import annotations

import gc
import logging
import os
import re
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Generator, TypeVar

__version__ = "0.2.0"

log = logging.getLogger("metal_guard")

T = TypeVar("T")

# Pattern to detect Metal OOM errors from the C++ runtime.
# These surface as RuntimeError in Python with this specific message.
_METAL_OOM_PATTERN = re.compile(
    r"Command buffer execution failed.*Insufficient Memory"
    r"|kIOGPUCommandBufferCallbackErrorOutOfMemory",
)


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
    def available_gb(self) -> float:
        return (self.limit_bytes - self.active_bytes) / 1e9

    @property
    def active_pct(self) -> float:
        return (self.active_bytes / self.limit_bytes * 100) if self.limit_bytes else 0

    @property
    def peak_pct(self) -> float:
        return (self.peak_bytes / self.limit_bytes * 100) if self.limit_bytes else 0

    def __str__(self) -> str:
        return (
            f"active={self.active_gb:.1f}GB peak={self.peak_gb:.1f}GB "
            f"limit={self.limit_gb:.0f}GB avail={self.available_gb:.1f}GB "
            f"(active {self.active_pct:.0f}% / peak {self.peak_pct:.0f}%)"
        )


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class MetalOOMError(RuntimeError):
    """Raised when Metal GPU runs out of memory.

    Unlike the raw C++ RuntimeError, this is catchable and recoverable.
    MetalGuard catches the raw error, cleans up GPU state, and re-raises
    as this so callers can handle it gracefully (e.g. return HTTP 503).
    """

    def __init__(self, message: str, stats: MemoryStats | None = None):
        super().__init__(message)
        self.stats = stats


# ---------------------------------------------------------------------------
# MetalGuard
# ---------------------------------------------------------------------------

class MetalGuard:
    """Centralized Metal GPU safety for MLX on Apple Silicon.

    Five responsibilities:
      1. **Thread tracking** — ensure no GPU thread is alive before cleanup
      2. **Safe cleanup** — gc + GPU sync + cache clear + cooldown
      3. **Memory pressure** — monitor and preemptively clean before overload
      4. **OOM recovery** — catch Metal OOM, cleanup, and raise recoverable error
      5. **Periodic maintenance** — background flush for long-running processes

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
        self._flush_timer: threading.Timer | None = None
        self._flush_interval: float = 0
        self._watchdog_warn_pct: float = 70.0
        self._watchdog_critical_pct: float = 85.0
        self._watchdog_on_critical: Callable[[], None] | None = None
        self._watchdog_baseline: int | None = None

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

    # ── OOM recovery (v0.2) ──────────────────────────────────────────
    # Addresses: mlx-lm#1015 (generate OOM crash), #854 (server OOM crash)

    @staticmethod
    def is_metal_oom(exc: BaseException) -> bool:
        """Check if an exception is a Metal GPU out-of-memory error.

        Metal OOM surfaces as RuntimeError with a specific message pattern
        from the C++ runtime.  This detects it reliably.
        """
        return isinstance(exc, RuntimeError) and bool(
            _METAL_OOM_PATTERN.search(str(exc))
        )

    def oom_protected(
        self,
        fn: Callable[..., T],
        *args: Any,
        max_retries: int = 1,
        **kwargs: Any,
    ) -> T:
        """Run a function with Metal OOM protection.

        If the function raises a Metal OOM error:
          1. Catch it (prevent process termination)
          2. Run safe_cleanup() to recover GPU state
          3. Retry once (if max_retries > 0)
          4. If still OOM, raise MetalOOMError (catchable, not fatal)

        Usage:
            result = metal_guard.oom_protected(
                generate, model, tokenizer, prompt=prompt, max_tokens=4096
            )

        For servers, catch MetalOOMError and return HTTP 503:
            try:
                result = metal_guard.oom_protected(generate, ...)
            except MetalOOMError:
                return Response(status_code=503, body="GPU memory exhausted")
        """
        last_exc: BaseException | None = None

        for attempt in range(max_retries + 1):
            try:
                return fn(*args, **kwargs)
            except RuntimeError as exc:
                if not self.is_metal_oom(exc):
                    raise  # Not OOM — don't swallow other errors
                last_exc = exc
                stats = self.memory_stats()
                log.error(
                    "Metal OOM caught (attempt %d/%d): %s | %s",
                    attempt + 1, max_retries + 1, exc, stats,
                )
                self.breadcrumb(f"OOM_RECOVERY: attempt {attempt + 1}, cleaning up")
                self.safe_cleanup()

                if attempt < max_retries:
                    log.info("Retrying after OOM cleanup...")

        stats = self.memory_stats()
        raise MetalOOMError(
            f"Metal GPU out of memory after {max_retries + 1} attempts. "
            f"Memory: {stats}. Original error: {last_exc}",
            stats=stats,
        )

    @contextmanager
    def oom_protected_context(self) -> Generator[None, None, None]:
        """Context manager version of OOM protection.

        Usage:
            with metal_guard.oom_protected_context():
                result = generate(model, tokenizer, prompt=prompt)
        """
        try:
            yield
        except RuntimeError as exc:
            if not self.is_metal_oom(exc):
                raise
            stats = self.memory_stats()
            log.error("Metal OOM caught in context: %s | %s", exc, stats)
            self.breadcrumb("OOM_RECOVERY: context manager cleanup")
            self.safe_cleanup()
            raise MetalOOMError(
                f"Metal GPU out of memory. Memory: {stats}. Original: {exc}",
                stats=stats,
            ) from exc

    # ── Pre-load memory check (v0.2) ─────────────────────────────────
    # Addresses: mlx-lm#427 (M1 MBA crash), #1047 (KV cache OOM)

    def can_fit(
        self,
        model_size_gb: float,
        overhead_gb: float = 2.0,
    ) -> bool:
        """Check if a model of the given size can fit in available Metal memory.

        Accounts for currently active memory + a safety overhead (default 2GB
        for KV cache, OS, and other allocations).

        Usage:
            if not metal_guard.can_fit(model_size_gb=24.0):
                metal_guard.safe_cleanup()  # Try freeing first
                if not metal_guard.can_fit(model_size_gb=24.0):
                    raise MemoryError("Cannot fit 24GB model")
            model = load("my-24gb-model")
        """
        stats = self.memory_stats()
        if not stats.limit_bytes:
            return True  # Can't check — assume OK

        needed_bytes = (model_size_gb + overhead_gb) * 1e9
        available_bytes = stats.limit_bytes - stats.active_bytes
        fits = available_bytes >= needed_bytes

        if not fits:
            log.warning(
                "Model needs %.1fGB (%.1f + %.1f overhead) but only %.1fGB available "
                "(active=%.1fGB, limit=%.0fGB)",
                model_size_gb + overhead_gb, model_size_gb, overhead_gb,
                available_bytes / 1e9, stats.active_gb, stats.limit_gb,
            )
        return fits

    def require_fit(
        self,
        model_size_gb: float,
        model_name: str = "",
        overhead_gb: float = 2.0,
    ) -> None:
        """Ensure a model can fit, cleaning up first if needed.

        Raises MemoryError if it still doesn't fit after cleanup.

        Usage:
            metal_guard.require_fit(24.0, model_name="Mistral-24B-8bit")
            model = load("Mistral-24B-8bit")
        """
        if self.can_fit(model_size_gb, overhead_gb):
            return

        log.info("Cleaning up to make room for %s (%.1fGB)...", model_name, model_size_gb)
        self.safe_cleanup()

        if not self.can_fit(model_size_gb, overhead_gb):
            stats = self.memory_stats()
            raise MemoryError(
                f"Cannot fit {model_name or 'model'} ({model_size_gb:.1f}GB) "
                f"even after cleanup. Available: {stats.available_gb:.1f}GB, "
                f"needed: {model_size_gb + overhead_gb:.1f}GB"
            )

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

    # ── Periodic flush (v0.2) ────────────────────────────────────────
    # Addresses: mlx-lm#854 (server memory leak), mlx-examples#1124

    def start_periodic_flush(self, interval_secs: float = 300.0) -> None:
        """Start a background timer that flushes GPU cache periodically.

        Prevents memory accumulation in long-running servers/daemons.
        The flush only runs if no GPU threads are active (safe).

        Call stop_periodic_flush() to cancel.

        Usage:
            metal_guard.start_periodic_flush(interval_secs=300)  # Every 5 min
        """
        self.stop_periodic_flush()
        self._flush_interval = interval_secs
        self._schedule_next_flush()
        log.info("Periodic Metal flush started (every %.0fs)", interval_secs)

    def stop_periodic_flush(self) -> None:
        """Stop the background periodic flush timer."""
        if self._flush_timer is not None:
            self._flush_timer.cancel()
            self._flush_timer = None
            self._flush_interval = 0
            log.info("Periodic Metal flush stopped")

    def _schedule_next_flush(self) -> None:
        """Schedule the next periodic flush."""
        if self._flush_interval <= 0:
            return
        self._flush_timer = threading.Timer(
            self._flush_interval, self._periodic_flush_tick,
        )
        self._flush_timer.daemon = True
        self._flush_timer.start()

    def _periodic_flush_tick(self) -> None:
        """Execute one periodic flush cycle."""
        try:
            with self._lock:
                alive = [t for t in self._threads if t.is_alive()]
            if alive:
                log.debug(
                    "Periodic flush skipped: %d GPU thread(s) active", len(alive),
                )
            else:
                stats = self.memory_stats()
                if stats.active_bytes > 0 or stats.peak_pct > 50:
                    log.info(
                        "Periodic flush: %s", stats,
                    )
                    self.flush_gpu()
        except Exception as exc:
            log.debug("Periodic flush error (non-fatal): %s", exc)
        finally:
            self._schedule_next_flush()

    # ── Memory drift watchdog (v0.2) ───────────────────────────────
    # For long-running processes (OpenClaw-style agents, mlx_lm.server,
    # 24/7 daemons) where memory drifts upward over hours/days.

    def start_watchdog(
        self,
        interval_secs: float = 300.0,
        warn_pct: float = 70.0,
        critical_pct: float = 85.0,
        on_critical: Callable[[], None] | None = None,
    ) -> None:
        """Start a background watchdog that monitors memory drift.

        Combines periodic flush with escalating memory pressure response:
          - Below warn_pct: periodic flush only (lightweight)
          - Above warn_pct: flush + log warning
          - Above critical_pct: full safe_cleanup + call on_critical callback

        The on_critical callback is where you put your app-level response:
        e.g. drop the oldest conversation, shrink KV cache, restart worker.

        Usage:
            def on_critical():
                kv_cache.clear()
                log.error("Memory critical — KV cache dropped")

            metal_guard.start_watchdog(
                interval_secs=60,
                warn_pct=70.0,
                critical_pct=85.0,
                on_critical=on_critical,
            )
        """
        self._watchdog_warn_pct = warn_pct
        self._watchdog_critical_pct = critical_pct
        self._watchdog_on_critical = on_critical
        self._watchdog_baseline: int | None = None
        # Reuse periodic flush infrastructure
        self.start_periodic_flush(interval_secs)
        # Override the tick function
        self._periodic_flush_tick = self._watchdog_tick  # type: ignore[assignment]
        log.info(
            "Memory watchdog started (every %.0fs, warn=%.0f%%, critical=%.0f%%)",
            interval_secs, warn_pct, critical_pct,
        )

    def _watchdog_tick(self) -> None:
        """Execute one watchdog cycle with escalating response."""
        try:
            with self._lock:
                alive = [t for t in self._threads if t.is_alive()]
            if alive:
                log.debug("Watchdog skipped: %d GPU thread(s) active", len(alive))
                return

            stats = self.memory_stats()
            if not stats.limit_bytes:
                return

            # Track baseline for drift detection
            if self._watchdog_baseline is None:
                self._watchdog_baseline = stats.active_bytes
                log.info("Watchdog baseline: %.1fGB active", stats.active_gb)

            drift_gb = (stats.active_bytes - self._watchdog_baseline) / 1e9

            if stats.active_pct > self._watchdog_critical_pct:
                log.error(
                    "WATCHDOG CRITICAL: active=%.1fGB (%.0f%%), drift=+%.1fGB — "
                    "full cleanup + callback",
                    stats.active_gb, stats.active_pct, drift_gb,
                )
                self.breadcrumb(
                    f"WATCHDOG: CRITICAL active={stats.active_pct:.0f}% "
                    f"drift={drift_gb:+.1f}GB"
                )
                self.safe_cleanup()
                if self._watchdog_on_critical:
                    try:
                        self._watchdog_on_critical()
                    except Exception as exc:
                        log.error("Watchdog on_critical callback error: %s", exc)
                # Reset baseline after critical cleanup
                new_stats = self.memory_stats()
                self._watchdog_baseline = new_stats.active_bytes

            elif stats.active_pct > self._watchdog_warn_pct:
                log.warning(
                    "WATCHDOG WARN: active=%.1fGB (%.0f%%), drift=+%.1fGB — flushing",
                    stats.active_gb, stats.active_pct, drift_gb,
                )
                self.flush_gpu()

            elif stats.active_bytes > 0 or stats.peak_pct > 50:
                self.flush_gpu()

        except Exception as exc:
            log.debug("Watchdog error (non-fatal): %s", exc)
        finally:
            self._schedule_next_flush()

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
