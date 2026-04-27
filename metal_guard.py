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
  3. Metal CommandBuffer completion error — C++ exception thrown from
     mlx::core::gpu::check_error() inside Metal's GCD CompletionQueueDispatch,
     which cannot be caught by Python (triggers std::terminate → abort).
     Observed in production: SIGABRT on Thread "com.Metal.CompletionQueueDispatch"

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

import datetime
import gc
import glob
import json
import logging
import multiprocessing
import os
import pathlib
import re
import shutil
import signal
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any, Callable, Generator, Iterator, Tuple, TypeVar

__version__ = "0.10.0"

log = logging.getLogger("metal_guard")

T = TypeVar("T")

# ── Apple GPU driver workaround ─────────────────────────────────────────
# Suggested by @zcbenz (MLX maintainer) in mlx#3267:
# Relaxes the command buffer context store timeout to reduce kernel panics
# on long-running GPU workloads.  Zero-cost — just an env var hint to the
# IOGPUFamily driver.  Safe to set unconditionally.
if "AGX_RELAX_CDM_CTXSTORE_TIMEOUT" not in os.environ:
    os.environ["AGX_RELAX_CDM_CTXSTORE_TIMEOUT"] = "1"

# Pattern to detect Metal GPU errors from the C++ runtime.
# Sources:
#   - "Command buffer execution failed...Insufficient Memory" (mlx-lm#883, #1015)
#   - "kIOGPUCommandBufferCallbackErrorOutOfMemory" (mlx-lm#854)
#   - "fPendingMemorySet" (mlx#3346 @yoyaku155 — IOGPUGroupMemory race)
#   - "ImpactingInteractivity" (mlx#3267 @zackedds — GPU watchdog kills
#     MLX when command buffers block WindowServer compositing on MacBook)
_METAL_OOM_PATTERN = re.compile(
    r"Command buffer execution failed.*Insufficient Memory"
    r"|kIOGPUCommandBufferCallbackErrorOutOfMemory"
    r"|fPendingMemorySet"
    r"|ImpactingInteractivity",
    re.IGNORECASE | re.DOTALL,
)


# Kernel-panic classifier.  Maps a human-readable signature name to the
# regex that fingerprints it in `kernel.log` / `log show` output after a
# reboot.  Used by `detect_panic_signature()` so incidents can be tagged
# and triaged rather than lumped together as "panic".
#
# Each entry is keyed by signature name → (regex, short explanation).
# Regexes are case-insensitive and matched with DOTALL so multi-line
# panic backtraces work.
_KERNEL_PANIC_SIGNATURES: dict[str, tuple[re.Pattern[str], str]] = {
    # mlx-lm#883 / #1015 — the original panic reported in summer 2025.
    # Underflow when Metal's IOGPUMemory prepare count drops below zero
    # because a command buffer was released before its dependent ops
    # finished. MetalGuard mitigates via cooldown + thread tracking.
    "prepare_count_underflow": (
        re.compile(
            r"completeMemory\(\)\s*prepare\s*count\s*underflow",
            re.IGNORECASE | re.DOTALL,
        ),
        "IOGPUMemory.cpp:492 — command buffer released before Metal "
        "finished. Reproduces under back-to-back load/unload cycles.",
    ),
    # mlx#3346 @yoyaku155 — second panic signature (IOGPUGroupMemory.cpp:219).
    # Lingering entries in the pending-memory set when a new command
    # buffer is submitted. Typically triggered by rapid model swaps
    # without sufficient cooldown.
    "pending_memory_set": (
        re.compile(
            r"fPendingMemorySet"
            r"|IOGPUGroupMemory\.cpp:219",
            re.IGNORECASE | re.DOTALL,
        ),
        "IOGPUGroupMemory.cpp:219 — pending memory set not drained "
        "before reuse. Mitigation: longer post-unload cooldown.",
    ),
    # mlx#3267 @zcbenz — command-buffer context store timeout on long
    # GPU workloads. Partial fix is AGX_RELAX_CDM_CTXSTORE_TIMEOUT=1
    # (set above at import time).
    "ctxstore_timeout": (
        re.compile(
            r"IOGPUCommandQueue.*context\s*store\s*timeout",
            re.IGNORECASE | re.DOTALL,
        ),
        "AGX command-buffer context store timeout. MetalGuard sets "
        "AGX_RELAX_CDM_CTXSTORE_TIMEOUT=1 at import.",
    ),
    # Generic OOM / command buffer execution failure from the C++ layer.
    # Falls through here if none of the more specific patterns match but
    # the panic still talks about IOGPU + OOM wording.
    "metal_oom": (
        re.compile(
            r"kIOGPUCommandBufferCallbackErrorOutOfMemory"
            r"|Command\s+buffer\s+execution\s+failed.*Insufficient\s+Memory",
            re.IGNORECASE | re.DOTALL,
        ),
        "Metal OOM at command-buffer callback. Raise headroom or "
        "reduce active model footprint.",
    ),
}


def detect_panic_signature(text: str) -> tuple[str | None, str | None]:
    """Classify a kernel-panic log snippet into a known signature.

    Args:
        text: Raw text from ``log show --predicate 'eventMessage CONTAINS "panic"'``,
            ``sudo dmesg``, or the contents of ``/Library/Logs/DiagnosticReports/*.panic``.

    Returns:
        ``(signature_name, explanation)`` — or ``(None, None)`` if no known
        pattern matched. Callers can route unknown panics to a catch-all
        bucket.
    """
    for name, (pattern, explanation) in _KERNEL_PANIC_SIGNATURES.items():
        if pattern.search(text):
            return name, explanation
    return None, None


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
# KV cache pressure helpers
# ---------------------------------------------------------------------------

def kv_cache_clear_on_pressure(
    available_gb: float,
    growth_rate_gb_per_min: float,
) -> None:
    """Default ``on_pressure`` callback for :meth:`MetalGuard.start_kv_cache_monitor`.

    Clears MLX's cache when headroom is low or growth is too fast. Safe
    to pass directly::

        metal_guard.start_kv_cache_monitor(
            on_pressure=kv_cache_clear_on_pressure,
        )

    Does NOT evict application-level KV caches (those are held by the
    caller's server process — only it can invalidate them). This only
    asks Metal to return as much unused GPU memory as possible.
    """
    try:
        import mlx.core as mx
    except ImportError:
        return

    log.warning(
        "KV pressure callback firing: avail=%.1fGB growth=%.2fGB/min — "
        "calling mx.clear_cache()",
        available_gb, growth_rate_gb_per_min,
    )
    try:
        mx.clear_cache()
    except Exception as exc:
        log.error("mx.clear_cache() from pressure callback failed: %s", exc)


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
        self._lock = threading.RLock()  # Reentrant — safe if cleanup calls nest
        self._cooldown = cooldown_secs
        self._thread_timeout = thread_timeout_secs
        self._breadcrumb_path = breadcrumb_path
        self._flush_timer: threading.Timer | None = None
        self._flush_interval: float = 0
        self._tick_fn: Callable[[], None] = self._periodic_flush_tick
        self._watchdog_warn_pct: float = 70.0
        self._watchdog_critical_pct: float = 85.0
        self._watchdog_on_critical: Callable[[], None] | None = None
        self._watchdog_baseline: int | None = None
        # KV cache monitor state (initialized by start_kv_cache_monitor)
        self._kv_growth_rate_warn: float = 1.0
        self._kv_headroom_gb: float = 4.0
        self._kv_on_pressure: Callable[[float, float], None] | None = None
        self._kv_samples: list[tuple[float, float]] = []
        self._kv_interval: float = 30.0
        self._kv_timer: threading.Timer | None = None

    # ── Hardware-aware auto-configuration ────────────────────────────
    # Addresses: community feedback that users with different hardware
    # (8GB MBA to 512GB Mac Studio) need different safety thresholds.

    @staticmethod
    def detect_hardware() -> dict[str, Any]:
        """Detect Apple Silicon hardware and return safety-relevant info.

        Returns a dict with:
          - gpu_memory_gb: Total GPU memory (unified memory on Apple Silicon)
          - chip: Chip name (e.g. "Apple M1 Ultra")
          - recommended_working_set_gb: Apple's recommended working set limit
          - tier: "low" (8-16GB), "mid" (32-64GB), "high" (96-512GB)

        Works without MLX installed (falls back to sysctl).
        """
        gpu_memory_gb = 0.0
        chip = "unknown"
        recommended_gb = 0.0

        # Try MLX first (most accurate)
        try:
            import mlx.core as mx
            info = mx.device_info()
            recommended = info.get("max_recommended_working_set_size", 0)
            recommended_gb = recommended / (1024 ** 3)
            # Total memory ≈ recommended / 0.75 (Apple reserves ~25%)
            gpu_memory_gb = recommended_gb / 0.75
        except (ImportError, Exception):
            pass

        # Fallback: sysctl for total memory
        if gpu_memory_gb == 0:
            try:
                import subprocess
                result = subprocess.run(
                    ["sysctl", "-n", "hw.memsize"],
                    capture_output=True, text=True, timeout=5,
                )
                if result.returncode == 0:
                    gpu_memory_gb = int(result.stdout.strip()) / (1024 ** 3)
                    recommended_gb = gpu_memory_gb * 0.75
            except Exception:
                pass

        # Detect chip name
        try:
            import subprocess
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                chip = result.stdout.strip()
        except Exception:
            pass

        # Classify tier
        if gpu_memory_gb <= 16:
            tier = "low"
        elif gpu_memory_gb <= 64:
            tier = "mid"
        else:
            tier = "high"

        # IOGPUFamily kext version. mlx#3186 pins the panic fault to
        # this specific kext, so recording the driver revision gives
        # forensic context for future crash correlation. Never raises
        # — returns None on any failure. Forward reference is fine;
        # the function is module-level and resolved lazily at call
        # time.
        gpu_driver_version: str | None = None
        try:
            gpu_driver_version = read_gpu_driver_version()
        except Exception as exc:  # pragma: no cover — defensive
            log.debug("read_gpu_driver_version failed: %s", exc)

        return {
            "gpu_memory_gb": round(gpu_memory_gb, 1),
            "chip": chip,
            "recommended_working_set_gb": round(recommended_gb, 1),
            "tier": tier,
            "gpu_driver_version": gpu_driver_version,
        }

    @classmethod
    def recommended_config(cls) -> dict[str, Any]:
        """Return hardware-appropriate configuration values.

        Call this to get safe defaults for your machine, then pass them
        to the relevant MetalGuard methods.

        Returns a dict with recommended values for:
          - watchdog_warn_pct / watchdog_critical_pct
          - kv_headroom_gb / kv_growth_rate_warn
          - cooldown_secs
          - max_concurrent_models
          - overhead_gb (for require_fit)

        Example::

            config = MetalGuard.recommended_config()
            print(f"Your {config['chip']} ({config['gpu_memory_gb']}GB)")
            print(f"Recommended watchdog: warn={config['watchdog_warn_pct']}%")

            metal_guard.start_watchdog(
                warn_pct=config["watchdog_warn_pct"],
                critical_pct=config["watchdog_critical_pct"],
            )
            metal_guard.start_kv_cache_monitor(
                headroom_gb=config["kv_headroom_gb"],
            )
        """
        hw = cls.detect_hardware()
        mem = hw["gpu_memory_gb"]
        tier = hw["tier"]

        if tier == "low":  # 8-16GB (MBA, base MBP)
            config = {
                "watchdog_warn_pct": 60.0,
                "watchdog_critical_pct": 75.0,
                "kv_headroom_gb": 2.0,
                "kv_growth_rate_warn": 0.5,
                "cooldown_secs": 3.0,
                "max_concurrent_models": 1,
                "overhead_gb": 3.0,
            }
        elif tier == "mid":  # 32-64GB (Mac Studio M1/M2/M4, MBP Max)
            config = {
                "watchdog_warn_pct": 67.0,
                "watchdog_critical_pct": 82.0,
                "kv_headroom_gb": 4.0,
                "kv_growth_rate_warn": 1.0,
                "cooldown_secs": 2.0,
                "max_concurrent_models": 2,
                "overhead_gb": 2.0,
            }
        else:  # 96-512GB (Ultra, Max Pro)
            config = {
                "watchdog_warn_pct": 70.0,
                "watchdog_critical_pct": 85.0,
                "kv_headroom_gb": 8.0,
                "kv_growth_rate_warn": 2.0,
                "cooldown_secs": 2.0,
                "max_concurrent_models": 3,
                "overhead_gb": 2.0,
            }

        config.update(hw)
        return config

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
        timeout = timeout if timeout is not None else self._thread_timeout
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
        except ImportError:
            return
        self.breadcrumb("FLUSH: mx.eval sync")
        mx.eval(mx.zeros(1))
        self.breadcrumb("FLUSH: mx.clear_cache()")
        mx.clear_cache()
        self.breadcrumb("FLUSH: done")

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

        Note: model_size_gb uses decimal gigabytes (1 GB = 1e9 bytes), which
        matches mx.get_active_memory() and HuggingFace model card sizes.
        Binary GiB (1024^3) would overestimate by ~7%.

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
        *,
        cache_clear_cb: Callable[[], None] | None = None,
        escalated_cooldown_sec: float = 0.0,
    ) -> None:
        """Ensure a model can fit, cleaning up first if needed.

        Two-tier retry strategy:

        1. **Standard** — if ``can_fit`` fails, run ``safe_cleanup`` once and
           re-check. This handles transient GPU buffer pressure where a
           ``wait_for_threads + gc + flush + internal cooldown`` is enough.

        2. **Escalated** (opt-in via ``escalated_cooldown_sec > 0``) — if the
           standard retry still can't fit, invoke the caller-supplied
           ``cache_clear_cb`` to drop Python-side references (their model
           dict), run a second ``safe_cleanup``, reset ``mlx.core.reset_peak_memory``,
           sleep ``escalated_cooldown_sec`` to let Metal's internal pool
           actually return pages to the OS, then re-check a final time.

        Escalation is **opt-in** because the cache clear callback is
        application-specific (MetalGuard has no global knowledge of your
        ModelCache implementation). Pass ``escalated_cooldown_sec=5.0`` and
        ``cache_clear_cb=my_cache.clear`` to enable.

        Raises ``MemoryError`` if the model still doesn't fit after the
        highest-level retry that was enabled.

        Usage::

            # Standard (no escalation):
            metal_guard.require_fit(24.0, model_name="Mistral-24B-8bit")

            # Escalated retry for tight-memory ensembles:
            metal_guard.require_fit(
                24.0,
                model_name="Mistral-24B-8bit",
                cache_clear_cb=my_model_cache.clear,
                escalated_cooldown_sec=5.0,
            )
        """
        if self.can_fit(model_size_gb, overhead_gb):
            return

        log.info("Cleaning up to make room for %s (%.1fGB)...", model_name, model_size_gb)
        self.safe_cleanup()

        if self.can_fit(model_size_gb, overhead_gb):
            return

        # Standard cleanup insufficient. Raise now unless caller opted into
        # escalated retry.
        if escalated_cooldown_sec <= 0:
            stats = self.memory_stats()
            raise MemoryError(
                f"Cannot fit {model_name or 'model'} ({model_size_gb:.1f}GB) "
                f"even after cleanup. Available: {stats.available_gb:.1f}GB, "
                f"needed: {model_size_gb + overhead_gb:.1f}GB"
            )

        # Escalated retry: drop caller's cache references, second cleanup,
        # longer sleep to force Metal page release, re-check.
        log.warning(
            "Standard cleanup insufficient for %s — escalating "
            "(cache_clear_cb=%s, cooldown=%.1fs)",
            model_name or "model",
            "yes" if cache_clear_cb else "no",
            escalated_cooldown_sec,
        )
        if cache_clear_cb is not None:
            try:
                cache_clear_cb()
            except Exception as exc:  # noqa: BLE001 — caller failure must not abort escalation
                log.warning("cache_clear_cb raised: %s (continuing escalation)", exc)
        self.safe_cleanup()
        try:
            import mlx.core as mx
            mx.reset_peak_memory()
        except (ImportError, AttributeError):
            pass
        time.sleep(escalated_cooldown_sec)

        if self.can_fit(model_size_gb, overhead_gb):
            log.info(
                "Escalated cleanup succeeded — %s (%.1fGB) now fits",
                model_name or "model", model_size_gb,
            )
            return

        stats = self.memory_stats()
        raise MemoryError(
            f"Cannot fit {model_name or 'model'} ({model_size_gb:.1f}GB) "
            f"even after escalated cleanup ({escalated_cooldown_sec}s cooldown). "
            f"active={stats.active_gb:.1f}GB "
            f"available={stats.available_gb:.1f}GB "
            f"needed={model_size_gb + overhead_gb:.1f}GB. "
            f"Consider reducing max loaded models or switching to a smaller quant."
        )

    @staticmethod
    def estimate_model_size_from_name(model_name: str) -> float | None:
        """Heuristically estimate a model's Metal footprint from its name.

        Parses patterns like ``Mistral-Small-24B-8bit`` → ``24 * 1.0 = 24 GB``
        or ``Phi-4-mini-instruct-4bit`` → ``4 * 0.5 = 2 GB`` (mini-class fallback
        estimate). Returns ``None`` when no size hint is found — callers should
        treat that as "unknown, skip the fit check" and fall back to the
        threshold-based ``ensure_headroom`` path.

        Designed to pair with ``require_fit`` so multi-model batch workloads
        (e.g. an MLX ensemble that loads several debaters sequentially) can
        proactively evict cached models before hitting the Metal working-set
        limit, instead of failing with an uncaught ``std::runtime_error``
        from the Metal completion queue.

        Supported param-count patterns (case-insensitive):
          <N>B    → N billion parameters   (e.g. "24B", "31B")
          <N>M    → N million parameters (×0.001 GB)
          mini    → 4 billion fallback (phi-4-mini class)
          small   → 7 billion fallback
          medium  → 13 billion fallback
          large   → 70 billion fallback
          xl      → 13 billion fallback

        Supported quantization multipliers (bytes per parameter, decimal GB):
          16bit / fp16 / bf16  → 2.0
          8bit / int8          → 1.0
          6bit                 → 0.75
          4bit / int4 / q4 / mxfp4 / TQ4 / UD-MLX-4bit → 0.5
          3bit / int3 / TQ3    → 0.375
          2bit / int2          → 0.25
          (default when unspecified: 2.0, assumes fp16)

        TurboQuant (TQ) models use quantized KV cache compression. The model
        weights themselves follow the same per-parameter sizing, but TQ models
        can sustain much longer context windows (50K-200K tokens) due to
        compressed KV cache.  The estimator reports *model weight* footprint
        only — actual runtime memory will be higher with long contexts.

        The result is a rough upper bound intended for pre-load gating —
        callers should not use it for precise accounting.

        Usage::

            size = MetalGuard.estimate_model_size_from_name(
                "mlx-community/Mistral-Small-24B-8bit"
            )
            if size is not None:
                metal_guard.require_fit(size, model_name="Mistral-24B")
            model = load("mlx-community/Mistral-Small-24B-8bit")

        """
        if not model_name:
            return None
        name = model_name.lower()

        # Quantization multiplier (bytes per param)
        bit_mult: float | None = None
        for pat, mult in (
            (r"16bit|fp16|bf16", 2.0),
            (r"8bit|int8", 1.0),
            (r"6bit", 0.75),
            (r"4bit|int4|(?<![a-z])q4|mxfp4|tq4|ud-mlx-4bit", 0.5),
            (r"3bit|int3|tq3", 0.375),
            (r"2bit|int2", 0.25),
        ):
            if re.search(pat, name):
                bit_mult = mult
                break

        # Parameter count (billions)
        # Explicit B/M suffix: "24b", "31b", "350m" — word-boundary anchors
        # require the token to be standalone, which rejects things like
        # "8bit" (where 'b' is inside "bit") and "some-24beauty" while
        # still matching "24b", "1.5B", "350m" when they appear as full
        # tokens. More robust than the earlier (?![a-z]) lookahead.
        params_b: float | None = None
        match = re.search(r"\b(\d+(?:\.\d+)?)\s*([bm])\b", name)
        if match:
            value = float(match.group(1))
            unit = match.group(2)
            params_b = value if unit == "b" else value / 1000.0
        else:
            # Size class fallback when the name has a keyword but no explicit
            # parameter count (e.g. "phi-4-mini-instruct-4bit").
            for pat, size in (
                ("mini", 4.0),
                ("small", 7.0),
                ("medium", 13.0),
                ("large", 70.0),
                ("xl", 13.0),
            ):
                if pat in name:
                    params_b = size
                    break

        if params_b is None:
            # Without a param count, we can't estimate at all — let the
            # caller fall back to ensure_headroom threshold checks.
            return None

        # Default to fp16 (conservative upper bound) when bits are unspecified.
        if bit_mult is None:
            bit_mult = 2.0

        return params_b * bit_mult

    # ── Memory pressure ──────────────────────────────────────────────

    def memory_stats(self) -> MemoryStats:
        """Get current Metal GPU memory stats.

        Returns a zero MemoryStats if MLX is not available.
        """
        try:
            import mlx.core as mx
        except ImportError:
            return MemoryStats()
        info = mx.device_info()
        return MemoryStats(
            active_bytes=mx.get_active_memory(),
            peak_bytes=mx.get_peak_memory(),
            limit_bytes=info.get("max_recommended_working_set_size", 0),
        )

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
        self._tick_fn = self._periodic_flush_tick
        self._schedule_next_flush()
        log.info("Periodic Metal flush started (every %.0fs)", interval_secs)

    def stop_periodic_flush(self) -> None:
        """Stop the background periodic flush timer."""
        if self._flush_timer is not None:
            self._flush_timer.cancel()
            self._flush_timer = None
        self._flush_interval = 0
        self._tick_fn = self._periodic_flush_tick  # Reset to default

    def _schedule_next_flush(self) -> None:
        """Schedule the next periodic flush."""
        if self._flush_interval <= 0:
            return
        self._flush_timer = threading.Timer(
            self._flush_interval, self._tick_fn,
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
        self._watchdog_baseline = None
        # Reuse periodic flush infrastructure with watchdog tick
        self.stop_periodic_flush()
        self._flush_interval = interval_secs
        self._tick_fn = self._watchdog_tick
        self._schedule_next_flush()
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

    # ── KV cache memory monitor (v0.3.1) ────────────────────────────
    # Addresses: mlx-lm#1047 (KV cache OOM on 512GB Mac Studio)
    # KV cache grows unbounded during long conversations. Standard memory
    # watchdog only sees total Metal active bytes, but KV cache growth
    # rate is the leading indicator of imminent OOM. This monitor tracks
    # the growth rate and fires a callback before the cache fills memory.

    def start_kv_cache_monitor(
        self,
        interval_secs: float = 30.0,
        growth_rate_warn_gb_per_min: float = 1.0,
        headroom_gb: float = 4.0,
        on_pressure: Callable[[float, float], None] | None = None,
    ) -> None:
        """Start a background monitor that tracks memory growth rate.

        Designed for long-running servers (mlx_lm.server, OpenClaw) where
        KV cache grows unbounded across conversations.  Detects rapid
        memory growth *before* it hits OOM, giving the application time
        to rotate caches or reject new requests.

        Args:
            interval_secs: How often to sample memory (default 30s).
            growth_rate_warn_gb_per_min: Warn when memory is growing
                faster than this rate (GB/min).  Default 1.0 GB/min.
            headroom_gb: Trigger on_pressure when available memory drops
                below this threshold.  Default 4.0 GB.
            on_pressure: Callback(available_gb, growth_rate_gb_per_min)
                called when headroom is low or growth rate is high.
                Typical action: clear KV cache, reject new requests, or
                restart the inference server gracefully.

        Usage::

            def handle_pressure(available_gb, growth_rate):
                log.warning("KV cache pressure: %.1fGB free, growing %.1fGB/min",
                            available_gb, growth_rate)
                kv_cache.clear()

            metal_guard.start_kv_cache_monitor(
                interval_secs=30,
                headroom_gb=8.0,
                on_pressure=handle_pressure,
            )
        """
        self._kv_growth_rate_warn = growth_rate_warn_gb_per_min
        self._kv_headroom_gb = headroom_gb
        self._kv_on_pressure = on_pressure
        self._kv_samples: list[tuple[float, float]] = []  # (timestamp, active_gb)
        self._kv_interval = interval_secs
        self._kv_timer: threading.Timer | None = None
        self._schedule_kv_tick()
        log.info(
            "KV cache monitor started (every %.0fs, warn_rate=%.1fGB/min, "
            "headroom=%.1fGB)",
            interval_secs, growth_rate_warn_gb_per_min, headroom_gb,
        )

    def stop_kv_cache_monitor(self) -> None:
        """Stop the KV cache monitor."""
        if self._kv_timer is not None:
            self._kv_timer.cancel()
            self._kv_timer = None
            log.info("KV cache monitor stopped")

    def _schedule_kv_tick(self) -> None:
        self._kv_timer = threading.Timer(self._kv_interval, self._kv_tick)
        self._kv_timer.daemon = True
        self._kv_timer.start()

    def _kv_tick(self) -> None:
        try:
            stats = self.memory_stats()
            now = time.monotonic()
            active_gb = stats.active_gb
            available_gb = stats.available_gb

            # Keep a sliding window of samples (last 5 minutes)
            self._kv_samples.append((now, active_gb))
            cutoff = now - 300  # 5-minute window
            self._kv_samples = [
                (t, v) for t, v in self._kv_samples if t >= cutoff
            ]

            # Calculate growth rate (GB/min) via linear slope
            growth_rate = 0.0
            if len(self._kv_samples) >= 2:
                oldest_t, oldest_v = self._kv_samples[0]
                elapsed_min = (now - oldest_t) / 60.0
                if elapsed_min > 0.1:
                    growth_rate = (active_gb - oldest_v) / elapsed_min

            # Check conditions
            pressure = False
            if available_gb < self._kv_headroom_gb:
                log.warning(
                    "KV MONITOR: low headroom %.1fGB (threshold %.1fGB), "
                    "growth_rate=%.2fGB/min",
                    available_gb, self._kv_headroom_gb, growth_rate,
                )
                self.breadcrumb(
                    f"KV_MONITOR: LOW_HEADROOM available={available_gb:.1f}GB "
                    f"rate={growth_rate:.2f}GB/min"
                )
                pressure = True

            if growth_rate > self._kv_growth_rate_warn:
                log.warning(
                    "KV MONITOR: rapid growth %.2fGB/min (threshold %.1f), "
                    "available=%.1fGB",
                    growth_rate, self._kv_growth_rate_warn, available_gb,
                )
                if not pressure:
                    self.breadcrumb(
                        f"KV_MONITOR: RAPID_GROWTH rate={growth_rate:.2f}GB/min "
                        f"available={available_gb:.1f}GB"
                    )
                pressure = True

            if pressure and self._kv_on_pressure:
                try:
                    self._kv_on_pressure(available_gb, growth_rate)
                except Exception as exc:
                    log.error("KV monitor on_pressure callback error: %s", exc)

        except Exception as exc:
            log.debug("KV monitor error (non-fatal): %s", exc)
        finally:
            self._schedule_kv_tick()

    # ── Pre-generate Metal health probe ─────────────────────────────

    def probe_metal_health(self) -> bool:
        """Run a tiny Metal operation to verify the command queue is alive.

        Call this before starting a generate() call. If the Metal driver
        is in a bad state from a prior crash (stale command queue, leaked
        buffers), this probe will crash *here* rather than during a long
        generate that has already consumed minutes of work.

        Returns True if the probe succeeded, False if MLX is not installed.
        Raises whatever Metal/MLX raises if the GPU is unhealthy — callers
        should wrap this in try/except if they want to handle the failure.

        Introduced after observing production SIGABRT (2026-04-12 18:30)
        where ``mlx::core::gpu::check_error(MTL::CommandBuffer*)`` threw a
        C++ exception on the Metal CompletionQueueDispatch queue. A health
        probe before generate would have caught this at a controlled point
        instead of mid-inference.

        Usage::

            metal_guard.probe_metal_health()  # crash here, not mid-generate
            result = generate(model, tokenizer, prompt=prompt)
        """
        try:
            import mlx.core as mx
        except ImportError:
            return False
        self.breadcrumb("PROBE: Metal health check START")
        mx.eval(mx.zeros(1))  # ~1ms — exercises command queue round-trip
        self.breadcrumb("PROBE: Metal health check OK")
        return True

    # ── SIGABRT signal handler (crash forensics) ─────────────────────

    def install_abort_handler(self) -> None:
        """Install a SIGABRT handler for crash forensics.

        When MLX's ``check_error(MTL::CommandBuffer*)`` detects a Metal
        command buffer error, it throws a C++ exception from inside the
        Metal GCD CompletionQueueDispatch queue. Since this happens outside
        any Python try/except scope, the C++ runtime calls
        ``std::terminate`` → ``abort()`` → SIGABRT.

        Python **cannot prevent** this crash (the C++ throw → terminate
        path is not catchable from Python), but it **can** install a
        ``signal.SIGABRT`` handler that runs before the process dies.

        This handler:
          1. Writes a breadcrumb with the last known context
          2. Logs the event at CRITICAL level
          3. Does NOT attempt recovery (Metal state is corrupt)
          4. Re-raises SIGABRT with the default handler to produce a
             proper core dump / crash report

        Call this once at process startup (before any MLX work)::

            metal_guard.install_abort_handler()

        .. note::

           The handler may not run in all cases — if ``abort()`` is called
           from a non-main thread (which is the case for Metal's
           CompletionQueueDispatch), Python's signal handling has
           limitations. On macOS, ``signal.signal`` handlers are only
           guaranteed to run on the main thread. For GCD queue crashes,
           the breadcrumb from ``probe_metal_health()`` or the last
           ``breadcrumb()`` call is more reliable for forensics.
        """
        self._original_sigabrt = signal.getsignal(signal.SIGABRT)

        def _abort_handler(signum: int, frame: Any) -> None:
            # Best-effort forensics — Metal state is already corrupt.
            try:
                self.breadcrumb(
                    "SIGABRT: Metal command buffer error detected. "
                    "This is a C++ exception from mlx::core::gpu::check_error() "
                    "on the Metal CompletionQueueDispatch queue. "
                    "Cannot recover — writing forensic breadcrumb."
                )
                log.critical(
                    "SIGABRT caught by MetalGuard. Metal GPU state is corrupt. "
                    "Last breadcrumb written. Process will terminate."
                )
            except Exception:
                pass  # Cannot risk another exception during signal handling
            # Restore default handler and re-raise for proper crash report
            signal.signal(signal.SIGABRT, signal.SIG_DFL)
            os.kill(os.getpid(), signal.SIGABRT)

        signal.signal(signal.SIGABRT, _abort_handler)
        self.breadcrumb("SIGABRT handler installed")
        log.info("MetalGuard SIGABRT handler installed (crash forensics)")

    # ── Breadcrumb ───────────────────────────────────────────────────

    def breadcrumb(self, msg: str) -> None:
        """Write a crash-safe breadcrumb to disk.

        Uses fsync to ensure the line survives a kernel panic.
        After a crash, read the last line of the breadcrumb log to
        identify which Metal operation triggered the panic.
        """
        if not self._breadcrumb_path:
            return
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n"
        try:
            dirname = os.path.dirname(self._breadcrumb_path)
            if dirname:
                os.makedirs(dirname, exist_ok=True)
            with open(self._breadcrumb_path, "a") as f:
                f.write(line)
                f.flush()
                os.fsync(f.fileno())
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Cross-process mutual exclusion (Layer 8)
# ---------------------------------------------------------------------------

# Lock file path — configurable via MLX_LOCK_PATH env var.
# Default: ~/.metal-guard/locks/mlx_exclusive.lock
_LOCK_PATH_ENV = os.getenv("MLX_LOCK_PATH")
_PROCESS_LOCK_PATH = (
    Path(_LOCK_PATH_ENV).expanduser()
    if _LOCK_PATH_ENV
    else Path.home() / ".metal-guard" / "locks" / "mlx_exclusive.lock"
)

# How long to wait for the previous lock holder to exit after receiving
# SIGTERM during a ``force=True`` acquire. A well-behaved MLX process
# takes a few seconds to flush Metal buffers and run atexit hooks.
# Anything beyond this is hung or actively ignoring SIGTERM, in which
# case we abort rather than silently unlink — unlinking while the peer
# still holds Metal buffers is the kernel-panic path (see FORCE hardening
# notes in CHANGELOG v0.6.0). Tunable via env var for tests / tight CI.
_FORCE_WAIT_SEC = float(os.getenv("MLX_FORCE_WAIT_SEC", "30"))
_FORCE_POLL_INTERVAL_SEC = 0.25

# After a FORCE acquire has SIGTERM'd the previous holder and confirmed
# it exited, sleep this long before returning — gives the kernel's Metal
# buffer GC a window to release the peer's VRAM allocations before the
# new process starts loading models into the same GPU. Without this gap
# the new model load can race the buffer GC and still recreate the panic
# conditions. Only applies when we actually reclaimed from a live peer;
# normal (lock-free) acquires stay zero-latency. Set to 0 in tests.
_RECLAIM_COOLDOWN_SEC = float(os.getenv("MLX_RECLAIM_COOLDOWN_SEC", "8"))


class MLXLockConflict(RuntimeError):
    """Raised when another live process already holds the MLX lock.

    The exception message includes the holder's label, pid, timestamp,
    and cmdline so operators can decide whether to wait or intervene.
    The holder info is exposed via ``.holder`` as a dict.

    When raised from the ``force=True`` path, ``.holder`` may include:
      * ``force_timeout=True`` — SIGTERM was delivered but the peer did
        not exit within ``MLX_FORCE_WAIT_SEC``. Lock deliberately left
        intact so that Metal buffers are not double-allocated.
      * ``force_permission_denied=True`` — we lack permission to signal
        the peer (e.g. pid 1, different user). Lock deliberately left
        intact. Caller should escalate manually rather than retry.
    """

    def __init__(self, holder: dict[str, Any]):
        self.holder = holder
        label = holder.get("label", "unknown")
        pid = holder.get("pid", "?")
        started = holder.get("started_at", "?")
        cmdline = holder.get("cmdline", "")
        super().__init__(
            f"MLX lock held by {label} (pid={pid}, since {started}). "
            f"cmdline: {cmdline!r}. "
            f"Concurrent MLX workloads trigger Metal kernel panics — "
            f"wait for the other process to finish, or pass force=True "
            f"if you know the other process is hung."
        )


def _is_pid_alive(pid: int) -> bool:
    """True if ``pid`` is a running, non-zombie process.

    A process that has called ``exit()`` but has not yet been reaped by
    its parent shows up as a zombie in ``ps``. ``os.kill(pid, 0)``
    succeeds for zombies, but they have already released every kernel
    resource including Metal buffers. For the purposes of the
    kernel-panic invariant ("is it safe to take this MLX lock?"),
    zombies must be treated as dead — otherwise FORCE override would
    block forever waiting for a peer whose Metal buffers are already
    freed.
    """
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but owned by another user. We cannot inspect
        # its state, so conservatively say alive.
        return True
    if _is_zombie(pid):
        return False
    return True


def _is_zombie(pid: int) -> bool:
    """Return True if ``ps`` reports the pid is a zombie (state 'Z').

    macOS and Linux both expose this via ``ps -p <pid> -o state=``; the
    first character is the process state, 'Z' for zombies. Falls back
    to False on exotic platforms without ``ps`` — the conservative
    choice, preserving pre-v0.6.0 behaviour.
    """
    try:
        import subprocess

        out = subprocess.run(
            ["ps", "-p", str(pid), "-o", "state="],
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (FileNotFoundError, OSError):
        return False
    except Exception as exc:  # noqa: BLE001 — diagnostic fallback
        log.debug("_is_zombie: ps failed for pid %d: %s", pid, exc)
        return False
    state = out.stdout.strip()
    return bool(state) and state[0].upper() == "Z"


def _force_terminate_and_wait(existing: dict[str, Any]) -> None:
    """SIGTERM the previous lock holder and wait for it to exit.

    Raises ``MLXLockConflict`` when the holder does not exit within
    ``_FORCE_WAIT_SEC`` or when we lack permission to signal it. The
    caller only unlinks the lock after this returns cleanly —
    unlinking while the holder is still alive is the kernel-panic
    path (Metal buffers double-allocated across two live MLX
    processes → ``IOGPUMemory.cpp:492`` underflow).
    """
    existing_pid = existing.get("pid")
    if not isinstance(existing_pid, int) or existing_pid <= 0:
        log.warning(
            "_force_terminate_and_wait: invalid pid %r in lock, "
            "unlinking anyway", existing_pid,
        )
        return

    log.warning(
        "acquire_mlx_lock: FORCE requested, sending SIGTERM to %s "
        "(pid %s) and waiting up to %.1fs for Metal buffer release",
        existing.get("label", "?"), existing_pid, _FORCE_WAIT_SEC,
    )
    try:
        os.kill(existing_pid, signal.SIGTERM)
    except ProcessLookupError:
        # Already dead — nothing to wait for, safe to unlink.
        log.info(
            "_force_terminate_and_wait: pid %d already exited before SIGTERM",
            existing_pid,
        )
        return
    except PermissionError as exc:
        # Process exists but we cannot signal it. We must not unlink
        # because we cannot guarantee the peer released Metal buffers.
        raise MLXLockConflict({
            **existing,
            "force_permission_denied": True,
            "message": (
                f"FORCE override cannot signal pid {existing_pid}: {exc}. "
                f"Peer's MLX buffer state is unknown — refusing to unlink "
                f"the lock to avoid a kernel panic. Stop the peer "
                f"manually, or run with elevated privileges if you are "
                f"certain it is safe to kill."
            ),
        }) from exc

    deadline = time.monotonic() + _FORCE_WAIT_SEC
    while time.monotonic() < deadline:
        if not _is_pid_alive(existing_pid):
            log.info(
                "_force_terminate_and_wait: pid %d exited after SIGTERM",
                existing_pid,
            )
            return
        time.sleep(_FORCE_POLL_INTERVAL_SEC)

    raise MLXLockConflict({
        **existing,
        "force_timeout": True,
        "message": (
            f"FORCE override sent SIGTERM to pid {existing_pid} but it "
            f"did not exit within {_FORCE_WAIT_SEC:.1f}s. MLX buffers "
            f"are still allocated on the GPU — aborting to avoid a "
            f"kernel panic. Kill pid {existing_pid} manually (SIGKILL) "
            f"and retry."
        ),
    })


def read_mlx_lock() -> dict[str, Any] | None:
    """Return the current lock holder's info dict, or None if free.

    Self-healing: if the recorded pid is dead (including zombies that
    have released their Metal buffers), the stale lock is removed and
    None is returned.
    """
    if not _PROCESS_LOCK_PATH.exists():
        return None
    try:
        raw = _PROCESS_LOCK_PATH.read_text()
    except OSError:
        return None
    try:
        info = json.loads(raw)
    except json.JSONDecodeError:
        _PROCESS_LOCK_PATH.unlink(missing_ok=True)
        return None
    pid = info.get("pid")
    if not isinstance(pid, int) or not _is_pid_alive(pid):
        _PROCESS_LOCK_PATH.unlink(missing_ok=True)
        return None
    return info


def acquire_mlx_lock(label: str, *, force: bool = False) -> dict[str, Any]:
    """Acquire the cross-process MLX exclusive lock.

    Raises ``MLXLockConflict`` when another live process holds the lock
    and ``force=False``.

    When ``force=True`` (v0.6.0 hardened semantics): sends ``SIGTERM``
    to the current holder, polls up to ``MLX_FORCE_WAIT_SEC`` seconds
    for it to exit, then reclaims the lock. Raises ``MLXLockConflict``
    with ``holder["force_timeout"]=True`` if the peer does not exit,
    or ``holder["force_permission_denied"]=True`` if the SIGTERM was
    denied. **Never unlinks while the holder is still alive** — that
    is the documented kernel-panic path. After a successful reclaim
    from a live peer, sleeps ``MLX_RECLAIM_COOLDOWN_SEC`` to allow
    Metal buffer GC to finish before returning.

    Args:
        label: short identifier for the caller (e.g. ``"bench_harness"``,
            ``"long_running_job"``). Shown to any other process that
            attempts to acquire while you hold it.
        force: if True, terminate the existing holder (see above).
            Use only when you have independently verified that killing
            the other holder is safe.

    Returns:
        The info dict written to the lock file.

    Raises:
        MLXLockConflict: see above for the three failure modes.
    """
    _PROCESS_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)

    force_reclaimed = False

    if not force:
        holder = read_mlx_lock()
        if holder is not None and holder.get("pid") != os.getpid():
            raise MLXLockConflict(holder)
    else:
        existing = read_mlx_lock()
        force_reclaimed = (
            existing is not None and existing.get("pid") != os.getpid()
        )
        if force_reclaimed:
            _force_terminate_and_wait(existing)
            _PROCESS_LOCK_PATH.unlink(missing_ok=True)

    info: dict[str, Any] = {
        "pid": os.getpid(),
        "label": label,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cmdline": " ".join(sys.argv)[:300] if hasattr(sys, "argv") else f"pid:{os.getpid()}",
    }
    tmp = _PROCESS_LOCK_PATH.with_suffix(f".tmp.{os.getpid()}")
    try:
        tmp.write_text(json.dumps(info, indent=2))
        os.replace(str(tmp), str(_PROCESS_LOCK_PATH))
    except OSError:
        tmp.unlink(missing_ok=True)
        raise

    # Post-reclaim cooldown — let the kernel's Metal buffer GC catch up
    # before the caller starts loading new models into the same GPU.
    if force_reclaimed and _RECLAIM_COOLDOWN_SEC > 0:
        log.info(
            "acquire_mlx_lock: post-reclaim cooldown %.1fs (Metal buffer GC)",
            _RECLAIM_COOLDOWN_SEC,
        )
        time.sleep(_RECLAIM_COOLDOWN_SEC)

    return info


def release_mlx_lock() -> bool:
    """Release the MLX lock if this process holds it."""
    info = read_mlx_lock()
    if info is None:
        return False
    if info.get("pid") != os.getpid():
        return False
    _PROCESS_LOCK_PATH.unlink(missing_ok=True)
    return True


@contextmanager
def mlx_exclusive_lock(
    label: str,
    *,
    force: bool = False,
) -> Generator[dict[str, Any], None, None]:
    """Context manager: acquire on enter, always release on exit.

    See ``acquire_mlx_lock`` for ``force=True`` hardened semantics
    (SIGTERM-and-wait, never silent unlink).

    Example::

        with mlx_exclusive_lock("my_script"):
            model, tokenizer = mlx_lm.load("mlx-community/gemma-4-31b-it-8bit")
            result = mlx_lm.generate(model, tokenizer, prompt="Hello")
    """
    info = acquire_mlx_lock(label, force=force)
    try:
        yield info
    finally:
        release_mlx_lock()


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

metal_guard = MetalGuard()


# ---------------------------------------------------------------------------
# Version advisory system (v0.6.0)
# ---------------------------------------------------------------------------
# Static database of known-bad (mlx, mlx-lm, mlx-vlm) version combos
# mapped to upstream issue numbers + severity. Use
# ``check_version_advisories()`` to get the active advisories for the
# packages installed in the current environment.
#
# Maintenance policy:
#   * Only list issues that are OPEN upstream, or CLOSED but not yet
#     fixed in a shipped release. Remove entries once the fix reaches
#     a PyPI release.
#   * Severity: "critical" — crash / kernel panic / data loss
#                "high"    — feature broken, workaround exists
#                "medium"  — warning noise, correctness preserved
#                "info"    — observability note
#   * Keep entries short; link to the upstream issue for detail.
# ---------------------------------------------------------------------------

# Each entry is (package, version_spec, advisory_dict).
# version_spec is a string compared with ``packaging.specifiers.SpecifierSet``;
# e.g. "==0.31.2" matches only that exact version, ">=0.31.2,<0.32"
# matches that range.
_VERSION_ADVISORIES: list[tuple[str, str, dict[str, Any]]] = [
    (
        "mlx-lm",
        "==0.31.2",
        {
            "issue": "ml-explore/mlx-lm#1128",
            "severity": "high",
            "title": "TokenizerWrapper.think_start_id crashes when _think_start_tokens is None",
            "url": "https://github.com/ml-explore/mlx-lm/issues/1128",
            "symptom": "TypeError: object of type 'NoneType' has no len()",
            "mitigation": (
                "Call ``install_upstream_defensive_patches()`` to install "
                "a safe accessor, or downgrade to mlx-lm 0.31.1."
            ),
        },
    ),
    (
        "mlx-lm",
        "==0.31.2",
        {
            "issue": "ml-explore/mlx-lm#1139",
            "severity": "high",
            "title": "0.31.2 regression: broadcast errors under multi-agent voting",
            "url": "https://github.com/ml-explore/mlx-lm/issues/1139",
            "symptom": "broadcast error after second voting round in 0.31.2 (not 0.31.1)",
            "mitigation": "Downgrade to mlx-lm 0.31.1 until upstream fix ships.",
        },
    ),
    (
        "mlx-lm",
        "==0.31.2",
        {
            "issue": "ml-explore/mlx-lm#1081",
            "severity": "medium",
            "title": "ArraysCache.is_trimmable() returns True but trim() method missing",
            "url": "https://github.com/ml-explore/mlx-lm/issues/1081",
            "symptom": (
                "AttributeError on speculative-decoding MTP perfect cache hits"
            ),
            "mitigation": (
                "Affects speculative-decoding callers only. Guard with "
                "``hasattr(cache, 'trim')`` before invoking."
            ),
        },
    ),
    (
        "mlx",
        "<0.31.2",
        {
            "issue": "ml-explore/mlx#3348",
            "severity": "info",
            "title": "CommandEncoder thread-local fix merged 2026-04-01 but not in this release",
            "url": "https://github.com/ml-explore/mlx/pull/3348",
            "symptom": (
                "Concurrent MLX dispatches across threads still risk the "
                "IOGPUFamily kernel panic on Apple Silicon."
            ),
            "mitigation": (
                "Keep METALGUARD_MODE defensive (default). Observer mode "
                "is only safe once mlx ships a release containing #3348."
            ),
        },
    ),
    # ── 2026-04-15 mlx-vlm 0.4.4 cluster ──────────────────────────────────
    (
        "mlx-vlm",
        "<0.4.5",
        {
            "issue": "Blaizzy/mlx-vlm#967",
            "severity": "critical",
            "title": "TurboQuant fused quantize race condition (decode T=1 silent corruption)",
            "url": "https://github.com/Blaizzy/mlx-vlm/pull/967",
            "symptom": (
                "Race on Metal threadgroup `packed_shared[w] |= idx_val << shift` "
                "(non-atomic) corrupts decode-time T=1 kernels. 4-bit "
                "TurboQuant output measurably worse than 2-bit; existing "
                "tests (T>1 prefill path) pass unchanged, making this a "
                "silent regression."
            ),
            "mitigation": (
                "Upgrade to mlx-vlm >= 0.4.5 when released (fix merged "
                "2026-04-07). Until then, avoid kv_quant_scheme=\"turboquant\" "
                "for decode-heavy workloads. Recommend PPL / logit cosine "
                "quality gate before trusting any TQ speedup numbers."
            ),
        },
    ),
    (
        "mlx-vlm",
        "==0.4.4",
        {
            "issue": "Blaizzy/mlx-vlm#1016",
            "severity": "high",
            "title": "prefill_attention always returns None after #909 (silent full dequantize)",
            "url": "https://github.com/Blaizzy/mlx-vlm/issues/1016",
            "symptom": (
                "Downstream projects that create TurboQuantKVCache instances "
                "*before* prefill silently dequantize the entire KV to fp16 "
                "during prefill (~31 GB activation memory at 128K context "
                "on gemma-4-31B per PR #939 benchmarks)."
            ),
            "mitigation": (
                "Only affects callers that build TQ caches externally. "
                "mlx-vlm's own generate() uses post-conversion and is "
                "unaffected. If affected, pin mlx-vlm < the upgrade "
                "containing the fix PR or keep TQ disabled at prefill-time."
            ),
        },
    ),
    (
        "mlx-vlm",
        "==0.4.4",
        {
            "issue": "Blaizzy/mlx-vlm#1011",
            "severity": "high",
            "title": "Gemma 4 loading fails with transformers 5.5.x (ReasoningEffort ImportError)",
            "url": "https://github.com/Blaizzy/mlx-vlm/issues/1011",
            "symptom": (
                "ImportError: cannot import name 'ReasoningEffort' from "
                "'transformers' when loading any Gemma 4 variant via "
                "mlx_vlm.load()."
            ),
            "mitigation": (
                "Pin transformers < 5.5 in venvs that load Gemma 4 via "
                "mlx-vlm. No monkey-patch available — transformers API "
                "removed the attribute."
            ),
        },
    ),
    (
        "mlx-vlm",
        "==0.4.4",
        {
            "issue": "Blaizzy/mlx-vlm#943",
            "severity": "critical",
            "title": "Gemma 4 26b-a4b-it-4bit vision produces garbage (text-only unaffected)",
            "url": "https://github.com/Blaizzy/mlx-vlm/issues/943",
            "symptom": (
                "NaN propagation in all-masked SDPA padding rows on Gemma 4 "
                "vision branch → degenerate output on mlx-community/"
                "gemma-4-26b-a4b-it-4bit image inputs. Text-only path OK."
            ),
            "mitigation": (
                "Avoid gemma-4-26b-a4b-it-4bit for vision tasks until PR "
                "#1006 ships. Substitute pixtral-12b-4bit or "
                "gemma-4-e4b-it-4bit for vision."
            ),
            "scope": "model:mlx-community/gemma-4-26b-a4b-it-4bit backend:mlx-vlm",
        },
    ),
    # ── 2026-04-16 community survey additions ─────────────────────────────
    # Scanned every MLX / mlx-lm / mlx-vlm issue since mlx-lm 0.31.2
    # released (2026-02-01+). These five are actionable safety /
    # correctness reports affecting typical MLX usage patterns.
    (
        "mlx",
        "<=0.31.1",
        {
            "issue": "ml-explore/mlx#3384",
            "severity": "critical",
            "title": "fast.scaled_dot_product_attention numerical divergence on 4-bit quantized models",
            "url": "https://github.com/ml-explore/mlx/issues/3384",
            "symptom": (
                "Token repetition / divergent logits from the fused SDPA "
                "kernel when inputs are 4-bit quantized. Silent — "
                "generation does not crash, just produces worse output. "
                "Non-quantised correctness tests pass unchanged; this is "
                "a regression only visible at the 4-bit decode step."
            ),
            "mitigation": (
                "4-bit is the default deployment format for many users. "
                "No safe in-place shim; fall back to a non-fused attention "
                "path or avoid affected sizes if quality regression "
                "appears. Monitor PPL / logit cosine before trusting "
                "4-bit speedups until a fix lands."
            ),
            "scope": "backend:mlx",
        },
    ),
    (
        "mlx-lm",
        ">=0.31.0,<=0.31.2",
        {
            "issue": "ml-explore/mlx-lm#897",
            "severity": "high",
            "title": "mlx_lm.server: chat completions crash with transformers >= 5.0 (return_dict=False missing)",
            "url": "https://github.com/ml-explore/mlx-lm/issues/897",
            "symptom": (
                "Server chat endpoint raises because call to HF transformers "
                "no longer accepts the positional signature used here. "
                "Only affects `mlx_lm.server`; CLI `generate` is unaffected."
            ),
            "mitigation": (
                "Pin transformers < 5.0 in venvs that launch the HTTP "
                "server, or wait for a mlx-lm release that includes the "
                "server-side fix."
            ),
            "scope": "backend:mlx-lm component:server",
        },
    ),
    (
        "mlx-vlm",
        "==0.4.4",
        {
            "issue": "Blaizzy/mlx-vlm#999",
            "severity": "high",
            "title": "server clears Metal cache after every request — destroys KV prefix cache",
            "url": "https://github.com/Blaizzy/mlx-vlm/issues/999",
            "symptom": (
                "`mlx_vlm` server calls `mx.clear_cache()` (and sync "
                "barriers) at the end of every request handler. The "
                "per-conversation KV prefix cache is dropped, forcing "
                "full re-prefill every turn — major latency plus Metal "
                "allocator thrash that can cascade into IOGPUFamily "
                "pressure."
            ),
            "mitigation": (
                "If running the HTTP server, disable per-request cache "
                "clears and rely on a between-model cooldown policy "
                "instead. Upstream fix pending."
            ),
            "scope": "backend:mlx-vlm component:server",
        },
    ),
    (
        "mlx",
        "<=0.31.2",
        {
            "issue": "ml-explore/mlx#3350",
            "severity": "high",
            "title": "Metal allocator buffer pool grows unboundedly on monotonic-size allocations",
            "url": "https://github.com/ml-explore/mlx/issues/3350",
            "symptom": (
                "MetalAllocator::free() unconditionally recycles freed "
                "buffers into buffer_cache_ until max_pool_size_ (~1.5 × "
                "max_rec_size ≈ 192 GB on a 128 GB Mac). Sequential "
                "increasing-size allocations (monotonic KV cache growth "
                "across long-context requests) can never reuse cached "
                "buffers — process footprint grows unbounded. Reporter "
                "observed 108 GB used where 34 GB was expected."
            ),
            "mitigation": (
                "Maintainer closed won't-fix 2026-04-04 (framework-level "
                "eviction would kill inference perf). Caller-side: "
                "1) Call `mx.set_cache_limit(N)` at subprocess startup "
                "sized to the model's working set (2-4× weight bytes) "
                "instead of the 192 GB default. "
                "2) Call `mx.clear_cache()` when KV growth crosses a "
                "threshold (e.g. >64k tokens). "
                "3) The one-model-per-subprocess pattern in "
                "`MLXSubprocessRunner` avoids the pool-growth path because "
                "each subprocess exits entirely between models."
            ),
            "scope": "backend:mlx",
        },
    ),
    (
        "mlx",
        "<=0.31.2",
        {
            "issue": "ml-explore/mlx#3390",
            "severity": "high",
            "title": "Metal completion-handler check_error throws → std::terminate → uncatchable SIGABRT",
            "url": "https://github.com/ml-explore/mlx/issues/3390",
            "symptom": (
                "`mlx/backend/metal/eval.cpp::check_error(MTL::CommandBuffer*)` "
                "throws std::runtime_error when "
                "`cbuf->status() == CommandBufferStatusError`. Invoked from "
                "3 sites registered via `addCompletedHandler(...)`, "
                "executing on Apple's `com.Metal.CompletionQueueDispatch` "
                "(libdispatch/GCD) queue. libdispatch blocks are not "
                "exception-safe — throw hits __cxa_throw → std::terminate "
                "→ abort() → SIGABRT. Python cannot catch this; try/except "
                "around `mx.eval()` never fires. Duplicate reports: #3224 "
                "(M3 Ultra 6 hr), #3317 (M2 Ultra asyncio race). Umbrella: "
                "ml-explore/mlx#2670. This is the 7th kernel-panic "
                "root-cause class; metal-guard only partially mitigates "
                "(subprocess isolation keeps parent alive + module-import "
                "AGX_RELAX stopgap reduces trigger frequency)."
            ),
            "mitigation": (
                "PR #3318 (check_error_deferred pattern) CLOSED without "
                "merge — maintainer declined on the grounds that process "
                "state is undefined post-throw. No release targets this "
                "fix. metal-guard auto-sets `AGX_RELAX_CDM_CTXSTORE_TIMEOUT=1` "
                "at module import (reduces GPU watchdog false-positives "
                "that most commonly trigger the abort). Subprocess "
                "isolation (MLXSubprocessRunner) mitigates blast radius: "
                "parent survives, sibling subprocesses unaffected; "
                "in-flight request is lost. Do NOT attempt a C++ "
                "terminate handler — state is undefined post-throw."
            ),
            "scope": "backend:mlx",
        },
    ),
]


def _installed_version(package: str) -> str | None:
    """Return the installed version of ``package`` or None if not installed."""
    try:
        import importlib.metadata as metadata
    except ImportError:  # pragma: no cover — Python ≥ 3.8 has this
        return None
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None
    except Exception as exc:  # noqa: BLE001 — diagnostic fallback
        log.debug("_installed_version(%s) failed: %s", package, exc)
        return None


def _spec_matches(version: str, spec: str) -> bool:
    """Return True if ``version`` satisfies ``spec`` (PEP 440 specifier)."""
    try:
        from packaging.specifiers import SpecifierSet
        from packaging.version import Version, InvalidVersion
    except ImportError:
        # packaging is a pip/setuptools transitive dep — nearly universal —
        # but fall back to exact string match if it is missing.
        return version == spec.lstrip("=")
    try:
        return Version(version) in SpecifierSet(spec)
    except (InvalidVersion, ValueError):
        return False


def check_version_advisories(
    *, packages: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Return active advisories for the packages installed in this env.

    Args:
        packages: optional override map of ``{package_name: version}``
            to bypass ``importlib.metadata`` lookup. Useful in tests.

    Returns:
        List of advisory dicts. Each dict has keys ``package``,
        ``installed_version``, ``issue``, ``severity``, ``title``,
        ``url``, ``symptom``, ``mitigation``. Empty list if no known
        issues apply to the current environment.

    Example::

        for a in check_version_advisories():
            print(f"[{a['severity']}] {a['package']} {a['installed_version']} — {a['title']}")
            print(f"    {a['url']}")
    """
    resolved = packages or {}
    active: list[dict[str, Any]] = []
    for pkg, spec, advisory in _VERSION_ADVISORIES:
        installed = resolved.get(pkg) if packages is not None else _installed_version(pkg)
        if installed is None:
            continue
        if not _spec_matches(installed, spec):
            continue
        entry = dict(advisory)
        entry["package"] = pkg
        entry["installed_version"] = installed
        entry["affected_spec"] = spec
        active.append(entry)
    return active


# ---------------------------------------------------------------------------
# Upstream defensive patches (v0.6.0)
# ---------------------------------------------------------------------------
# Narrow, version-gated monkey-patches that install safe shims for known
# mlx-lm bugs without modifying upstream code. Opt-in only — callers
# invoke ``install_upstream_defensive_patches()`` explicitly at program
# start. Each patch is idempotent (tagged on first install, skipped on
# subsequent calls) and auto-skips when the installed package version
# is outside the affected range, so once upstream ships a fix this
# becomes a no-op without any caller change.
# ---------------------------------------------------------------------------


def install_upstream_defensive_patches(*, force: bool = False) -> dict[str, bool]:
    """Install safe accessors for known upstream bugs.

    Each patch is version-gated against the installed package version
    and skipped when upstream has shipped a fix. Idempotent — safe to
    call multiple times. Emits a WARNING log for each patch applied so
    operators can confirm what was installed.

    Args:
        force: bypass the installed-version check. Tests only.

    Returns:
        Dict of ``{patch_name: was_applied}``. ``False`` means the patch
        was skipped (already applied, package missing, version outside
        affected range, or upstream already fixed).

    Example::

        from metal_guard import install_upstream_defensive_patches
        status = install_upstream_defensive_patches()
        # {'mlx_lm_1128_think_start_id': True}
    """
    return {
        "mlx_lm_1128_think_start_id": _patch_mlx_lm_1128(force=force),
    }


def _patch_mlx_lm_1128(*, force: bool) -> bool:
    """Apply defensive shim for ml-explore/mlx-lm#1128.

    ``TokenizerWrapper.think_start_id`` calls ``len(self._think_start_tokens)``
    but ``_infer_thinking()`` can leave that attribute as ``None``, which
    raises ``TypeError`` rather than signalling no think tokens. We
    replace the property with one that returns ``None`` in that case.
    """
    mlx_lm_version = _installed_version("mlx-lm")
    if mlx_lm_version is None:
        return False
    if not force and not _spec_matches(mlx_lm_version, "==0.31.2"):
        return False

    # Resolve via sys.modules so test fixtures that inject a fake
    # ``mlx_lm.tokenizer_utils`` can intercept. A naive
    # ``from mlx_lm import tokenizer_utils`` reads the attribute off
    # the already-imported parent package and bypasses sys.modules.
    _tu = sys.modules.get("mlx_lm.tokenizer_utils")
    if _tu is None:
        try:
            import importlib
            _tu = importlib.import_module("mlx_lm.tokenizer_utils")
        except ImportError:
            return False

    TW = getattr(_tu, "TokenizerWrapper", None)
    if TW is None:
        return False

    existing = getattr(TW, "think_start_id", None)
    if not isinstance(existing, property):
        return False
    if getattr(existing.fget, "_metal_guard_safe", False):
        return False  # idempotent

    def safe_think_start_id(self):  # type: ignore[no-untyped-def]
        tokens = getattr(self, "_think_start_tokens", None)
        if tokens is None:
            return None
        if len(tokens) > 1:
            raise ValueError(
                "multiple think start tokens in TokenizerWrapper — ambiguous"
            )
        return tokens[0] if tokens else None

    safe_think_start_id._metal_guard_safe = True  # type: ignore[attr-defined]
    TW.think_start_id = property(safe_think_start_id)
    log.warning(
        "metal_guard: installed defensive patch for mlx-lm#1128 "
        "(think_start_id None-safety) on mlx-lm %s", mlx_lm_version,
    )
    return True


# ---------------------------------------------------------------------------
# Layer 5: bench_scoped_load — sequential benchmark harness guard (v0.5.0)
# ---------------------------------------------------------------------------
# Layers 1-4 target concurrency and stale-thread hazards. They do NOT
# protect a single-threaded, single-process harness that loads 4+ large
# MLX models sequentially within one Python lifetime — the scenario
# where residual Metal driver buffer accounting drifts above the working
# set recommended limit and triggers an IOGPUMemory.cpp:492 kernel panic.
#
# L5 closes that gap via ``bench_scoped_load``: every entry acquires
# the cross-process lock and loads fresh; every exit runs safe_cleanup +
# extra cooldown + post-unload memory verification.
#
# Empirical: Metal lazy release needs ~8s on 64GB Apple Silicon after
# unloading a 24GB model before the buffer pool pages return to the OS.
# safe_cleanup's internal 1s cooldown is insufficient; 8s lets the OS
# reclaimer catch up in almost all cases.

_BENCH_POST_UNLOAD_COOLDOWN_SEC = 8.0

# ---------------------------------------------------------------------------
# System-level audits (R2 / R3 — 2026-04-15)
# ---------------------------------------------------------------------------
#
# Read-only audits of the runtime host that can be correlated with known
# kernel-panic patterns. Purely informational — results are logged at
# startup and exposed via dashboard endpoints; nothing here changes
# system state.

_WIRED_LIMIT_WARN_RATIO = 0.85


def _sysctl(name: str, *, timeout: float = 2.0) -> str | None:
    """Run ``sysctl -n <name>`` and return stdout stripped, or None on failure."""
    import subprocess
    try:
        out = subprocess.run(
            ["sysctl", "-n", name],
            capture_output=True, text=True, timeout=timeout, check=False,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        log.debug("sysctl -n %s failed: %s", name, exc)
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def audit_wired_limit() -> dict[str, Any]:
    """Audit the ``iogpu.wired_limit_mb`` sysctl for panic-prone values.

    Returns a dict with keys ``limit_mb`` / ``total_gb`` / ``ratio`` /
    ``mode`` (``"default"`` | ``"override"`` | ``"unknown"``) / ``advisory``.

    Rationale: mlx-lm maintainer ``angeloskath`` commented on issue
    ``ml-explore/mlx-lm#1047`` that IOGPUFamily kernel panics are often
    correlated with too-high wired-memory overrides. Values >
    ``_WIRED_LIMIT_WARN_RATIO`` (85%) of physical memory stress the
    allocator path implicated in the #3186 panic signature.

    Never raises. Subprocess failures return ``mode="unknown"``.
    """
    raw_limit = _sysctl("iogpu.wired_limit_mb")
    raw_total = _sysctl("hw.memsize")

    limit_mb: int | None = None
    if raw_limit is not None:
        try:
            limit_mb = int(raw_limit)
        except ValueError:
            pass

    total_gb: float | None = None
    if raw_total is not None:
        try:
            total_gb = int(raw_total) / (1024 ** 3)
        except ValueError:
            pass

    if limit_mb is None or total_gb is None or total_gb <= 0:
        return {
            "limit_mb": limit_mb,
            "total_gb": total_gb,
            "ratio": 0.0,
            "mode": "unknown",
            "advisory": None,
        }

    if limit_mb == 0:
        return {
            "limit_mb": 0,
            "total_gb": total_gb,
            "ratio": 0.0,
            "mode": "default",
            "advisory": None,
        }

    ratio = (limit_mb / 1024.0) / total_gb
    advisory: str | None = None
    if ratio > _WIRED_LIMIT_WARN_RATIO:
        advisory = (
            f"iogpu.wired_limit_mb={limit_mb}MB is {ratio:.0%} of "
            f"{total_gb:.0f}GB unified memory. Too-high overrides have "
            f"been correlated with IOGPUFamily kernel panics "
            f"(mlx-lm#1047 maintainer comment). Consider: "
            f"sudo sysctl iogpu.wired_limit_mb=0 to return to Apple "
            f"default, or pick a value < "
            f"{int(total_gb * _WIRED_LIMIT_WARN_RATIO * 1024)}MB."
        )

    return {
        "limit_mb": limit_mb,
        "total_gb": total_gb,
        "ratio": ratio,
        "mode": "override",
        "advisory": advisory,
    }


def read_gpu_driver_version(*, timeout: float = 3.0) -> str | None:
    """Return the ``IOGPUFamily`` kext bundle version, or None if unreadable.

    Panic reports on ``ml-explore/mlx#3186`` pin the fault to the
    ``com.apple.iokit.IOGPUFamily`` kext. Logging the driver revision
    at startup is useful forensic context for future crash correlation.

    Probes ``kextstat`` first; falls back to
    ``ioreg -rc IOGPUFamily -d 1`` CFBundleVersion. First non-empty
    reading wins. Never raises.
    """
    import subprocess
    try:
        out = subprocess.run(
            ["kextstat"], capture_output=True, text=True,
            timeout=timeout, check=False,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        log.debug("kextstat failed: %s", exc)
    else:
        if out.returncode == 0:
            for line in out.stdout.splitlines():
                if "com.apple.iokit.IOGPUFamily" not in line:
                    continue
                lo, _, _ = line.partition(")")
                _, _, rhs = lo.rpartition("(")
                version = rhs.strip()
                if version:
                    return version

    try:
        out = subprocess.run(
            ["ioreg", "-rc", "IOGPUFamily", "-d", "1"],
            capture_output=True, text=True, timeout=timeout, check=False,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        log.debug("ioreg IOGPUFamily failed: %s", exc)
        return None
    if out.returncode != 0:
        return None
    for line in out.stdout.splitlines():
        if "CFBundleVersion" not in line:
            continue
        _, _, rhs = line.partition("=")
        value = rhs.strip().strip('"').strip()
        if value:
            return value
    return None


def log_system_audit_at_startup() -> dict[str, Any]:
    """Run all host-level audits and WARNING-log any advisory.

    Returns the combined audit dict. Intended for program entry points
    (CLI ``main()``, FastAPI lifespan) so operators see host-level
    stability warnings in routine logs.
    """
    wired = audit_wired_limit()
    kext = read_gpu_driver_version()

    if wired.get("advisory"):
        log.warning("system audit: %s", wired["advisory"])
    else:
        log.info(
            "system audit: wired_limit mode=%s limit_mb=%s total_gb=%s",
            wired.get("mode"), wired.get("limit_mb"), wired.get("total_gb"),
        )

    if kext:
        log.info("system audit: IOGPUFamily kext version=%s", kext)
    else:
        log.debug("system audit: IOGPUFamily kext version unreadable")

    return {
        "wired_limit": wired,
        "gpu_driver_version": kext,
    }


# Post-unload active memory above this threshold logs an informational
# message (not a failure). Metal's lazy page reclaimer only returns
# pages when the NEXT load() allocates — stale-looking memory here is
# expected behaviour, not a leak.
_BENCH_POST_UNLOAD_ACTIVE_LIMIT_GB = 4.0


@contextmanager
def bench_scoped_load(
    model_id: str,
    *,
    backend: str = "mlx",
) -> Iterator[Tuple[Any, Any]]:
    """Safe sequential model loading for long-running benchmark harnesses.

    MetalGuard Layer 5. Wraps ``mlx_lm.load`` / ``mlx_vlm.load`` so a
    harness that must load many large MLX models inside one Python
    process gets the full defensive stack — cross-process lock, thread
    registry via safe_cleanup, gc + mx.clear_cache + cooldown — on
    every iteration, plus post-unload memory-drop verification.

    Rationale: benchmark harnesses that bypass this library (calling
    ``mlx_lm.load`` + ``mlx_lm.generate`` directly in a loop) can drift
    above the working-set limit on 64GB Apple Silicon after loading 6+
    large models sequentially — residual Metal driver buffer accounting
    keeps stale allocations visible even after ``mx.clear_cache()``.
    This has been observed to trigger an IOGPUMemory.cpp:492
    ``completeMemory() prepare count underflow`` kernel panic.

    Usage::

        from metal_guard import bench_scoped_load

        for model_id in candidate_models:  # 8 large models
            with bench_scoped_load(model_id) as (model, tokenizer):
                score = run_eval(model, tokenizer, items)
                save_atomic_checkpoint(model_id, score)

    Parameters
    ----------
    model_id:
        Hugging Face model ID that ``mlx_lm.load`` / ``mlx_vlm.load`` accepts,
        e.g. ``"mlx-community/Mistral-Small-3.2-24B-Instruct-2506-8bit"``.
    backend:
        ``"mlx"`` / ``"mlx-lm"`` → text model via ``mlx_lm.load``.
        ``"mlx-vlm"`` → vision model via ``mlx_vlm.load``.

    Yields
    ------
    The ``(model, tokenizer)`` tuple returned by the underlying loader.

    Raises
    ------
    ValueError
        If ``backend`` is not a supported MLX backend.
    MLXLockConflict
        If another process already holds the MLX exclusive lock.

    Notes
    -----
    Does NOT replace Layers 1-4. It delegates to them and adds
    post-unload verification specific to sequential benchmark loops.
    """
    metal_guard.breadcrumb(f"BENCH_SCOPED_LOAD: ENTER {model_id} backend={backend}")
    log.info("bench_scoped_load: entering %s (backend=%s)", model_id, backend)

    # R4 advisory (v0.7.0): if model has known dims, log prefill safety
    # envelope at 131 072 context so the caller sees which cells would
    # trip require_prefill_fit. Advisory only; never blocks load.
    try:
        _stats = metal_guard.memory_stats()
        _plan = describe_prefill_plan(
            context_tokens=131072, model_id=model_id,
            available_gb=_stats.available_gb if _stats.limit_bytes > 0 else None,
        )
        if _plan["dims_known"]:
            if _plan["fits_ceiling"] is False:
                log.warning(
                    "bench_scoped_load: %s at 131k ctx would alloc ~%s GB "
                    "> 5 GB ceiling — R4 will refuse; recommend chunk=%s",
                    model_id, _plan["peak_alloc_gb"],
                    _plan["recommended_chunk_size"],
                )
            else:
                log.info(
                    "bench_scoped_load: %s prefill plan OK at 131k "
                    "(peak ~%s GB)", model_id, _plan["peak_alloc_gb"],
                )
    except Exception as _exc:
        log.debug("bench_scoped_load: prefill plan advisory failed: %s", _exc)

    # Acquire cross-process lock for the entire bench scope (load + yield + unload).
    # Unlike per-call acquisition, bench holds the lock across multiple
    # generate() calls on the same model — preventing other processes from
    # loading a competing model mid-benchmark.
    acquire_mlx_lock("bench_scoped_load")

    # Direct load. In bench scope we manage the full lifecycle
    # (load → yield → unload) sequentially, and Metal's lazy page reclaimer
    # only returns stale pages to the OS when the next load() actually
    # allocates. Pre-checking memory_stats() here would report stale pages
    # as "active" and incorrectly block legitimate back-to-back loads of
    # 25GB+ models on 64GB machines.
    if backend in ("mlx", "mlx-lm"):
        from mlx_lm import load as _mlx_load
        metal_guard.breadcrumb(f"BENCH_SCOPED_LOAD: direct load {model_id}")
        loaded = _mlx_load(model_id)
    elif backend == "mlx-vlm":
        from mlx_vlm import load as _vlm_load
        metal_guard.breadcrumb(f"BENCH_SCOPED_LOAD: direct vlm load {model_id}")
        loaded = _vlm_load(model_id)
    else:
        release_mlx_lock()
        raise ValueError(
            f"bench_scoped_load: unsupported backend {backend!r} "
            f"(expected 'mlx', 'mlx-lm', or 'mlx-vlm')"
        )

    try:
        yield loaded
    finally:
        metal_guard.breadcrumb(
            f"BENCH_SCOPED_LOAD: EXIT {model_id} — unloading + cooldown"
        )
        log.info("bench_scoped_load: exiting %s, unloading", model_id)

        # Full cleanup: wait_for_threads + gc + flush_gpu + internal cooldown.
        metal_guard.safe_cleanup()
        gc.collect()

        # Reset peak tracker so the NEXT iteration's memory_stats().peak_gb
        # reflects only that model, not a stale high-water mark from a
        # previous large model that Metal hasn't fully released yet.
        try:
            import mlx.core as mx
            mx.reset_peak_memory()
        except (ImportError, AttributeError):
            pass

        time.sleep(_BENCH_POST_UNLOAD_COOLDOWN_SEC)

        # Diagnostic only: Metal lazy release means active memory may still
        # report high here — the pages are reclaimed when the NEXT load()
        # allocates. This is informational, not load-blocking.
        stats = metal_guard.memory_stats()
        active_after_gb = stats.active_gb
        if active_after_gb > _BENCH_POST_UNLOAD_ACTIVE_LIMIT_GB:
            log.info(
                "bench_scoped_load: %s active memory %.1fGB after unload "
                "(expected eventually < %.1fGB via Metal lazy release on next load)",
                model_id, active_after_gb, _BENCH_POST_UNLOAD_ACTIVE_LIMIT_GB,
            )
        metal_guard.breadcrumb(
            f"BENCH_SCOPED_LOAD: EXIT {model_id} done, active={active_after_gb:.1f}GB"
        )
        log.info(
            "bench_scoped_load: exited %s, active memory %.1fGB peak %.1fGB",
            model_id, active_after_gb, stats.peak_gb,
        )
        release_mlx_lock()


# ---------------------------------------------------------------------------
# Layer 6: Dual-mode switcher — defensive / observer (v0.5.0)
# ---------------------------------------------------------------------------
# MetalGuard has two operating modes:
#
#   - Defensive (default): actively block dangerous Metal GPU operations.
#     process_lock refuses conflicting runs, sequential dispatch enforced.
#
#   - Observer: monitor and log, let the caller proceed. Used after
#     upstream MLX ships #3348 (CommandEncoder thread-local) and empirical
#     validation confirms parallel inference is safe on the target hardware.
#
# The five long-term primitives (safe_cleanup, daemon thread registry,
# oom_protected_context, breadcrumb logging, memory_stats) are active
# in BOTH modes — they address concerns orthogonal to thread safety.
#
# Set METALGUARD_MODE env var to switch:
#   export METALGUARD_MODE=defensive    # default
#   export METALGUARD_MODE=observer     # opt-in after #3348

DEFENSIVE = "defensive"
OBSERVER = "observer"

_VALID_MODES = frozenset({DEFENSIVE, OBSERVER})
_MODE_ENV_VAR = "METALGUARD_MODE"


def current_mode() -> str:
    """Return the current MetalGuard operating mode.

    Reads the ``METALGUARD_MODE`` environment variable fresh on every
    call. Values are case-insensitive. Unknown values fall back to
    ``defensive`` (fail-safe default).
    """
    raw = os.environ.get(_MODE_ENV_VAR, DEFENSIVE).strip().lower()
    if raw in _VALID_MODES:
        return raw
    log.warning(
        "Unknown %s value %r — falling back to 'defensive'. "
        "Valid values: %s",
        _MODE_ENV_VAR, raw, sorted(_VALID_MODES),
    )
    return DEFENSIVE


def is_defensive() -> bool:
    """True when defensive mode is active (the default)."""
    return current_mode() == DEFENSIVE


def is_observer() -> bool:
    """True when observer mode is active (opt-in via env var)."""
    return current_mode() == OBSERVER


def describe_mode() -> dict[str, str]:
    """Return a structured description suitable for health endpoints.

    Example output::

        {
            "mode": "defensive",
            "description": "Actively blocks dangerous Metal operations",
            "env_var": "METALGUARD_MODE=<unset> (default → defensive)",
        }
    """
    mode = current_mode()
    env_raw = os.environ.get(_MODE_ENV_VAR)
    if env_raw is None:
        env_display = f"{_MODE_ENV_VAR}=<unset> (default → defensive)"
    else:
        env_display = f"{_MODE_ENV_VAR}={env_raw!r}"
    descriptions = {
        DEFENSIVE: (
            "Actively blocks dangerous Metal operations — process_lock refuses "
            "conflicts, sequential dispatch enforced."
        ),
        OBSERVER: (
            "Monitoring only — process_lock logs warnings but does not refuse, "
            "parallel dispatch permitted. Requires mlx >= 0.31.2 (includes "
            "#3348 CommandEncoder thread-local)."
        ),
    }
    return {
        "mode": mode,
        "description": descriptions.get(mode, "(unknown)"),
        "env_var": env_display,
    }


# ---------------------------------------------------------------------------
# Layer 7: Subprocess isolation — crash-safe MLX inference (v0.5.0)
# ---------------------------------------------------------------------------
# When MLX's C++ runtime throws from Metal's GCD CompletionQueueDispatch
# (mlx::core::gpu::check_error), Python cannot catch it — std::terminate
# calls abort() and the process dies.
#
# Subprocess isolation is the only way to protect the parent process:
#   - Each MLX model runs in its own subprocess
#   - Worker loads model once, accepts multiple prompts via pipe
#   - If worker crashes (SIGABRT) → pipe breaks → parent detects → new worker
#   - Parent's Python/Metal state is never corrupted
#
# Usage::
#
#     from metal_guard import MLXSubprocessRunner
#
#     runner = MLXSubprocessRunner("mlx-community/Mistral-Small-3.2-24B-8bit")
#     for prompt in prompts:
#         result = runner.generate(prompt, max_tokens=4096)
#     runner.shutdown()
#
# Or use the auto-managed pool::
#
#     from metal_guard import call_model_isolated
#     result = call_model_isolated(prompt, model="mlx-community/Phi-4-mini-4bit")

# Default timeouts (seconds). MLX text generate rarely exceeds 120s; VLM can
# take 180s. Add generous buffer for model loading on first call.
_DEFAULT_GENERATE_TIMEOUT = 300.0
_DEFAULT_LOAD_TIMEOUT = 120.0


class SubprocessCrashError(RuntimeError):
    """Raised when the MLX worker subprocess crashed (SIGABRT / unexpected exit)."""

    def __init__(self, model_id: str, exitcode: int | None, detail: str = ""):
        self.model_id = model_id
        self.exitcode = exitcode
        sig_name = ""
        if exitcode is not None and exitcode < 0:
            try:
                sig_name = f" ({signal.Signals(-exitcode).name})"
            except (ValueError, AttributeError):
                pass
        msg = (
            f"MLX worker subprocess for {model_id!r} crashed "
            f"(exit code {exitcode}{sig_name})"
        )
        if detail:
            msg += f": {detail}"
        super().__init__(msg)


class SubprocessTimeoutError(TimeoutError):
    """Raised when the worker subprocess did not respond within the timeout."""


def _worker_main(
    model_id: str,
    backend: str,
    recv_conn: Connection,
    send_conn: Connection,
) -> None:
    """Worker loop running in a forked subprocess.

    Loads the model once, then processes prompts until shutdown or crash.
    All MLX / Metal operations happen here — if Metal SIGABRT fires,
    only this process dies.
    """
    # Suppress TOKENIZERS_PARALLELISM warning in subprocess
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # R6 process-mode marker — let detect_process_mode() classify this
    # child as ``subprocess_worker`` (skip_process_lock=True in defaults,
    # since the parent already owns the cross-process lock).
    os.environ["METALGUARD_SUBPROCESS_WORKER"] = "1"

    # Install MetalGuard's SIGABRT handler for forensic breadcrumb
    try:
        from metal_guard import metal_guard as _mg
        _mg.install_abort_handler()
        _mg.breadcrumb(f"SUBPROCESS_WORKER: loading {model_id}")
    except Exception:
        pass

    try:
        if backend in ("mlx", "mlx-lm"):
            from mlx_lm import load, generate
            model, tokenizer = load(model_id)
            gen_fn = generate
        elif backend == "mlx-vlm":
            from mlx_vlm import load, generate
            model, tokenizer = load(model_id)
            gen_fn = generate
        else:
            send_conn.send({"type": "error", "error": f"unsupported backend: {backend}"})
            return
    except Exception as exc:
        send_conn.send({"type": "error", "error": f"load failed: {type(exc).__name__}: {exc}"})
        return

    # Signal ready to parent
    send_conn.send({"type": "ready", "model_id": model_id, "pid": os.getpid()})

    try:
        from metal_guard import metal_guard as _mg
        _mg.breadcrumb(f"SUBPROCESS_WORKER: {model_id} ready, pid={os.getpid()}")
    except Exception:
        pass

    # Main loop: receive prompts, generate, send results
    while True:
        try:
            if not recv_conn.poll(timeout=600):  # 10 min idle timeout
                break  # No work for 10 minutes → self-terminate
            request = recv_conn.recv()
        except (EOFError, OSError):
            break  # Parent closed pipe → exit

        if request is None or request.get("type") == "shutdown":
            break

        prompt = request.get("prompt", "")
        max_tokens = request.get("max_tokens", 4096)
        temperature = request.get("temperature", 0.3)
        system_prompt = request.get("system", "")

        try:
            # Apply chat template if available. Fall back to model-family
            # templates when tokenizer.chat_template is unset (observed on
            # some Mistral/Gemma4 quantized variants where the mlx-community
            # upload strips the template). Falling back to raw prompt causes
            # the model to treat the text as completion continuation and
            # produce empty / incoherent output.
            try:
                formatted = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    add_generation_prompt=True,
                    tokenize=False,
                )
            except Exception:
                mid = model_id.lower()
                if "gemma" in mid:
                    formatted = (
                        f"<start_of_turn>user\n{prompt}<end_of_turn>\n"
                        f"<start_of_turn>model\n"
                    )
                elif "mistral" in mid:
                    formatted = f"[INST] {prompt} [/INST]"
                elif "phi" in mid:
                    formatted = f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n"
                else:
                    formatted = prompt

            # R4 auto-wire (v0.7.0): refuse prefills whose estimated peak
            # allocation would risk IOGPUFamily kernel panic. Silently
            # skips if model dims aren't in KNOWN_MODELS.
            _dims = lookup_dims(model_id)
            if _dims is not None:
                try:
                    _prompt_tok = tokenizer.encode(formatted)
                    _prompt_len = len(_prompt_tok)
                except Exception:
                    _prompt_len = max(1, len(formatted) // 4)
                _stats = metal_guard.memory_stats()
                if _stats.limit_bytes > 0:
                    # Raises MetalOOMError on breach → caught by the except
                    # below and returned as an error reply, not a crash.
                    require_prefill_fit(
                        context_tokens=_prompt_len + max_tokens,
                        dims=_dims,
                        available_gb=_stats.available_gb,
                    )
                    metal_guard.breadcrumb(
                        f"PREFILL_GUARD: OK model={model_id} "
                        f"ctx={_prompt_len + max_tokens}"
                    )

            result = gen_fn(
                model, tokenizer,
                prompt=formatted,
                max_tokens=max_tokens,
                verbose=False,
            )
            if not isinstance(result, str):
                result = str(result)
            send_conn.send({"type": "result", "text": result})
        except Exception as exc:
            send_conn.send({
                "type": "error",
                "error": f"{type(exc).__name__}: {exc}",
            })

    # Clean exit
    send_conn.send({"type": "shutdown_ack"})


class MLXSubprocessRunner:
    """Manage an isolated MLX worker subprocess for a specific model.

    The worker loads the model once and stays alive for multiple generate
    calls. If it crashes (Metal SIGABRT), the parent detects the broken
    pipe and can create a new runner.

    Thread-safe: uses a lock to serialize generate calls (Metal cannot
    handle concurrent generate in a single process anyway).
    """

    def __init__(
        self,
        model_id: str,
        *,
        backend: str = "mlx",
        load_timeout: float = _DEFAULT_LOAD_TIMEOUT,
        generate_timeout: float = _DEFAULT_GENERATE_TIMEOUT,
    ) -> None:
        self.model_id = model_id
        self.backend = backend
        self._load_timeout = load_timeout
        self._generate_timeout = generate_timeout
        self._lock = multiprocessing.Lock()
        self._process: multiprocessing.Process | None = None
        self._parent_send: Connection | None = None
        self._parent_recv: Connection | None = None
        self._worker_pid: int | None = None
        self._started = False
        self._holds_mlx_lock = False

        # H7 (2026-04-16): acquire the cross-process MLX lock BEFORE
        # spawning the worker. Without this, any concurrent MLX
        # acquirer (pytest running bench_scoped_load, a second bench
        # CLI, an acceptance test) can legally overwrite the lock
        # mid-run while we are still holding Metal buffers. Two
        # processes sharing IOGPU = IOGPUMemory.cpp:492 kernel panic.
        acquire_mlx_lock(f"mlx_subprocess_runner:{self.model_id}")
        self._holds_mlx_lock = True

        try:
            self._start_worker()
        except BaseException:
            # Worker failed to load / timed out — release the lock so
            # the next caller is not locked out by a dead runner.
            self._release_mlx_lock_if_held()
            raise

    def _start_worker(self) -> None:
        """Fork the worker subprocess and wait for 'ready' signal."""
        parent_recv, child_send = multiprocessing.Pipe(duplex=False)
        child_recv, parent_send = multiprocessing.Pipe(duplex=False)

        self._parent_send = parent_send
        self._parent_recv = parent_recv

        self._process = multiprocessing.Process(
            target=_worker_main,
            args=(self.model_id, self.backend, child_recv, child_send),
            daemon=True,
        )
        self._process.start()
        log.info(
            "MLXSubprocessRunner: started worker pid=%d for %s",
            self._process.pid, self.model_id,
        )

        if not parent_recv.poll(timeout=self._load_timeout):
            self.kill()
            raise SubprocessTimeoutError(
                f"Worker for {self.model_id} did not respond within "
                f"{self._load_timeout}s during model load"
            )

        msg = parent_recv.recv()
        if msg.get("type") == "error":
            self.kill()
            raise RuntimeError(
                f"Worker for {self.model_id} failed to load: {msg.get('error')}"
            )
        if msg.get("type") == "ready":
            self._worker_pid = msg.get("pid")
            self._started = True
            log.info(
                "MLXSubprocessRunner: worker pid=%d ready for %s",
                self._worker_pid, self.model_id,
            )

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 4096,
        temperature: float = 0.3,
        system: str = "",
        timeout: float | None = None,
    ) -> str:
        """Send a prompt to the worker and return the generated text.

        Raises:
            SubprocessCrashError: Worker process died (SIGABRT, etc.)
            SubprocessTimeoutError: Worker did not respond in time
            RuntimeError: Worker returned an error
        """
        if not self.is_alive():
            raise SubprocessCrashError(
                self.model_id,
                self._process.exitcode if self._process else None,
                "worker not alive at generate() entry",
            )

        effective_timeout = timeout if timeout is not None else self._generate_timeout

        with self._lock:
            try:
                self._parent_send.send({
                    "type": "generate",
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "system": system,
                })
            except (BrokenPipeError, OSError) as exc:
                raise SubprocessCrashError(
                    self.model_id,
                    self._process.exitcode if self._process else None,
                    f"send failed: {exc}",
                ) from exc

            if not self._parent_recv.poll(timeout=effective_timeout):
                log.error(
                    "MLXSubprocessRunner: worker pid=%s timed out after %.0fs for %s",
                    self._worker_pid, effective_timeout, self.model_id,
                )
                self.kill()
                raise SubprocessTimeoutError(
                    f"Worker for {self.model_id} did not respond within "
                    f"{effective_timeout}s"
                )

            try:
                msg = self._parent_recv.recv()
            except (EOFError, OSError) as exc:
                raise SubprocessCrashError(
                    self.model_id,
                    self._process.exitcode if self._process else None,
                    f"recv failed (worker likely crashed): {exc}",
                ) from exc

        if msg.get("type") == "result":
            return msg["text"]
        if msg.get("type") == "error":
            raise RuntimeError(
                f"Worker error for {self.model_id}: {msg.get('error')}"
            )
        raise RuntimeError(f"Unexpected message from worker: {msg}")

    def is_alive(self) -> bool:
        """Check if the worker subprocess is still running."""
        return self._process is not None and self._process.is_alive()

    def shutdown(self, timeout: float = 10.0) -> None:
        """Gracefully shut down the worker subprocess."""
        if self._process is None:
            return
        if not self._process.is_alive():
            self._cleanup()
            return

        try:
            self._parent_send.send({"type": "shutdown"})
        except (BrokenPipeError, OSError):
            pass

        self._process.join(timeout=timeout)
        if self._process.is_alive():
            log.warning(
                "MLXSubprocessRunner: worker pid=%s did not exit gracefully, killing",
                self._worker_pid,
            )
            self._process.kill()
            self._process.join(timeout=5)

        self._cleanup()
        log.info("MLXSubprocessRunner: worker for %s shut down", self.model_id)

    def kill(self) -> None:
        """Force-kill the worker subprocess."""
        if self._process is not None and self._process.is_alive():
            self._process.kill()
            self._process.join(timeout=5)
        self._cleanup()

    def _cleanup(self) -> None:
        """Close pipes, clear references, and release the MLX lock."""
        for conn in (self._parent_send, self._parent_recv):
            if conn is not None:
                try:
                    conn.close()
                except OSError:
                    pass
        self._parent_send = None
        self._parent_recv = None
        self._process = None
        self._started = False
        self._release_mlx_lock_if_held()

    def _release_mlx_lock_if_held(self) -> None:
        """Release the MLX lock if this runner acquired it. Idempotent.

        Best-effort: ``release_mlx_lock`` never raises, so this is safe
        from ``__del__``, error handlers, and normal cleanup paths.
        """
        if not self._holds_mlx_lock:
            return
        try:
            release_mlx_lock()
        except Exception as exc:  # pragma: no cover — defensive
            log.warning(
                "MLXSubprocessRunner: release_mlx_lock failed for %s: %s",
                self.model_id, exc,
            )
        self._holds_mlx_lock = False

    @property
    def worker_pid(self) -> int | None:
        """PID of the worker subprocess, or None if not running."""
        return self._worker_pid if self.is_alive() else None

    def __del__(self) -> None:
        try:
            self.kill()
        except Exception:
            pass

    def __repr__(self) -> str:
        alive = "alive" if self.is_alive() else "dead"
        return (
            f"MLXSubprocessRunner(model={self.model_id!r}, "
            f"pid={self._worker_pid}, {alive})"
        )


# ---------------------------------------------------------------------------
# Layer 7: Auto-managed runner pool
# ---------------------------------------------------------------------------

_runner_pool: dict[str, MLXSubprocessRunner] = {}
_pool_lock = multiprocessing.Lock()


def call_model_isolated(
    prompt: str,
    model: str,
    *,
    backend: str = "mlx",
    max_tokens: int = 4096,
    temperature: float = 0.3,
    system: str = "",
    timeout: float | None = None,
) -> str:
    """Drop-in replacement for direct MLX calls, running inference in a subprocess.

    Automatically manages a pool of worker subprocesses — one per model.
    If a worker crashes (Metal SIGABRT), it is replaced on the next call.

    Returns empty string on crash so the caller can continue the loop
    (matching the conventional error-handling pattern of in-process calls).
    """
    key = f"{model}:{backend}"

    with _pool_lock:
        runner = _runner_pool.get(key)
        if runner is not None and not runner.is_alive():
            exitcode = runner._process.exitcode if runner._process else None
            log.warning(
                "call_model_isolated: previous worker for %s crashed "
                "(exitcode=%s), creating new one",
                model, exitcode,
            )
            runner.kill()
            runner = None

        if runner is None:
            try:
                runner = MLXSubprocessRunner(model, backend=backend)
                _runner_pool[key] = runner
            except (SubprocessTimeoutError, RuntimeError) as exc:
                log.error("call_model_isolated: failed to start worker for %s: %s", model, exc)
                return ""

    try:
        return runner.generate(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            timeout=timeout,
        )
    except SubprocessCrashError as exc:
        log.error("call_model_isolated: %s", exc)
        with _pool_lock:
            runner.kill()
            _runner_pool.pop(key, None)
        return ""
    except SubprocessTimeoutError as exc:
        log.error("call_model_isolated: %s", exc)
        with _pool_lock:
            runner.kill()
            _runner_pool.pop(key, None)
        return ""


def shutdown_all_workers() -> None:
    """Gracefully shut down all worker subprocesses in the pool."""
    with _pool_lock:
        for key, runner in list(_runner_pool.items()):
            runner.shutdown()
        _runner_pool.clear()


# ---------------------------------------------------------------------------
# R4 + R7 — Prefill allocation guard (2026-04-16)
# ---------------------------------------------------------------------------
#
# Single-allocation ceiling guard: for context_tokens × model attention
# geometry, estimate the peak Metal buffer a prefill pass would need
# and refuse the load if it risks IOGPUFamily state corruption
# (mlx#3186 cluster). ``require_fit`` guards model weights;
# ``require_prefill_fit`` guards the single-pass attention tensor which
# scales quadratically with context.
#
# A 131 k context on a 32-head 4-bit model allocates a score tensor
# ~55 GB in one dispatch — IOGPUFamily panics even when the device
# has 60+ GB free. Hard-cap at 5 GB single-allocation by default.
# ``recommend_chunk_size`` (R7) answers "if a full prefill would blow,
# what chunk would fit?" via binary search.


@dataclass(frozen=True)
class ModelDims:
    """Attention geometry needed to estimate prefill peak allocation.

    GQA / MQA aware: ``n_kv_heads ≤ n_heads`` for grouped variants;
    ``n_kv_heads == n_heads`` for MHA.
    """

    n_layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    dtype_bytes: int = 2  # fp16 / bf16 default; 1 for int8, 0.5 for int4


# Curated table of popular Apple-Silicon-deployed model dimensions.
# Not a complete catalog — only well-known shapes likely to appear
# in real workloads. Users can extend at runtime by inserting into
# ``KNOWN_MODELS`` or passing dims directly to ``require_prefill_fit``.
#
# Sources: HuggingFace model cards + mlx-community repacks (2026-04-16).
KNOWN_MODELS: dict[str, ModelDims] = {
    "gemma-4-26b-a4b-it-4bit": ModelDims(n_layers=62, n_heads=16, n_kv_heads=16, head_dim=256),
    "gemma-4-e2b-it-4bit": ModelDims(n_layers=30, n_heads=8, n_kv_heads=4, head_dim=256),
    "gemma-4-e4b-it-4bit": ModelDims(n_layers=34, n_heads=16, n_kv_heads=8, head_dim=256),
    "gemma-4-31b-it-8bit": ModelDims(n_layers=48, n_heads=32, n_kv_heads=16, head_dim=256),
    "Mistral-Small-3.2-24B": ModelDims(n_layers=56, n_heads=32, n_kv_heads=8, head_dim=128),
    "pixtral-12b-4bit": ModelDims(n_layers=40, n_heads=32, n_kv_heads=8, head_dim=128),
    "Hermes-3-Llama-3.1-8B-4bit": ModelDims(n_layers=32, n_heads=32, n_kv_heads=8, head_dim=128),
    "LFM2-VL-3B-4bit": ModelDims(n_layers=24, n_heads=16, n_kv_heads=16, head_dim=128),
}


def lookup_dims(model_id: str) -> ModelDims | None:
    """Best-effort lookup by substring — ``mlx-community/<model>`` prefix ok."""
    basename = model_id.rsplit("/", 1)[-1]
    if basename in KNOWN_MODELS:
        return KNOWN_MODELS[basename]
    for key, dims in KNOWN_MODELS.items():
        if key in model_id:
            return dims
    return None


def estimate_prefill_peak_alloc_gb(
    *, context_tokens: int, dims: ModelDims,
) -> float:
    """Estimate peak single-allocation during prefill, in GB.

    Returns the larger of:

    * Per-layer attention score tensor
      ``n_heads × ctx × ctx × dtype_bytes`` (one fused dispatch)
    * Whole-model KV cache
      ``2 × n_layers × n_kv_heads × ctx × head_dim × dtype_bytes``

    Attention scores scale quadratically in context; KV linearly.
    Scores overtake KV around long contexts on deep models.
    Either can be what tips IOGPUFamily.
    """
    scores_bytes = dims.n_heads * context_tokens * context_tokens * dims.dtype_bytes
    kv_bytes = (
        2 * dims.n_layers * dims.n_kv_heads
        * context_tokens * dims.head_dim * dims.dtype_bytes
    )
    return max(scores_bytes, kv_bytes) / 1e9


def require_prefill_fit(
    *,
    context_tokens: int,
    dims: ModelDims,
    available_gb: float,
    single_alloc_ceiling_gb: float = 5.0,
    headroom_pct: float = 0.30,
) -> None:
    """Raise :class:`MetalOOMError` if a prefill pass risks kernel panic.

    Two fail paths:

    * ``peak > single_alloc_ceiling_gb`` — IOGPUFamily state corruption
      risk (``ml-explore/mlx#3186`` root cause). Hard cap regardless of
      total memory available.
    * ``peak > available_gb × (1 - headroom_pct)`` — not enough headroom
      for the actual generate pass + KV cache overhead + scratch.

    Never called on an unknown model (``dims`` is required). Caller
    responsibility to either ``lookup_dims(model_id)`` or pass dims;
    if dims are unknown, caller should ``log.info`` and skip.
    """
    peak = estimate_prefill_peak_alloc_gb(context_tokens=context_tokens, dims=dims)

    if peak > single_alloc_ceiling_gb:
        raise MetalOOMError(
            f"Prefill peak alloc {peak:.1f} GB > single-alloc ceiling "
            f"{single_alloc_ceiling_gb:.1f} GB at context={context_tokens}. "
            f"IOGPUFamily state corruption risk (mlx#3186). "
            f"Use recommend_chunk_size() for safe segmentation."
        )

    threshold = available_gb * (1.0 - headroom_pct)
    if peak > threshold:
        raise MetalOOMError(
            f"Prefill peak alloc {peak:.1f} GB > {threshold:.1f} GB "
            f"(available {available_gb:.1f} GB × {1 - headroom_pct:.0%} "
            f"headroom) at context={context_tokens}."
        )

    log.debug(
        "require_prefill_fit OK: peak=%.2f GB avail=%.1f GB ctx=%d",
        peak, available_gb, context_tokens,
    )


def recommend_chunk_size(
    *,
    context_tokens: int,
    dims: ModelDims,
    single_alloc_ceiling_gb: float = 4.0,
    min_chunk: int = 512,
) -> int:
    """Return a chunk size whose peak alloc estimate fits the ceiling.

    If the full ``context_tokens`` already fits, returns
    ``context_tokens`` unchanged. Otherwise binary-searches the largest
    chunk that satisfies ``estimate ≤ ceiling``.

    Purely advisory — metal-guard does not chunk on behalf of the
    caller. A batched prefill wrapper picks up the recommendation and
    does its own segmentation.

    Uses a tighter default ceiling (4 GB) than
    :func:`require_prefill_fit` (5 GB) so post-chunk slack remains for
    downstream dispatches.
    """
    if estimate_prefill_peak_alloc_gb(
        context_tokens=context_tokens, dims=dims,
    ) <= single_alloc_ceiling_gb:
        return context_tokens

    lo, hi = min_chunk, context_tokens
    while hi - lo > min_chunk:
        mid = (lo + hi) // 2
        if estimate_prefill_peak_alloc_gb(
            context_tokens=mid, dims=dims,
        ) <= single_alloc_ceiling_gb:
            lo = mid
        else:
            hi = mid
    return lo


def describe_prefill_plan(
    *,
    context_tokens: int,
    model_id: str | None = None,
    dims: ModelDims | None = None,
    available_gb: float | None = None,
) -> dict[str, Any]:
    """Dashboard-ready dict summarising prefill safety for a plan.

    Null-safe for unknown models — never raises. Keys::

        model_id / context_tokens / dims_known /
        peak_alloc_gb / fits_ceiling / fits_headroom /
        recommended_chunk_size / advisory
    """
    resolved_dims = dims
    if resolved_dims is None and model_id is not None:
        resolved_dims = lookup_dims(model_id)

    if resolved_dims is None:
        return {
            "model_id": model_id,
            "context_tokens": context_tokens,
            "dims_known": False,
            "peak_alloc_gb": None,
            "fits_ceiling": None,
            "fits_headroom": None,
            "recommended_chunk_size": None,
            "advisory": (
                "model dims unknown — add to KNOWN_MODELS to enable check"
            ),
        }

    peak = estimate_prefill_peak_alloc_gb(
        context_tokens=context_tokens, dims=resolved_dims,
    )
    fits_ceiling = peak <= 5.0
    fits_headroom: bool | None = None
    if available_gb is not None:
        fits_headroom = peak <= available_gb * 0.70

    advisory: str | None = None
    chunk: int | None = None
    if not fits_ceiling:
        chunk = recommend_chunk_size(
            context_tokens=context_tokens, dims=resolved_dims,
        )
        advisory = (
            f"peak {peak:.1f} GB > 5 GB ceiling — recommend chunk_size={chunk}"
        )
    elif fits_headroom is False:
        advisory = (
            f"peak {peak:.1f} GB > {available_gb * 0.70:.1f} GB headroom"
        )

    return {
        "model_id": model_id,
        "context_tokens": context_tokens,
        "dims_known": True,
        "peak_alloc_gb": round(peak, 3),
        "fits_ceiling": fits_ceiling,
        "fits_headroom": fits_headroom,
        "recommended_chunk_size": chunk,
        "advisory": advisory,
    }


# ---------------------------------------------------------------------------
# R5 — Per-request KV cumulative byte tracker (2026-04-16)
# ---------------------------------------------------------------------------
#
# ``MetalGuard.start_kv_cache_monitor`` watches *global* Metal pressure —
# which is late: a long-running request that steadily grows its KV
# cache can push the device past the IOGPUFamily threshold while the
# global metric still looks fine. Per-request tracking catches that
# specific request early, before the global metric crosses.
#
# Caller opts in; no automatic instrumentation. Pattern::
#
#     from metal_guard import kv_tracker
#     kv_tracker.start(request_id, ceiling_gb=10.0)
#     try:
#         for token in generate(...):
#             kv_tracker.add_bytes(request_id, bytes_for_this_step)
#             yield token
#     finally:
#         kv_tracker.finalize(request_id)


@dataclass
class _RequestRecord:
    """Mutable per-request state for :class:`KVGrowthTracker`."""

    ceiling_bytes: int
    cumulative_bytes: int = 0
    step_count: int = 0
    aborted: bool = False


@dataclass
class KVGrowthTracker:
    """Thread-safe registry of per-request KV growth.

    Not a singleton by type — callers that want a shared registry use
    the module-level :data:`kv_tracker`. Tests can construct their own
    to stay isolated.
    """

    def __post_init__(self) -> None:
        self._requests: dict[str, _RequestRecord] = {}
        self._lock = threading.Lock()

    def start(self, request_id: str, *, ceiling_gb: float = 20.0) -> None:
        """Begin tracking ``request_id``. Overwrites any stale entry."""
        if ceiling_gb <= 0:
            raise ValueError(f"ceiling_gb must be > 0, got {ceiling_gb}")
        with self._lock:
            self._requests[request_id] = _RequestRecord(
                ceiling_bytes=int(ceiling_gb * 1e9),
            )
        log.debug("kv_tracker: started %s ceiling=%.1f GB", request_id, ceiling_gb)

    def add_bytes(self, request_id: str, bytes_added: int) -> int:
        """Record ``bytes_added`` for ``request_id``.

        Returns new cumulative byte count. Raises
        :class:`MetalOOMError` if cumulative would exceed the ceiling;
        the record is marked ``aborted`` and future ``add_bytes`` calls
        re-raise immediately.

        Silently no-ops if ``request_id`` was never ``start``-ed — so
        opt-in instrumentation does not error out of the generate loop.
        """
        if bytes_added < 0:
            raise ValueError(f"bytes_added must be ≥ 0, got {bytes_added}")

        with self._lock:
            rec = self._requests.get(request_id)
            if rec is None:
                return 0
            if rec.aborted:
                self._raise_oom(request_id, rec)
            rec.cumulative_bytes += bytes_added
            rec.step_count += 1
            if rec.cumulative_bytes > rec.ceiling_bytes:
                rec.aborted = True
                self._raise_oom(request_id, rec)
            return rec.cumulative_bytes

    def finalize(self, request_id: str) -> dict[str, Any] | None:
        """Stop tracking; return the final record as a dict (or None)."""
        with self._lock:
            rec = self._requests.pop(request_id, None)
        if rec is None:
            return None
        log.debug(
            "kv_tracker: finalized %s cumulative=%.2f GB steps=%d aborted=%s",
            request_id, rec.cumulative_bytes / 1e9, rec.step_count, rec.aborted,
        )
        return {
            "request_id": request_id,
            "cumulative_gb": round(rec.cumulative_bytes / 1e9, 3),
            "ceiling_gb": round(rec.ceiling_bytes / 1e9, 3),
            "step_count": rec.step_count,
            "aborted": rec.aborted,
        }

    def snapshot(self) -> list[dict[str, Any]]:
        """Return currently tracked requests (dashboard helper)."""
        with self._lock:
            return [
                {
                    "request_id": rid,
                    "cumulative_gb": round(rec.cumulative_bytes / 1e9, 3),
                    "ceiling_gb": round(rec.ceiling_bytes / 1e9, 3),
                    "step_count": rec.step_count,
                    "aborted": rec.aborted,
                }
                for rid, rec in self._requests.items()
            ]

    @staticmethod
    def _raise_oom(request_id: str, rec: _RequestRecord) -> None:
        raise MetalOOMError(
            f"Request {request_id!r} KV cache "
            f"{rec.cumulative_bytes / 1e9:.1f} GB > ceiling "
            f"{rec.ceiling_bytes / 1e9:.1f} GB after {rec.step_count} steps. "
            f"Aborting to prevent Metal kernel panic."
        )


# Module-level shared instance for most callers.
kv_tracker = KVGrowthTracker()


# ---------------------------------------------------------------------------
# R6 — Process-mode detection (2026-04-16)
# ---------------------------------------------------------------------------
#
# Different runtime contexts need different MetalGuard defaults:
#
# * ``server`` — multi-tenant, short-lived requests, tight budgets.
#   mlx-lm#883 / #854 clustered panic reports around long-lived server
#   loops that never flushed between concurrent requests.
# * ``subprocess_worker`` — spawned by :class:`MLXSubprocessRunner`;
#   parent already owns the cross-process MLX lock, child must skip.
# * ``notebook`` — human-driven, relaxed timeouts.
# * ``cli`` — one-shot script; library defaults.
# * ``embedded`` — imported by a host application; defaults match server.
#
# Detection is heuristic and cheap. Never raises; unknown → ``cli``.


ProcessMode = str  # literal: "server" | "embedded" | "notebook" | "cli" | "subprocess_worker"

# Set by MLXSubprocessRunner in the spawned child before it imports
# metal_guard — opt-in explicit marker is more reliable than grepping
# sys.argv for process IDs.
_SUBPROCESS_WORKER_ENV = "METALGUARD_SUBPROCESS_WORKER"


def detect_process_mode() -> ProcessMode:
    """Classify the current process. Called on demand — no caching."""
    argv = sys.argv if sys.argv else []
    argv_str = " ".join(argv)
    argv0 = os.path.basename(argv[0]) if argv else ""

    if os.environ.get(_SUBPROCESS_WORKER_ENV) == "1":
        return "subprocess_worker"

    if "mlx_lm.server" in argv_str or "mlx.server" in argv_str:
        return "server"

    if any(
        token in argv0 or token in argv_str.split()
        for token in ("uvicorn", "gunicorn", "hypercorn", "fastapi")
    ):
        return "server"

    if "ipykernel" in sys.modules or "ipykernel_launcher" in argv_str:
        return "notebook"

    if "pytest" in argv0 or argv0.endswith("_pytest"):
        return "cli"

    if argv0.endswith(".py") or argv0 == "python" or argv0.startswith("python"):
        return "cli"

    return "embedded"


_MODE_DEFAULTS: dict[str, dict[str, Any]] = {
    "server": {
        "generate_timeout_sec": 60.0,
        "periodic_flush_interval_sec": 60.0,
        "kv_ceiling_gb": 10.0,
        "prefill_single_alloc_ceiling_gb": 4.0,
    },
    "embedded": {
        "generate_timeout_sec": 120.0,
        "periodic_flush_interval_sec": 120.0,
        "kv_ceiling_gb": 15.0,
        "prefill_single_alloc_ceiling_gb": 4.5,
    },
    "notebook": {
        "generate_timeout_sec": 600.0,
        "periodic_flush_interval_sec": 900.0,
        "kv_ceiling_gb": 30.0,
        "prefill_single_alloc_ceiling_gb": 5.0,
    },
    "cli": {
        "generate_timeout_sec": 300.0,
        "periodic_flush_interval_sec": 300.0,
        "kv_ceiling_gb": 20.0,
        "prefill_single_alloc_ceiling_gb": 5.0,
    },
    "subprocess_worker": {
        "generate_timeout_sec": 300.0,
        "periodic_flush_interval_sec": 300.0,
        "kv_ceiling_gb": 20.0,
        "prefill_single_alloc_ceiling_gb": 5.0,
        # Worker inherits parent's cross-process lock ownership and
        # must not re-acquire.
        "skip_process_lock": True,
    },
}


def apply_mode_defaults(mode: ProcessMode | None = None) -> dict[str, Any]:
    """Return recommended config dict for ``mode``.

    If ``mode`` is ``None``, detects via :func:`detect_process_mode`.
    Returned dict is a fresh copy.
    """
    resolved = mode if mode is not None else detect_process_mode()
    defaults = _MODE_DEFAULTS.get(resolved, _MODE_DEFAULTS["cli"])
    result = dict(defaults)
    result["mode"] = resolved
    return result


def describe_process_mode() -> dict[str, Any]:
    """Dashboard-ready summary of current process classification."""
    mode = detect_process_mode()
    return {
        "mode": mode,
        "argv0": os.path.basename(sys.argv[0]) if sys.argv else "",
        "has_ipykernel": "ipykernel" in sys.modules,
        "subprocess_worker_env": os.environ.get(_SUBPROCESS_WORKER_ENV, ""),
        "defaults": apply_mode_defaults(mode),
    }


# ---------------------------------------------------------------------------
# R8 — Apple Feedback Assistant panic report formatter (2026-04-16)
# ---------------------------------------------------------------------------
#
# ``ml-explore/mlx#3186`` (FB22091885) is the canonical template Apple
# accepts for IOGPUFamily kernel panics. This function converts a
# forensics dict into a ready-to-paste Feedback Assistant report so
# operators don't hand-stitch it during a reboot recovery. Pure
# formatting, no I/O.


def format_panic_for_apple_feedback(
    forensics: dict[str, Any],
    *,
    include_breadcrumb: bool = True,
    max_breadcrumb_lines: int = 60,
) -> str:
    """Return a multi-line string ready to paste into Feedback Assistant.

    Expected ``forensics`` keys (all optional — missing values render
    as ``unknown``): ``panic_string``, ``panic_time``, ``hardware``,
    ``gpu_driver``, ``os_version``, ``kernel_version``, ``mlx_versions``,
    ``repro_steps``, ``breadcrumbs``, ``advisories``.
    """
    hw = forensics.get("hardware", {}) or {}
    panic = forensics.get("panic_string", "unknown")
    panic_time = forensics.get("panic_time", "unknown")
    kext = forensics.get("gpu_driver", "unknown")
    os_ver = forensics.get("os_version", "unknown")
    kernel = forensics.get("kernel_version", "unknown")
    repro = forensics.get("repro_steps", []) or []
    breadcrumbs = (
        forensics.get("breadcrumbs", []) or [] if include_breadcrumb else []
    )
    advisories = forensics.get("advisories", []) or []
    mlx_versions = forensics.get("mlx_versions", {}) or {}

    lines: list[str] = []
    lines.append(f"**PANIC STRING:** {panic}")
    lines.append(f"**TIME:** {panic_time}")
    lines.append("")

    lines.append("## System")
    lines.append(f"- Hardware: {hw.get('chip', 'unknown')} "
                 f"({hw.get('gpu_memory_gb', '?')} GB unified)")
    lines.append(f"- macOS: {os_ver}")
    lines.append(f"- Kernel: {kernel}")
    lines.append(f"- IOGPUFamily kext: {kext}")
    if mlx_versions:
        lines.append("- MLX stack:")
        for pkg, ver in sorted(mlx_versions.items()):
            lines.append(f"  - {pkg}: {ver}")
    lines.append("")

    lines.append("## Reproducer")
    if repro:
        for i, step in enumerate(repro, 1):
            lines.append(f"{i}. {step}")
    else:
        lines.append("(No reproducer supplied — see breadcrumb tail below)")
    lines.append("")

    if advisories:
        lines.append("## Active advisories at panic time")
        for a in advisories:
            sev = a.get("severity", "?")
            pkg = a.get("package", "?")
            ver = a.get("installed_version", "?")
            url = a.get("url", "")
            title = a.get("title", "")
            lines.append(f"- [{sev}] {pkg} {ver} — {title} ({url})")
        lines.append("")

    if include_breadcrumb and breadcrumbs:
        tail = breadcrumbs[-max_breadcrumb_lines:]
        lines.append(f"## Breadcrumb tail (last {len(tail)} lines)")
        lines.append("```")
        lines.extend(str(b).rstrip() for b in tail)
        lines.append("```")
        lines.append("")

    lines.append("## What MetalGuard observed")
    lines.append("MetalGuard is a Python-side defensive layer for MLX on "
                 "Apple Silicon. At the time of the panic it was enforcing "
                 "the defensive-mode MLX process lock, subprocess worker "
                 "isolation, prefill allocation guard, and per-request KV "
                 "growth tracking. The panic still occurred despite these "
                 "guards, which suggests an in-kernel IOGPUFamily state "
                 "issue rather than a Python-side race.")
    lines.append("")
    lines.append("## Related upstream issues")
    lines.append("- mlx#3186 (canonical IOGPUFamily panic)")
    lines.append("- mlx#3348 (CommandEncoder thread-local)")
    lines.append("- mlx#3346 (merged into #3186)")
    lines.append("- mlx-lm#1047 (wired_limit correlation)")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# L9 — Cadence Guard (2026-04-16)
#
# Eighth-hole patch. Production kernel panic at 2026-04-16 23:33:27 was
# IOGPUMemory.cpp:492 "completeMemory() prepare count underflow" during
# the first generate() after a freshly-loaded subprocess worker — i.e.
# the SIGABRT handler was installed (L6) but the panic happened at
# kernel level before any user-space signal could fire. Root-cause
# matches the Apple backend log note: "Reproduces under back-to-back
# load/unload cycles". Only way to avoid it is to not do back-to-back
# loads.
# ---------------------------------------------------------------------------


class CadenceViolation(RuntimeError):
    """Raised when a back-to-back MLX load would risk IOGPU underflow."""

    def __init__(self, model_id: str, last_ts: float, min_interval: float) -> None:
        self.model_id = model_id
        self.last_ts = last_ts
        self.min_interval = min_interval
        self.delta = max(0.0, time.time() - last_ts)
        super().__init__(
            f"CadenceGuard blocked load of {model_id!r}: last cycle "
            f"{self.delta:.1f}s ago < min_interval {min_interval:.0f}s"
        )


class CrossModelCadenceViolation(CadenceViolation):
    """Raised when loading a *different* model too soon after another.

    Added v0.9.0 (2026-04-25). Harper's panic timeline (8 IOGPU kernel
    panics between 2026-04-16 and 2026-04-24, see CHANGELOG) showed that
    back-to-back loads of **different** models trigger the same
    ``IOGPUMemory.cpp:492 prepare_count_underflow`` panic as back-to-back
    same-model loads. The classic same-model ``min_interval`` check let
    these through because every ``model_id`` was different.

    Inherits from :class:`CadenceViolation` so ``except CadenceViolation``
    still catches both variants.
    """

    def __init__(
        self,
        model_id: str,
        last_model: str,
        last_ts: float,
        cross_model_interval: float,
    ) -> None:
        self.last_model = last_model
        self.cross_model_interval = cross_model_interval
        super().__init__(model_id, last_ts, cross_model_interval)
        self.args = (
            f"CadenceGuard blocked cross-model load of {model_id!r}: "
            f"last load was {last_model!r} {self.delta:.1f}s ago "
            f"< cross_model_interval {cross_model_interval:.0f}s",
        )


_CADENCE_PATH_DEFAULT = os.path.expanduser("~/.cache/metal-guard/cadence.json")
_CADENCE_FILE_LOCK = threading.Lock()
_CADENCE_MAX_AGE_SEC = 14400.0  # GC entries older than 4h — far past any min_interval

# --- Cross-model cadence (v0.9.0, C5 phase-1, 2026-04-25) ------------------
# Non-gemma-4 baseline cadence between loads of *different* models. Set to 0
# to disable. Env var ``METALGUARD_CROSS_MODEL_INTERVAL`` overrides default.
_CROSS_MODEL_ENV_VAR = "METALGUARD_CROSS_MODEL_INTERVAL"
_C5_DEFAULT_CROSS_MODEL_INTERVAL_SEC = 60.0

# Gemma-4 family floor (v0.9.0, C5+C7, 2026-04-25). 8/8 IOGPUMemory.cpp:492
# panics in Harper's timeline had at-panic model in the gemma-4 family; panic
# #6 landed 66s after prior unload, so 90s = 66s + ~36% safety margin. Floor
# applies regardless of configured base (users who set cross_model_interval=0
# still get this minimum for gemma-4).
GEMMA4_MIN_CROSS_MODEL_INTERVAL_SEC = 90.0

# Size-anchored regex for gemma-4 family match. Anchors prevent false-positive
# matches on ``gemma-4b-*`` (4 billion, not Gen4) or ``google/gemma-4-*`` (not
# a real model) while admitting both standard and MoE variants:
#   mlx-community/gemma-4-26b-a4b-it-4bit
#   mlx-community/gemma-4-31b-it-8bit
#   mlx-community/gemma-4-e4b-it-4bit
#   unsloth/gemma-4-31b-it-UD-MLX-4bit
_GEMMA4_NAME_PATTERN = re.compile(
    r"^gemma-4-(?:\d+b|e\d+b)-"    # size-anchored: digits+b or e+digit+b
    r"[a-z0-9\-]*"                 # variant tail
    r"$",
    re.IGNORECASE,
)
_GEMMA4_VENDOR_ALLOWLIST = frozenset({"mlx-community", "unsloth", "mlx-models"})


def _is_gemma4_family(model_id: str) -> bool:
    """Return True for known gemma-4 family model identifiers.

    Vendor allowlist + size-anchored basename match. Used by
    :class:`CadenceGuard` to enforce the 90s floor on cross-model cadence
    and by :func:`gemma4_generation_flush` to decide whether to insert a
    first-generate settle window.
    """
    if not model_id or "/" not in model_id:
        return False
    vendor, _, name = model_id.lower().partition("/")
    if vendor not in _GEMMA4_VENDOR_ALLOWLIST:
        return False
    return bool(_GEMMA4_NAME_PATTERN.match(name))


class CadenceGuard:
    """Per-model load timestamp store + min-interval enforcement.

    Persisted to JSON on disk so timestamps survive subprocess spawn and
    kernel panics. Cross-process safe via atomic write (``os.replace``).

    Typical use (inside the worker subprocess, right before
    ``mlx_lm.load``)::

        from metal_guard import require_cadence_clear, CadenceViolation
        try:
            require_cadence_clear(model_id)
        except CadenceViolation:
            # Exit cleanly — the parent will propagate as a worker error
            sys.exit(2)
    """

    def __init__(
        self,
        path: str | None = None,
        *,
        min_interval_sec: float = 180.0,
        cross_model_interval_sec: float = 0.0,
    ) -> None:
        # Resolve default at call time so test monkeypatches of the
        # module-level constant take effect.
        self._path = os.path.expanduser(path or _CADENCE_PATH_DEFAULT)
        self._min_interval = float(min_interval_sec)
        # Zero disables the check (backwards-compat default). Non-zero means
        # reject any load of a *different* model_id within this many seconds
        # of the most-recent load. gemma-4 family is always floored at
        # GEMMA4_MIN_CROSS_MODEL_INTERVAL_SEC regardless of this value.
        self._cross_model_interval = max(0.0, float(cross_model_interval_sec))

    @property
    def path(self) -> str:
        return self._path

    @property
    def min_interval_sec(self) -> float:
        return self._min_interval

    @property
    def cross_model_interval_sec(self) -> float:
        return self._cross_model_interval

    def _effective_cross_model_interval(self, model_id: str) -> float:
        """Return effective cross-model cadence for target model.

        Gemma-4 family floor is ``GEMMA4_MIN_CROSS_MODEL_INTERVAL_SEC``
        regardless of the configured base (see class docstring for why).
        """
        base = self._cross_model_interval
        if _is_gemma4_family(model_id):
            return max(base, GEMMA4_MIN_CROSS_MODEL_INTERVAL_SEC)
        return base

    def _read(self) -> dict[str, float]:
        try:
            with open(self._path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError, ValueError):
            return {}
        if not isinstance(data, dict):
            return {}
        result: dict[str, float] = {}
        for k, v in data.items():
            if isinstance(v, (int, float)):
                result[str(k)] = float(v)
        return result

    def _write(self, data: dict[str, float]) -> None:
        dirname = os.path.dirname(self._path)
        if dirname:
            try:
                os.makedirs(dirname, exist_ok=True)
            except OSError:
                return
        tmp = f"{self._path}.tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, sort_keys=True)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self._path)
        except OSError as exc:
            log.debug("CadenceGuard write failed: %s", exc)

    def _gc_stale(self, data: dict[str, float], now: float) -> dict[str, float]:
        return {k: v for k, v in data.items() if now - v < _CADENCE_MAX_AGE_SEC}

    def last_ts(self, model_id: str) -> float | None:
        with _CADENCE_FILE_LOCK:
            return self._read().get(model_id)

    def mark_load(self, model_id: str, *, ts: float | None = None) -> None:
        now = ts if ts is not None else time.time()
        with _CADENCE_FILE_LOCK:
            data = self._gc_stale(self._read(), now)
            data[model_id] = now
            self._write(data)

    def check(self, model_id: str) -> None:
        """Raise CadenceViolation / CrossModelCadenceViolation if unsafe.

        Two independent checks (same-model has priority):

        1. ``min_interval_sec`` — same ``model_id`` within that window.
        2. cross-model interval — any *different* ``model_id`` within
           ``_effective_cross_model_interval(model_id)``. gemma-4 family
           enforces a 90s floor regardless of configured base.
        """
        with _CADENCE_FILE_LOCK:
            data = self._read()
        now = time.time()
        # 1. Same-model back-to-back (classic L9).
        last = data.get(model_id)
        if last is not None and now - last < self._min_interval:
            raise CadenceViolation(model_id, last, self._min_interval)
        # 2. Cross-model back-to-back (v0.9.0 C5).
        effective_cross = self._effective_cross_model_interval(model_id)
        if effective_cross > 0.0:
            most_recent_other: tuple[str, float] | None = None
            for other_id, other_ts in data.items():
                if other_id == model_id:
                    continue
                if most_recent_other is None or other_ts > most_recent_other[1]:
                    most_recent_other = (other_id, other_ts)
            if most_recent_other is not None:
                other_id, other_ts = most_recent_other
                if now - other_ts < effective_cross:
                    raise CrossModelCadenceViolation(
                        model_id, other_id, other_ts, effective_cross,
                    )


def _resolve_cross_model_interval(explicit: float | None) -> float:
    """Explicit arg wins; else env var; else default.

    Resolution order:

    1. Explicit ``explicit`` argument (including ``0.0`` to disable).
    2. ``METALGUARD_CROSS_MODEL_INTERVAL`` environment variable.
    3. ``_C5_DEFAULT_CROSS_MODEL_INTERVAL_SEC`` (60s).

    Invalid env var values fall back to the default with a warning.
    """
    if explicit is not None:
        return max(0.0, float(explicit))
    raw = os.getenv(_CROSS_MODEL_ENV_VAR, "").strip()
    if not raw:
        return _C5_DEFAULT_CROSS_MODEL_INTERVAL_SEC
    try:
        return max(0.0, float(raw))
    except ValueError:
        log.warning(
            "Invalid %s=%r, falling back to default %.0fs",
            _CROSS_MODEL_ENV_VAR, raw, _C5_DEFAULT_CROSS_MODEL_INTERVAL_SEC,
        )
        return _C5_DEFAULT_CROSS_MODEL_INTERVAL_SEC


def require_cadence_clear(
    model_id: str,
    *,
    min_interval_sec: float = 180.0,
    cross_model_interval_sec: float = 0.0,
    guard: CadenceGuard | None = None,
) -> None:
    """Check + mark atomic helper.

    Checks cadence; on pass, marks the load timestamp (so the NEXT call
    within ``min_interval`` will be blocked, including retries after a
    kernel panic since the mark is persisted to disk before the crash).

    v0.9.0 additions:

    - ``cross_model_interval_sec`` (new): seconds required between loads
      of *different* models. Default is ``0.0`` (disabled) to preserve
      backwards-compatibility with v0.8.x callers. To opt in globally
      without code changes, set ``METALGUARD_CROSS_MODEL_INTERVAL=<sec>``
      in the environment; the CadenceGuard this helper creates will read
      that env value. Explicit ``cross_model_interval_sec`` argument
      takes priority over the env var when non-zero; pass ``0.0`` (or
      omit) to defer to the env var. Ignored when ``guard`` is supplied
      (the guard's own ``cross_model_interval_sec`` wins).
    - Raises :class:`CrossModelCadenceViolation` (subclass of
      :class:`CadenceViolation`) when a cross-model violation fires.

    Note on subclassing: in v0.9.0 ``CadenceGuard.check()`` reads the
    JSON store directly under ``_CADENCE_FILE_LOCK`` instead of calling
    ``self.last_ts()``, so subclasses that overrode ``last_ts()`` to
    customise the read path will no longer be invoked by ``check()``.
    This is a tiny subclassing-contract change; the public API on the
    instance is unchanged.
    """
    if guard is None:
        # v0.9.0: explicit arg wins when non-zero; zero defers to env var
        # (which itself falls back to disabled when unset).
        resolved_cross = (
            cross_model_interval_sec
            if cross_model_interval_sec > 0
            else _resolve_cross_model_interval_for_require()
        )
        cg = CadenceGuard(
            min_interval_sec=min_interval_sec,
            cross_model_interval_sec=resolved_cross,
        )
    else:
        cg = guard
    cg.check(model_id)
    cg.mark_load(model_id)


def _resolve_cross_model_interval_for_require() -> float:
    """v0.9.0 require_cadence_clear env-var fallback.

    Returns env var value when set (including explicit ``0`` to disable),
    else ``0.0`` to preserve v0.8.x behaviour. This intentionally differs
    from :func:`_resolve_cross_model_interval` (which defaults to 60s)
    because :func:`require_cadence_clear` is the back-compat entry point;
    the 60s C5 default is opt-in via ``CadenceGuard`` constructor or env.
    """
    raw = os.getenv(_CROSS_MODEL_ENV_VAR, "").strip()
    if not raw:
        return 0.0
    try:
        return max(0.0, float(raw))
    except ValueError:
        log.warning(
            "Invalid %s=%r, falling back to 0.0 (disabled)",
            _CROSS_MODEL_ENV_VAR, raw,
        )
        return 0.0


# ---------------------------------------------------------------------------
# L9 — Panic Report Ingest (2026-04-16)
# ---------------------------------------------------------------------------

_PANIC_REPORT_DIR = "/Library/Logs/DiagnosticReports"
_PANIC_JSONL_PATH = os.path.expanduser("~/.cache/metal-guard/panics.jsonl")
_PANIC_CALENDAR_RE = re.compile(r"Calendar:\s*0x([0-9a-fA-F]+)\s+0x([0-9a-fA-F]+)")
_PANIC_PID_RE = re.compile(r"pid\s+(\d+):\s*\S+")
_PANIC_FILE_MAX_READ = 200_000  # chars — panic files can be 500k+, only need header + signatures


def _parse_panic_timestamp(text: str) -> float | None:
    """Extract ``Calendar: 0x<sec> 0x<usec>`` as float seconds-since-epoch."""
    m = _PANIC_CALENDAR_RE.search(text)
    if not m:
        return None
    try:
        sec = int(m.group(1), 16)
        usec = int(m.group(2), 16)
        return float(sec) + usec / 1_000_000.0
    except ValueError:
        return None


def _parse_panic_pid(text: str) -> int | None:
    m = _PANIC_PID_RE.search(text)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def parse_panic_reports(
    directory: str | None = None,
    *,
    since_ts: float | None = None,
) -> list[dict[str, Any]]:
    """Scan ``*.panic`` files and classify them.

    Silently returns ``[]`` if the directory is unreadable (common —
    admin permissions required on modern macOS for
    ``/Library/Logs/DiagnosticReports``). Callers are expected to also
    look at backend-side / application panic text when available.
    """
    directory = directory or _PANIC_REPORT_DIR
    results: list[dict[str, Any]] = []

    # Scan both the top-level directory and its ``Retired/`` subdir.
    # macOS moves older panic reports into ``Retired/`` after a few days;
    # missing it means the archive never sees panics more than ~48h old.
    scan_dirs = [directory, os.path.join(directory, "Retired")]
    scanned: set[str] = set()

    for scan_dir in scan_dirs:
        if scan_dir in scanned:
            continue
        scanned.add(scan_dir)
        try:
            entries = sorted(os.listdir(scan_dir))
        except OSError:
            continue

        for name in entries:
            if not name.endswith(".panic"):
                continue
            path = os.path.join(scan_dir, name)
            try:
                with open(path, encoding="utf-8", errors="ignore") as fh:
                    text = fh.read(_PANIC_FILE_MAX_READ)
            except OSError:
                continue

            ts = _parse_panic_timestamp(text)
            if ts is None:
                try:
                    ts = os.path.getmtime(path)
                except OSError:
                    continue
            if since_ts is not None and ts < since_ts:
                continue

            signature, explanation = detect_panic_signature(text)
            results.append({
                "ts": ts,
                "signature": signature or "unknown",
                "explanation": explanation,
                "pid": _parse_panic_pid(text),
                "source_file": path,
            })
    return results


def ingest_panics_jsonl(
    *,
    report_dir: str | None = None,
    jsonl_path: str | None = None,
) -> int:
    """Append any new panic records to the JSONL archive.

    Dedupes by ``source_file`` **and** ``(ts_bucket, pid)`` event key.
    Idempotent: repeated calls append zero new records once the archive
    catches up. Returns number of new records written. Never raises —
    best-effort; panic archival must never itself crash the caller.
    """
    report_dir = report_dir or _PANIC_REPORT_DIR
    jsonl_path = os.path.expanduser(jsonl_path or _PANIC_JSONL_PATH)
    seen_paths: set[str] = set()
    seen_events: set[tuple[int, int | None]] = set()

    def _event_key(rec: dict[str, Any]) -> tuple[int, int | None]:
        """Deduplicate across multiple DiagnosticReports copies of the same
        panic (macOS writes both ``.contents.panic`` and
        ``panic-full-...`` for the most recent event — both parse to the
        same ts+pid).
        """
        ts = rec.get("ts")
        pid = rec.get("pid")
        ts_bucket = int(ts) if isinstance(ts, (int, float)) else 0
        return (ts_bucket, pid if isinstance(pid, int) else None)

    try:
        with open(jsonl_path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(rec, dict):
                    continue
                src = rec.get("source_file")
                if isinstance(src, str):
                    seen_paths.add(src)
                seen_events.add(_event_key(rec))
    except OSError:
        pass

    new_records: list[dict[str, Any]] = []
    for rec in parse_panic_reports(report_dir):
        if rec.get("source_file") in seen_paths:
            continue
        key = _event_key(rec)
        if key in seen_events:
            continue
        seen_events.add(key)
        new_records.append(rec)
    if not new_records:
        return 0

    dirname = os.path.dirname(jsonl_path)
    if dirname:
        try:
            os.makedirs(dirname, exist_ok=True)
        except OSError:
            pass

    try:
        with open(jsonl_path, "a", encoding="utf-8") as fh:
            for rec in new_records:
                fh.write(json.dumps(rec, sort_keys=True))
                fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
    except OSError as exc:
        log.warning("ingest_panics_jsonl: write failed path=%s err=%s", jsonl_path, exc)
        return 0

    return len(new_records)


# ---------------------------------------------------------------------------
# L9 — Circuit Breaker (2026-04-16)
# ---------------------------------------------------------------------------


class MLXCooldownActive(RuntimeError):
    """Raised when MetalGuard refuses to spawn a worker during a cooldown.

    Callers should surface this (HTTP 503, task runner decline, CLI
    exit) rather than silently fall through — the kernel is in a bad
    state and another load is likely to panic the machine again.
    """

    def __init__(self, panic_count: int, window_sec: float, cooldown_until: float) -> None:
        self.panic_count = panic_count
        self.window_sec = window_sec
        self.cooldown_until = cooldown_until
        self.remaining_sec = max(0.0, cooldown_until - time.time())
        super().__init__(
            f"MLX cooldown active — {panic_count} panics in last "
            f"{window_sec / 3600:.0f}h. Retry in {self.remaining_sec:.0f}s "
            f"(until {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(cooldown_until))})."
        )


_BREAKER_STATE_PATH = os.path.expanduser("~/.cache/metal-guard/breaker.json")


class CircuitBreaker:
    """Refuse new MLX workers after repeated panics within a rolling window.

    Default policy: ≥2 kernel panics within **1h** → 1h cooldown. Tuned
    around the empirical observation that CadenceGuard (L9 sibling)
    already enforces ≥180s between same-model loads — so two panics
    within a single hour means cadence guard failed AND the kernel's
    IOGPU accounting is compromised. A wider window (e.g. 24h) was
    tried first but tripped chronically on a machine that averages
    ~1.4 panics/day over a week: over-defensive to the point of
    blocking all MLX work. The 1h window catches catastrophic
    clusters without punishing the steady-state background rate.

    Humans should still reboot + investigate after any cooldown trip.
    """

    def __init__(
        self,
        *,
        jsonl_path: str | None = None,
        state_path: str | None = None,
        window_sec: float = 3600,
        panic_threshold: int = 2,
        cooldown_sec: float = 3600,
    ) -> None:
        self._jsonl_path = os.path.expanduser(jsonl_path or _PANIC_JSONL_PATH)
        self._state_path = os.path.expanduser(state_path or _BREAKER_STATE_PATH)
        self._window = float(window_sec)
        self._threshold = int(panic_threshold)
        self._cooldown = float(cooldown_sec)

    def _load_state(self) -> dict[str, Any]:
        try:
            with open(self._state_path, encoding="utf-8") as fh:
                data = json.load(fh)
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def _save_state(self, data: dict[str, Any]) -> None:
        dirname = os.path.dirname(self._state_path)
        if dirname:
            try:
                os.makedirs(dirname, exist_ok=True)
            except OSError:
                return
        tmp = f"{self._state_path}.tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(data, fh, sort_keys=True)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self._state_path)
        except OSError as exc:
            log.debug("CircuitBreaker save_state failed: %s", exc)

    def _recent_panic_count(self, now: float) -> int:
        cutoff = now - self._window
        count = 0
        try:
            with open(self._jsonl_path, encoding="utf-8") as fh:
                for line in fh:
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    ts = rec.get("ts") if isinstance(rec, dict) else None
                    if isinstance(ts, (int, float)) and ts >= cutoff:
                        count += 1
        except OSError:
            return 0
        return count

    def check(self, *, now: float | None = None) -> None:
        """Raise MLXCooldownActive if the breaker is currently tripped."""
        current = now if now is not None else time.time()
        state = self._load_state()
        cooldown_until = state.get("cooldown_until")
        if isinstance(cooldown_until, (int, float)) and cooldown_until > current:
            raise MLXCooldownActive(
                int(state.get("panic_count", self._threshold)),
                self._window,
                float(cooldown_until),
            )

        count = self._recent_panic_count(current)
        if count >= self._threshold:
            until = current + self._cooldown
            self._save_state({
                "cooldown_until": until,
                "panic_count": count,
                "entered_at": current,
                "window_sec": self._window,
                "threshold": self._threshold,
            })
            raise MLXCooldownActive(count, self._window, until)

    def clear(self) -> None:
        """Operator override — remove the current cooldown state."""
        try:
            os.remove(self._state_path)
        except OSError:
            pass

    def status(self, *, now: float | None = None) -> dict[str, Any]:
        current = now if now is not None else time.time()
        state = self._load_state()
        cooldown_until = state.get("cooldown_until")
        in_cooldown = isinstance(cooldown_until, (int, float)) and cooldown_until > current
        return {
            "in_cooldown": bool(in_cooldown),
            "cooldown_until": float(cooldown_until) if in_cooldown else None,
            "cooldown_remaining_sec": max(0.0, float(cooldown_until) - current) if in_cooldown else 0.0,
            "panic_threshold": self._threshold,
            "window_sec": self._window,
            "recent_panic_count": self._recent_panic_count(current),
        }


# ---------------------------------------------------------------------------
# B2 — subprocess_inference_guard (per-inference Metal flush)
# ---------------------------------------------------------------------------
#
# Background: if you use subprocess isolation for MLX inference (a common
# pattern to contain C++ crashes — see L7 in Harper's internal fork), the
# worker process runs ``gen_fn(...)`` in its own address space. Parent-
# process hooks such as ``mx.clear_cache()`` or post-generate
# ``mx.eval(result)`` do NOT reach the subprocess. Empirically this
# reproduces the ``IOGPUMemory.cpp:492 "completeMemory() prepare count
# underflow"`` kernel panic: the GPU command buffer is still in flight
# when the subprocess releases its memory, and Metal accounting drifts
# until the next load trips an underflow → kernel panic.
#
# See upstream reports:
#   - mlx-lm: ml-explore/mlx-lm#1128  (prefill guard design)
#   - mlx:    ml-explore/mlx#3186     (subprocess isolation guidance)
#   - mlx:    ml-explore/mlx#3346     (kernel panic reproducers)
#
# Fix: wrap every ``gen_fn`` call in the worker with this guard. Order:
#
#   PRE  — mx.clear_cache()          release prior iter's cached buffers
#   body — generate runs
#   POST — mx.synchronize()          block until GPU command buffer drained
#   POST — mx.clear_cache()          release this iter's buffers
#
# ``mx.synchronize()`` is chosen over ``mx.eval(result)`` because generate
# return types vary (``str`` for mlx-lm, ``GenerateResult`` for mlx-vlm).
# ``synchronize`` is return-type agnostic and semantically correct: wait
# for all pending GPU ops to complete.
#
# This API is a pure addition — no breaking changes. It is useful to any
# project doing subprocess-based MLX isolation; Harper's internal B1 fix
# ended a streak of 6 panics in 3 days after wiring this into the worker.
@contextmanager
def subprocess_inference_guard(model_id: str) -> Generator[None, None, None]:
    """Per-inference Metal flush for subprocess workers.

    Wrap each ``gen_fn(...)`` call inside your subprocess worker::

        from metal_guard import subprocess_inference_guard

        with subprocess_inference_guard(model_id):
            result = gen_fn(model, processor, prompt=..., ...)

    The guard is best-effort: if ``mlx.core`` cannot be imported (test
    environment without MLX) the context manager no-ops. PRE hook
    failures do NOT block the body — a broken guard is worse than no
    guard because it would silently stop all inference. POST hook
    failures are logged but never mask a body exception.

    Args:
        model_id: model identifier used only for breadcrumb forensics.
            A post-mortem of a kernel panic can grep the breadcrumb log
            for ``SUBPROC_PRE`` / ``SUBPROC_POST`` entries to confirm
            whether the guard was engaged on the fatal call.
    """
    try:
        import mlx.core as mx
    except ImportError:
        yield
        return
    except Exception as exc:
        log.warning(
            "subprocess_inference_guard: mlx.core import failed (%s); "
            "guard disabled for %s", exc, model_id,
        )
        yield
        return

    # PRE: release any cached buffers from the previous iteration before
    # starting the new generate. Best-effort — a failure here must not
    # block the body (see docstring rationale).
    try:
        mx.clear_cache()
    except Exception as exc:
        log.warning(
            "subprocess_inference_guard PRE clear_cache failed for %s: %s",
            model_id, exc,
        )
    try:
        metal_guard.breadcrumb(f"SUBPROC_PRE: {model_id}")
    except Exception as exc:
        log.debug(
            "subprocess_inference_guard PRE breadcrumb failed for %s: %s",
            model_id, exc,
        )

    try:
        yield
    finally:
        # POST: synchronize FIRST so the command buffer is fully drained
        # before we release the cache. Reversing this order reproduces
        # the IOGPUMemory.cpp:492 underflow this guard exists to prevent.
        try:
            mx.synchronize()
        except Exception as exc:
            log.warning(
                "subprocess_inference_guard POST synchronize failed for %s: %s",
                model_id, exc,
            )
        try:
            mx.clear_cache()
        except Exception as exc:
            log.warning(
                "subprocess_inference_guard POST clear_cache failed for %s: %s",
                model_id, exc,
            )
        try:
            metal_guard.breadcrumb(f"SUBPROC_POST: {model_id}")
        except Exception as exc:
            log.debug(
                "subprocess_inference_guard POST breadcrumb failed for %s: %s",
                model_id, exc,
            )


# ---------------------------------------------------------------------------
# v0.9.0 — Gemma-4 first-generate settle (C7, 2026-04-25)
# ---------------------------------------------------------------------------
#
# Empirical analysis of Harper's 8-panic timeline: 7 of 8 panics landed on
# the FIRST ``generate()`` call of a gemma-4 family model, within 7–66s of
# the worker becoming ready. Four pre-existing flush barriers (parent-side
# clear_cache, subprocess_inference_guard PRE, load barrier, shutdown
# barrier) failed to catch these — the race is between Metal command buffer
# completion on model load and the first forward pass.
#
# ``gemma4_generation_flush`` inserts a mandatory synchronize + clear_cache
# + sleep window between worker load and first generate. It is NOT a block:
# it cannot stop a panic that is already in flight at the kernel level.
# What it does is extend the settle window so the kernel has time to finish
# its own bookkeeping before the first user-space Metal allocation. In
# combination with cross-model cadence (v0.9.0 C5) it meaningfully reduces
# panic frequency on gemma-4 loads but does not eliminate it — see
# KNOWN_PANIC_MODELS for the escape hatch when metal-guard is not enough.
#
# Env knobs:
#   METALGUARD_GEMMA4_FIRSTGEN_DISABLED=1   -> disable entirely
#   METALGUARD_GEMMA4_FIRSTGEN_SLEEP_SEC    -> override sleep (default 3.0)
def gemma4_generation_flush(model_id: str, generate_call_count: int) -> None:
    """First-generate Metal settle window for the gemma-4 family.

    Call this immediately before the *first* ``generate()`` on a freshly
    loaded worker. Subsequent calls (``generate_call_count != 0``) are
    no-ops. Non-gemma-4 models are no-ops.

    The function performs a best-effort ``mx.synchronize()`` +
    ``mx.clear_cache()`` + ``time.sleep(3.0)``. All steps are optional;
    the function never raises and never blocks on failure of any step.

    Args:
        model_id: the model identifier to match against the gemma-4 family.
        generate_call_count: number of prior successful generate calls on
            this worker. ``0`` means this call will be the first.

    Renamed in v0.9.0 from ``gemma4_firstgen_guard`` — the name "guard"
    incorrectly suggested a block; the function is a flush + settle window,
    not a gate. If you need a gate, use :class:`CircuitBreaker` or
    :func:`require_cadence_clear`.
    """
    if os.environ.get("METALGUARD_GEMMA4_FIRSTGEN_DISABLED") == "1":
        return
    if generate_call_count != 0:
        return
    if not _is_gemma4_family(model_id):
        return

    try:
        import mlx.core as mx
    except ImportError:
        return
    except Exception as exc:
        log.warning(
            "gemma4_generation_flush: mlx.core import failed (%s); "
            "flush disabled for %s", exc, model_id,
        )
        return

    try:
        metal_guard.breadcrumb(f"GEMMA4_FLUSH: {model_id}")
    except Exception as exc:
        log.debug("gemma4_generation_flush breadcrumb PRE failed: %s", exc)

    try:
        mx.synchronize()
    except Exception as exc:
        # log.debug rather than warning: failures of the best-effort
        # flush are dominated by the kernel-panic signal itself.
        # Emitting warnings per-load produces log spam when the driver
        # is already in trouble.
        log.debug(
            "gemma4_generation_flush synchronize failed for %s: %s",
            model_id, exc,
        )
    try:
        mx.clear_cache()
    except Exception as exc:
        log.debug(
            "gemma4_generation_flush clear_cache failed for %s: %s",
            model_id, exc,
        )

    try:
        sleep_sec = float(
            os.environ.get("METALGUARD_GEMMA4_FIRSTGEN_SLEEP_SEC", "3.0")
        )
    except ValueError:
        sleep_sec = 3.0
    time.sleep(max(0.0, sleep_sec))

    try:
        metal_guard.breadcrumb(f"GEMMA4_FLUSH_DONE: {model_id}")
    except Exception as exc:
        log.debug("gemma4_generation_flush breadcrumb POST failed: %s", exc)


# ---------------------------------------------------------------------------
# v0.9.0 — Known panic models (2026-04-25)
# ---------------------------------------------------------------------------
#
# When the same model kernel-panics multiple times in production even with
# every metal-guard defence engaged, we record it here. metal-guard narrows
# the race window; it does NOT fix the underlying Apple IOGPU driver bug,
# and for some models the race window is wide enough that narrowing is not
# enough. For those models, the right operational answer is to switch
# backends (Ollama / llama.cpp) or pivot to a different model family (MoE
# variants have a smaller active-param footprint and are markedly safer).
#
# Callers should invoke :func:`check_known_panic_model(model_id)` at load
# time and escalate — log, refuse, or re-route — based on their own policy.
# :func:`warn_if_known_panic_model` is provided as a fire-and-forget
# convenience that emits a single ``log.warning`` per process per model_id.

KNOWN_PANIC_MODELS: dict[str, dict[str, Any]] = {
    "mlx-community/gemma-4-31b-it-8bit": {
        "panic_signature": "IOGPUMemory.cpp:492 prepare_count_underflow",
        "first_observed": "2026-04-23",
        "last_observed": "2026-04-24",
        "reproductions": [
            "Harper production 2026-04-23 03:14 local, PID 67840, "
            "~6 min worker-ready to panic, pre-cross-model-cadence",
            "Harper production 2026-04-24 03:14 local, PID 26608, "
            "~1.5 min worker-ready to panic, same pipeline as prior",
        ],
        "community": [
            "Hannecke — M4 Max 64GB — ml-explore/mlx#3186, "
            "pivoted to Qwen3-Coder-30B-A3B MoE",
            "lmstudio bug #1740 — hybrid attention (50 sliding + 10 global) "
            "KV cache 8-bit weights 34GB + full ctx KV 20GB+ > 54GB",
            "ml-explore/mlx-lm#883 (M3 Ultra 96GB)",
        ],
        "recommendation": (
            "metal-guard v0.9.0 narrows the race window via cross-model "
            "cadence (C5) + gemma4_generation_flush (C7) + "
            "subprocess_inference_guard (B1), but does NOT eliminate "
            "panic on this model in Harper's production workload. If you "
            "see repeat panics with metal-guard fully engaged, switch "
            "backend (Ollama / llama.cpp) or pivot to an MoE variant "
            "(e.g. mlx-community/gemma-4-26b-a4b-it-4bit)."
        ),
        "upstream": [
            "https://github.com/ml-explore/mlx/issues/3186",
            "https://github.com/ml-explore/mlx-lm/issues/883",
            "https://github.com/ml-explore/mlx/issues/3346",
        ],
    },
}

_WARNED_PANIC_MODELS: set[str] = set()


def check_known_panic_model(model_id: str) -> dict[str, Any] | None:
    """Return the advisory dict for a known-panic model, else ``None``.

    Policy is the caller's — metal-guard does not refuse loads on its own.
    Typical pattern::

        advisory = metal_guard.check_known_panic_model(model_id)
        if advisory is not None:
            log.warning(
                "Loading known-panic model %s: %s",
                model_id, advisory["recommendation"],
            )
    """
    return KNOWN_PANIC_MODELS.get(model_id)


def warn_if_known_panic_model(model_id: str) -> bool:
    """Emit a single ``log.warning`` per process per model_id.

    Returns True if a warning was emitted (or would have been on a prior
    call), False if the model is not in the known-panic list. Safe to
    call on every load — idempotent via a module-level set.
    """
    advisory = check_known_panic_model(model_id)
    if advisory is None:
        return False
    if model_id in _WARNED_PANIC_MODELS:
        return True
    _WARNED_PANIC_MODELS.add(model_id)
    log.warning(
        "metal-guard: %s is in KNOWN_PANIC_MODELS — %s",
        model_id, advisory["recommendation"],
    )
    return True


# ===========================================================================
# v0.10.0 — L10/L11/L12/L13 Harper-private features promoted to public
# (2026-04-27)
#
# These four layers were developed in Harper's private fork during the
# 2026-04 panic series and used in production for two weeks. They cover
# scenarios the public L1-L9 didn't address:
#   L10: panic cooldown gate — auto re-panic prevention after reboot
#   L11: subprocess orphan monitor — pre-panic SUBPROC_PRE/POST signal
#   L12: postmortem auto-collect — bundle .panic + breadcrumb after reboot
#   L13: status snapshot — JSON for menu bar / dashboard consumers
#
# Path conventions: all state lives under ``~/.cache/metal-guard/`` (XDG-
# compatible). User-facing ack file is ``~/.metal-guard-ack`` so a single
# ``touch`` clears the lockout without spelunking caches.
# ===========================================================================

# ---------------------------------------------------------------------------
# L10 — Panic cooldown gate (C1 in Harper fork)
# ---------------------------------------------------------------------------

PANIC_REPORTS_GLOBS: tuple[str, ...] = (
    "/Library/Logs/DiagnosticReports/Retired/panic-full-*.panic",
    "/Library/Logs/DiagnosticReports/panic-full-*.panic",
)

_PANIC_SENTINEL_PATH = os.path.expanduser("~/.cache/metal-guard/panic-sentinel.json")
_PANIC_LOCKOUT_ACK_PATH = os.path.expanduser("~/.metal-guard-ack")

# AND-pattern: core signature + IOGPUMemory.cpp context. Avoids matching
# unrelated panics that mention "underflow" or "IOGPUMemory" in isolation.
_PANIC_CORE_RE = re.compile(r"prepare[_ ]count[_ ]underflow", re.IGNORECASE)
_PANIC_CONTEXT_RE = re.compile(
    r"IOGPUMemory\.cpp:\d+|completeMemory\(\)",
    re.IGNORECASE,
)

# Staircase cooldown defaults (env-overridable)
_GATE_STAGE1_ENV = "METALGUARD_PANIC_COOLDOWN_STAGE1_H"
_GATE_LOCKOUT_MAX_ENV = "METALGUARD_PANIC_LOCKOUT_MAX_H"
_GATE_LOCKOUT_24H_N_ENV = "METALGUARD_PANIC_LOCKOUT_24H_N"
_GATE_LOCKOUT_72H_N_ENV = "METALGUARD_PANIC_LOCKOUT_72H_N"
_GATE_KILL_SWITCH_ENV = "METALGUARD_PANIC_GATE_DISABLED"

_GATE_STAGE1_DEFAULT_H = 2.0
_GATE_LOCKOUT_MAX_DEFAULT_H = 72.0
_GATE_LOCKOUT_24H_N_DEFAULT = 2
_GATE_LOCKOUT_72H_N_DEFAULT = 3


@dataclass(frozen=True)
class PanicRecord:
    """A single IOGPU panic observation (filtered by AND-pattern match)."""
    path: pathlib.Path
    timestamp: datetime.datetime


@dataclass(frozen=True)
class CooldownVerdict:
    """Evaluation result returned by :func:`evaluate_panic_cooldown`.

    ``exit_code`` semantics mirror Harper's plist wrapper protocol:
        0 → proceed (no panic / cooldown expired / kill-switch)
        2 → skip this run (cooldown active or lockout)
        ≥3 → gate broken; wrappers should treat as fail-open
    """
    exit_code: int
    reason: str
    recent_panics_24h: int
    recent_panics_72h: int
    cooldown_until: datetime.datetime | None


def _gate_env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        log.warning("metal-guard: invalid %s=%r, using default %.1f", name, raw, default)
        return default


def _gate_env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        log.warning("metal-guard: invalid %s=%r, using default %d", name, raw, default)
        return default


def _iter_panic_files() -> list[str]:
    """Enumerate panic report files across Retired + active dirs."""
    out: list[str] = []
    for pattern in PANIC_REPORTS_GLOBS:
        out.extend(glob.glob(pattern))
    return out


def _file_matches_iogpu_signature(path: str) -> bool:
    """AND-match core + context. False positive guard for unrelated panics."""
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            content = f.read()
    except OSError as exc:
        # Some .panic files are root-owned 0600; cannot read = cannot confirm.
        log.debug("metal-guard: cannot read %s: %s", path, exc)
        return False
    return bool(_PANIC_CORE_RE.search(content)) and bool(_PANIC_CONTEXT_RE.search(content))


def scan_recent_panics(
    hours: float = 72.0,
    *,
    now: datetime.datetime | None = None,
) -> list[PanicRecord]:
    """Return MLX-matching IOGPU panic records within the last ``hours``.

    Uses file mtime as the panic timestamp. Sorted newest-first. Files
    that fail to read or fail the AND-pattern are skipped silently —
    this is best-effort detection, callers must not rely on completeness.
    """
    now = now or datetime.datetime.now()
    cutoff = now - datetime.timedelta(hours=hours)
    records: list[PanicRecord] = []
    for p in _iter_panic_files():
        try:
            mtime_ts = os.path.getmtime(p)
        except OSError:
            continue
        mtime = datetime.datetime.fromtimestamp(mtime_ts)
        if mtime < cutoff:
            continue
        if not _file_matches_iogpu_signature(p):
            continue
        records.append(PanicRecord(pathlib.Path(p), mtime))
    records.sort(key=lambda r: r.timestamp, reverse=True)
    return records


def _staircase_cooldown_hours(count_24h: int, count_72h: int) -> float | None:
    """Map (24h count, 72h count) to cooldown duration.

    Returns:
        None — no cooldown
        positive float — wait this many hours since latest panic
        ``-1.0`` — lockout (require ack or absolute-max elapsed)
    """
    lock_24 = _gate_env_int(_GATE_LOCKOUT_24H_N_ENV, _GATE_LOCKOUT_24H_N_DEFAULT)
    lock_72 = _gate_env_int(_GATE_LOCKOUT_72H_N_ENV, _GATE_LOCKOUT_72H_N_DEFAULT)
    if count_24h >= lock_24 or count_72h >= lock_72:
        return -1.0
    if count_24h == 1:
        return _gate_env_float(_GATE_STAGE1_ENV, _GATE_STAGE1_DEFAULT_H)
    return None


def _read_panic_sentinel_until(now: datetime.datetime) -> datetime.datetime | None:
    """Read sentinel cooldown_until; None if absent / corrupt / expired."""
    path = pathlib.Path(_PANIC_SENTINEL_PATH)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        log.warning("metal-guard: panic sentinel corrupt (%s), ignoring", exc)
        return None
    raw = data.get("cooldown_until")
    if not isinstance(raw, str):
        return None
    try:
        until = datetime.datetime.fromisoformat(raw)
    except ValueError:
        return None
    if until <= now:
        return None
    return until


def _read_lockout_ack_valid(
    now: datetime.datetime,
    latest_panic: datetime.datetime | None = None,
) -> bool:
    """Return True if ack is valid (24h TTL AND mtime > latest panic).

    The ``mtime > latest_panic`` constraint blocks the "ack survived a new
    panic" race: user touches ack at T, panic occurs at T+23h, reboot at
    T+23.5h. Pure-TTL semantics would let the ack still pass, exactly when
    the gate should be locked. Requiring ack newer than the latest panic
    forces re-acknowledgement after every fresh panic.
    """
    ack_path = pathlib.Path(_PANIC_LOCKOUT_ACK_PATH)
    if not ack_path.exists():
        return False
    try:
        mtime = datetime.datetime.fromtimestamp(ack_path.stat().st_mtime)
    except OSError:
        return False
    if (now - mtime) >= datetime.timedelta(hours=24):
        return False
    if latest_panic is not None and mtime < latest_panic:
        return False
    return True


def _lockout_absolute_max_reached(
    earliest_panic: datetime.datetime, now: datetime.datetime
) -> bool:
    """After ``METALGUARD_PANIC_LOCKOUT_MAX_H`` since earliest panic, auto-clear."""
    max_h = _gate_env_float(_GATE_LOCKOUT_MAX_ENV, _GATE_LOCKOUT_MAX_DEFAULT_H)
    return (now - earliest_panic) >= datetime.timedelta(hours=max_h)


def evaluate_panic_cooldown(
    *,
    now: datetime.datetime | None = None,
) -> CooldownVerdict:
    """Evaluate whether the caller should defer Metal work right now.

    Inspects ``/Library/Logs/DiagnosticReports/`` for AND-pattern IOGPU
    panics within 72h, applies the staircase policy, and consults the
    sentinel + ack files. Stdlib-only by design — works even when MLX is
    wedged mid-recovery.

    Kill switch: ``METALGUARD_PANIC_GATE_DISABLED=1`` short-circuits to
    exit 0 unconditionally (used for emergency override).
    """
    now = now or datetime.datetime.now()

    if os.environ.get(_GATE_KILL_SWITCH_ENV) == "1":
        return CooldownVerdict(0, "kill-switch active", 0, 0, None)

    sentinel_until = _read_panic_sentinel_until(now)
    if sentinel_until is not None:
        return CooldownVerdict(
            2,
            f"sentinel cooldown until {sentinel_until.isoformat(timespec='seconds')}",
            0, 0, sentinel_until,
        )

    panics_72h = scan_recent_panics(72.0, now=now)
    cutoff_24h = now - datetime.timedelta(hours=24)
    panics_24h = [p for p in panics_72h if p.timestamp >= cutoff_24h]
    count_24h = len(panics_24h)
    count_72h = len(panics_72h)

    hours = _staircase_cooldown_hours(count_24h, count_72h)

    if hours is None:
        return CooldownVerdict(0, "no recent IOGPU panics", count_24h, count_72h, None)

    if hours < 0:
        latest = panics_72h[0].timestamp if panics_72h else None
        if _read_lockout_ack_valid(now, latest_panic=latest):
            return CooldownVerdict(
                0, "lockout cleared by ack", count_24h, count_72h, None,
            )
        # critic R1 P1-3 guard: if user sets METALGUARD_PANIC_LOCKOUT_72H_N=0
        # the gate enters lockout-on-empty-list path. Default to "no auto-clear"
        # (treat earliest as `now` so MAX_H window never elapses).
        earliest = panics_72h[-1].timestamp if panics_72h else now
        if _lockout_absolute_max_reached(earliest, now):
            return CooldownVerdict(
                0, "lockout absolute-max elapsed", count_24h, count_72h, None,
            )
        return CooldownVerdict(
            2,
            f"lockout: 24h={count_24h} 72h={count_72h}; "
            f"touch {_PANIC_LOCKOUT_ACK_PATH} to clear",
            count_24h, count_72h, None,
        )

    latest = panics_72h[0].timestamp
    cooldown_until = latest + datetime.timedelta(hours=hours)
    if cooldown_until > now:
        return CooldownVerdict(
            2,
            f"cooldown {hours:.1f}h until {cooldown_until.isoformat(timespec='seconds')}",
            count_24h, count_72h, cooldown_until,
        )
    return CooldownVerdict(
        0,
        f"cooldown expired (latest panic {latest.isoformat(timespec='seconds')})",
        count_24h, count_72h, None,
    )


def mark_panic_sentinel_cooldown(
    duration_hours: float,
    *,
    now: datetime.datetime | None = None,
) -> pathlib.Path:
    """Write a sentinel cooldown extending exit=2 for ``duration_hours``.

    Atomic write (rename) so concurrent gate readers never see partial JSON.
    Typically called by the postmortem collector (L12) right after writing
    a bundle, to extend the cooldown beyond DiagnosticReports rotation lag.
    """
    now = now or datetime.datetime.now()
    until = now + datetime.timedelta(hours=duration_hours)
    path = pathlib.Path(_PANIC_SENTINEL_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = json.dumps({
        "cooldown_until": until.isoformat(timespec="seconds"),
        "created_at": now.isoformat(timespec="seconds"),
    })
    tmp.write_text(payload, encoding="utf-8")
    os.replace(tmp, path)
    return path


def ack_panic_lockout() -> pathlib.Path:
    """Touch ``~/.metal-guard-ack`` (atomic) to clear an active lockout.

    Atomic rename so concurrent gate evaluations never read a half-written
    ack. Operator should write a brief reason line into the file before
    touching — future versions may mandate this content for audit.
    """
    ack = pathlib.Path(_PANIC_LOCKOUT_ACK_PATH)
    tmp = ack.with_suffix(ack.suffix + ".pending")
    tmp.write_text(datetime.datetime.now().isoformat(timespec="seconds") + "\n")
    os.replace(tmp, ack)
    return ack


def clear_panic_ack() -> bool:
    """Remove ``~/.metal-guard-ack``. Returns True if removed, False if absent."""
    ack = pathlib.Path(_PANIC_LOCKOUT_ACK_PATH)
    try:
        ack.unlink()
        return True
    except FileNotFoundError:
        return False
    except OSError as exc:
        log.warning("metal-guard: clear_panic_ack failed: %s", exc)
        return False


def clear_panic_sentinel() -> bool:
    """Remove the sentinel cooldown file."""
    path = pathlib.Path(_PANIC_SENTINEL_PATH)
    try:
        path.unlink()
        return True
    except FileNotFoundError:
        return False
    except OSError as exc:
        log.warning("metal-guard: clear_panic_sentinel failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# L11 — Subprocess orphan monitor (C6 in Harper fork)
# ---------------------------------------------------------------------------

# Breadcrumb line format:
#   [2026-04-23 17:39:43] SUBPROC_PRE: mlx-community/gemma-4-26b-a4b-8bit
_BREADCRUMB_LINE_RE = re.compile(
    r"^\[(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\] "
    r"(?P<tag>[A-Z_0-9]+): (?P<payload>.*)$"
)
_WORKER_READY_RE = re.compile(
    r"SUBPROCESS_WORKER: (?P<model>\S+) ready, pid=(?P<pid>\d+)"
)

_ORPHAN_KILL_SWITCH_ENV = "METALGUARD_SUBPROC_ORPHAN_WATCH_DISABLED"
_ORPHAN_THRESHOLD_ENV = "METALGUARD_SUBPROC_ORPHAN_THRESHOLD_SEC"
_ORPHAN_DEFAULT_THRESHOLD_SEC = 90.0
_ORPHAN_TAIL_LINES = 2000


@dataclass(frozen=True)
class OrphanPre:
    """A SUBPROC_PRE entry without a matching SUBPROC_POST after threshold."""
    model_id: str
    pre_ts: datetime.datetime
    age_sec: float
    pid: int | None  # from SUBPROCESS_WORKER ready line if traceable


def _orphan_read_tail(path: pathlib.Path, n_lines: int) -> list[str]:
    """Return last ``n_lines`` of ``path`` (cheap seek for large files)."""
    try:
        size = path.stat().st_size
    except OSError:
        return []
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            if size < 1024 * 1024:
                lines = f.readlines()
            else:
                approx_bytes = n_lines * 200
                f.seek(max(0, size - approx_bytes))
                _ = f.readline()  # discard partial line
                lines = f.readlines()
    except OSError as exc:
        log.warning("metal-guard: cannot read %s: %s", path, exc)
        return []
    return lines[-n_lines:]


def _orphan_parse_line(line: str) -> tuple[datetime.datetime, str, str] | None:
    m = _BREADCRUMB_LINE_RE.match(line.strip())
    if not m:
        return None
    try:
        ts = datetime.datetime.strptime(m.group("ts"), "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None
    return ts, m.group("tag"), m.group("payload")


def scan_orphan_subproc_pre(
    *,
    threshold_sec: float | None = None,
    now: datetime.datetime | None = None,
    breadcrumb_path: str | None = None,
) -> list[OrphanPre]:
    """Return SUBPROC_PRE entries with no matching POST within threshold.

    Pre-panic signal: SUBPROC_PRE without POST after 90s strongly suggests
    Metal is stuck. Caller can then SIGKILL the offending worker pid before
    the kernel does (saves a reboot).

    Algorithm:
        1. Read tail (~2000 lines) of breadcrumb log
        2. Track latest WORKER_READY pid per model_id
        3. FIFO-pair PRE↔POST per model_id
        4. Unpaired PREs older than threshold are orphans

    Disabled by ``METALGUARD_SUBPROC_ORPHAN_WATCH_DISABLED=1``.
    """
    if os.environ.get(_ORPHAN_KILL_SWITCH_ENV) == "1":
        return []

    if threshold_sec is None:
        raw = os.environ.get(_ORPHAN_THRESHOLD_ENV, "").strip()
        if raw:
            try:
                threshold_sec = float(raw)
            except ValueError:
                threshold_sec = _ORPHAN_DEFAULT_THRESHOLD_SEC
        else:
            threshold_sec = _ORPHAN_DEFAULT_THRESHOLD_SEC

    now = now or datetime.datetime.now()
    path_str = breadcrumb_path if breadcrumb_path is not None else (
        metal_guard._breadcrumb_path or "logs/metal_breadcrumb.log"
    )
    path = pathlib.Path(os.path.expanduser(path_str))
    if not path.exists():
        return []

    tail = _orphan_read_tail(path, _ORPHAN_TAIL_LINES)
    pending: dict[str, list[datetime.datetime]] = {}
    model_pid: dict[str, int] = {}

    for line in tail:
        parsed = _orphan_parse_line(line)
        if parsed is None:
            continue
        ts, tag, payload = parsed
        if tag == "SUBPROCESS_WORKER":
            m = _WORKER_READY_RE.search(line)
            if m:
                model_pid[m.group("model")] = int(m.group("pid"))
            continue
        if tag == "SUBPROC_PRE":
            model = payload.strip()
            pending.setdefault(model, []).append(ts)
        elif tag == "SUBPROC_POST":
            model = payload.strip()
            queue = pending.get(model)
            if queue:
                queue.pop(0)

    cutoff = now - datetime.timedelta(seconds=threshold_sec)
    orphans: list[OrphanPre] = []
    for model, timestamps in pending.items():
        for ts in timestamps:
            if ts <= cutoff:
                age = (now - ts).total_seconds()
                orphans.append(OrphanPre(
                    model_id=model,
                    pre_ts=ts,
                    age_sec=age,
                    pid=model_pid.get(model),
                ))
    orphans.sort(key=lambda o: o.age_sec, reverse=True)
    return orphans


# ---------------------------------------------------------------------------
# L12 — Postmortem auto-collect (C2 in Harper fork)
# ---------------------------------------------------------------------------

_POSTMORTEM_KILL_SWITCH_ENV = "METALGUARD_POSTMORTEM_DISABLED"
_POSTMORTEM_SENTINEL_DURATION_ENV = "METALGUARD_POSTMORTEM_SENTINEL_H"
_POSTMORTEM_SENTINEL_DURATION_DEFAULT_H = 2.0

_POSTMORTEM_MAX_PANIC_FILES = 5
_POSTMORTEM_MAX_BREADCRUMB_LINES = 500
_POSTMORTEM_MAX_COPY_BYTES = 5 * 1024 * 1024  # 5MB cap per file


def _postmortem_collect_panic_files(
    output_dir: pathlib.Path,
    *,
    within_hours: float = 24.0,
    now: datetime.datetime | None = None,
) -> list[pathlib.Path]:
    """Copy up to N IOGPU panic files into ``output_dir``."""
    now = now or datetime.datetime.now()
    records = scan_recent_panics(within_hours, now=now)[:_POSTMORTEM_MAX_PANIC_FILES]
    copied: list[pathlib.Path] = []
    for rec in records:
        dest = output_dir / rec.path.name
        try:
            if rec.path.stat().st_size > _POSTMORTEM_MAX_COPY_BYTES:
                with open(rec.path, "rb") as src, open(dest, "wb") as dst:
                    dst.write(src.read(_POSTMORTEM_MAX_COPY_BYTES))
                    dst.write(b"\n\n--- TRUNCATED at 5MB ---\n")
            else:
                shutil.copy2(rec.path, dest)
            copied.append(dest)
        except OSError as exc:
            log.warning("metal-guard postmortem: copy %s failed: %s", rec.path, exc)
    return copied


def _postmortem_collect_breadcrumb(output_dir: pathlib.Path) -> pathlib.Path | None:
    """Copy last N lines of breadcrumb log into ``output_dir``."""
    breadcrumb_path = metal_guard._breadcrumb_path or "logs/metal_breadcrumb.log"
    src = pathlib.Path(os.path.expanduser(breadcrumb_path))
    if not src.exists():
        return None
    dest = output_dir / "breadcrumb_tail.log"
    try:
        with open(src, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        tail = lines[-_POSTMORTEM_MAX_BREADCRUMB_LINES:]
        dest.write_text("".join(tail), encoding="utf-8")
        return dest
    except OSError as exc:
        log.warning("metal-guard postmortem: breadcrumb tail failed: %s", exc)
        return None


def _postmortem_collect_panics_jsonl(output_dir: pathlib.Path) -> pathlib.Path | None:
    """Copy panics.jsonl history (cross-ref panic ledger with this bundle)."""
    src = pathlib.Path(_PANIC_JSONL_PATH)
    if not src.exists():
        return None
    dest = output_dir / "panics.jsonl"
    try:
        shutil.copy2(src, dest)
        return dest
    except OSError as exc:
        log.warning("metal-guard postmortem: panics.jsonl copy failed: %s", exc)
        return None


def _postmortem_collect_mlx_stats(output_dir: pathlib.Path) -> pathlib.Path:
    """Best-effort mx.metal stats snapshot. File always written."""
    dest = output_dir / "mlx_stats.txt"
    try:
        import mlx.core as mx  # type: ignore[import-not-found]
    except ImportError as exc:
        dest.write_text(f"mlx.core import failed: {exc}\n")
        return dest
    except Exception as exc:  # noqa: BLE001 — post-panic env may be wedged
        dest.write_text(f"mlx.core import error: {type(exc).__name__}: {exc}\n")
        return dest

    lines: list[str] = []
    # Try v0.32+ flat API first, fall back to legacy mx.metal namespace.
    # critic R1 P1-1 fix: on exception, `continue` to legacy fallback rather
    # than `break` — bug-by-bug API breakage shouldn't lose the whole stat.
    for fn_name in ("get_active_memory", "get_cache_memory", "get_peak_memory"):
        captured = False
        for owner_name in ("mx", "mx.metal"):
            owner = mx if owner_name == "mx" else getattr(mx, "metal", None)
            if owner is None:
                continue
            fn = getattr(owner, fn_name, None)
            if fn is None:
                continue
            try:
                lines.append(f"{owner_name}.{fn_name}: {fn()}")
                captured = True
                break
            except Exception as exc:  # noqa: BLE001
                # Try next owner instead of giving up on this stat
                lines.append(
                    f"{owner_name}.{fn_name}: ERROR {type(exc).__name__}: {exc}"
                )
                continue
        if not captured:
            lines.append(f"{fn_name}: no working API path on this MLX version")
    dest.write_text("\n".join(lines) + "\n")
    return dest


def _postmortem_write_index(
    output_dir: pathlib.Path,
    *,
    now: datetime.datetime,
    collected_panics: list[pathlib.Path],
    breadcrumb: pathlib.Path | None,
    panics_jsonl: pathlib.Path | None,
    mlx_stats: pathlib.Path,
) -> pathlib.Path:
    """Write ``index.md`` summarising the bundle for human review."""
    dest = output_dir / "index.md"
    lines: list[str] = [
        f"# metal-guard Postmortem {now.isoformat(timespec='seconds')}",
        "",
        "## Collected artifacts",
        "",
    ]
    if collected_panics:
        lines.append(f"### Panic files ({len(collected_panics)})")
        for p in collected_panics:
            try:
                ts = datetime.datetime.fromtimestamp(p.stat().st_mtime).isoformat(
                    timespec="seconds"
                )
            except OSError:
                ts = "unknown"
            lines.append(f"- `{p.name}` (mtime {ts})")
        lines.append("")
    else:
        lines.append("### Panic files: NONE FOUND IN WINDOW")
        lines.append("")
    if breadcrumb:
        lines.append(
            f"### Breadcrumb tail: `{breadcrumb.name}` "
            f"(last {_POSTMORTEM_MAX_BREADCRUMB_LINES} lines)"
        )
        lines.append("")
    if panics_jsonl:
        lines.append(f"### Panic JSONL history: `{panics_jsonl.name}`")
        lines.append("")
    lines.append(f"### MLX stats: `{mlx_stats.name}`")
    lines.append("")
    lines.append("## Next steps")
    lines.append("")
    lines.append(
        "1. `grep -E 'SUBPROC_PRE|SUBPROC_POST|GEMMA4_FLUSH' breadcrumb_tail.log` "
        "to trace last MLX ops before panic"
    )
    lines.append(
        "2. `grep 'IOGPUMemory' *.panic` to confirm signature"
    )
    lines.append(
        "3. If panic was on a specific model, file a Known Panic Model report: "
        "https://github.com/Harperbot/metal-guard/issues/new?template=known-panic-report.yml"
    )
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return dest


def run_postmortem(
    output_dir: pathlib.Path | str,
    *,
    now: datetime.datetime | None = None,
) -> dict[str, Any]:
    """Orchestrate full postmortem collection.

    When at least one IOGPU panic is found inside the 24h window, a
    sentinel cooldown is also written so L10's gate keeps deferring runs
    even if the DiagnosticReports rotation races the next plist start.

    Kill-switch: ``METALGUARD_POSTMORTEM_DISABLED=1`` short-circuits.

    Returns a dict with keys: ``status`` (``"collected"`` / ``"disabled"``),
    ``output_dir``, ``panic_count``, ``index`` (path), ``sentinel`` (path or None).
    """
    if os.environ.get(_POSTMORTEM_KILL_SWITCH_ENV) == "1":
        return {"status": "disabled", "output_dir": str(output_dir)}
    now = now or datetime.datetime.now()
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    panics = _postmortem_collect_panic_files(output_dir, now=now)
    breadcrumb = _postmortem_collect_breadcrumb(output_dir)
    jsonl = _postmortem_collect_panics_jsonl(output_dir)
    stats = _postmortem_collect_mlx_stats(output_dir)
    index = _postmortem_write_index(
        output_dir,
        now=now,
        collected_panics=panics,
        breadcrumb=breadcrumb,
        panics_jsonl=jsonl,
        mlx_stats=stats,
    )

    # Extend cooldown if we actually saw a panic — prevents the gate from
    # re-opening before the human has a chance to look at the bundle.
    sentinel_path: str | None = None
    if panics:
        try:
            raw = os.environ.get(_POSTMORTEM_SENTINEL_DURATION_ENV)
            duration = float(raw) if raw else _POSTMORTEM_SENTINEL_DURATION_DEFAULT_H
        except ValueError:
            duration = _POSTMORTEM_SENTINEL_DURATION_DEFAULT_H
        try:
            sentinel_path = str(mark_panic_sentinel_cooldown(duration, now=now))
        except OSError as exc:
            log.warning("metal-guard postmortem: sentinel write failed: %s", exc)

    return {
        "status": "collected",
        "output_dir": str(output_dir),
        "panic_count": len(panics),
        "index": str(index),
        "sentinel": sentinel_path,
    }


# ---------------------------------------------------------------------------
# L13 — Status snapshot writer (Harper-private status_api + status_writer)
# ---------------------------------------------------------------------------

STATUS_SNAPSHOT_SCHEMA_VERSION = "1.0"
_STATUS_SNAPSHOT_DEFAULT_PATH = os.path.expanduser(
    "~/.cache/metal-guard/status.json"
)
_STATUS_WRITER_DEFAULT_INTERVAL_S = 30.0


def get_status_snapshot(
    *,
    include_panics: bool = True,
    breadcrumb_lines: int = 20,
) -> dict[str, Any]:
    """Build a versioned JSON-serialisable snapshot of metal-guard state.

    Designed for cross-process consumers (menu bar apps, dashboards, ssh
    inspection scripts) that should not import ``metal_guard`` directly.
    Schema is append-only across minor versions; breaking changes bump
    ``STATUS_SNAPSHOT_SCHEMA_VERSION``.

    Top-level keys:
        - ``schema_version``
        - ``captured_at`` (ISO-8601 UTC)
        - ``memory`` (active/peak/limit/available + pct)
        - ``kv_monitor`` (running, interval, samples, growth_rate)
        - ``recent_panics`` (signature/time/explanation list)
        - ``breadcrumb_tail`` (last N lines of metal_breadcrumb.log)
        - ``lock`` (held + holder details from cross-process file lock)
        - ``mode`` (defensive / observer)
        - ``panic_cooldown`` (exit_code / reason / 24h+72h counts)
        - ``errors`` (only when a sub-collector failed; never raises)
    """
    snapshot: dict[str, Any] = {
        "schema_version": STATUS_SNAPSHOT_SCHEMA_VERSION,
        "captured_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "errors": {},
    }

    # 1a. Memory stats — public MetalGuard.memory_stats() returns MemoryStats
    #     (or zero MemoryStats when MLX not importable, e.g. running outside
    #     a venv with mlx installed). critic R1 P0-2 fix.
    try:
        stats = metal_guard.memory_stats()
        if stats.limit_bytes > 0:
            snapshot["memory"] = {
                "available": True,
                "active_gb": round(stats.active_gb, 3),
                "peak_gb": round(stats.peak_gb, 3),
                "limit_gb": round(stats.limit_gb, 3),
                "available_gb": round(stats.available_gb, 3),
                "active_pct": round(stats.active_pct, 1),
                "peak_pct": round(stats.peak_pct, 1),
            }
        else:
            # MLX not loaded / no Metal device; surface but don't error.
            snapshot["memory"] = {"available": False, "reason": "mlx not loaded"}
    except Exception as exc:  # noqa: BLE001
        snapshot["memory"] = {"available": False}
        snapshot["errors"]["memory"] = f"{type(exc).__name__}: {exc}"[:200]

    # 1b. KV growth tracker live state (module-level singleton)
    try:
        snapshot["kv_monitor"] = {
            "running": True,
            "active_requests": kv_tracker.snapshot(),
        }
    except Exception as exc:  # noqa: BLE001
        snapshot["kv_monitor"] = {"running": False}
        snapshot["errors"]["kv_monitor"] = f"{type(exc).__name__}: {exc}"[:200]

    # 1c. Recent panics via existing public parser
    if include_panics:
        try:
            since_ts = time.time() - 72 * 3600.0
            snapshot["recent_panics"] = parse_panic_reports(since_ts=since_ts) or []
        except Exception as exc:  # noqa: BLE001
            snapshot["recent_panics"] = []
            snapshot["errors"]["recent_panics"] = f"{type(exc).__name__}: {exc}"[:200]
    else:
        snapshot["recent_panics"] = []

    # 1d. Breadcrumb tail
    if breadcrumb_lines > 0:
        try:
            bc_path = metal_guard._breadcrumb_path or "logs/metal_breadcrumb.log"
            bc_full = pathlib.Path(os.path.expanduser(bc_path))
            if bc_full.is_file():
                with bc_full.open("r", encoding="utf-8", errors="replace") as fh:
                    lines = fh.readlines()
                snapshot["breadcrumb_path"] = str(bc_full)
                snapshot["breadcrumb_tail"] = [
                    ln.rstrip() for ln in lines[-breadcrumb_lines:]
                ]
            else:
                snapshot["breadcrumb_path"] = None
                snapshot["breadcrumb_tail"] = []
        except OSError as exc:
            snapshot["breadcrumb_tail"] = []
            snapshot["errors"]["breadcrumb"] = f"{type(exc).__name__}: {exc}"[:200]
    else:
        snapshot["breadcrumb_tail"] = []

    # 2. Lock holder
    try:
        lock_info = read_mlx_lock()
        if lock_info:
            started = lock_info.get("started_at")
            elapsed: float | None = None
            if isinstance(started, str):
                try:
                    started_dt = datetime.datetime.fromisoformat(
                        started.replace("Z", "+00:00")
                    )
                    elapsed = (
                        datetime.datetime.now(datetime.timezone.utc) - started_dt
                    ).total_seconds()
                except (ValueError, TypeError):
                    pass
            snapshot["lock"] = {
                "held": True,
                "pid": lock_info.get("pid"),
                "label": lock_info.get("label"),
                "started_at": started,
                "elapsed_s": round(elapsed, 1) if elapsed is not None else None,
                "cmdline": (lock_info.get("cmdline") or "")[:500],
                "host": lock_info.get("host"),
            }
        else:
            snapshot["lock"] = {"held": False}
    except Exception as exc:  # noqa: BLE001
        snapshot["lock"] = {"held": False, "available": False}
        snapshot["errors"]["lock"] = f"{type(exc).__name__}: {exc}"[:200]

    # 3. Mode
    try:
        mode_desc = describe_mode()  # returns dict[str, str]
        snapshot["mode"] = {
            "current": current_mode(),
            "env_var": "METALGUARD_MODE",
            "description": mode_desc.get("description", ""),
            "details": mode_desc,
        }
    except Exception as exc:  # noqa: BLE001
        snapshot["mode"] = {"current": "unknown"}
        snapshot["errors"]["mode"] = f"{type(exc).__name__}: {exc}"[:200]

    # 4. L10 panic cooldown verdict
    try:
        verdict = evaluate_panic_cooldown()
        snapshot["panic_cooldown"] = {
            "exit_code": verdict.exit_code,
            "reason": verdict.reason,
            "recent_panics_24h": verdict.recent_panics_24h,
            "recent_panics_72h": verdict.recent_panics_72h,
            "cooldown_until": (
                verdict.cooldown_until.isoformat(timespec="seconds")
                if verdict.cooldown_until else None
            ),
        }
    except Exception as exc:  # noqa: BLE001
        snapshot["panic_cooldown"] = {"available": False}
        snapshot["errors"]["panic_cooldown"] = f"{type(exc).__name__}: {exc}"[:200]

    if not snapshot["errors"]:
        del snapshot["errors"]
    return snapshot


def write_status_snapshot(
    out_path: pathlib.Path | str | None = None,
) -> pathlib.Path:
    """Atomic snapshot write to ``out_path`` (default ``~/.cache/metal-guard/status.json``).

    Atomic via ``tmp.write + os.replace`` so concurrent readers never see
    a partial JSON payload.
    """
    out_path = pathlib.Path(out_path or _STATUS_SNAPSHOT_DEFAULT_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    snap = get_status_snapshot()
    tmp = out_path.with_suffix(out_path.suffix + f".tmp.{os.getpid()}")
    try:
        tmp.write_text(
            json.dumps(snap, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(tmp, out_path)
    except OSError:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise
    return out_path
