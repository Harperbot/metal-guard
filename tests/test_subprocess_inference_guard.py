"""Tests for subprocess_inference_guard (B2 port — per-inference guard
for subprocess MLX workers).

Context: Harper's internal fork hit 6 consecutive ``prepare_count_underflow``
IOGPUMemory.cpp:492 kernel panics in 3 days. Root cause: a subprocess
worker ran ``gen_fn(...)`` without any per-inference Metal flush. Parent-
process hooks (PRE_INFERENCE ``clear_cache`` / post-gen ``mx.eval`` sync)
do NOT reach subprocess memory, so GPU command buffers were released
before Metal finished → kernel panic.

This guard wraps every ``gen_fn`` call inside the worker. Order:

  PRE:  mx.clear_cache()         — release any leftover cached buffers
  (yield — generate runs)
  POST: mx.synchronize()         — block until GPU drained (critical)
  POST: mx.clear_cache()         — release this iter's buffers

``synchronize`` chosen over ``mx.eval(result)`` because the generate
return type varies (``str`` for mlx-lm, ``GenerateResult`` for
mlx-vlm) — ``synchronize`` is return-type agnostic and semantically
means "wait for GPU to finish".
"""
from __future__ import annotations

import sys
import unittest.mock as mock

import pytest


class TestSubprocessInferenceGuard:
    """Module-level context manager that wraps each subprocess generate."""

    def test_symbol_exists_at_module_level(self):
        """API must be importable from metal_guard."""
        from metal_guard import subprocess_inference_guard  # noqa: F401

    def test_pre_calls_clear_cache_before_yield(self, monkeypatch):
        """PRE hook must mx.clear_cache() BEFORE the with-body runs."""
        from metal_guard import subprocess_inference_guard

        fake_mx = mock.MagicMock()
        call_order: list[str] = []
        fake_mx.clear_cache.side_effect = lambda: call_order.append("clear_cache")
        fake_mx.synchronize.side_effect = lambda: call_order.append("synchronize")
        fake_mlx_pkg = mock.MagicMock()
        fake_mlx_pkg.core = fake_mx
        monkeypatch.setitem(sys.modules, "mlx", fake_mlx_pkg)
        monkeypatch.setitem(sys.modules, "mlx.core", fake_mx)

        with subprocess_inference_guard("test-model"):
            call_order.append("body")

        # Expected: clear_cache → body → synchronize → clear_cache
        assert call_order[0] == "clear_cache", (
            f"PRE must clear_cache first; got {call_order}"
        )
        assert "body" in call_order

    def test_post_synchronize_before_clear_cache(self, monkeypatch):
        """POST must synchronize() BEFORE the final clear_cache() —
        synchronize flushes in-flight GPU work; clearing cache before
        sync would reproduce the IOGPUMemory.cpp:492 underflow."""
        from metal_guard import subprocess_inference_guard

        fake_mx = mock.MagicMock()
        call_order: list[str] = []
        fake_mx.clear_cache.side_effect = lambda: call_order.append("clear_cache")
        fake_mx.synchronize.side_effect = lambda: call_order.append("synchronize")
        fake_mlx_pkg = mock.MagicMock()
        fake_mlx_pkg.core = fake_mx
        monkeypatch.setitem(sys.modules, "mlx", fake_mlx_pkg)
        monkeypatch.setitem(sys.modules, "mlx.core", fake_mx)

        with subprocess_inference_guard("test-model"):
            pass

        # clear_cache (pre), synchronize (post), clear_cache (post)
        assert call_order == ["clear_cache", "synchronize", "clear_cache"], (
            f"POST sync must precede POST clear; got {call_order}"
        )

    def test_post_runs_even_on_exception(self, monkeypatch):
        """POST synchronize+clear MUST run in finally semantics so a
        failed generate still flushes the command buffer."""
        from metal_guard import subprocess_inference_guard

        fake_mx = mock.MagicMock()
        call_order: list[str] = []
        fake_mx.clear_cache.side_effect = lambda: call_order.append("clear_cache")
        fake_mx.synchronize.side_effect = lambda: call_order.append("synchronize")
        fake_mlx_pkg = mock.MagicMock()
        fake_mlx_pkg.core = fake_mx
        monkeypatch.setitem(sys.modules, "mlx", fake_mlx_pkg)
        monkeypatch.setitem(sys.modules, "mlx.core", fake_mx)

        with pytest.raises(RuntimeError, match="boom"):
            with subprocess_inference_guard("test-model"):
                raise RuntimeError("boom")

        assert "synchronize" in call_order, (
            f"POST must run on exception; got {call_order}"
        )
        # clear_cache appears twice (pre + post)
        assert call_order.count("clear_cache") == 2, (
            f"clear_cache should run twice (pre + post); got {call_order}"
        )

    def test_degrades_gracefully_without_mlx(self, monkeypatch):
        """If mlx.core is not importable the guard must no-op, not crash —
        so tests in environments without MLX still run and the production
        subprocess worker can fail loud on its own load() call instead of
        on the guard."""
        from metal_guard import subprocess_inference_guard

        # Simulate mlx not installed
        monkeypatch.setitem(sys.modules, "mlx", None)
        monkeypatch.setitem(sys.modules, "mlx.core", None)

        # Should not raise
        with subprocess_inference_guard("test-model"):
            pass

    def test_pre_failure_does_not_block_body(self, monkeypatch):
        """If clear_cache itself throws (defensive), the body must still
        run — a broken guard is worse than no guard because it would
        silently stop all inference."""
        from metal_guard import subprocess_inference_guard

        fake_mx = mock.MagicMock()
        fake_mx.clear_cache.side_effect = RuntimeError("cache miss")
        fake_mx.synchronize.side_effect = lambda: None
        fake_mlx_pkg = mock.MagicMock()
        fake_mlx_pkg.core = fake_mx
        monkeypatch.setitem(sys.modules, "mlx", fake_mlx_pkg)
        monkeypatch.setitem(sys.modules, "mlx.core", fake_mx)

        ran = False
        with subprocess_inference_guard("test-model"):
            ran = True
        assert ran, "body must run even if PRE hook fails"

    def test_post_failure_does_not_mask_body_exception(self, monkeypatch):
        """If the body raises, that exception must surface — POST hook
        errors must not swallow the real error."""
        from metal_guard import subprocess_inference_guard

        fake_mx = mock.MagicMock()
        fake_mx.synchronize.side_effect = RuntimeError("sync dead")
        fake_mlx_pkg = mock.MagicMock()
        fake_mlx_pkg.core = fake_mx
        monkeypatch.setitem(sys.modules, "mlx", fake_mlx_pkg)
        monkeypatch.setitem(sys.modules, "mlx.core", fake_mx)

        with pytest.raises(ValueError, match="body error"):
            with subprocess_inference_guard("test-model"):
                raise ValueError("body error")

    def test_breadcrumb_emits_pre_and_post(self, monkeypatch):
        """Forensics: we must see SUBPROC_PRE and SUBPROC_POST in the
        breadcrumb log so post-mortem analysis of a panic can confirm
        the guard was or was not engaged on the fatal call."""
        import importlib
        mg_module = importlib.import_module("metal_guard")

        fake_mx = mock.MagicMock()
        fake_mlx_pkg = mock.MagicMock()
        fake_mlx_pkg.core = fake_mx
        monkeypatch.setitem(sys.modules, "mlx", fake_mlx_pkg)
        monkeypatch.setitem(sys.modules, "mlx.core", fake_mx)

        seen: list[str] = []
        monkeypatch.setattr(
            mg_module.metal_guard,
            "breadcrumb",
            lambda msg: seen.append(msg),
        )

        with mg_module.subprocess_inference_guard("mlx-community/test-31b-8bit"):
            pass

        assert any("SUBPROC_PRE" in m and "test-31b-8bit" in m for m in seen), (
            f"expected SUBPROC_PRE breadcrumb; got {seen}"
        )
        assert any("SUBPROC_POST" in m and "test-31b-8bit" in m for m in seen), (
            f"expected SUBPROC_POST breadcrumb; got {seen}"
        )

    def test_model_id_escapes_cleanly(self, monkeypatch):
        """Breadcrumb must not crash on unusual model_id strings."""
        from metal_guard import subprocess_inference_guard

        fake_mx = mock.MagicMock()
        fake_mlx_pkg = mock.MagicMock()
        fake_mlx_pkg.core = fake_mx
        monkeypatch.setitem(sys.modules, "mlx", fake_mlx_pkg)
        monkeypatch.setitem(sys.modules, "mlx.core", fake_mx)

        # Should not raise
        with subprocess_inference_guard("mlx-community/weird name with spaces"):
            pass
        with subprocess_inference_guard(""):
            pass
