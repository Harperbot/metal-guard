"""mlx-safe-python (Python entry point) tests (v0.10.0)。

The bash wrapper at scripts/mlx-safe-python is kept for source-tree users.
The Python entry safe_python_main is what pip-installed users get
(`pyproject.toml [project.scripts] mlx-safe-python = ...`). critic R1
P0-1/P0-3 fixes both routed through safe_python_main.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from unittest.mock import patch

import pytest

import metal_guard as mg
from metal_guard_cli import safe_python_main


def test_safe_python_no_python_bin(monkeypatch):
    """No python on PATH → rc=127 + FATAL stderr."""
    monkeypatch.setenv("MLX_SAFE_PYTHON_BIN", "/nonexistent/python")
    rc = safe_python_main(["-c", "print('x')"])
    assert rc == 127


def test_safe_python_pip_passthrough(monkeypatch, capsys):
    """`-m pip ...` must bypass gate without touching cooldown logic."""
    # Force a "lockout" by patching evaluate_panic_cooldown — gate would
    # block, but pip path should bypass entirely.
    fake_verdict = mg.CooldownVerdict(2, "fake lockout", 5, 5, None)
    monkeypatch.setattr(mg, "evaluate_panic_cooldown", lambda: fake_verdict)
    monkeypatch.setenv("MLX_SAFE_PYTHON_BIN", sys.executable)
    # `-m pip --version` doesn't import torch/mlx → must pass through
    # We can't run os.execv in a test without forking, so use subprocess
    # round-trip via the public Python entry through subprocess module
    result = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.path.insert(0, '.'); "
         "from metal_guard_cli import safe_python_main; "
         "sys.exit(safe_python_main(['-m', 'pip', '--version']))"],
        capture_output=True, text=True, cwd=os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        ),
    )
    assert result.returncode == 0
    assert "pip" in result.stdout.lower()


def test_safe_python_blocks_when_cooldown(monkeypatch, capsys):
    """Cooldown active + no FORCE → rc=10."""
    fake_verdict = mg.CooldownVerdict(2, "fake lockout", 3, 3, None)
    monkeypatch.setattr(mg, "evaluate_panic_cooldown", lambda: fake_verdict)
    monkeypatch.setenv("MLX_SAFE_PYTHON_BIN", sys.executable)
    monkeypatch.delenv("MLX_SAFE_PYTHON_FORCE", raising=False)

    rc = safe_python_main(["-c", "import sys; sys.exit(0)"])
    assert rc == 10
    captured = capsys.readouterr()
    assert "BLOCKED" in captured.err
    assert "fake lockout" in captured.err
    assert "MLX_SAFE_PYTHON_FORCE=1" in captured.err


def test_safe_python_force_overrides_cooldown(monkeypatch, capsys):
    """FORCE=1 + cooldown → exec python anyway (we replace os.execv with
    a marker so the test process doesn't actually exec).
    """
    fake_verdict = mg.CooldownVerdict(2, "fake lockout", 3, 3, None)
    monkeypatch.setattr(mg, "evaluate_panic_cooldown", lambda: fake_verdict)
    monkeypatch.setenv("MLX_SAFE_PYTHON_BIN", sys.executable)
    monkeypatch.setenv("MLX_SAFE_PYTHON_FORCE", "1")

    exec_calls = []
    def _fake_execv(path, argv):
        exec_calls.append((path, argv))
    monkeypatch.setattr(os, "execv", _fake_execv)

    safe_python_main(["-c", "print('x')"])
    captured = capsys.readouterr()
    assert "MLX_SAFE_PYTHON_FORCE=1" in captured.err
    assert "WARN" in captured.err
    assert exec_calls, "should have called os.execv after WARN"
    assert exec_calls[0][1][0] == sys.executable
    assert "-c" in exec_calls[0][1]


def test_safe_python_proceeds_when_no_cooldown(monkeypatch):
    """Gate clean → exec python without WARN."""
    fake_verdict = mg.CooldownVerdict(0, "no panics", 0, 0, None)
    monkeypatch.setattr(mg, "evaluate_panic_cooldown", lambda: fake_verdict)
    monkeypatch.setenv("MLX_SAFE_PYTHON_BIN", sys.executable)

    exec_calls = []
    monkeypatch.setattr(os, "execv", lambda p, argv: exec_calls.append(argv))

    safe_python_main(["-c", "print('x')"])
    assert exec_calls, "should have called os.execv to run user script"


def test_safe_python_gate_broken_fails_open(monkeypatch, capsys):
    """Gate raises → fail-open WARN + exec python (rc=11)."""
    def _raise(*_a, **_k):
        raise RuntimeError("gate machinery broken")
    monkeypatch.setattr(mg, "evaluate_panic_cooldown", _raise)
    monkeypatch.setenv("MLX_SAFE_PYTHON_BIN", sys.executable)

    exec_calls = []
    monkeypatch.setattr(os, "execv", lambda p, argv: exec_calls.append(argv))

    safe_python_main(["-c", "print('x')"])
    captured = capsys.readouterr()
    assert "WARN" in captured.err
    assert "panic-gate raised" in captured.err
    assert exec_calls
