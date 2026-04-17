"""detect_panic_signature tests — kernel-panic log classifier (2026-04-16)."""
from __future__ import annotations

from metal_guard import detect_panic_signature


def test_detects_prepare_count_underflow():
    text = 'panic(cpu 3): "completeMemory() prepare count underflow" @IOGPUMemory.cpp:492'
    name, explanation = detect_panic_signature(text)
    assert name == "prepare_count_underflow"
    assert explanation is not None
    assert "IOGPUMemory.cpp:492" in explanation


def test_detects_pending_memory_set():
    text = "something fPendingMemorySet lingering"
    name, explanation = detect_panic_signature(text)
    assert name == "pending_memory_set"
    assert explanation is not None


def test_detects_ctxstore_timeout():
    text = "IOGPUCommandQueue hit context store timeout"
    name, explanation = detect_panic_signature(text)
    assert name == "ctxstore_timeout"


def test_detects_metal_oom_fallback():
    text = "kIOGPUCommandBufferCallbackErrorOutOfMemory"
    name, _ = detect_panic_signature(text)
    assert name == "metal_oom"


def test_unknown_returns_none():
    text = "plain-vanilla crash, nothing metal about it"
    name, explanation = detect_panic_signature(text)
    assert name is None
    assert explanation is None


def test_multiline_match_works():
    text = (
        "panic(cpu 2):\n"
        '  "completeMemory() prepare count underflow"\n'
        "  @IOGPUMemory.cpp:492\n"
    )
    name, _ = detect_panic_signature(text)
    assert name == "prepare_count_underflow"
