"""v0.11.0 error_classifier — central regex table tests."""
from __future__ import annotations

import metal_guard as mg


def test_kernel_panic_prepare_count_underflow():
    body = (
        "panic(cpu 4 caller 0x...):\n"
        '"completeMemory() prepare count underflow" '
        "@IOGPUMemory.cpp:492\n"
    )
    cls = mg.classify_mlx_error(body)
    assert cls is not None
    assert cls.severity == "kernel_panic"
    assert cls.recovery_hint == "wait_lockout"


def test_kernel_panic_pending_memory_set():
    body = "IOGPUGroupMemory.cpp:219 pending memory set panic..."
    cls = mg.classify_mlx_error(body)
    assert cls is not None
    assert cls.severity == "kernel_panic"
    assert cls.name == "iogpu_pending_memory_set"


def test_command_buffer_oom():
    cls = mg.classify_mlx_error(
        "kIOGPUCommandBufferCallbackErrorOutOfMemory"
    )
    assert cls.severity == "command_buffer_oom"
    assert cls.recovery_hint == "respawn_now"


def test_gpu_hang():
    cls = mg.classify_mlx_error("kIOGPUCommandBufferCallbackErrorHang")
    assert cls.severity == "gpu_hang"


def test_gpu_page_fault():
    cls = mg.classify_mlx_error(
        "[METAL] Command buffer execution failed: GPU Address Fault Error "
        "(0000000b:kIOGPUCommandBufferCallbackErrorPageFault)"
    )
    assert cls.severity == "gpu_page_fault"


def test_descriptor_leak():
    cls = mg.classify_mlx_error(
        "[metal::malloc] Resource limit (499000) exceeded"
    )
    assert cls.severity == "descriptor_leak"
    assert cls.recovery_hint == "force_reload"


def test_metal_completion_sigabrt():
    body = "libc++abi: ... std::terminate ... MetalStream::add_temporary"
    cls = mg.classify_mlx_error(body)
    assert cls.name == "metal_completion_sigabrt"
    assert cls.severity == "process_abort"


def test_unknown_returns_none():
    assert mg.classify_mlx_error("totally unrelated") is None
    assert mg.classify_mlx_error("") is None
    assert mg.classify_mlx_error(None) is None


def test_kernel_panic_priority_over_abort():
    """When both kernel + abort signatures appear, kernel must win."""
    body = (
        "kIOGPUCommandBufferCallbackErrorOutOfMemory cascading\n"
        "panic: prepare_count_underflow IOGPUMemory.cpp:492"
    )
    cls = mg.classify_mlx_error(body)
    assert cls.severity == "kernel_panic"


def test_helpers():
    assert mg.is_kernel_panic_signature("prepare_count_underflow IOGPUMemory.cpp:492")
    assert not mg.is_kernel_panic_signature("kIOGPUCommandBufferCallbackErrorHang")
    assert mg.is_process_abort_signature("kIOGPUCommandBufferCallbackErrorHang")
    assert not mg.is_process_abort_signature("prepare_count_underflow IOGPUMemory.cpp:492")


def test_subprocess_crash_error_classifies_detail():
    exc = mg.SubprocessCrashError(
        "model-x", -9,
        "kIOGPUCommandBufferCallbackErrorHang",
    )
    assert exc.error_class is not None
    assert exc.error_class.severity == "gpu_hang"
    assert exc.recovery_hint == "respawn_now"
    assert "gpu_hang" in str(exc)


def test_subprocess_crash_error_no_detail_no_class():
    exc = mg.SubprocessCrashError("model-x", -9, "")
    assert exc.error_class is None
    assert exc.recovery_hint == "unknown"


def test_subprocess_crash_error_unknown_detail_no_class():
    exc = mg.SubprocessCrashError("model-x", 1, "totally unrelated stderr")
    assert exc.error_class is None
    assert exc.recovery_hint == "unknown"
