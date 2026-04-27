"""v0.11.0 — L10b abort scanner / L13b apple_gpu_family / L14 ResourceTracker /
breadcrumb_with_meta / KNOWN_PANIC_MODELS schema upgrade."""
from __future__ import annotations

import datetime
import os

import pytest

import metal_guard as mg


# ─── L14 ResourceTracker ──────────────────────────────────────


def test_resource_tracker_records():
    t = mg.ResourceTracker(cold_restart_after=10)
    assert t.record_inference() == 1
    assert t.record_inference() == 2


def test_resource_tracker_threshold():
    t = mg.ResourceTracker(cold_restart_after=5)
    for _ in range(4):
        t.record_inference()
    assert t.should_cold_restart() is False
    t.record_inference()
    assert t.should_cold_restart() is True


def test_resource_tracker_reset():
    t = mg.ResourceTracker(cold_restart_after=5)
    for _ in range(3):
        t.record_inference()
    prior = t.reset()
    assert prior == 3
    assert t.snapshot()["current_count"] == 0
    assert t.snapshot()["last_restart_at_count"] == 3


def test_resource_tracker_kill_switch(monkeypatch):
    monkeypatch.setenv("METALGUARD_COLD_RESTART_DISABLED", "1")
    t = mg.ResourceTracker(cold_restart_after=2)
    t.record_inference()
    t.record_inference()
    t.record_inference()
    assert t.should_cold_restart() is False  # kill switch overrides


def test_resource_tracker_env_override(monkeypatch):
    monkeypatch.setenv("METALGUARD_COLD_RESTART_AFTER_N", "100")
    t = mg.ResourceTracker()
    assert t.cold_restart_after == 100


def test_resource_tracker_thread_safe():
    import threading
    t = mg.ResourceTracker(cold_restart_after=100_000)

    def _w():
        for _ in range(100):
            t.record_inference()

    threads = [threading.Thread(target=_w) for _ in range(10)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert t.snapshot()["current_count"] == 1000


def test_global_resource_tracker_singleton():
    mg._reset_global_resource_tracker()
    a = mg.global_resource_tracker()
    b = mg.global_resource_tracker()
    assert a is b


# ─── L13b apple_gpu_family ────────────────────────────────────


def test_classify_gpu_family_m1():
    assert mg._classify_gpu_family("applegpu_g13") == "M1"
    assert mg._classify_gpu_family("applegpu_g13d") == "M1"


def test_classify_gpu_family_m5():
    assert mg._classify_gpu_family("applegpu_g17s") == "M5"


def test_classify_gpu_family_unknown():
    assert mg._classify_gpu_family("") == "unknown"
    assert mg._classify_gpu_family(None) == "unknown"
    assert mg._classify_gpu_family("foo_bar") == "foo_bar"


def test_apple_gpu_family_returns_dict():
    info = mg.apple_gpu_family()
    assert isinstance(info, dict)
    assert "available" in info
    assert "family" in info


def test_apple_gpu_family_no_mlx_graceful(monkeypatch):
    """When MLX import fails, returns structured fallback (no raise)."""
    import builtins
    real_import = builtins.__import__

    def _fake_import(name, *a, **kw):
        if name == "mlx.core" or name.startswith("mlx."):
            raise ImportError("simulated")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    info = mg.apple_gpu_family()
    assert info["available"] is False
    assert "error" in info


# ─── L10b scan_recent_aborts ───────────────────────────────────


def _write_panic_file(panic_dir, name, body, mtime_dt=None):
    p = panic_dir / name
    p.write_text(body)
    if mtime_dt is not None:
        ts = mtime_dt.timestamp()
        os.utime(p, (ts, ts))
    return p


@pytest.fixture
def isolated_panic_dir(tmp_path, monkeypatch):
    panic_dir = tmp_path / "panics"
    panic_dir.mkdir()
    # v0.11: scan_recent_panics uses PANIC_REPORTS_GLOBS;
    # scan_recent_aborts uses _ABORT_REPORTS_GLOBS. Both must point at our
    # synthetic dir so the abort-via-cascade path matches in tests.
    monkeypatch.setattr(
        mg, "PANIC_REPORTS_GLOBS", (str(panic_dir / "*.panic"),),
    )
    monkeypatch.setattr(
        mg, "_ABORT_REPORTS_GLOBS", (str(panic_dir / "*.panic"),),
    )
    return panic_dir


def test_kernel_panic_classified_panic_not_abort(isolated_panic_dir):
    _write_panic_file(
        isolated_panic_dir, "p.panic",
        "panic: prepare_count_underflow IOGPUMemory.cpp:492",
    )
    panics = mg.scan_recent_panics(72.0)
    aborts = mg.scan_recent_aborts(24.0)
    assert len(panics) == 1
    assert len(aborts) == 0


def test_command_buffer_oom_classified_abort(isolated_panic_dir):
    _write_panic_file(
        isolated_panic_dir, "a.panic",
        "kIOGPUCommandBufferCallbackErrorOutOfMemory",
    )
    panics = mg.scan_recent_panics(72.0)
    aborts = mg.scan_recent_aborts(24.0)
    assert len(panics) == 0
    assert len(aborts) == 1
    assert aborts[0].error_class == "cmd_buffer_oom"


def test_descriptor_leak_classified_abort(isolated_panic_dir):
    _write_panic_file(
        isolated_panic_dir, "leak.panic",
        "[metal::malloc] Resource limit (499000) exceeded",
    )
    aborts = mg.scan_recent_aborts(24.0)
    assert len(aborts) == 1
    assert aborts[0].error_class == "metal_descriptor_leak"


def test_abort_default_window_24h(isolated_panic_dir):
    now = datetime.datetime.now()
    old = now - datetime.timedelta(hours=30)
    _write_panic_file(
        isolated_panic_dir, "old.panic",
        "kIOGPUCommandBufferCallbackErrorHang", mtime_dt=old,
    )
    aborts_24h = mg.scan_recent_aborts(24.0, now=now)
    aborts_72h = mg.scan_recent_aborts(72.0, now=now)
    assert len(aborts_24h) == 0
    assert len(aborts_72h) == 1


# ─── KNOWN_PANIC_MODELS v0.11 schema ───────────────────────────


def test_models_by_tier_panic():
    panics = mg.models_by_tier("panic")
    assert "mlx-community/gemma-4-31b-it-8bit" in panics


def test_models_by_tier_abort():
    aborts = mg.models_by_tier("abort")
    assert "mlx-community/Qwen3-VL-2B-Instruct" in aborts
    assert "mlx-community/Qwen3.6-35B-A3B-8bit" in aborts


def test_models_by_tier_degradation():
    degraded = mg.models_by_tier("degradation")
    assert "mlx-community/Qwen3.5-27B-4bit" in degraded
    assert "mlx-community/Qwen3.5-35B-A3B-8bit" in degraded


def test_models_affecting_gpu_family_m5():
    affected = mg.models_affecting_gpu_family("M5")
    assert "mlx-community/Qwen3-VL-2B-Instruct" in affected
    # Pure M4 entries should not appear
    assert "mlx-community/Qwen3.6-35B-A3B-8bit" not in affected


def test_check_known_panic_model_for_gpu_m1_filters_m5_only():
    advise = mg.check_known_panic_model_for_gpu(
        "mlx-community/Qwen3-VL-2B-Instruct", gpu_family="M1",
    )
    assert advise is None  # M5-only entries → filtered out


def test_check_known_panic_model_for_gpu_m5_returns_classes():
    advise = mg.check_known_panic_model_for_gpu(
        "mlx-community/Qwen3-VL-2B-Instruct", gpu_family="M5",
    )
    assert advise is not None
    assert len(advise["error_classes"]) == 2  # both Hang + PageFault


def test_check_known_panic_model_for_gpu_unfiltered_legacy():
    """No gpu_family arg → legacy unfiltered behaviour."""
    advise = mg.check_known_panic_model_for_gpu(
        "mlx-community/gemma-4-31b-it-8bit",
    )
    assert advise is not None
    assert advise["tier"] == "panic"


def test_legacy_check_known_panic_model_still_works():
    """v0.10 caller calling check_known_panic_model() unchanged."""
    advise = mg.check_known_panic_model("mlx-community/gemma-4-31b-it-8bit")
    assert advise is not None
    assert "panic_signature" in advise  # legacy field preserved


# ─── breadcrumb_with_meta ──────────────────────────────────────


def test_breadcrumb_with_meta_writes_format(tmp_path):
    bc = tmp_path / "metal_breadcrumb.log"
    g = mg.MetalGuard(breadcrumb_path=str(bc))
    g.breadcrumb_with_meta("SUBPROC_PRE", "model-x", ctx=2048, kv_bytes=12_000_000)
    line = bc.read_text().strip()
    assert "SUBPROC_PRE: model-x" in line
    assert "ctx=2048" in line
    assert "kv_bytes=12000000" in line


def test_breadcrumb_with_meta_no_payload(tmp_path):
    bc = tmp_path / "bc.log"
    g = mg.MetalGuard(breadcrumb_path=str(bc))
    g.breadcrumb_with_meta("RAM_WARN", elapsed_ms=8421)
    line = bc.read_text().strip()
    # No double colon when payload empty
    assert "RAM_WARN |" in line
    assert "RAM_WARN:" not in line


def test_breadcrumb_with_meta_no_kwargs(tmp_path):
    bc = tmp_path / "bc.log"
    g = mg.MetalGuard(breadcrumb_path=str(bc))
    g.breadcrumb_with_meta("CHECK", "all-clear")
    line = bc.read_text().strip()
    assert " | " not in line  # no separator when no meta
    assert "CHECK: all-clear" in line


def test_breadcrumb_with_meta_keys_sorted(tmp_path):
    bc = tmp_path / "bc.log"
    g = mg.MetalGuard(breadcrumb_path=str(bc))
    g.breadcrumb_with_meta("X", "y", zebra=1, alpha=2, mango=3)
    line = bc.read_text().strip()
    assert line.endswith("alpha=2 mango=3 zebra=1")


def test_breadcrumb_compat_with_lazy_regex(tmp_path):
    """Critical: lazy regex captures payload without polluting with meta."""
    bc = tmp_path / "bc.log"
    g = mg.MetalGuard(breadcrumb_path=str(bc))
    g.breadcrumb_with_meta("SUBPROC_PRE", "model-x", ctx=2048)
    line = bc.read_text().strip()
    m = mg._BREADCRUMB_LINE_RE.match(line)
    assert m is not None
    assert m.group("tag") == "SUBPROC_PRE"
    assert m.group("payload").strip() == "model-x"  # NOT "model-x | ctx=2048"
    assert "ctx=2048" in (m.group("meta") or "")


# ─── CooldownVerdict.abort_count_24h ──────────────────────────


def test_cooldown_verdict_has_abort_count_field():
    """v0.11.0 added abort_count_24h informational field."""
    v = mg.evaluate_panic_cooldown()
    assert hasattr(v, "abort_count_24h")
    assert isinstance(v.abort_count_24h, int)


def test_cooldown_verdict_5_arg_construction_still_works():
    """Backward-compat: 5-positional-arg construction remains valid (default 0)."""
    v = mg.CooldownVerdict(
        0, "test", 0, 0, None,
    )
    assert v.exit_code == 0
    assert v.abort_count_24h == 0


# ─── critic R1 fix regressions ────────────────────────────────


def test_breadcrumb_pipe_in_payload_sanitized(tmp_path):
    """critic R1 P1-3: pipe in payload would break reader FIFO pairing.

    Writer must replace `|` so reader's lazy regex finds the real meta
    separator. Otherwise model_id 'a|b' gets split into payload='a',
    meta='b' and PRE/POST never pair.
    """
    bc = tmp_path / "bc.log"
    g = mg.MetalGuard(breadcrumb_path=str(bc))
    g.breadcrumb_with_meta("PRE", "model|with|pipe", ctx=10)
    line = bc.read_text().strip()
    m = mg._BREADCRUMB_LINE_RE.match(line)
    assert m is not None
    # Pipe in payload converted to '/'
    assert "|" not in m.group("payload")
    assert "model/with/pipe" in m.group("payload")


def test_breadcrumb_pipe_in_meta_value_sanitized(tmp_path):
    """Pipe in meta value also breaks the parser — same defense applies."""
    bc = tmp_path / "bc.log"
    g = mg.MetalGuard(breadcrumb_path=str(bc))
    g.breadcrumb_with_meta("PRE", "model-x", url="http://x|y/path")
    line = bc.read_text().strip()
    # `|` in value gets replaced; only the meta separator's `|` remains
    assert line.count("|") == 1


def test_known_panic_models_schema_sanity():
    """critic R1 P1-4: rigid invariants on KNOWN_PANIC_MODELS schema.

    Locks legacy `panic_signature` field equal to first error_class signature
    so dual-write entries don't drift. Locks tier vocabulary.
    """
    allowed_tiers = {"panic", "abort", "degradation"}
    for mid, e in mg.KNOWN_PANIC_MODELS.items():
        # Tier vocabulary
        if "tier" in e:
            assert e["tier"] in allowed_tiers, (
                f"{mid}: tier={e['tier']!r} not in {allowed_tiers}"
            )
        # Legacy panic_signature should match first error_class signature
        if "error_classes" in e and e["error_classes"]:
            first_sig = e["error_classes"][0].get("signature", "")
            legacy_sig = e.get("panic_signature", "")
            if legacy_sig:
                assert legacy_sig == first_sig, (
                    f"{mid}: legacy panic_signature {legacy_sig!r} drifted "
                    f"from error_classes[0].signature {first_sig!r}"
                )
        # Required v0.11 fields when present
        if "error_classes" in e:
            for ec in e["error_classes"]:
                required = {"type", "signature", "first_seen_via",
                            "hardware", "gpu_family", "workload", "mitigation"}
                missing = required - set(ec.keys())
                assert not missing, f"{mid}: error_class missing {missing}"


def test_scan_recent_aborts_includes_ips_glob(monkeypatch, tmp_path):
    """critic R1 P0-1: ensure abort scanner reads `.ips` reports too,
    not just `panic-full-*.panic` (where real macOS process aborts go).
    """
    ips_dir = tmp_path / "ips"
    ips_dir.mkdir()
    # Write a synthetic .ips file with abort signature
    (ips_dir / "MyApp-2026.ips").write_text(
        '"signal":"SIGABRT" ... kIOGPUCommandBufferCallbackErrorOutOfMemory'
    )
    monkeypatch.setattr(
        mg, "_ABORT_REPORTS_GLOBS", (str(ips_dir / "*.ips"),),
    )
    aborts = mg.scan_recent_aborts(24.0)
    assert len(aborts) == 1
    assert aborts[0].error_class == "cmd_buffer_oom"


def test_scan_recent_aborts_dedupes_overlapping_globs(monkeypatch, tmp_path):
    """If a panic file appears in both panic + abort globs (cascade case),
    don't double-count it.
    """
    panic_dir = tmp_path / "panics"
    panic_dir.mkdir()
    p = panic_dir / "panic-full-1.panic"
    p.write_text("kIOGPUCommandBufferCallbackErrorHang")
    # Same path matches both globs (panic + cascade)
    monkeypatch.setattr(
        mg, "_ABORT_REPORTS_GLOBS",
        (str(panic_dir / "*.panic"), str(panic_dir / "*.panic")),  # dupe!
    )
    aborts = mg.scan_recent_aborts(24.0)
    assert len(aborts) == 1  # not 2


# ─── v0.11.4: MLX_VERSION_BLOCKLIST + 2026-04-28 panic registry additions ─


def test_mlx_version_blocklist_known_entry():
    block = mg.check_mlx_version_blocked("0.31.2")
    assert block is not None
    assert block["severity"] == "critical"
    assert "SIGSEGV" in block["signature"] or "segfault" in block["reason"].lower()
    assert block["upstream"][0].startswith("https://github.com/ml-explore/mlx")


def test_mlx_version_blocklist_unknown_returns_none():
    assert mg.check_mlx_version_blocked("0.31.1") is None
    assert mg.check_mlx_version_blocked("0.32.0") is None
    assert mg.check_mlx_version_blocked("") is None


def test_mlx_version_blocklist_schema_sanity():
    """Lock required fields on every blocklist entry."""
    required = {"severity", "error_class", "signature", "reason",
                "first_observed", "upstream", "workaround"}
    allowed_severity = {"critical", "high", "medium"}
    for ver, e in mg.MLX_VERSION_BLOCKLIST.items():
        missing = required - set(e.keys())
        assert not missing, f"MLX {ver}: blocklist entry missing {missing}"
        assert e["severity"] in allowed_severity, (
            f"MLX {ver}: severity={e['severity']!r} not in {allowed_severity}"
        )
        assert e["upstream"], f"MLX {ver}: upstream must be non-empty"


def test_v0114_new_panic_registry_entries_present():
    """Sanity that the 2026-04-28 sweep entries landed in the registry."""
    expected = {
        "mlx-community/Qwen3.5-122B-A10B-VLM-MTP-5bit",
        "mlx-community/Qwen3-Coder-Next-4bit",
        "mlx-community/Qwen3.5-9B-4bit",
        "mlx-community/Qwen3.6-35B-A3B-VLM-MTP-8bit",
        "mlx-community/kimi-k2.5",
    }
    missing = expected - set(mg.KNOWN_PANIC_MODELS.keys())
    assert not missing, f"v0.11.4 sweep entries missing: {missing}"


def test_v0114_silent_corruption_error_class_recorded():
    """The Qwen3.6 VLM-as-text entry uses the new silent_corruption type."""
    e = mg.KNOWN_PANIC_MODELS["mlx-community/Qwen3.6-35B-A3B-VLM-MTP-8bit"]
    types = {ec["type"] for ec in e["error_classes"]}
    assert "silent_corruption" in types
