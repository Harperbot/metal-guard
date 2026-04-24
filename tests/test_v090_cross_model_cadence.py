"""v0.9.0 — cross-model cadence + gemma-4 family + known-panic models.

Covers the C5/C7 additions ported from Harper's internal lib on 2026-04-25:
CrossModelCadenceViolation, gemma-4 floor, _resolve_cross_model_interval,
gemma4_generation_flush, KNOWN_PANIC_MODELS / warn_if_known_panic_model.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import pytest

# Allow running from checkout without install.
sys.path.insert(0, str(Path(__file__).parent.parent))

import metal_guard as mg  # noqa: E402


# ---------------------------------------------------------------------------
# _is_gemma4_family
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("model_id", "expected"),
    [
        # Real gemma-4 model sizes we want to match (as of 2026-04).
        ("mlx-community/gemma-4-1b-it-4bit", True),
        ("mlx-community/gemma-4-4b-it-4bit", True),
        ("mlx-community/gemma-4-12b-it-4bit", True),
        ("mlx-community/gemma-4-27b-it-4bit", True),
        ("mlx-community/gemma-4-31b-it-8bit", True),
        ("mlx-community/gemma-4-26b-a4b-it-4bit", True),     # MoE variant
        ("mlx-community/gemma-4-e4b-it-4bit", True),         # effective-size MoE
        ("mlx-community/gemma-4-e2b-it-4bit", True),
        ("unsloth/gemma-4-31b-it-UD-MLX-4bit", True),
        # Non-canonical sizes: regex is tolerant of future/synthetic names;
        # flush runs harmlessly as a no-op anyway.
        ("mlx-models/gemma-4-9b-it-bf16", True),
        # Negatives — critical that these DO NOT match.
        ("mlx-community/gemma-4b-it", False),                # 4B, not Gen4
        ("mlx-community/gemma-4b-it-8bit", False),           # same
        ("google/gemma-4-31b", False),                        # vendor not allowlisted
        ("mlx-community/gemma-3-27b-it-8bit", False),         # Gen3
        ("mlx-community/gemma-5-31b-it-8bit", False),         # Gen5 (future)
        ("mlx-community/mistral-small-24b", False),
        ("", False),
        ("no-slash", False),
        ("gemma-4-31b-it-8bit", False),                       # missing vendor
    ],
)
def test_is_gemma4_family(model_id: str, expected: bool) -> None:
    assert mg._is_gemma4_family(model_id) is expected


# ---------------------------------------------------------------------------
# _resolve_cross_model_interval
# ---------------------------------------------------------------------------


def test_resolve_cross_model_interval_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(mg._CROSS_MODEL_ENV_VAR, raising=False)
    assert mg._resolve_cross_model_interval(None) == mg._C5_DEFAULT_CROSS_MODEL_INTERVAL_SEC


def test_resolve_cross_model_interval_explicit_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(mg._CROSS_MODEL_ENV_VAR, "999")
    assert mg._resolve_cross_model_interval(120.0) == 120.0


def test_resolve_cross_model_interval_explicit_zero_disables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(mg._CROSS_MODEL_ENV_VAR, "999")
    assert mg._resolve_cross_model_interval(0.0) == 0.0


def test_resolve_cross_model_interval_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(mg._CROSS_MODEL_ENV_VAR, "42")
    assert mg._resolve_cross_model_interval(None) == 42.0


def test_resolve_cross_model_interval_invalid_env_falls_back(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv(mg._CROSS_MODEL_ENV_VAR, "not-a-number")
    with caplog.at_level(logging.WARNING):
        result = mg._resolve_cross_model_interval(None)
    assert result == mg._C5_DEFAULT_CROSS_MODEL_INTERVAL_SEC
    assert any("Invalid" in rec.message for rec in caplog.records)


def test_resolve_cross_model_interval_negative_clamped() -> None:
    assert mg._resolve_cross_model_interval(-5.0) == 0.0


# ---------------------------------------------------------------------------
# CrossModelCadenceViolation
# ---------------------------------------------------------------------------


def test_cross_model_violation_inherits_cadence_violation() -> None:
    """except CadenceViolation still catches the cross-model variant."""
    import time
    exc = mg.CrossModelCadenceViolation(
        model_id="mlx-community/gemma-4-31b-it-8bit",
        last_model="mlx-community/mistral-small",
        last_ts=time.time() - 10,
        cross_model_interval=90.0,
    )
    assert isinstance(exc, mg.CadenceViolation)
    assert isinstance(exc, RuntimeError)
    assert exc.last_model == "mlx-community/mistral-small"
    assert exc.cross_model_interval == 90.0
    assert "cross-model load" in str(exc)


# ---------------------------------------------------------------------------
# CadenceGuard with cross-model cadence
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_guard(tmp_path: Path) -> mg.CadenceGuard:
    return mg.CadenceGuard(
        path=str(tmp_path / "cadence.json"),
        min_interval_sec=180.0,
        cross_model_interval_sec=60.0,
    )


def test_cross_model_interval_sec_property(tmp_guard: mg.CadenceGuard) -> None:
    assert tmp_guard.cross_model_interval_sec == 60.0


def test_effective_cross_model_interval_gemma4_floor(tmp_path: Path) -> None:
    guard = mg.CadenceGuard(
        path=str(tmp_path / "c.json"),
        cross_model_interval_sec=30.0,  # below gemma-4 floor
    )
    # gemma-4 floored to 90
    assert guard._effective_cross_model_interval(
        "mlx-community/gemma-4-31b-it-8bit"
    ) == mg.GEMMA4_MIN_CROSS_MODEL_INTERVAL_SEC
    # non-gemma4 uses configured base
    assert guard._effective_cross_model_interval("mlx-community/mistral-small") == 30.0


def test_effective_cross_model_interval_respects_higher_base(tmp_path: Path) -> None:
    guard = mg.CadenceGuard(
        path=str(tmp_path / "c.json"),
        cross_model_interval_sec=300.0,  # above gemma-4 floor
    )
    assert guard._effective_cross_model_interval(
        "mlx-community/gemma-4-31b-it-8bit"
    ) == 300.0


def test_cross_model_violation_raised(tmp_guard: mg.CadenceGuard) -> None:
    tmp_guard.mark_load("mlx-community/mistral-small")
    with pytest.raises(mg.CrossModelCadenceViolation) as exc_info:
        tmp_guard.check("mlx-community/gemma-4-31b-it-8bit")
    assert exc_info.value.last_model == "mlx-community/mistral-small"
    # gemma-4 target → floor applies even though configured base was 60
    assert exc_info.value.cross_model_interval == mg.GEMMA4_MIN_CROSS_MODEL_INTERVAL_SEC


def test_cross_model_same_model_no_violation(tmp_guard: mg.CadenceGuard) -> None:
    """Same-model reload fires the classic check, not cross-model."""
    tmp_guard.mark_load("mlx-community/mistral-small")
    with pytest.raises(mg.CadenceViolation) as exc_info:
        tmp_guard.check("mlx-community/mistral-small")
    # same-model -> base class, NOT CrossModelCadenceViolation
    assert not isinstance(exc_info.value, mg.CrossModelCadenceViolation)


def test_cross_model_zero_disabled(tmp_path: Path) -> None:
    """cross_model_interval=0 + non-gemma-4 → no cross-model check."""
    guard = mg.CadenceGuard(
        path=str(tmp_path / "c.json"),
        cross_model_interval_sec=0.0,
    )
    guard.mark_load("mlx-community/mistral-small")
    # Different non-gemma model — no violation expected.
    guard.check("mlx-community/phi-4-mini")


def test_cross_model_zero_still_floors_gemma4(tmp_path: Path) -> None:
    """Even with cross_model_interval=0, gemma-4 family enforces the floor.

    This is the key safety property: users who opt out of cross-model
    cadence still get protection on the one family known to panic.
    """
    guard = mg.CadenceGuard(
        path=str(tmp_path / "c.json"),
        cross_model_interval_sec=0.0,
    )
    guard.mark_load("mlx-community/mistral-small")
    with pytest.raises(mg.CrossModelCadenceViolation):
        guard.check("mlx-community/gemma-4-31b-it-8bit")


def test_cross_model_passes_after_interval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import time as _time
    guard = mg.CadenceGuard(
        path=str(tmp_path / "c.json"),
        cross_model_interval_sec=60.0,
    )
    old_ts = _time.time() - 3600
    guard.mark_load("mlx-community/mistral-small", ts=old_ts)
    # 1h > 60s — cross-model check passes.
    guard.check("mlx-community/phi-4-mini")


def test_require_cadence_clear_accepts_cross_model_param(tmp_path: Path) -> None:
    path = str(tmp_path / "c.json")
    guard = mg.CadenceGuard(
        path=path,
        min_interval_sec=180.0,
        cross_model_interval_sec=60.0,
    )
    guard.mark_load("mlx-community/mistral-small")
    with pytest.raises(mg.CrossModelCadenceViolation):
        mg.require_cadence_clear(
            "mlx-community/gemma-4-31b-it-8bit",
            guard=guard,
        )


# ---------------------------------------------------------------------------
# require_cadence_clear no-guard hot path (P0-2 regression coverage)
# ---------------------------------------------------------------------------
# These exercise the env-var resolution path that v0.9.0's P0-2 fix
# introduced via _resolve_cross_model_interval_for_require. Previously
# the helper defaulted to 60s and silently broke v0.8.x callers; the fix
# defaults to 0.0 (disabled) unless the env var opts in.


def test_require_cadence_clear_no_env_no_args_cross_model_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """v0.8.x-compatible default: no env, no args → cross-model disabled.

    Regression test for P0-2. Loading non-gemma model B right after A
    (different vendor scope) must NOT raise when neither env nor
    explicit arg opts in. gemma-4 family still gets its 90s floor via
    the CadenceGuard `_effective_cross_model_interval` path; this test
    uses non-gemma IDs to isolate the default behaviour.
    """
    monkeypatch.delenv(mg._CROSS_MODEL_ENV_VAR, raising=False)
    monkeypatch.setattr(mg, "_CADENCE_PATH_DEFAULT", str(tmp_path / "c.json"))
    mg.require_cadence_clear("mlx-community/mistral-small")
    # Immediately load a different model — must pass without violation.
    mg.require_cadence_clear("mlx-community/phi-4-mini")


def test_require_cadence_clear_env_enables_cross_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Env var opt-in activates cross-model cadence in require_cadence_clear."""
    monkeypatch.setenv(mg._CROSS_MODEL_ENV_VAR, "60")
    monkeypatch.setattr(mg, "_CADENCE_PATH_DEFAULT", str(tmp_path / "c.json"))
    mg.require_cadence_clear("mlx-community/mistral-small")
    with pytest.raises(mg.CrossModelCadenceViolation) as exc_info:
        mg.require_cadence_clear("mlx-community/phi-4-mini")
    assert exc_info.value.last_model == "mlx-community/mistral-small"
    # Non-gemma target → effective == env value, not the gemma-4 floor.
    assert exc_info.value.cross_model_interval == 60.0


def test_require_cadence_clear_explicit_arg_wins_over_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Explicit positive arg overrides env; zero arg defers to env.

    Locks in the resolution order documented in
    :func:`require_cadence_clear` v0.9.0 docstring: explicit arg wins
    when > 0; 0 (the default) falls through to env; env unset → 0.0.
    """
    monkeypatch.setenv(mg._CROSS_MODEL_ENV_VAR, "600")
    monkeypatch.setattr(mg, "_CADENCE_PATH_DEFAULT", str(tmp_path / "c.json"))
    mg.require_cadence_clear("mlx-community/mistral-small")
    # Explicit 30s overrides env 600s → 30s window → but 1s elapsed here
    # so 30s still blocks cross-model.
    with pytest.raises(mg.CrossModelCadenceViolation) as exc_info:
        mg.require_cadence_clear(
            "mlx-community/phi-4-mini", cross_model_interval_sec=30.0,
        )
    assert exc_info.value.cross_model_interval == 30.0


# ---------------------------------------------------------------------------
# gemma4_generation_flush
# ---------------------------------------------------------------------------


def test_gemma4_generation_flush_non_gemma_noop() -> None:
    """Non-gemma models exit before any side effects."""
    # No exception, no side effect.
    mg.gemma4_generation_flush("mlx-community/mistral-small", 0)


def test_gemma4_generation_flush_count_gt_zero_noop() -> None:
    mg.gemma4_generation_flush("mlx-community/gemma-4-31b-it-8bit", 1)
    mg.gemma4_generation_flush("mlx-community/gemma-4-31b-it-8bit", 42)


def test_gemma4_generation_flush_disabled_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("METALGUARD_GEMMA4_FIRSTGEN_DISABLED", "1")
    mg.gemma4_generation_flush("mlx-community/gemma-4-31b-it-8bit", 0)


def test_gemma4_generation_flush_sleep_env_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Override sleep via env — verified by measuring elapsed time < 1s."""
    import time as _time
    # Disable full path so we don't actually sleep 3s.
    monkeypatch.setenv("METALGUARD_GEMMA4_FIRSTGEN_SLEEP_SEC", "0.05")
    t0 = _time.monotonic()
    mg.gemma4_generation_flush("mlx-community/gemma-4-31b-it-8bit", 0)
    elapsed = _time.monotonic() - t0
    # Should be well under 1s (0.05s sleep + mlx import + no-op tolerant).
    assert elapsed < 1.0


def test_gemma4_generation_flush_invalid_sleep_env_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("METALGUARD_GEMMA4_FIRSTGEN_DISABLED", "1")
    monkeypatch.setenv("METALGUARD_GEMMA4_FIRSTGEN_SLEEP_SEC", "not-a-number")
    # Shouldn't raise; disabled path returns immediately anyway.
    mg.gemma4_generation_flush("mlx-community/gemma-4-31b-it-8bit", 0)


# ---------------------------------------------------------------------------
# KNOWN_PANIC_MODELS
# ---------------------------------------------------------------------------


def test_check_known_panic_model_hit() -> None:
    advisory = mg.check_known_panic_model("mlx-community/gemma-4-31b-it-8bit")
    assert advisory is not None
    for key in ("panic_signature", "reproductions", "community", "recommendation", "upstream"):
        assert key in advisory
    # reproductions should include both 2026-04-23 and 2026-04-24 panics
    # to document the repeat-offender status that motivated this entry.
    reproductions_str = " ".join(advisory["reproductions"])
    assert "2026-04-23" in reproductions_str
    assert "2026-04-24" in reproductions_str
    assert len(advisory["reproductions"]) >= 2  # repeat offender required


def test_check_known_panic_model_miss() -> None:
    assert mg.check_known_panic_model("mlx-community/mistral-small") is None
    assert mg.check_known_panic_model("") is None


def test_warn_if_known_panic_model_unknown() -> None:
    assert mg.warn_if_known_panic_model("mlx-community/mistral-small") is False


def test_warn_if_known_panic_model_emits_once(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Second call should not emit another warning record.

    Uses monkeypatch to replace the module-level set so state is
    auto-restored after the test — prevents cross-test pollution
    when tests run in arbitrary order (e.g. pytest-randomly).
    """
    monkeypatch.setattr(mg, "_WARNED_PANIC_MODELS", set())
    with caplog.at_level(logging.WARNING):
        assert mg.warn_if_known_panic_model("mlx-community/gemma-4-31b-it-8bit") is True
        first_count = sum(
            1 for r in caplog.records if "KNOWN_PANIC_MODELS" in r.message
        )
        assert mg.warn_if_known_panic_model("mlx-community/gemma-4-31b-it-8bit") is True
        second_count = sum(
            1 for r in caplog.records if "KNOWN_PANIC_MODELS" in r.message
        )
    assert first_count == 1
    assert second_count == 1  # no additional record emitted
