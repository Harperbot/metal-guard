# Contributing to metal-guard

Thanks for considering a contribution. This document covers the two highest-leverage contribution paths: **panic reports for the registry** and **code / docs PRs**.

## Known Panic Models — schema

`KNOWN_PANIC_MODELS` is a community-curated dict of MLX model IDs that kernel-panic Apple Silicon Macs *even with metal-guard's defensive layers engaged*. The schema is intentionally rich so the entry stays useful when others read it months later.

### Required fields

| Field | Type | Description |
|---|---|---|
| `panic_signature` | str | Exact `IOGPUMemory.cpp:NNN` line + keyword. Match the C++ source location, not just the panic string — Apple sometimes renames the human-readable text but keeps the line number. |
| `first_observed` | str (ISO date `YYYY-MM-DD`) | First reproduction. |
| `last_observed` | str (ISO date `YYYY-MM-DD`) | Most recent reproduction. Bump on each new data point. |
| `reproductions` | list[str] | Production data points. Each entry must include hardware + RAM + time-to-panic + workload summary. Format: `"<hardware> <ram>GB — <date> — <duration> from worker-ready — <workload one-liner>"`. |
| `recommendation` | str | Actionable workaround. Specific (backend / model / config) is more useful than generic ("be careful"). Cite the metal-guard version that was tried — recommendations age. |
| `upstream` | list[str] | URLs of upstream tracking issues (mlx / mlx-lm / mlx-vlm GitHub). At least one. |

### Optional fields

| Field | Type | Description |
|---|---|---|
| `community` | list[str] | External cross-references (GitHub comments by other users, lmstudio bugs, forum threads). Strengthens "this isn't just one user". |
| `panic_by_hardware` | dict | Reserved for v0.10+ schema upgrade — per-hardware observation matrix. Don't add yet. |
| `notes` | str | Caveats, environmental specifics, anything that would surprise the next reader. |

### Quality bar

Entries are conservative by design. We accept either:

1. **A clean production reproduction** — same hardware reproducing the same panic signature on the same model, with metal-guard's L7/L8/L9 layers active. One-shot anecdotes go in `community` not `reproductions`.
2. **A confirmed upstream issue** with the same panic signature where the model is named in the issue body or a maintainer comment.

We **do not** accept:
- "Sometimes panics, sometimes doesn't" without a reproduction recipe
- Models that only panic without metal-guard engaged (those go in the README's "who is affected" section, not the registry)
- Models whose panic was clearly a different root cause (OOM-on-load, transformers ImportError, etc.) — those have separate handling in `_VERSION_ADVISORIES`

### Example entry

```python
"mlx-community/gemma-4-31b-it-8bit": {
    "panic_signature": "IOGPUMemory.cpp:492 prepare_count_underflow",
    "first_observed": "2026-04-23",
    "last_observed": "2026-04-24",
    "reproductions": [
        "M1 Ultra 64GB — 2026-04-23 03:14 local — ~6 min from worker-ready — "
        "subprocess worker, pre-cross-model-cadence, gemma-4 first-gen flush absent",
        "M1 Ultra 64GB — 2026-04-24 03:14 local — ~1.5 min from worker-ready — "
        "same pipeline, post-fix attempt, panicked sooner",
    ],
    "community": [
        "Hannecke (M4 Max 64GB) — ml-explore/mlx#3186 — pivoted to "
        "Qwen3-Coder-30B-A3B MoE",
        "lmstudio bug #1740 — hybrid attention (50 sliding + 10 global) "
        "KV cache 8-bit weights 34GB + full ctx KV 20GB+ > 54GB",
        "ml-explore/mlx-lm#883 (M3 Ultra 96GB)",
    ],
    "recommendation": (
        "metal-guard v0.9.0 narrows the race window via cross-model cadence "
        "(C5) + gemma4_generation_flush (C7) + subprocess_inference_guard "
        "(B1), but does NOT eliminate panic on this model in production "
        "workloads. Switch backend (Ollama / llama.cpp) or pivot to MoE "
        "variant (e.g. mlx-community/gemma-4-26b-a4b-it-4bit)."
    ),
    "upstream": [
        "https://github.com/ml-explore/mlx/issues/3186",
        "https://github.com/ml-explore/mlx-lm/issues/883",
        "https://github.com/ml-explore/mlx/issues/3346",
    ],
},
```

### How to submit

1. **File a [Known Panic Model report](https://github.com/Harperbot/metal-guard/issues/new?template=known-panic-report.yml)** — issue template walks through the schema. Maintainers will draft the dict entry from your report.
2. **OR** open a PR directly modifying `KNOWN_PANIC_MODELS` in `metal_guard.py`. Include the issue number you opened first so reviewers can cross-check.

Maintainers may ask for additional data — typically the redacted panic-full-*.panic file (Full Disk Access on macOS required to read) — to confirm the signature before merging.

## Code / docs PRs

Standard GitHub flow. Run `pytest` before submitting. CHANGELOG.md update is required for behavioural changes; not required for typo fixes / docs polish.

If your PR adds a new defence layer (L10+), please also extend the test matrix to cover the new layer's failure modes.

## License

By contributing you agree your contribution is licensed under the same MIT license as the project.
