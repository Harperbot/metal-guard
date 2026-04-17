"""L9 panic report ingest tests (2026-04-16)."""
from __future__ import annotations

import json

from metal_guard import (
    ingest_panics_jsonl,
    parse_panic_reports,
)


_SAMPLE_PANIC_HEADER = """panic(cpu 3 caller 0xfffffe0027ffd0f8): "completeMemory() prepare count underflow" @IOGPUMemory.cpp:492
Debugger message: panic
OS release type: User
OS version: 24G624

mach_absolute_time: 0x1d6f71e9ecf
Epoch Time:        sec       usec
  Boot    : 0x69dfb814 0x00085220
  Sleep   : 0x00000000 0x00000000
  Wake    : 0x00000000 0x00000000
  Calendar: 0x69e10147 0x000161c7

Panicked task 0xfffffe2b57f03248: 67178 pages, 13 threads: pid 25467: Python
"""

_SAMPLE_OTHER_PANIC = """panic(cpu 2): a completely different kind of panic
OS version: 24G624
Calendar: 0x68000000 0x00000000
Panicked task 0xffffabcd: 123 pages, 4 threads: pid 8888: kernel_task
"""


def _write_panic_file(path, content):
    path.write_text(content)


def test_parse_finds_iogpu_signature(tmp_path):
    panic = tmp_path / "20260416_233327.panic"
    _write_panic_file(panic, _SAMPLE_PANIC_HEADER)

    records = parse_panic_reports(str(tmp_path))
    assert len(records) == 1
    rec = records[0]
    assert rec["signature"] == "prepare_count_underflow"
    assert rec["pid"] == 25467
    assert rec["source_file"] == str(panic)
    assert rec["ts"] > 0


def test_parse_missing_directory_returns_empty(tmp_path):
    missing = tmp_path / "does-not-exist"
    assert parse_panic_reports(str(missing)) == []


def test_parse_empty_directory(tmp_path):
    assert parse_panic_reports(str(tmp_path)) == []


def test_parse_ignores_non_panic_files(tmp_path):
    (tmp_path / "something.log").write_text("noise")
    (tmp_path / "another.txt").write_text("more noise")
    assert parse_panic_reports(str(tmp_path)) == []


def test_parse_since_ts_filter(tmp_path):
    older = tmp_path / "old.panic"
    _write_panic_file(older, _SAMPLE_OTHER_PANIC)  # Calendar 0x68000000 = ~2025
    newer = tmp_path / "new.panic"
    _write_panic_file(newer, _SAMPLE_PANIC_HEADER)  # Calendar 0x69e10147 = 2026-04

    cutoff = 1_700_000_000  # ~2023; both pass
    records = parse_panic_reports(str(tmp_path), since_ts=cutoff)
    assert len(records) == 2

    cutoff2 = 1_770_000_000  # ~2026-02; only newer
    records2 = parse_panic_reports(str(tmp_path), since_ts=cutoff2)
    assert len(records2) == 1
    assert records2[0]["signature"] == "prepare_count_underflow"


def test_ingest_writes_jsonl(tmp_path):
    panic_dir = tmp_path / "reports"
    panic_dir.mkdir()
    _write_panic_file(panic_dir / "crash.panic", _SAMPLE_PANIC_HEADER)

    jsonl = tmp_path / "panics.jsonl"
    written = ingest_panics_jsonl(
        report_dir=str(panic_dir),
        jsonl_path=str(jsonl),
    )
    assert written == 1
    assert jsonl.exists()
    with open(jsonl) as f:
        line = f.readline()
    rec = json.loads(line)
    assert rec["signature"] == "prepare_count_underflow"
    assert rec["pid"] == 25467


def test_ingest_is_idempotent(tmp_path):
    panic_dir = tmp_path / "reports"
    panic_dir.mkdir()
    _write_panic_file(panic_dir / "crash.panic", _SAMPLE_PANIC_HEADER)

    jsonl = tmp_path / "panics.jsonl"
    first = ingest_panics_jsonl(
        report_dir=str(panic_dir),
        jsonl_path=str(jsonl),
    )
    second = ingest_panics_jsonl(
        report_dir=str(panic_dir),
        jsonl_path=str(jsonl),
    )
    assert first == 1
    assert second == 0  # no duplicates


def test_ingest_picks_up_new_panic_after_existing(tmp_path):
    panic_dir = tmp_path / "reports"
    panic_dir.mkdir()
    _write_panic_file(panic_dir / "old.panic", _SAMPLE_OTHER_PANIC)

    jsonl = tmp_path / "panics.jsonl"
    assert ingest_panics_jsonl(report_dir=str(panic_dir), jsonl_path=str(jsonl)) == 1

    # New panic appears
    _write_panic_file(panic_dir / "new.panic", _SAMPLE_PANIC_HEADER)
    assert ingest_panics_jsonl(report_dir=str(panic_dir), jsonl_path=str(jsonl)) == 1

    # Third call picks up nothing
    assert ingest_panics_jsonl(report_dir=str(panic_dir), jsonl_path=str(jsonl)) == 0

    lines = open(jsonl).read().strip().split("\n")
    assert len(lines) == 2
