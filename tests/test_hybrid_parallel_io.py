"""Hybrid parallel report IO admits only bounded canonical sealed bytes."""

from __future__ import annotations

import json

import pytest

from ai_journey.hybrid_parallel_io import (
    MAX_HYBRID_PARALLEL_REPORT_BYTES,
    load_hybrid_parallel_report,
    save_hybrid_parallel_report,
)
from ai_journey.hybrid_parallel_report import (
    HybridParallelReportError,
    canonical_hybrid_parallel_report_bytes,
)
from test_hybrid_parallel_report import sealed_report


def test_canonical_report_round_trip_uses_one_exact_encoding(tmp_path):
    report = sealed_report()
    path = tmp_path / "hybrid-parallel.json"

    save_hybrid_parallel_report(path, report)

    assert path.read_bytes() == canonical_hybrid_parallel_report_bytes(report)
    assert load_hybrid_parallel_report(path) == report


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (
            b'{"schema":"first","schema":"second"}',
            "duplicate hybrid parallel report field",
        ),
        (b'{ "schema": "noncanonical" }', "not canonical JSON"),
    ],
)
def test_loader_rejects_duplicate_and_noncanonical_encodings(
    tmp_path, payload, message
):
    path = tmp_path / "hybrid-parallel.json"
    path.write_bytes(payload)

    with pytest.raises(HybridParallelReportError, match=message):
        load_hybrid_parallel_report(path)


def test_loader_rejects_unsupported_schema_even_when_canonically_encoded(tmp_path):
    report = sealed_report()
    report["schema"] = "ai-journey.hybrid-parallel-report.v2"
    path = tmp_path / "hybrid-parallel.json"
    path.write_bytes(canonical_hybrid_parallel_report_bytes(report))

    with pytest.raises(HybridParallelReportError, match="schema"):
        load_hybrid_parallel_report(path)


def test_loader_rejects_oversized_reports_before_json_decode(tmp_path):
    path = tmp_path / "hybrid-parallel.json"
    path.write_bytes(b"{" + b" " * MAX_HYBRID_PARALLEL_REPORT_BYTES)

    with pytest.raises(HybridParallelReportError, match="exceeds 1 MiB"):
        load_hybrid_parallel_report(path)


def test_save_rejects_unsealed_and_preserves_existing_report(tmp_path):
    report = sealed_report()
    path = tmp_path / "hybrid-parallel.json"
    save_hybrid_parallel_report(path, report)
    before = path.read_bytes()
    invalid = json.loads(json.dumps(report))
    invalid["integrity"]["digest"] = "0" * 64

    with pytest.raises(HybridParallelReportError, match="integrity"):
        save_hybrid_parallel_report(path, invalid)

    assert path.read_bytes() == before
