from pathlib import Path

import pytest

from ai_journey.zero_io import load_zero_comparison_report, save_zero_comparison_report
from ai_journey.zero_report import (
    ZeroReportError,
    build_zero_comparison_report,
    seal_zero_comparison_report,
)
from test_zero_report import _plan


def _sealed_report() -> dict[str, object]:
    return seal_zero_comparison_report(build_zero_comparison_report(_plan()))


def test_zero_report_storage_round_trips_verified_evidence(tmp_path: Path):
    path = tmp_path / "nested" / "zero-report.json"
    report = _sealed_report()

    save_zero_comparison_report(path, report)

    assert load_zero_comparison_report(path) == report
    assert path.read_bytes().endswith(b"\n")


def test_zero_report_loader_rejects_duplicate_fields(tmp_path: Path):
    path = tmp_path / "zero-report.json"
    path.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")

    with pytest.raises(ZeroReportError, match="duplicate"):
        load_zero_comparison_report(path)


def test_invalid_zero_report_cannot_replace_existing_evidence(tmp_path: Path):
    path = tmp_path / "zero-report.json"
    report = _sealed_report()
    save_zero_comparison_report(path, report)
    original = path.read_bytes()
    tampered = {**report, "measured": True}

    with pytest.raises(ZeroReportError, match="unmeasured"):
        save_zero_comparison_report(path, tampered)

    assert path.read_bytes() == original
