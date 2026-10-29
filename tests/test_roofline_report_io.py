import json
from pathlib import Path

import pytest

from ai_journey.resource_accounting import ResourceAccountingError
from ai_journey.roofline_report import build_roofline_report, seal_roofline_report
from ai_journey.roofline_report_io import (
    MAX_ROOFLINE_REPORT_BYTES,
    load_roofline_report,
    save_roofline_report,
)
from test_roofline_report import make_plan


def test_roofline_report_io_round_trips_verified_evidence(tmp_path: Path):
    path = tmp_path / "nested" / "roofline.json"
    report = seal_roofline_report(build_roofline_report(make_plan()))

    save_roofline_report(path, report)

    assert load_roofline_report(path) == report
    assert path.read_bytes().endswith(b"\n")
    assert not list(path.parent.glob(f".{path.name}.*"))


def test_roofline_report_io_rejects_duplicates_tampering_and_oversize(
    tmp_path: Path,
):
    path = tmp_path / "roofline.json"
    path.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")
    with pytest.raises(ResourceAccountingError, match="duplicate"):
        load_roofline_report(path)

    report = seal_roofline_report(build_roofline_report(make_plan()))
    report["summary"]["operation_count"] += 1
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ResourceAccountingError, match="integrity"):
        load_roofline_report(path)

    path.write_bytes(b"{" + b" " * MAX_ROOFLINE_REPORT_BYTES)
    with pytest.raises(ResourceAccountingError, match="exceeds"):
        load_roofline_report(path)


def test_roofline_report_save_does_not_replace_existing_file_on_invalid_input(
    tmp_path: Path,
):
    path = tmp_path / "roofline.json"
    path.write_bytes(b"protected\n")
    report = seal_roofline_report(build_roofline_report(make_plan()))
    report["operations"]["attention_query"]["flops"] += 1

    with pytest.raises(ResourceAccountingError, match="integrity"):
        save_roofline_report(path, report)

    assert path.read_bytes() == b"protected\n"
