import json
from pathlib import Path

import pytest

from ai_journey.resource_accounting import ResourceAccountingError
from ai_journey.resource_report import build_resource_report, seal_resource_report
from ai_journey.resource_report_io import (
    MAX_RESOURCE_REPORT_BYTES,
    load_resource_report,
    save_resource_report,
)
from test_resource_report import make_plan


def test_resource_report_io_round_trips_verified_evidence(tmp_path: Path):
    path = tmp_path / "nested" / "report.json"
    report = seal_resource_report(build_resource_report(make_plan()))
    save_resource_report(path, report)
    assert load_resource_report(path) == report
    assert not list(path.parent.glob(f".{path.name}.*"))


def test_resource_report_io_rejects_duplicates_tampering_and_oversize(
    tmp_path: Path,
):
    path = tmp_path / "report.json"
    path.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")
    with pytest.raises(ResourceAccountingError, match="duplicate"):
        load_resource_report(path)

    report = seal_resource_report(build_resource_report(make_plan()))
    report["memory"]["per_device"]["total"] += 1
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ResourceAccountingError, match="recomputation"):
        load_resource_report(path)

    path.write_bytes(b" " * (MAX_RESOURCE_REPORT_BYTES + 1))
    with pytest.raises(ResourceAccountingError, match="1 MiB"):
        load_resource_report(path)
