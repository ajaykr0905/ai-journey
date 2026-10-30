import json

import pytest

from ai_journey.distributed_capacity_io import (
    MAX_DISTRIBUTED_CAPACITY_BYTES,
    load_distributed_capacity_report,
    save_distributed_capacity_report,
)
from ai_journey.distributed_capacity_report import (
    DistributedCapacityReportError,
    build_distributed_capacity_report,
    seal_distributed_capacity_report,
)
from test_distributed_capacity_report import _plan


def _sealed_report():
    return seal_distributed_capacity_report(build_distributed_capacity_report(_plan()))


def test_capacity_report_io_round_trips_verified_evidence(tmp_path):
    path = tmp_path / "nested" / "capacity.json"
    report = _sealed_report()
    save_distributed_capacity_report(path, report)
    assert load_distributed_capacity_report(path) == report


def test_capacity_report_loader_rejects_duplicate_fields(tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")
    with pytest.raises(DistributedCapacityReportError, match="duplicate"):
        load_distributed_capacity_report(path)


def test_capacity_report_loader_bounds_input_before_json_decode(tmp_path):
    path = tmp_path / "oversize.json"
    path.write_bytes(b" " * (MAX_DISTRIBUTED_CAPACITY_BYTES + 1))
    with pytest.raises(DistributedCapacityReportError, match="exceeds"):
        load_distributed_capacity_report(path)


def test_capacity_report_loader_rejects_valid_json_tampering(tmp_path):
    path = tmp_path / "tampered.json"
    report = _sealed_report()
    report["gradient_sync"]["bucket_count"] += 1
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(DistributedCapacityReportError, match="integrity"):
        load_distributed_capacity_report(path)
