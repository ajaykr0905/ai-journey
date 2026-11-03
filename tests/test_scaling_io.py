import json

import pytest

from ai_journey.scaling_io import (
    MAX_SCALING_REPORT_BYTES,
    load_scaling_report,
    save_scaling_report,
)
from ai_journey.scaling_laws import ScalingLawError
from ai_journey.scaling_report import build_scaling_report
from ai_journey.scaling_verify import seal_scaling_report
from test_scaling_report import make_protocol


def sealed_report():
    return seal_scaling_report(build_scaling_report(make_protocol()))


def test_scaling_io_round_trips_verified_reports(tmp_path):
    path = tmp_path / "nested" / "scaling.json"
    report = sealed_report()

    save_scaling_report(path, report)

    assert load_scaling_report(path) == report
    assert path.read_bytes().endswith(b"\n")
    assert not list(path.parent.glob(f".{path.name}.*"))


def test_scaling_io_rejects_duplicates_tampering_and_oversize(tmp_path):
    path = tmp_path / "scaling.json"
    path.write_text('{"schema_version":1,"schema_version":1}')
    with pytest.raises(ScalingLawError, match="duplicate"):
        load_scaling_report(path)

    report = sealed_report()
    report["target"]["tokens"] += 1
    path.write_text(json.dumps(report))
    with pytest.raises(ScalingLawError, match="integrity"):
        load_scaling_report(path)

    path.write_bytes(b"{" + b" " * MAX_SCALING_REPORT_BYTES)
    with pytest.raises(ScalingLawError, match="exceeds"):
        load_scaling_report(path)


def test_invalid_save_preserves_existing_report_path(tmp_path):
    path = tmp_path / "scaling.json"
    path.write_bytes(b"protected\n")
    report = sealed_report()
    report["target"]["tokens"] += 1

    with pytest.raises(ScalingLawError, match="integrity"):
        save_scaling_report(path, report)

    assert path.read_bytes() == b"protected\n"
