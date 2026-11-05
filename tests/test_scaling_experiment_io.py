import json

import pytest

from ai_journey.scaling_experiment_io import (
    MAX_EXPERIMENT_REPORT_BYTES,
    load_experiment_report,
    save_experiment_report,
)
from ai_journey.scaling_experiment_verify import seal_experiment_report
from ai_journey.scaling_laws import ScalingLawError
from test_scaling_experiment_verify import report


def test_experiment_report_round_trips_through_atomic_storage(tmp_path):
    path = tmp_path / "report.json"
    sealed = seal_experiment_report(report())

    save_experiment_report(path, sealed)

    assert load_experiment_report(path) == json.loads(path.read_text())


def test_loader_rejects_duplicate_and_oversized_reports(tmp_path):
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema_version":1,"schema_version":1}')
    with pytest.raises(ScalingLawError, match="duplicate"):
        load_experiment_report(duplicate)

    oversized = tmp_path / "oversized.json"
    oversized.write_bytes(b"{" + b" " * MAX_EXPERIMENT_REPORT_BYTES)
    with pytest.raises(ScalingLawError, match="exceeds"):
        load_experiment_report(oversized)


def test_invalid_report_does_not_replace_existing_output(tmp_path):
    path = tmp_path / "report.json"
    path.write_text("preserve-me")

    with pytest.raises(ScalingLawError):
        save_experiment_report(path, {"invalid": True})

    assert path.read_text() == "preserve-me"
