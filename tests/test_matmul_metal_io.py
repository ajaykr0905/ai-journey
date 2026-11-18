import itertools
import json

import pytest

from ai_journey.matmul_metal_config import config_from_mapping
from ai_journey.matmul_metal_io import load_report, save_report
from ai_journey.matmul_metal_mapping import MatmulMetalError
from ai_journey.matmul_metal_report import measure_report
from ai_journey.matmul_metal_verify import MatmulMetalVerificationError
from ai_journey.memory_access_plan import canonical_json_bytes
from ai_journey.memory_access_report import DIGEST_FIELD
from test_matmul_metal_config import make_config_mapping
from test_matmul_metal_recorded import RecordedBackend


def sealed():
    config = config_from_mapping(make_config_mapping())
    clock = itertools.count(0, 1_000).__next__
    return config, measure_report(config, backend=RecordedBackend(), clock=clock)


def test_a_saved_report_is_canonical_and_loads_with_its_summary(tmp_path):
    config, report = sealed()
    path = tmp_path / "nested" / "report.json"

    saved = save_report(path, report, config=config)
    loaded, summary = load_report(path, config=config)

    assert path.read_bytes() == canonical_json_bytes(report) + b"\n"
    assert loaded == report
    assert saved == summary
    assert summary["report_sha256"] == report[DIGEST_FIELD]
    assert not [item for item in path.parent.iterdir() if item.name != "report.json"]


def test_loading_names_tampering_before_formatting(tmp_path):
    _, report = sealed()
    path = tmp_path / "report.json"

    path.write_text(json.dumps(report, indent=1), encoding="utf-8")
    with pytest.raises(MatmulMetalVerificationError, match="not canonical"):
        load_report(path)
    report["run"]["elapsed_ns"] += 1
    path.write_text(json.dumps(report, indent=1), encoding="utf-8")
    with pytest.raises(MatmulMetalVerificationError, match="digest"):
        load_report(path)


def test_unreadable_files_are_bad_input_not_rejected_reports(tmp_path):
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"kind": 1, "kind": 2}', encoding="utf-8")
    link = tmp_path / "link.json"
    link.symlink_to(duplicate)

    for path, message in (
        (tmp_path / "missing.json", "regular non-symlink"),
        (duplicate, "duplicate JSON field"),
        (link, "regular non-symlink"),
    ):
        with pytest.raises(MatmulMetalError, match=message) as raised:
            load_report(path)
        assert not isinstance(raised.value, MatmulMetalVerificationError)


def test_saving_verifies_first_and_never_writes_over_the_config(tmp_path):
    config, report = sealed()
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(make_config_mapping()), encoding="utf-8")

    with pytest.raises(MatmulMetalError, match="must not alias a protected input"):
        save_report(config_path, report, protected=(config_path,))
    assert json.loads(config_path.read_text(encoding="utf-8"))["seed"] == 74
    broken = {**report, "measured": True}
    output = tmp_path / "report.json"
    with pytest.raises(MatmulMetalVerificationError, match="digest"):
        save_report(output, broken)
    assert not output.exists()
