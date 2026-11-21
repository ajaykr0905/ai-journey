import json

import pytest

from ai_journey.memory_access_plan import canonical_json_bytes
from ai_journey.metal_fused_io import load_report, save_report
from ai_journey.metal_fused_source import MetalFusedError
from ai_journey.metal_fused_verify import MetalFusedVerificationError
from test_metal_fused_ceiling import recorded_ceiling
from test_metal_fused_report import small_config, synthetic_report


def test_a_saved_report_is_canonical_and_loads_with_its_summary(tmp_path):
    report = synthetic_report(tmp_path)
    path = tmp_path / "out" / "report.json"

    summary = save_report(path, report, config=small_config())

    assert path.read_bytes() == canonical_json_bytes(report) + b"\n"
    loaded, again = load_report(
        path, config=small_config(), ceiling=recorded_ceiling(tmp_path)
    )
    assert loaded == report and again == summary
    assert summary["report_sha256"] == report["report_sha256"]


def test_save_verifies_first_and_keeps_protected_inputs(tmp_path):
    report = synthetic_report(tmp_path)
    config = tmp_path / "config.json"
    config.write_text("{}", encoding="utf-8")

    with pytest.raises(MetalFusedError, match="protected input"):
        save_report(config, report, protected=(config,))
    assert config.read_text(encoding="utf-8") == "{}"
    broken = {**report, "measured": True}
    target = tmp_path / "broken.json"
    with pytest.raises(MetalFusedVerificationError):
        save_report(target, broken)
    assert not target.exists()


def test_load_names_bad_files_apart_from_rejected_reports(tmp_path):
    report = synthetic_report(tmp_path)
    pretty = tmp_path / "pretty.json"
    pretty.write_text(json.dumps(report, indent=1), encoding="utf-8")
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"kind": 1, "kind": 2}', encoding="utf-8")

    with pytest.raises(MetalFusedVerificationError, match="not canonical JSON"):
        load_report(pretty)
    with pytest.raises(MetalFusedError, match="duplicate") as caught:
        load_report(duplicate)
    assert not isinstance(caught.value, MetalFusedVerificationError)
