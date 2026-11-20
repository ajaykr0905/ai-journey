import json

import pytest

from ai_journey.gpu_scan_io import load_report, save_report
from ai_journey.gpu_scan_source import GpuScanError
from ai_journey.gpu_scan_verify import GpuScanVerificationError
from ai_journey.memory_access_plan import canonical_json_bytes
from test_gpu_scan_report import sealed


def test_a_saved_report_loads_back_byte_for_byte(tmp_path):
    report = sealed()
    path = tmp_path / "reports" / "scan.json"

    summary = save_report(path, report)

    assert path.read_bytes() == canonical_json_bytes(report) + b"\n"
    loaded, again = load_report(path)
    assert loaded == report and again == summary
    assert summary["report_sha256"] == report["report_sha256"]
    assert [entry.name for entry in tmp_path.joinpath("reports").iterdir()] == [
        "scan.json"
    ]


def test_the_reader_refuses_reformatted_tampered_and_unusual_files(tmp_path):
    report = sealed()
    pretty = tmp_path / "pretty.json"
    pretty.write_text(json.dumps(report, indent=2), encoding="utf-8")
    tampered = tmp_path / "tampered.json"
    tampered.write_bytes(canonical_json_bytes({**report, "measured": True}) + b"\n")
    link = tmp_path / "link.json"
    link.symlink_to(pretty)

    with pytest.raises(GpuScanVerificationError, match="not canonical"):
        load_report(pretty)
    with pytest.raises(GpuScanVerificationError, match="digest"):
        load_report(tampered)
    for path in (link, tmp_path, tmp_path / "missing.json"):
        with pytest.raises(GpuScanError):
            load_report(path)


def test_the_writer_verifies_first_and_keeps_protected_inputs(tmp_path):
    report = sealed()
    config = tmp_path / "config.json"
    config.write_text("{}", encoding="utf-8")
    target = tmp_path / "out.json"

    with pytest.raises(GpuScanVerificationError):
        save_report(target, {**report, "measured": True})
    assert not target.exists()
    with pytest.raises(GpuScanError, match="alias a protected input"):
        save_report(config, report, protected=(config,))
    assert config.read_text(encoding="utf-8") == "{}"
