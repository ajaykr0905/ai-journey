import json
import os
from pathlib import Path

import pytest

import ai_journey.simt_report_io as simt_report_io
from ai_journey.simt_device import SimtError
from ai_journey.simt_json import canonical_json_bytes
from ai_journey.simt_report_io import (
    MAX_SIMT_REPORT_BYTES,
    load_simt_report,
    save_simt_report,
)
from ai_journey.simt_verify import SimtVerificationError
from test_simt_verify import sealed


def test_reports_round_trip_as_canonical_bytes_without_temporary_files(
    tmp_path: Path,
):
    path = tmp_path / "nested" / "report.json"
    report = sealed()

    save_simt_report(path, report)

    assert path.read_bytes() == canonical_json_bytes(report) + b"\n"
    assert load_simt_report(path) == report
    assert sorted(item.name for item in path.parent.iterdir()) == ["report.json"]
    save_simt_report(path, report)
    assert load_simt_report(path) == report


def test_save_refuses_unverified_reports_and_keeps_the_existing_file(tmp_path: Path):
    path = tmp_path / "report.json"
    path.write_bytes(b"previous\n")
    report = sealed()
    report["summary"]["devices"] = 99

    with pytest.raises(SimtVerificationError, match="integrity"):
        save_simt_report(path, report)
    assert path.read_bytes() == b"previous\n"


def test_save_never_overwrites_a_protected_input_through_any_alias(tmp_path: Path):
    plan = tmp_path / "plan.json"
    plan.write_bytes(b"protected plan\n")
    symlink = tmp_path / "symlink.json"
    os.symlink(plan, symlink)
    hardlink = tmp_path / "hardlink.json"
    os.link(plan, hardlink)
    report = sealed()

    for target in (plan, tmp_path / "." / "plan.json", hardlink):
        with pytest.raises(SimtError, match="must not alias a protected input"):
            save_simt_report(target, report, protected=(plan,))
    with pytest.raises(SimtError, match="must not alias a protected input"):
        save_simt_report(symlink, report, protected=(plan,))
    assert plan.read_bytes() == b"protected plan\n"
    with pytest.raises(SimtError, match="regular non-symlink file"):
        save_simt_report(symlink, report)
    with pytest.raises(SimtError, match="regular non-symlink file"):
        save_simt_report(tmp_path, report)
    assert plan.read_bytes() == b"protected plan\n"
    with pytest.raises(TypeError, match="pathlib.Path"):
        save_simt_report(str(tmp_path / "report.json"), report)
    with pytest.raises(TypeError, match="protected inputs"):
        save_simt_report(tmp_path / "report.json", report, protected=("plan.json",))


def test_failed_replace_cleans_up_and_preserves_the_previous_report(
    tmp_path: Path, monkeypatch
):
    path = tmp_path / "report.json"
    path.write_bytes(b"previous\n")

    def failing_replace(source, target):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(simt_report_io.os, "replace", failing_replace)
    with pytest.raises(OSError, match="simulated replace failure"):
        save_simt_report(path, sealed())

    assert path.read_bytes() == b"previous\n"
    assert sorted(item.name for item in tmp_path.iterdir()) == ["report.json"]


def test_load_rejects_duplicate_non_canonical_oversize_and_tampered_reports(
    tmp_path: Path,
):
    path = tmp_path / "report.json"
    report = sealed()

    path.write_bytes(b'{"kind":"a","kind":"b"}')
    with pytest.raises(SimtError, match="duplicate JSON field: kind"):
        load_simt_report(path)

    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    with pytest.raises(SimtError, match="not canonical JSON"):
        load_simt_report(path)

    path.write_bytes(b"[]")
    with pytest.raises(SimtError, match="must be a JSON object"):
        load_simt_report(path)

    path.write_bytes(b" " * (MAX_SIMT_REPORT_BYTES + 1))
    with pytest.raises(SimtError, match="exceeds"):
        load_simt_report(path)

    tampered = dict(report, summary=dict(report["summary"], devices=3))
    path.write_bytes(canonical_json_bytes(tampered) + b"\n")
    with pytest.raises(SimtVerificationError, match="integrity"):
        load_simt_report(path)

    path.write_bytes(canonical_json_bytes(dict(report, measured=True)) + b"\n")
    with pytest.raises(SimtVerificationError, match="unmeasured analysis"):
        load_simt_report(path)

    save_simt_report(path, report)
    alias = tmp_path / "alias.json"
    os.symlink(path, alias)
    with pytest.raises(SimtError, match="regular non-symlink file"):
        load_simt_report(alias)
