import json
import os

import pytest

from ai_journey.memory_access_plan import canonical_json_bytes
from ai_journey.profile_trace_events import ProfileTraceError
from ai_journey.profile_trace_io import (
    REPORT_FILE,
    load_report,
    output_paths,
    save_capture,
)
from ai_journey.profile_trace_verify import ProfileTraceVerificationError
from test_profile_trace_report import synthetic_report


def test_a_saved_capture_loads_and_verifies(tmp_path):
    config, report, captured = synthetic_report()
    directory = tmp_path / "capture"
    summary = save_capture(directory, report, captured, config=config)

    assert sorted(path.name for path in directory.iterdir()) == sorted(
        [REPORT_FILE] + [item.file_name for item in captured]
    )
    for item in captured:
        assert (directory / item.file_name).read_bytes() == item.trace_bytes
    payload = (directory / REPORT_FILE).read_bytes()
    assert payload == canonical_json_bytes(report) + b"\n"
    loaded, again = load_report(directory / REPORT_FILE, config=config)
    assert loaded == report
    assert again == summary


def test_a_report_that_fails_verification_writes_nothing(tmp_path):
    config, report, captured = synthetic_report()
    directory = tmp_path / "capture"
    with pytest.raises(ProfileTraceVerificationError, match="digest"):
        save_capture(directory, {**report, "label": "x"}, captured)
    assert not directory.exists()


def test_outputs_may_not_alias_protected_inputs_or_links(tmp_path):
    config, report, captured = synthetic_report()
    config_path = tmp_path / REPORT_FILE
    config_path.write_text("{}")
    with pytest.raises(ProfileTraceError, match="protected input"):
        save_capture(tmp_path, report, captured, protected=(config_path,))
    target = tmp_path / "elsewhere"
    target.mkdir()
    link = tmp_path / "link"
    os.symlink(target, link)
    with pytest.raises(ProfileTraceError, match="real directory"):
        save_capture(link, report, captured)
    with pytest.raises(TypeError):
        output_paths(str(tmp_path), captured)


def test_a_reformatted_or_unreadable_report_is_refused(tmp_path):
    config, report, captured = synthetic_report()
    save_capture(tmp_path, report, captured)
    path = tmp_path / REPORT_FILE
    path.write_text(json.dumps(report, indent=1))
    with pytest.raises(ProfileTraceVerificationError, match="canonical"):
        load_report(path)
    path.write_text("{")
    with pytest.raises(ProfileTraceError, match="strict UTF-8 JSON"):
        load_report(path)
    with pytest.raises(ProfileTraceError, match="regular"):
        load_report(tmp_path / "missing.json")
