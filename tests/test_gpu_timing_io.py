import json

import pytest

from ai_journey.gpu_timing_clocks import GpuTimingError
from ai_journey.gpu_timing_config import config_from_mapping
from ai_journey.gpu_timing_io import load_report, save_report
from ai_journey.gpu_timing_verify import GpuTimingVerificationError
from ai_journey.memory_access_plan import canonical_json_bytes
from ai_journey.memory_access_report import DIGEST_FIELD
from test_gpu_timing_config import small
from test_gpu_timing_verify import report, reseal


def test_a_saved_report_is_canonical_and_loads_with_its_summary(tmp_path):
    config = config_from_mapping(small())
    sealed = report()
    path = tmp_path / "nested" / "report.json"

    saved = save_report(path, sealed, config=config)
    loaded, summary = load_report(path, config=config)

    assert path.read_bytes() == canonical_json_bytes(sealed) + b"\n"
    assert loaded == sealed
    assert saved == summary
    assert summary["report_sha256"] == sealed[DIGEST_FIELD]
    assert not [item for item in path.parent.iterdir() if item.name != "report.json"]


def test_loading_names_tampering_before_formatting(tmp_path):
    sealed = report()
    path = tmp_path / "report.json"

    path.write_text(json.dumps(sealed, indent=1), encoding="utf-8")
    with pytest.raises(GpuTimingVerificationError, match="not canonical"):
        load_report(path)
    sealed["run"]["elapsed_ns"] += 1
    path.write_text(json.dumps(sealed, indent=1), encoding="utf-8")
    with pytest.raises(GpuTimingVerificationError, match="digest"):
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
        with pytest.raises(GpuTimingError, match=message) as raised:
            load_report(path)
        assert not isinstance(raised.value, GpuTimingVerificationError)


def test_a_report_that_fails_verification_is_never_written(tmp_path):
    sealed = report()
    sealed["measured"] = True
    path = tmp_path / "report.json"

    with pytest.raises(GpuTimingVerificationError):
        save_report(path, reseal(sealed))
    assert not path.exists()


def test_a_report_cannot_overwrite_its_config(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(small()), encoding="utf-8")

    with pytest.raises(GpuTimingError, match="protected input"):
        save_report(config_path, report(), protected=(config_path,))
    assert json.loads(config_path.read_text(encoding="utf-8")) == small()
