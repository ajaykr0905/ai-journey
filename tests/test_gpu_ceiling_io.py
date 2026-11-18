import json

import pytest

from ai_journey.gpu_ceiling_io import load_report, save_report
from ai_journey.gpu_ceiling_source import GpuCeilingError
from ai_journey.gpu_ceiling_verify import GpuCeilingVerificationError
from ai_journey.memory_access_plan import canonical_json_bytes
from ai_journey.memory_access_report_io import MAX_REPORT_BYTES
from test_gpu_ceiling_report import CONFIG
from test_gpu_ceiling_verify import SYNTHETIC, fresh, reseal


def test_a_saved_report_is_canonical_and_loads_with_the_same_summary(tmp_path):
    path = tmp_path / "nested" / "report.json"

    saved = save_report(path, fresh(), config=CONFIG)
    report, loaded = load_report(path, config=CONFIG)

    assert path.read_bytes() == canonical_json_bytes(SYNTHETIC) + b"\n"
    assert report == SYNTHETIC
    assert saved == loaded
    assert loaded["report_sha256"] == SYNTHETIC["report_sha256"]
    assert [item.name for item in path.parent.iterdir()] == ["report.json"]


def test_reformatted_or_tampered_files_are_rejected_reports(tmp_path):
    path = tmp_path / "report.json"
    path.write_text(json.dumps(SYNTHETIC, indent=1), encoding="ascii")
    with pytest.raises(GpuCeilingVerificationError, match="not canonical"):
        load_report(path)

    changed = fresh()
    changed["summary"]["fma_points"] += 1
    path.write_bytes(canonical_json_bytes(reseal(changed)) + b"\n")
    with pytest.raises(GpuCeilingVerificationError, match="summary"):
        load_report(path)


def test_unreadable_inputs_are_bad_input_not_rejected_reports(tmp_path):
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="ascii")
    large = tmp_path / "large.json"
    large.write_bytes(b" " * (MAX_REPORT_BYTES + 1))
    link = tmp_path / "link.json"
    link.symlink_to(broken)

    for path, message in (
        (broken, "strict UTF-8 JSON"),
        (large, "exceeds"),
        (link, "regular non-symlink"),
        (tmp_path / "missing.json", "regular non-symlink"),
    ):
        with pytest.raises(GpuCeilingError, match=message) as raised:
            load_report(path)
        assert not isinstance(raised.value, GpuCeilingVerificationError)


def test_saving_verifies_first_and_never_overwrites_a_protected_input(tmp_path):
    config = tmp_path / "config.json"
    config.write_text("{}", encoding="ascii")
    unsealed = fresh()
    unsealed["summary"]["bank_points"] += 1

    with pytest.raises(GpuCeilingVerificationError, match="digest"):
        save_report(tmp_path / "report.json", unsealed)
    with pytest.raises(GpuCeilingError, match="alias a protected input"):
        save_report(config, fresh(), protected=(config,))
    with pytest.raises(GpuCeilingError, match="regular non-symlink"):
        save_report(tmp_path, fresh())
    assert config.read_text(encoding="ascii") == "{}"
    assert not (tmp_path / "report.json").exists()
