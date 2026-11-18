import json
import os
from pathlib import Path

import pytest

from ai_journey.kernel_metal_config import config_from_mapping
from ai_journey.kernel_metal_geometry import KernelMetalError
from ai_journey.kernel_metal_io import load_report, save_report
from ai_journey.kernel_metal_verify import KernelMetalVerificationError
from ai_journey.memory_access_plan import canonical_json_bytes
from test_kernel_metal_config import make_config_mapping
from test_kernel_metal_recorded import RECORDED_REPORT_JSON
from test_kernel_metal_verify import SYNTHETIC, fresh

CONFIG = config_from_mapping(make_config_mapping())
RECORDED = json.loads(RECORDED_REPORT_JSON)


def test_reports_round_trip_through_atomic_canonical_files(tmp_path: Path):
    path = tmp_path / "nested" / "report.json"

    summary = save_report(path, fresh(SYNTHETIC), config=CONFIG)

    assert path.read_bytes() == canonical_json_bytes(SYNTHETIC) + b"\n"
    report, loaded = load_report(path, config=CONFIG)
    assert report == SYNTHETIC
    assert loaded == summary
    assert not list(path.parent.glob(f".{path.name}.*"))
    save_report(path, fresh())
    assert load_report(path)[0] == RECORDED
    assert path.read_text(encoding="ascii") == RECORDED_REPORT_JSON + "\n"


def test_loading_separates_unreadable_input_from_rejected_reports(tmp_path: Path):
    path = tmp_path / "report.json"

    for payload, message in (
        (b'{"kind": "x", "kind": "y"}', "duplicate JSON field: kind"),
        (b"[]", "root must be a JSON object"),
        (b"{" + b" " * (1024 * 1024), "exceeds"),
    ):
        path.write_bytes(payload)
        with pytest.raises(KernelMetalError, match=message) as error:
            load_report(path)
        assert not isinstance(error.value, KernelMetalVerificationError)
    for missing in (tmp_path / "missing.json", tmp_path):
        with pytest.raises(KernelMetalError, match="regular non-symlink"):
            load_report(missing)

    path.write_text(json.dumps(RECORDED, indent=1), encoding="utf-8")
    with pytest.raises(KernelMetalVerificationError, match="not canonical"):
        load_report(path)
    tampered = fresh()
    tampered["summary"]["timed_samples"] += 1
    path.write_bytes(canonical_json_bytes(tampered) + b"\n")
    with pytest.raises(KernelMetalVerificationError, match="digest"):
        load_report(path)
    path.write_bytes(canonical_json_bytes(RECORDED) + b"\n")
    with pytest.raises(KernelMetalVerificationError, match="supplied config"):
        load_report(path, config=CONFIG)


def test_saving_refuses_unverified_reports_and_keeps_the_old_file(tmp_path: Path):
    path = tmp_path / "report.json"
    path.write_bytes(b"previous\n")
    drifted = fresh(SYNTHETIC)
    drifted["measured"] = True

    for invalid in (drifted, {}, []):
        with pytest.raises(KernelMetalVerificationError):
            save_report(path, invalid)

    assert path.read_bytes() == b"previous\n"
    assert not list(tmp_path.glob(".report.json.*"))


def test_saving_never_replaces_a_protected_config_or_follows_a_link(tmp_path: Path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps(make_config_mapping(), indent=2), encoding="utf-8")
    before = config.read_bytes()
    hardlink = tmp_path / "hardlink.json"
    os.link(config, hardlink)
    symlink = tmp_path / "symlink.json"
    symlink.symlink_to(config)

    for output in (config, hardlink, symlink):
        with pytest.raises(KernelMetalError, match="protected input"):
            save_report(output, fresh(SYNTHETIC), protected=(config,))
    unrelated = tmp_path / "unrelated.json"
    unrelated.write_bytes(b"other\n")
    link = tmp_path / "link.json"
    link.symlink_to(unrelated)
    with pytest.raises(KernelMetalError, match="non-symlink"):
        save_report(link, fresh(SYNTHETIC), protected=(config,))

    assert config.read_bytes() == before
    assert unrelated.read_bytes() == b"other\n"
