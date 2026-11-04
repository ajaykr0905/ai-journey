from pathlib import Path

import pytest
from test_serving_report import _report

from ai_journey.serving_report import ServingReportError, seal_serving_report
from ai_journey.serving_report_io import read_sealed_report, write_sealed_report


def test_sealed_report_is_written_atomically_and_verified(tmp_path: Path):
    path = tmp_path / "evidence" / "serving.json"
    sealed = seal_serving_report(_report())

    assert write_sealed_report(path, sealed) == path
    assert read_sealed_report(path) == sealed
    assert not list(path.parent.glob("*.tmp"))


def test_sealed_report_reader_rejects_oversized_input(tmp_path: Path):
    path = tmp_path / "serving.json"
    path.write_text("{}")

    with pytest.raises(ServingReportError, match="size limit"):
        read_sealed_report(path, max_bytes=1)


def test_sealed_report_reader_rejects_symlinks(tmp_path: Path):
    target = tmp_path / "target.json"
    target.write_text("{}")
    link = tmp_path / "serving.json"
    link.symlink_to(target)

    with pytest.raises(ServingReportError, match="regular file"):
        read_sealed_report(link)
