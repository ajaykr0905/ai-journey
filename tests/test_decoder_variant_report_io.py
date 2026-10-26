from concurrent.futures import ThreadPoolExecutor
import json
import os

import pytest

from ai_journey.decoder_variant_report_io import write_new_report


def test_failed_fsync_does_not_publish_partial_output(tmp_path, monkeypatch):
    def fail(descriptor):
        raise OSError("injected fsync failure")

    monkeypatch.setattr(os, "fsync", fail)
    output = tmp_path / "report.json"
    with pytest.raises(OSError, match="injected"):
        write_new_report(output, {"complete": True})
    assert list(tmp_path.iterdir()) == []


def test_concurrent_writers_publish_one_complete_report(tmp_path):
    output = tmp_path / "report.json"

    def attempt(index):
        try:
            write_new_report(output, {"writer": index, "payload": "x" * 10000})
            return index
        except FileExistsError:
            return None

    with ThreadPoolExecutor(max_workers=4) as pool:
        winners = [index for index in pool.map(attempt, range(4)) if index is not None]
    assert len(winners) == 1
    assert json.loads(output.read_text()) == {
        "writer": winners[0],
        "payload": "x" * 10000,
    }
    assert list(tmp_path.iterdir()) == [output]


def test_invalid_numbers_and_existing_symlinks_are_not_published(tmp_path):
    output = tmp_path / "report.json"
    with pytest.raises(ValueError):
        write_new_report(output, {"loss": float("nan")})
    assert not output.exists()
    target = tmp_path / "user.json"
    target.write_text("preserved")
    output.symlink_to(target)
    with pytest.raises(FileExistsError):
        write_new_report(output, {"loss": 1.0})
    assert target.read_text() == "preserved"
    assert output.is_symlink()
