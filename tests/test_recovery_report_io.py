import os
from unittest.mock import patch

import pytest

from ai_journey.recovery_report_io import load_recovery_report, write_recovery_report


def report():
    return {"schema_version": 1, "report_sha256": "a" * 64}


def test_report_round_trip_is_canonical_and_atomic(tmp_path):
    path = tmp_path / "evidence" / "report.json"
    write_recovery_report(path, report())

    assert load_recovery_report(path) == report()
    assert path.read_bytes() == (
        b'{"report_sha256":"' + b"a" * 64 + b'","schema_version":1}'
    )
    assert list(path.parent.iterdir()) == [path]


def test_failed_report_replace_preserves_prior_bytes(tmp_path):
    path = tmp_path / "report.json"
    path.write_bytes(b"prior")
    with (
        patch.object(os, "replace", side_effect=OSError("injected replace failure")),
        pytest.raises(OSError, match="injected"),
    ):
        write_recovery_report(path, report())
    assert path.read_bytes() == b"prior"
    assert list(tmp_path.iterdir()) == [path]


def test_report_loader_rejects_alias_invalid_json_and_size(tmp_path):
    real = tmp_path / "report.json"
    real.write_text("not-json")
    alias = tmp_path / "alias.json"
    alias.symlink_to(real)
    with pytest.raises(ValueError, match="regular non-symlink"):
        load_recovery_report(alias)
    with pytest.raises(ValueError, match="valid JSON"):
        load_recovery_report(real)
    real.write_bytes(b"12345")
    with pytest.raises(ValueError, match="exceeds"):
        load_recovery_report(real, max_bytes=4)


def test_report_writer_rejects_nonfinite_or_empty_payload(tmp_path):
    with pytest.raises(TypeError, match="non-empty"):
        write_recovery_report(tmp_path / "report.json", {})
    with pytest.raises(ValueError):
        write_recovery_report(tmp_path / "report.json", {"value": float("nan")})
