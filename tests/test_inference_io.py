import os

import pytest

from ai_journey.inference_io import read_inference_report, write_inference_report
from ai_journey.inference_protocol import InferenceProtocol
from ai_journey.inference_report import build_inference_report
from ai_journey.inference_seal import seal_inference_report
from ai_journey.inference_speculative import SpeculativePlan


def report():
    gib = 1024**3
    protocol = InferenceProtocol(
        "io-inference",
        7_000_000_000,
        32,
        32,
        8,
        128,
        32_768,
        (32_768,) * 16,
        2,
        80 * gib,
        14 * gib,
        6 * gib,
        16,
        SpeculativePlan(4, 4, 5),
    )
    return seal_inference_report(build_inference_report(protocol))


def test_verified_report_round_trips_through_atomic_json(tmp_path):
    path = tmp_path / "nested" / "report.json"

    write_inference_report(path, report())

    assert read_inference_report(path) == report()
    assert path.stat().st_mode & 0o777 == 0o600
    assert not list(path.parent.glob(f".{path.name}.*"))


def test_reader_rejects_symlink_aliases(tmp_path):
    target = tmp_path / "target.json"
    write_inference_report(target, report())
    alias = tmp_path / "alias.json"
    os.symlink(target, alias)

    with pytest.raises(ValueError, match="regular file"):
        read_inference_report(alias)


def test_writer_rejects_unverified_report(tmp_path):
    with pytest.raises(ValueError, match="contract"):
        write_inference_report(tmp_path / "report.json", {"schema_version": 1})
