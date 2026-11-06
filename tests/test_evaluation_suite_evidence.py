import json

import pytest

from ai_journey.evaluation_suite import EvaluationSuiteError, build_evaluation_suite
from ai_journey.evaluation_suite_evidence import read_suite, write_suite
from test_evaluation_suite import suite_results


def test_suite_round_trip_is_atomic_canonical_and_verified(tmp_path):
    suite = build_evaluation_suite(suite_results())
    path = tmp_path / "nested" / "suite.json"
    write_suite(path, suite)
    assert read_suite(path) == suite
    assert path.read_bytes().endswith(b"\n")


def test_suite_reader_rejects_symlinks_and_modified_files(tmp_path):
    suite = build_evaluation_suite(suite_results())
    target = tmp_path / "suite.json"
    write_suite(target, suite)
    link = tmp_path / "link.json"
    link.symlink_to(target)
    with pytest.raises(EvaluationSuiteError, match="constraints"):
        read_suite(link)
    payload = json.loads(target.read_text())
    payload["evaluations"][0]["value"] = 0.0
    target.write_text(json.dumps(payload))
    with pytest.raises(EvaluationSuiteError, match="digest"):
        read_suite(target)


def test_suite_writer_rejects_symlink_destinations(tmp_path):
    target = tmp_path / "target.json"
    target.write_text("preserve")
    link = tmp_path / "output.json"
    link.symlink_to(target)
    with pytest.raises(EvaluationSuiteError, match="symlink"):
        write_suite(link, build_evaluation_suite(suite_results()))
    assert target.read_text() == "preserve"
