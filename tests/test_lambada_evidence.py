import copy
import json

import pytest

from ai_journey.lambada import LambadaError, _payload_digest
from ai_journey.lambada_evidence import read_evidence, write_evidence
from test_lambada import evidence_fixture


def test_evidence_round_trip_is_canonical_and_validated(tmp_path):
    evidence = evidence_fixture()
    path = tmp_path / "nested" / "evidence.json"
    write_evidence(path, evidence)
    assert read_evidence(path) == evidence
    assert path.read_bytes().endswith(b"\n")


def test_evidence_reader_rejects_symlinks_and_modified_files(tmp_path):
    evidence = evidence_fixture()
    target = tmp_path / "evidence.json"
    write_evidence(target, evidence)
    link = tmp_path / "link.json"
    link.symlink_to(target)
    with pytest.raises(LambadaError, match="constraints"):
        read_evidence(link)
    payload = json.loads(target.read_text())
    payload["run"]["evaluation"]["exact_match_accuracy"] = 1.0
    target.write_text(json.dumps(payload))
    with pytest.raises(LambadaError, match="digest"):
        read_evidence(target)


def test_evidence_reader_rejects_semantically_resealed_metrics(tmp_path):
    evidence = copy.deepcopy(evidence_fixture())
    evidence["run"]["evaluation"]["mean_nll"] = 0.0
    evidence["evidence_sha256"] = _payload_digest(
        {key: value for key, value in evidence.items() if key != "evidence_sha256"}
    )
    path = tmp_path / "evidence.json"
    path.write_text(json.dumps(evidence))
    with pytest.raises(LambadaError, match="metrics"):
        read_evidence(path)


def test_evidence_writer_rejects_symlink_destinations(tmp_path):
    target = tmp_path / "target.json"
    target.write_text("preserve")
    link = tmp_path / "output.json"
    link.symlink_to(target)
    with pytest.raises(LambadaError, match="symlink"):
        write_evidence(link, evidence_fixture())
    assert target.read_text() == "preserve"
