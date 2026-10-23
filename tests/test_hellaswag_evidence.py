import json

import pytest
import torch

from ai_journey.gpt2_config import GPT2Config
from ai_journey.gpt2_model import GPT2LanguageModel
from ai_journey.hellaswag import HellaSwagError, HellaSwagProtocol
from ai_journey.hellaswag import run_evaluation, seal_evidence
from ai_journey.hellaswag_evidence import read_evidence, write_evidence
from test_hellaswag import example_payload, protocol_payload, server_payload


def evidence_fixture():
    response = json.dumps(
        server_payload(example_payload(label="0")), separators=(",", ":")
    ).encode()
    payload = protocol_payload()
    payload.update(count=1, world_size=1, bootstrap_samples=100)
    from hashlib import sha256

    payload["source"]["sample_sha256"] = sha256(response).hexdigest()
    payload["source"]["sample_bytes"] = len(response)
    protocol = HellaSwagProtocol.from_mapping(payload)
    model = GPT2LanguageModel(GPT2Config(256, 8, 8, 2, 1)).double()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    return seal_evidence(run_evaluation(model, response, protocol), protocol)


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
    with pytest.raises(HellaSwagError, match="constraints"):
        read_evidence(link)
    payload = json.loads(target.read_text())
    payload["run"]["evaluation"]["accuracy"] = 0.5
    target.write_text(json.dumps(payload))
    with pytest.raises(HellaSwagError, match="digest"):
        read_evidence(target)


def test_evidence_writer_rejects_symlink_destinations(tmp_path):
    target = tmp_path / "target.json"
    target.write_text("preserve")
    link = tmp_path / "output.json"
    link.symlink_to(target)
    with pytest.raises(HellaSwagError, match="symlink"):
        write_evidence(link, evidence_fixture())
    assert target.read_text() == "preserve"
