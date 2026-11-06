from hashlib import sha256
import json
from pathlib import Path

import pytest

from ai_journey.lambada_cli import main
from ai_journey.lambada_evidence import read_evidence
from test_lambada import protocol_payload, server_payload


def test_cli_evaluates_and_verifies_saved_evidence(tmp_path, capsys):
    root = Path(__file__).resolve().parents[1]
    response = json.dumps(
        server_payload(), sort_keys=True, separators=(",", ":")
    ).encode()
    dataset = tmp_path / "dataset.json"
    dataset.write_bytes(response)
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps(protocol_payload(response)))
    output = tmp_path / "evidence.json"
    arguments = [
        "evaluate",
        "--dataset-response",
        str(dataset),
        "--protocol",
        str(protocol),
        "--source",
        str(root / "data/day-19-demo-names.txt"),
        "--spec",
        str(root / "config/gpt2-training.json"),
        "--output",
        str(output),
    ]
    assert main(arguments) == 0
    receipt = json.loads(capsys.readouterr().out)
    evidence = read_evidence(output)
    assert receipt["evidence_sha256"] == evidence["evidence_sha256"]
    assert receipt["examples"] == 2
    assert receipt["contamination_passed"] is True
    assert evidence["run"]["sample_sha256"] == sha256(response).hexdigest()
    assert main(["verify", "--report", str(output)]) == 0
    assert json.loads(capsys.readouterr().out) == receipt


def test_cli_rejects_missing_reports(tmp_path):
    with pytest.raises(SystemExit) as result:
        main(["verify", "--report", str(tmp_path / "missing.json")])
    assert result.value.code == 2
