from hashlib import sha256
import json
from pathlib import Path
import tomllib

import pytest

from ai_journey.hellaswag_cli import main
from ai_journey.hellaswag_evidence import read_evidence
from test_hellaswag import example_payload, protocol_payload, server_payload


def test_cli_evaluates_fresh_model_and_verifies_saved_evidence(tmp_path, capsys):
    root = Path(__file__).resolve().parents[1]
    response = json.dumps(
        server_payload(example_payload(label="0")), separators=(",", ":")
    ).encode()
    dataset = tmp_path / "dataset.json"
    dataset.write_bytes(response)
    payload = protocol_payload()
    payload.update(count=1, world_size=1, bootstrap_samples=100)
    payload["source"]["sample_sha256"] = sha256(response).hexdigest()
    payload["source"]["sample_bytes"] = len(response)
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps(payload))
    report = tmp_path / "evidence.json"
    assert (
        main(
            [
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
                str(report),
            ]
        )
        == 0
    )
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["examples"] == 1
    assert receipt["evidence_sha256"] == read_evidence(report)["evidence_sha256"]
    assert main(["verify", "--report", str(report)]) == 0
    assert json.loads(capsys.readouterr().out) == receipt


def test_cli_rejects_missing_or_invalid_reports(tmp_path):
    with pytest.raises(SystemExit) as result:
        main(["verify", "--report", str(tmp_path / "missing.json")])
    assert result.value.code == 2


def test_package_installs_hellaswag_command():
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text())
    assert (
        project["project"]["scripts"]["ai-journey-hellaswag"]
        == "ai_journey.hellaswag_cli:main"
    )
