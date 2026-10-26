"""Architecture-aware benchmark inputs must preserve verified model identity."""

from copy import deepcopy
from hashlib import sha256
import json

import pytest

from ai_journey.gpt2_training import run_training_experiment
from ai_journey.hellaswag import HellaSwagError, verify_evidence
from ai_journey.hellaswag_cli import main
from ai_journey.hellaswag_evidence import read_evidence
from test_hellaswag import example_payload, protocol_payload, server_payload
from test_modern_training import modern_spec


def evaluation_files(tmp_path):
    source, spec = modern_spec()
    source_path = tmp_path / "source.txt"
    source_path.write_bytes(source)
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec.to_mapping()))
    response = json.dumps(
        server_payload(example_payload(label="0")), separators=(",", ":")
    ).encode()
    response_path = tmp_path / "response.json"
    response_path.write_bytes(response)
    protocol = protocol_payload()
    protocol.update(count=1, world_size=1, bootstrap_samples=100)
    protocol["source"].update(
        sample_sha256=sha256(response).hexdigest(), sample_bytes=len(response)
    )
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol))
    return source, spec, source_path, spec_path, response_path, protocol_path


@pytest.mark.parametrize("trained", [False, True])
def test_cli_evaluates_modern_initial_or_local_checkpoint_with_exact_identity(
    tmp_path, capsys, trained
):
    source, spec, source_path, spec_path, response, protocol = evaluation_files(
        tmp_path
    )
    arguments = [
        "evaluate",
        "--source",
        str(source_path),
        "--spec",
        str(spec_path),
        "--dataset-response",
        str(response),
        "--protocol",
        str(protocol),
        "--output",
        str(tmp_path / "result.json"),
    ]
    expected_hash = None
    if trained:
        checkpoint = tmp_path / "checkpoint.pt"
        run = run_training_experiment(source, spec, checkpoint_path=checkpoint)
        expected_hash = run.final_model_fingerprint
        arguments.extend(
            ["--checkpoint", str(checkpoint), "--checkpoint-format", "local"]
        )
    assert main(arguments) == 0
    receipt = json.loads(capsys.readouterr().out)
    evidence = read_evidence(tmp_path / "result.json")
    assert evidence["run"]["model_config"]["kind"] == "decoder_variant"
    assert evidence["run"]["parameter_count"] > 0
    assert evidence["run"]["model_dtype"] == (
        "torch.float32" if trained else "torch.float64"
    )
    if trained:
        assert receipt["model_sha256"] == expected_hash


def test_resealed_wrong_modern_parameter_accounting_is_rejected(tmp_path, capsys):
    _, _, source, spec, response, protocol = evaluation_files(tmp_path)
    main(
        [
            "evaluate",
            "--source",
            str(source),
            "--spec",
            str(spec),
            "--dataset-response",
            str(response),
            "--protocol",
            str(protocol),
            "--output",
            str(tmp_path / "result.json"),
        ]
    )
    capsys.readouterr()
    evidence = deepcopy(read_evidence(tmp_path / "result.json"))
    evidence["run"]["parameter_count"] += 1
    from ai_journey.hellaswag import _payload_digest

    evidence["evidence_sha256"] = _payload_digest(
        {key: value for key, value in evidence.items() if key != "evidence_sha256"}
    )
    with pytest.raises(HellaSwagError, match="model accounting"):
        verify_evidence(evidence)
