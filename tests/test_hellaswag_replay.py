"""Exact replay binds actual checkpoint, dataset and CPU runtime evidence."""

from copy import deepcopy
import json

import pytest
import torch

from ai_journey.gpt2_config import GPT2Config
from ai_journey.gpt2_model import GPT2LanguageModel
from ai_journey.gpt2_training import run_training_experiment
from ai_journey.hellaswag import (
    HellaSwagError,
    _payload_digest,
    summarize_scores,
    verify_evidence,
)
from ai_journey.hellaswag_cli import main
from ai_journey.hellaswag_evidence import read_evidence
from ai_journey.hellaswag_replay import verify_evaluation_replay
from test_hellaswag_evidence import evidence_fixture
from test_hellaswag_modern import evaluation_files


def zero_model():
    model = GPT2LanguageModel(GPT2Config(256, 8, 8, 2, 1)).double()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    return model


def reseal(payload):
    payload["evidence_sha256"] = _payload_digest(
        {key: value for key, value in payload.items() if key != "evidence_sha256"}
    )


@pytest.mark.parametrize("checkpoint", [False, True])
def test_cli_replays_modern_initial_and_verified_local_checkpoint(
    tmp_path, capsys, checkpoint
):
    source, spec, source_path, spec_path, response, protocol = evaluation_files(
        tmp_path
    )
    shared = [
        "--source",
        str(source_path),
        "--spec",
        str(spec_path),
        "--dataset-response",
        str(response),
    ]
    if checkpoint:
        path = tmp_path / "state.pt"
        run_training_experiment(source, spec, checkpoint_path=path)
        shared.extend(["--checkpoint", str(path), "--checkpoint-format", "local"])
    report = tmp_path / "result.json"
    main(["evaluate", *shared, "--protocol", str(protocol), "--output", str(report)])
    first = json.loads(capsys.readouterr().out)
    rng = torch.get_rng_state().clone()
    assert main(["replay", *shared, "--report", str(report)]) == 0
    assert json.loads(capsys.readouterr().out) == first
    assert torch.equal(torch.get_rng_state(), rng)
    evidence = read_evidence(report)
    assert evidence["runtime"]["model_dtype"] == evidence["run"]["model_dtype"]


def test_replay_rejects_different_model_and_resealed_runtime_metadata():
    evidence = evidence_fixture()
    model = zero_model()
    with torch.no_grad():
        model.wte.weight[0, 0] = 1
    with pytest.raises(HellaSwagError, match="model identity"):
        verify_evaluation_replay(model, b"unused", evidence)
    model = zero_model()
    evidence["runtime"]["torch_config_sha256"] = "0" * 64
    reseal(evidence)
    with pytest.raises(HellaSwagError, match="runtime"):
        verify_evaluation_replay(model, b"unused", evidence)


def test_legacy_evidence_remains_verifiable_but_cannot_claim_exact_replay():
    evidence = deepcopy(evidence_fixture())
    evidence["schema_version"] = evidence["run"]["schema_version"] = 1
    for key in ("parameter_count", "model_dtype", "torch_threads"):
        del evidence["run"][key]
    for key in (
        "model_dtype",
        "torch_threads",
        "cpu_capability",
        "torch_config_sha256",
    ):
        del evidence["runtime"][key]
    reseal(evidence)
    assert verify_evidence(evidence) == evidence
    with pytest.raises(HellaSwagError, match="legacy"):
        verify_evaluation_replay(zero_model(), b"unused", evidence)


def test_self_consistent_resealed_scores_are_detected_by_reexecution():
    from test_hellaswag import example_payload, server_payload

    evidence = evidence_fixture()
    row = evidence["run"]["evaluation"]["records"][0]
    for score in row["scores"]:
        score["mean_nll"] += 1
        score["loss_sum"] += score["tokens"]
    protocol = evidence["protocol"]
    summary = summarize_scores(
        [row],
        bootstrap_samples=protocol["bootstrap_samples"],
        bootstrap_seed=protocol["bootstrap_seed"],
    )
    evidence["run"]["evaluation"] = {**summary, "records": [row]}
    reseal(evidence)
    assert verify_evidence(evidence) == evidence
    response = json.dumps(
        server_payload(example_payload(label="0")), separators=(",", ":")
    ).encode()
    with pytest.raises(HellaSwagError, match="candidate scores"):
        verify_evaluation_replay(zero_model(), response, evidence)
