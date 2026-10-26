from dataclasses import replace
import json
import math
from pathlib import Path

import pytest
import torch

from ai_journey.ddp_acceptance import certify_recovery
from ai_journey.ddp_cli import main
from ai_journey.ddp_evaluation import Document, evaluate_documents, score_choices
from ai_journey.ddp_evidence import read_evidence
from ai_journey.ddp_launch import launch_training
from ai_journey.decoder_variant_config import DecoderVariantConfig
from ai_journey.gpt2_training import load_training_spec
from ai_journey.training_models import build_training_model, model_config_mapping


ROOT = Path(__file__).resolve().parents[1]


def _modern_spec(tmp_path, norm="rmsnorm", mlp="swiglu", position="rope", **kwargs):
    original = load_training_spec(ROOT / "config/ddp-lab.json")
    spec = replace(
        original,
        model=DecoderVariantConfig(original.model, norm, mlp, position, **kwargs),
    )
    path = tmp_path / "modern-spec.json"
    path.write_text(json.dumps(spec.to_mapping()))
    return ROOT / "data/day-19-demo-names.txt", path, spec


@pytest.mark.parametrize(
    "choices",
    [
        ("rmsnorm", "swiglu", "rope"),
        ("layernorm", "gelu", "learned"),
        ("rmsnorm", "gelu", "learned"),
    ],
)
def test_modern_two_rank_checkpoint_recovery_is_exact(tmp_path, choices):
    source, spec_path, spec = _modern_spec(tmp_path, *choices)
    result = certify_recovery(
        source=source, spec=spec_path, directory=tmp_path / "recovery"
    )
    assert result["world_size"] == 2
    assert all(result["exact_state"].values())
    assert not set(result["initial_worker_pids"]) & set(result["resumed_worker_pids"])
    payload = torch.load(tmp_path / "recovery/resumed.pt", weights_only=True)
    assert payload["model_config"] == model_config_mapping(spec.model)


def test_modern_ddp_cli_emits_verifiable_architecture_bound_evidence(tmp_path, capsys):
    source, spec_path, spec = _modern_spec(tmp_path)
    report = tmp_path / "evidence.json"
    checkpoint = tmp_path / "state.pt"
    assert (
        main(
            [
                "train",
                "--source",
                str(source),
                "--spec",
                str(spec_path),
                "--checkpoint",
                str(checkpoint),
                "--workdir",
                str(tmp_path / "ranks"),
                "--output",
                str(report),
                "--steps",
                "2",
                "--checkpoint-interval",
                "1",
            ]
        )
        == 0
    )
    evidence = read_evidence(report)
    assert json.loads(capsys.readouterr().out) == evidence
    assert evidence["spec"] == spec.to_mapping()
    assert evidence["completed_step"] == 2
    previous = checkpoint.read_bytes()
    changed = replace(spec, model=replace(spec.model, rope_base=1000))
    spec_path.write_text(json.dumps(changed.to_mapping()))
    with pytest.raises(RuntimeError, match="worker failed"):
        launch_training(
            source=source,
            spec=spec_path,
            checkpoint=checkpoint,
            output_dir=tmp_path / "mismatch",
            steps=2,
            resume=True,
        )
    assert checkpoint.read_bytes() == previous


def test_modern_swiglu_budget_is_checked_before_worker_creation(tmp_path, monkeypatch):
    source, spec_path, _ = _modern_spec(tmp_path, swiglu_hidden_dim=100_000)

    def unexpected(*args, **kwargs):
        raise AssertionError("must reject oversized architecture before launch")

    monkeypatch.setattr("subprocess.Popen", unexpected)
    with pytest.raises(ValueError, match="parameter"):
        launch_training(
            source=source,
            spec=spec_path,
            checkpoint=tmp_path / "state.pt",
            output_dir=tmp_path / "ranks",
        )


def test_modern_document_and_choice_scoring_preserve_uniform_oracle(tmp_path):
    _, _, spec = _modern_spec(tmp_path)
    model = build_training_model(spec.model).double().train()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    before = torch.get_rng_state().clone()
    result = evaluate_documents(model, [Document("public", b"abcdefg")])
    assert result["tokens"] == 6
    assert result["mean_nll"] == pytest.approx(math.log(256))
    choices = score_choices(model, b"long public prompt", [b"a", b"bbbb"])
    assert [score["mean_nll"] for score in choices["scores"]] == pytest.approx(
        [math.log(256)] * 2
    )
    assert model.training
    assert torch.equal(torch.get_rng_state(), before)
