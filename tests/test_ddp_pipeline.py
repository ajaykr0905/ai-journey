import json
from pathlib import Path
import pytest
from ai_journey.ddp_pipeline import run_pipeline


@pytest.mark.parametrize("modern", [False, True])
def test_pipeline_trains_only_on_training_documents_and_serves_verified_state(
    tmp_path, modern
):
    root = Path(__file__).resolve().parents[1]
    payload = json.loads((root / "config/gpt2-training.json").read_text())
    payload["training"].update(
        optimizer_steps=2, sequence_length=4, micro_batch_size=1, accumulation_steps=2
    )
    if modern:
        payload["schema_version"] = 2
        payload["model"] = {
            "kind": "decoder_variant",
            "config": {
                "gpt2": payload["model"],
                "norm_kind": "rmsnorm",
                "mlp_kind": "swiglu",
                "position_kind": "rope",
                "rope_base": 10000.0,
                "swiglu_hidden_dim": None,
            },
        }
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps(payload))
    result = run_pipeline(
        source=root / "data/day-19-demo-names.txt",
        spec_path=spec,
        directory=tmp_path / "pipeline",
    )
    assert not set(result["training_documents"]) & set(result["validation_documents"])
    assert len(result["validation_documents"]) == 4
    assert result["validation_after"]["tokens"] == result["validation_before"]["tokens"]
    assert result["generation"]["model_sha256"] == result["distributed"]["model_sha256"]
    assert result["serving_benchmark"]["output_parity"]
    from ai_journey.ddp_pipeline import load_serving_model
    from ai_journey.gpt2_training import load_training_spec

    directory = tmp_path / "pipeline"
    loaded = load_serving_model(
        directory / "trained.pt",
        load_training_spec(directory / "training-spec.json"),
        (directory / "train.txt").read_bytes(),
    )
    assert not loaded.training
    assert not any(parameter.requires_grad for parameter in loaded.parameters())
