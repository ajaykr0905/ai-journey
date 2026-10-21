import json
from pathlib import Path
import pytest
from ai_journey.ddp_launch import launch_training


def test_resource_preflight_rejects_large_models_before_process_creation(
    tmp_path, monkeypatch
):
    root = Path(__file__).resolve().parents[1]
    payload = json.loads((root / "config/gpt2-training.json").read_text())
    payload["model"]["embedding_dim"] = 4096
    spec = tmp_path / "large.json"
    spec.write_text(json.dumps(payload))

    def unexpected(*args, **kwargs):
        raise AssertionError("process creation must not run")

    monkeypatch.setattr("subprocess.Popen", unexpected)
    with pytest.raises(ValueError, match="parameter"):
        launch_training(
            source=root / "data/day-19-demo-names.txt",
            spec=spec,
            checkpoint=tmp_path / "state.pt",
            output_dir=tmp_path / "ranks",
        )
