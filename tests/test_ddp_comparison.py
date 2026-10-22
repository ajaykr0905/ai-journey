import json
from pathlib import Path
from ai_journey.ddp_comparison import compare_decompositions, verify_comparison


def test_three_controlled_batch_decompositions_agree_on_trained_parameters(tmp_path):
    root = Path(__file__).resolve().parents[1]
    payload = json.loads((root / "config/gpt2-training.json").read_text())
    payload["training"].update(optimizer_steps=4, sequence_length=4)
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps(payload))
    result = compare_decompositions(
        source=root / "data/day-19-demo-names.txt",
        spec_path=spec,
        directory=tmp_path / "comparison",
        steps=4,
    )
    assert verify_comparison(result) == result
    assert max(result["max_parameter_errors"]) < 1e-10
    assert [arm["world_size"] for arm in result["arms"]] == [1, 2, 2]
