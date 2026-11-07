import json
from pathlib import Path

from ai_journey.gpt2_training import load_training_spec
from ai_journey.gpt2_training_cli import load_source_bytes
from ai_journey.project1_contract import contract_from_mapping
from ai_journey.project1_preflight import admit_training_run

ROOT = Path(__file__).parents[1]


def test_checked_in_project1_contract_admits_exact_public_fixture():
    source = load_source_bytes(ROOT / "data/day-19-demo-names.txt")
    spec = load_training_spec(ROOT / "config/gpt2-training.json")
    contract = contract_from_mapping(
        json.loads((ROOT / "config/day-64-project1-run.json").read_bytes())
    )

    admitted = admit_training_run(contract, spec, source)
    assert admitted.optimizer_steps == 12
    assert admitted.total_target_tokens == 768
    assert admitted.parameter_count < contract.max_parameters
