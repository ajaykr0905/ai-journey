import json
from hashlib import sha256

from ai_journey.sft_finetune_contract import SFTFineTuneConfig
from ai_journey.sft_finetune_dataset import FineTuneDataset, FineTuneExample
from ai_journey.sft_finetune_experiment import run_finetune_experiment


def dataset():
    examples = tuple(
        FineTuneExample(f"train-{index}", "train", "prompt answer", ((7, 13),))
        for index in range(3)
    ) + (FineTuneExample("held", "validation", "prompt answer", ((7, 13),)),)
    return FineTuneDataset(examples, sha256(b"fixture").hexdigest())


def test_experiment_composes_reproducible_training_and_comparison():
    config = SFTFineTuneConfig(
        block_size=8,
        batch_size=2,
        steps=2,
        evaluation_examples=1,
        generation_tokens=2,
        checkpoint_interval=1,
    )
    left = run_finetune_experiment(dataset(), config)
    right = run_finetune_experiment(dataset(), config)
    assert json.dumps(left, sort_keys=True) == json.dumps(right, sort_keys=True)
    assert left["initial_model"]["model_sha256"] != left["final_model"]["model_sha256"]
    assert left["training"]["optimizer_steps"] == 2
