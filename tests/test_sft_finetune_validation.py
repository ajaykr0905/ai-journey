from ai_journey.sft_finetune_dataset import FineTuneExample
from ai_journey.sft_finetune_validation import validation_evidence, validation_windows


def test_validation_windows_cover_a_bounded_ordered_prefix():
    examples = tuple(
        FineTuneExample(f"held-{index}", "validation", "prompt answer", ((7, 13),))
        for index in range(3)
    )
    windows = validation_windows(examples, block_size=8, example_limit=2)
    evidence = validation_evidence(windows)
    assert {row.example_id for row in windows} == {"held-0", "held-1"}
    assert evidence["examples"] == 2
    assert evidence["target_tokens"] == sum(sum(row.target_mask) for row in windows)
