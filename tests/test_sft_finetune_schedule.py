from ai_journey.sft_finetune_dataset import FineTuneExample
from ai_journey.sft_finetune_schedule import build_schedule


def examples():
    return tuple(
        FineTuneExample(f"row-{index}", "train", f"prompt answer{index}", ((7, 14),))
        for index in range(4)
    )


def test_schedule_is_seeded_rectangular_and_reproducible():
    left = build_schedule(examples(), block_size=8, steps=3, batch_size=2, seed=68)
    right = build_schedule(examples(), block_size=8, steps=3, batch_size=2, seed=68)
    assert left == right
    assert len(left.batches) == 3
    assert all(len(batch) == 2 for batch in left.batches)
    assert (
        left.sha256
        != build_schedule(
            examples(), block_size=8, steps=3, batch_size=2, seed=69
        ).sha256
    )


def test_schedule_never_selects_validation_examples():
    validation = FineTuneExample("heldout", "validation", "prompt answer", ((7, 13),))
    try:
        build_schedule((validation,), block_size=8, steps=1, batch_size=1, seed=68)
    except ValueError as exc:
        assert "training examples" in str(exc)
    else:
        raise AssertionError("validation data entered the training schedule")
