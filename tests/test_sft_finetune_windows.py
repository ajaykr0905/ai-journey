from ai_journey.sft_finetune_dataset import FineTuneExample
from ai_journey.sft_finetune_windows import example_windows


def test_example_windows_map_character_spans_to_utf8_target_bytes():
    example = FineTuneExample("unicode", "train", "aéz", ((1, 2),))
    windows = example_windows(example, block_size=3)
    assert len(windows) == 1
    assert windows[0].inputs == (97, 195, 169)
    assert windows[0].targets == (195, 169, 122)
    assert windows[0].target_mask == (True, True, False)


def test_example_windows_never_cross_examples_and_skip_unsupervised_chunks():
    example = FineTuneExample("bounded", "train", "abcdefghi", ((7, 9),))
    windows = example_windows(example, block_size=4)
    assert [(row.offset, row.inputs) for row in windows] == [(4, (101, 102, 103, 104))]
    assert windows[0].target_mask == (False, False, True, True)


def test_example_windows_tail_align_to_cover_a_partial_final_chunk():
    example = FineTuneExample("tail", "train", "0123456789ABCDEFGHIJ", ((18, 20),))
    windows = example_windows(example, block_size=8)
    assert [(row.offset, row.inputs) for row in windows] == [(11, tuple(b"BCDEFGHI"))]
    assert windows[0].target_mask == (
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        True,
    )
