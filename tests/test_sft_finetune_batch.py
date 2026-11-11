import torch

from ai_journey.sft_finetune_batch import collate_windows
from ai_journey.sft_finetune_windows import FineTuneWindow


def test_collate_windows_preserves_token_and_mask_alignment():
    rows = (
        FineTuneWindow("a", 0, (1, 2), (2, 3), (False, True)),
        FineTuneWindow("b", 2, (4, 5), (5, 6), (True, True)),
    )
    batch = collate_windows(rows)
    assert torch.equal(batch.inputs, torch.tensor([[1, 2], [4, 5]]))
    assert torch.equal(batch.targets, torch.tensor([[2, 3], [5, 6]]))
    assert batch.target_mask.tolist() == [[False, True], [True, True]]
