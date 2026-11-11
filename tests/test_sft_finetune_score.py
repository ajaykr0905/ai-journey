import math

import pytest
import torch

from ai_journey.sft_finetune_contract import SFTFineTuneConfig
from ai_journey.sft_finetune_model import build_finetune_model
from ai_journey.sft_finetune_score import score_windows
from ai_journey.sft_finetune_windows import FineTuneWindow


def test_score_windows_counts_only_selected_assistant_targets():
    model = build_finetune_model(SFTFineTuneConfig(block_size=8))
    for parameter in model.parameters():
        torch.nn.init.zeros_(parameter)
    window = FineTuneWindow("row", 0, (1, 2, 3), (2, 3, 4), (False, True, True))
    score = score_windows(model, (window,))
    assert score["target_tokens"] == 2
    assert score["mean_nll"] == math.log(256)
    assert score["perplexity"] == pytest.approx(256.0)
