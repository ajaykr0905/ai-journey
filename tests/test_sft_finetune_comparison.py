import copy

from ai_journey.sft_finetune_comparison import compare_finetune_models
from ai_journey.sft_finetune_contract import SFTFineTuneConfig
from ai_journey.sft_finetune_dataset import FineTuneExample
from ai_journey.sft_finetune_model import build_finetune_model


def test_comparison_uses_matched_coverage_and_reports_nll_delta():
    config = SFTFineTuneConfig(block_size=8, evaluation_examples=1, generation_tokens=2)
    before = build_finetune_model(config)
    after = copy.deepcopy(before)
    validation = (FineTuneExample("held", "validation", "prompt answer", ((7, 13),)),)
    result = compare_finetune_models(before, after, validation, config)
    assert result["before"]["score"] == result["after"]["score"]
    assert result["delta_mean_nll"] == 0
    assert result["changed_generations"] == 0
