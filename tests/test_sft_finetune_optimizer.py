from ai_journey.sft_finetune_contract import SFTFineTuneConfig
from ai_journey.sft_finetune_model import build_finetune_model
from ai_journey.sft_finetune_optimizer import build_finetune_optimizer


def test_finetune_optimizer_applies_decay_only_to_matrix_parameters():
    config = SFTFineTuneConfig(weight_decay=0.2)
    model = build_finetune_model(config)
    optimizer = build_finetune_optimizer(model, config)
    assert [group["weight_decay"] for group in optimizer.param_groups] == [0.2, 0.0]
    grouped = [
        parameter for group in optimizer.param_groups for parameter in group["params"]
    ]
    assert {id(parameter) for parameter in grouped} == {
        id(parameter) for parameter in model.parameters()
    }
