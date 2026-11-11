from ai_journey.sft_finetune_contract import SFTFineTuneConfig
from ai_journey.sft_finetune_dataset import FineTuneExample
from ai_journey.sft_finetune_generation import generation_prompt, greedy_generation
from ai_journey.sft_finetune_model import build_finetune_model


def test_greedy_generation_uses_only_the_heldout_prompt_prefix():
    example = FineTuneExample("held", "validation", "useréANSWER", ((5, 11),))
    prompt = generation_prompt(example, block_size=4)
    assert bytes(prompt).decode("utf-8") == "eré"
    output = greedy_generation(
        build_finetune_model(SFTFineTuneConfig(block_size=8)), example, new_tokens=3
    )
    assert output["example_id"] == "held"
    assert len(output["token_ids"]) == 3
    assert len(output["sha256"]) == 64
