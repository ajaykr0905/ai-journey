from pathlib import Path

from ai_journey.sft_finetune_cli import load_finetune_config


def test_day_68_config_declares_the_bounded_reference_protocol():
    config = load_finetune_config(Path("config/day-68-sft-finetune.json"))
    assert config.seed == 68
    assert config.steps == 8
    assert config.evaluation_examples == 8
    assert config.work_units == 524_288
