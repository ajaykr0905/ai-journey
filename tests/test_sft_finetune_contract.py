from dataclasses import replace

import pytest

from ai_journey.sft_finetune_contract import (
    SFTFineTuneConfig,
    SFTFineTuneError,
)


def test_finetune_config_round_trips_all_explicit_controls():
    config = SFTFineTuneConfig()
    assert SFTFineTuneConfig.from_mapping(config.as_mapping()) == config
    assert config.work_units == 524_288


@pytest.mark.parametrize(
    "changes",
    [
        {"seed": True},
        {"block_size": 7},
        {"embedding_dim": 15},
        {"steps": 129},
        {"learning_rate": float("nan")},
        {"weight_decay": -0.1},
        {"checkpoint_interval": 9},
    ],
)
def test_finetune_config_rejects_unsafe_controls(changes):
    with pytest.raises((TypeError, SFTFineTuneError)):
        replace(SFTFineTuneConfig(), **changes)


def test_finetune_config_rejects_missing_or_extra_fields():
    with pytest.raises(SFTFineTuneError):
        SFTFineTuneConfig.from_mapping({})
