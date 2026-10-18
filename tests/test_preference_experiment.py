from dataclasses import asdict

import pytest

from ai_journey.preference_data import PreferenceDataError
from ai_journey.preference_experiment import (
    PreferenceComparisonConfig,
    preference_config_dict,
    preference_config_from_dict,
)


def test_preference_config_round_trips_all_controls():
    config = PreferenceComparisonConfig()

    restored = preference_config_from_dict(preference_config_dict(config))

    assert restored == config
    assert isinstance(preference_config_dict(config)["seeds"], list)
    assert config.work_units <= 80_000_000


def test_preference_config_builds_a_bounded_layer_norm_decoder():
    config = PreferenceComparisonConfig()

    model = config.model_config(53)

    assert model.vocab_size == 53
    assert model.block_size == config.block_size
    assert model.normalization_mode == "layer_norm"
    assert model.dropout == 0


@pytest.mark.parametrize(
    "changes",
    [
        {"seeds": (44,)},
        {"seeds": (44, 44)},
        {"fold_count": 1},
        {"block_size": 4},
        {"embedding_dim": 10, "head_count": 4},
        {"preference_steps": 0},
        {"beta": 0},
        {"preference_learning_rate": float("nan")},
    ],
)
def test_preference_config_rejects_invalid_controls(changes):
    values = asdict(PreferenceComparisonConfig())
    values.update(changes)

    with pytest.raises(PreferenceDataError):
        PreferenceComparisonConfig(**values)


def test_preference_config_rejects_unknown_fields():
    value = preference_config_dict(PreferenceComparisonConfig())
    value["ignored"] = True

    with pytest.raises(PreferenceDataError, match="exactly"):
        preference_config_from_dict(value)
