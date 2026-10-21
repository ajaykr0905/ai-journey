import pytest

from ai_journey.gpt2_training import GPT2TrainingConfig, GPT2TrainingError


def make_config(**overrides):
    values = {
        "optimizer_steps": 12,
        "micro_batch_size": 2,
        "accumulation_steps": 4,
        "sequence_length": 8,
        "max_learning_rate": 3e-3,
        "min_learning_rate": 3e-4,
        "warmup_steps": 2,
        "weight_decay": 0.1,
        "gradient_clip": 1.0,
        "seed": 47,
    }
    values.update(overrides)
    return GPT2TrainingConfig(**values)


def test_training_config_accounts_for_effective_batch_and_tokens():
    config = make_config()

    assert config.effective_batch_size == 8
    assert config.tokens_per_step == 64
    assert config.total_target_tokens == 768
    assert config.fingerprint() == make_config().fingerprint()
    assert config.fingerprint() != make_config(seed=48).fingerprint()


@pytest.mark.parametrize(
    ("name", "value", "exception"),
    [
        ("optimizer_steps", 0, GPT2TrainingError),
        ("micro_batch_size", True, TypeError),
        ("accumulation_steps", -1, GPT2TrainingError),
        ("sequence_length", 0, GPT2TrainingError),
        ("warmup_steps", 13, GPT2TrainingError),
        ("warmup_steps", 1.5, TypeError),
        ("max_learning_rate", float("nan"), GPT2TrainingError),
        ("max_learning_rate", 0.0, GPT2TrainingError),
        ("min_learning_rate", -1.0, GPT2TrainingError),
        ("min_learning_rate", 4e-3, GPT2TrainingError),
        ("weight_decay", -0.1, GPT2TrainingError),
        ("gradient_clip", 0.0, GPT2TrainingError),
        ("seed", True, TypeError),
        ("seed", 2**32, GPT2TrainingError),
    ],
)
def test_training_config_rejects_invalid_controls(name, value, exception):
    with pytest.raises(exception):
        make_config(**{name: value})
