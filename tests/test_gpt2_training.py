from hashlib import sha256

import pytest
import torch

from ai_journey.gpt2_training import (
    GPT2ByteCorpus,
    GPT2BatchCursor,
    GPT2TrainingConfig,
    GPT2TrainingError,
)


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


def test_byte_corpus_preserves_source_and_owns_its_tokens():
    source = b"public training text\n"
    corpus = GPT2ByteCorpus.from_bytes(source)

    assert corpus.tokens.tolist() == list(source)
    assert corpus.source_sha256 == sha256(source).hexdigest()
    assert corpus.fingerprint() == GPT2ByteCorpus.from_bytes(source).fingerprint()

    caller_tokens = torch.tensor(list(source), dtype=torch.long)
    independent = GPT2ByteCorpus(caller_tokens, sha256(source).hexdigest())
    caller_tokens.zero_()
    assert independent.tokens.tolist() == list(source)


@pytest.mark.parametrize("source", [b"", b"x"])
def test_byte_corpus_rejects_too_short_sources(source):
    with pytest.raises(GPT2TrainingError, match="at least two bytes"):
        GPT2ByteCorpus.from_bytes(source)


def test_byte_corpus_enforces_input_budget():
    with pytest.raises(GPT2TrainingError, match="exceeds max_bytes"):
        GPT2ByteCorpus.from_bytes(b"abc", max_bytes=2)


def test_byte_corpus_rejects_out_of_range_tokens():
    with pytest.raises(GPT2TrainingError, match=r"in \[0, 255\]"):
        GPT2ByteCorpus(torch.tensor([0, 256]), "0" * 64)


def test_batch_cursor_replays_complete_micro_batches_from_saved_state():
    corpus = GPT2ByteCorpus.from_bytes(bytes(range(25)))
    config = make_config(micro_batch_size=4, sequence_length=5)
    cursor = GPT2BatchCursor(corpus, config)

    first = cursor.next()
    state = cursor.state_dict()
    expected_second = cursor.next()

    resumed = GPT2BatchCursor(corpus, config)
    resumed.load_state_dict(state)
    actual_second = resumed.next()

    assert first[0].shape == (4, 5)
    assert first[1].shape == (4, 5)
    assert all(
        torch.equal(actual, expected)
        for actual, expected in zip(actual_second, expected_second, strict=True)
    )
    assert resumed.fingerprint() == cursor.fingerprint()


def test_batch_cursor_discards_only_incomplete_epoch_tail():
    corpus = GPT2ByteCorpus.from_bytes(bytes(range(24)))
    config = make_config(micro_batch_size=4, sequence_length=5)
    cursor = GPT2BatchCursor(corpus, config)

    assert cursor.sample_count == 19
    assert cursor.usable_samples == 16
    assert cursor.discarded_samples_per_epoch == 3
    for _ in range(4):
        cursor.next()
    assert cursor.state_dict() == {"epoch": 0, "offset": 16}
    cursor.next()
    assert cursor.state_dict() == {"epoch": 1, "offset": 4}


def test_batch_cursor_requires_one_complete_micro_batch():
    corpus = GPT2ByteCorpus.from_bytes(b"abcdef")
    with pytest.raises(GPT2TrainingError, match="complete micro-batch"):
        GPT2BatchCursor(corpus, make_config(micro_batch_size=4, sequence_length=4))


@pytest.mark.parametrize(
    "state",
    [
        {"epoch": 0},
        {"epoch": -1, "offset": 0},
        {"epoch": 0, "offset": 1},
        {"epoch": 0, "offset": 10_000},
    ],
)
def test_batch_cursor_rejects_invalid_restart_boundaries(state):
    cursor = GPT2BatchCursor(
        GPT2ByteCorpus.from_bytes(bytes(range(25))),
        make_config(micro_batch_size=4, sequence_length=5),
    )
    with pytest.raises((GPT2TrainingError, TypeError)):
        cursor.load_state_dict(state)
