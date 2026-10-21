from hashlib import sha256

import pytest
import torch

from ai_journey.gpt2_config import GPT2Config
from ai_journey.gpt2_model import GPT2LanguageModel
from ai_journey.gpt2_training import (
    GPT2ByteCorpus,
    GPT2BatchCursor,
    GPT2TrainingConfig,
    GPT2TrainingError,
    build_gpt2_optimizer,
    gpt2_model_fingerprint,
    language_model_loss,
    learning_rate_at,
    optimizer_group_summary,
    train_optimizer_step,
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


def make_model():
    torch.manual_seed(47)
    return GPT2LanguageModel(
        GPT2Config(
            vocab_size=256,
            context_length=8,
            embedding_dim=16,
            head_count=2,
            layer_count=1,
            embedding_dropout=0.0,
            attention_dropout=0.0,
            residual_dropout=0.0,
        )
    )


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


def test_learning_rate_warms_up_and_cosine_decays_to_floor():
    config = make_config(optimizer_steps=6, warmup_steps=2)
    rates = [learning_rate_at(step, config) for step in range(6)]

    assert rates[:2] == pytest.approx([1.5e-3, 3e-3])
    assert rates[2] == pytest.approx(config.max_learning_rate)
    assert rates[-1] == pytest.approx(config.min_learning_rate)
    assert rates[2:] == sorted(rates[2:], reverse=True)


def test_learning_rate_supports_no_warmup_and_all_warmup():
    no_warmup = make_config(optimizer_steps=3, warmup_steps=0)
    all_warmup = make_config(optimizer_steps=3, warmup_steps=3)

    assert [learning_rate_at(step, no_warmup) for step in range(3)] == pytest.approx(
        [3e-3, 1.65e-3, 3e-4]
    )
    assert [learning_rate_at(step, all_warmup) for step in range(3)] == pytest.approx(
        [1e-3, 2e-3, 3e-3]
    )


def test_single_decay_step_uses_learning_rate_floor():
    config = make_config(optimizer_steps=3, warmup_steps=2)
    assert learning_rate_at(2, config) == pytest.approx(config.min_learning_rate)


@pytest.mark.parametrize("step", [-1, 12, True, 1.5])
def test_learning_rate_rejects_steps_outside_schedule(step):
    with pytest.raises((GPT2TrainingError, TypeError)):
        learning_rate_at(step, make_config())


def test_optimizer_groups_decay_matrices_but_not_biases_or_norms():
    model = make_model()
    summary = optimizer_group_summary(model)

    assert "wte.weight" in summary.decay_names
    assert "lm_head.weight" not in summary.decay_names
    assert "h.0.attn.c_attn.bias" in summary.no_decay_names
    assert "h.0.ln_1.weight" in summary.no_decay_names
    assert summary.total_parameters == model.parameter_count
    assert set(summary.decay_names) | set(summary.no_decay_names) == {
        name for name, _ in model.named_parameters()
    }


def test_optimizer_applies_declared_decay_to_exact_groups():
    config = make_config(weight_decay=0.25)
    model = make_model()
    optimizer = build_gpt2_optimizer(model, config)
    summary = optimizer_group_summary(model)

    assert [group["weight_decay"] for group in optimizer.param_groups] == [0.25, 0.0]
    assert [group["lr"] for group in optimizer.param_groups] == [
        config.max_learning_rate,
        config.max_learning_rate,
    ]
    assert len(optimizer.param_groups[0]["params"]) == len(summary.decay_names)
    assert len(optimizer.param_groups[1]["params"]) == len(summary.no_decay_names)
    assert len(
        {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
    ) == len(list(model.parameters()))


def test_language_model_loss_matches_direct_cross_entropy():
    model = make_model()
    inputs = torch.tensor([[1, 2, 3], [4, 5, 6]])
    targets = torch.tensor([[2, 3, 4], [5, 6, 7]])

    loss = language_model_loss(model, inputs, targets)
    expected = torch.nn.functional.cross_entropy(
        model(inputs).reshape(-1, 256), targets.reshape(-1)
    )

    assert float(loss.detach()) == pytest.approx(float(expected.detach()))
    assert loss.ndim == 0


@pytest.mark.parametrize(
    ("inputs", "targets", "message"),
    [
        (torch.tensor([1, 2]), torch.tensor([2, 3]), "rank-two"),
        (torch.tensor([[1, 2]]), torch.tensor([[2]]), "same non-empty shape"),
        (
            torch.empty((0, 2), dtype=torch.long),
            torch.empty((0, 2), dtype=torch.long),
            "same non-empty shape",
        ),
        (torch.tensor([[1] * 9]), torch.tensor([[2] * 9]), "exceeds model context"),
        (torch.tensor([[256]]), torch.tensor([[1]]), "inputs contain"),
        (torch.tensor([[1]]), torch.tensor([[-1]]), "targets contain"),
    ],
)
def test_language_model_loss_rejects_invalid_batches(inputs, targets, message):
    with pytest.raises((GPT2TrainingError, TypeError), match=message):
        language_model_loss(make_model(), inputs, targets)


def test_gradient_accumulation_matches_one_large_batch_update():
    corpus = GPT2ByteCorpus.from_bytes(bytes(range(100)))
    accumulated_config = make_config(
        optimizer_steps=2,
        micro_batch_size=2,
        accumulation_steps=2,
        warmup_steps=0,
        gradient_clip=1_000.0,
    )
    large_batch_config = make_config(
        optimizer_steps=2,
        micro_batch_size=4,
        accumulation_steps=1,
        warmup_steps=0,
        gradient_clip=1_000.0,
    )
    accumulated = make_model()
    large_batch = make_model()
    accumulated_optimizer = build_gpt2_optimizer(accumulated, accumulated_config)
    large_batch_optimizer = build_gpt2_optimizer(large_batch, large_batch_config)

    accumulated_metric = train_optimizer_step(
        accumulated,
        GPT2BatchCursor(corpus, accumulated_config),
        accumulated_optimizer,
        accumulated_config,
        step=0,
    )
    large_batch_metric = train_optimizer_step(
        large_batch,
        GPT2BatchCursor(corpus, large_batch_config),
        large_batch_optimizer,
        large_batch_config,
        step=0,
    )

    assert accumulated_metric.target_tokens == large_batch_metric.target_tokens == 32
    assert accumulated_metric.mean_loss == pytest.approx(
        large_batch_metric.mean_loss, abs=2e-6
    )
    for accumulated_parameter, large_batch_parameter in zip(
        accumulated.parameters(), large_batch.parameters(), strict=True
    ):
        torch.testing.assert_close(
            accumulated_parameter, large_batch_parameter, rtol=2e-6, atol=5e-7
        )


def test_optimizer_step_records_schedule_and_gradient_clipping():
    config = make_config(
        optimizer_steps=2,
        warmup_steps=0,
        gradient_clip=1e-3,
        micro_batch_size=2,
        accumulation_steps=2,
    )
    model = make_model()
    metric = train_optimizer_step(
        model,
        GPT2BatchCursor(GPT2ByteCorpus.from_bytes(bytes(range(100))), config),
        build_gpt2_optimizer(model, config),
        config,
        step=0,
    )

    assert metric.step == 1
    assert metric.learning_rate == pytest.approx(config.max_learning_rate)
    assert metric.gradient_norm_before_clip > config.gradient_clip
    assert metric.gradient_norm_after_clip <= config.gradient_clip
    assert metric.clipped is True
    assert metric.micro_batches == 2


def test_optimizer_step_rolls_back_model_optimizer_cursor_rng_and_mode(monkeypatch):
    config = make_config(
        optimizer_steps=2,
        warmup_steps=0,
        micro_batch_size=2,
        accumulation_steps=2,
    )
    corpus = GPT2ByteCorpus.from_bytes(bytes(range(100)))
    model = make_model().eval()
    cursor = GPT2BatchCursor(corpus, config)
    optimizer = build_gpt2_optimizer(model, config)
    model_before = {name: value.clone() for name, value in model.state_dict().items()}
    optimizer_before = optimizer.state_dict()
    cursor_before = cursor.state_dict()
    rng_before = torch.get_rng_state().clone()
    real_step = optimizer.step

    def corrupt_step():
        real_step()
        with torch.no_grad():
            next(model.parameters()).view(-1)[0] = float("inf")

    monkeypatch.setattr(optimizer, "step", corrupt_step)
    with pytest.raises(GPT2TrainingError, match=r"updated model.*must be finite"):
        train_optimizer_step(model, cursor, optimizer, config, step=0)

    assert model.training is False
    assert cursor.state_dict() == cursor_before
    assert optimizer.state_dict() == optimizer_before
    assert torch.equal(torch.get_rng_state(), rng_before)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, model_before[name], rtol=0, atol=0)


def test_model_fingerprint_binds_names_shapes_dtypes_and_values():
    model = make_model()
    original = gpt2_model_fingerprint(model)

    assert original == gpt2_model_fingerprint(make_model())
    with torch.no_grad():
        model.h[0].ln_1.weight[0] += 1
    assert gpt2_model_fingerprint(model) != original
