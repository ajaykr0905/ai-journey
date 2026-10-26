from dataclasses import replace

import pytest
import torch

from ai_journey.decoder_variant_config import DecoderVariantConfig
from ai_journey.decoder_variant_budget import require_comparison_budget
from ai_journey.decoder_variant_experiment import run_variant_comparison
from ai_journey.decoder_variant_runtime import isolated_decoder_runtime
from ai_journey.gpt2_config import GPT2Config


def config():
    return DecoderVariantConfig(
        GPT2Config(
            vocab_size=256,
            context_length=4,
            embedding_dim=8,
            head_count=2,
            layer_count=1,
            embedding_dropout=0.0,
            attention_dropout=0.0,
            residual_dropout=0.0,
        )
    )


def compare(**overrides):
    base = config()
    controls = dict(
        train_bytes=b"public training material for decoder experiments",
        heldout_bytes=b"separate heldout text with different contents",
        variants={"baseline": base, "rmsnorm": replace(base, norm_kind="rmsnorm")},
        seed=7,
        steps=1,
        batch_size=2,
        learning_rate=0.001,
        evaluation_windows=3,
    )
    return run_variant_comparison(**(controls | overrides))


def test_comparison_ignores_ambient_dtype_and_default_device():
    expected = compare()
    dtype = torch.get_default_dtype()
    rng = torch.get_rng_state().clone()
    try:
        torch.set_default_dtype(torch.float64)
        with torch.device("meta"):
            actual = compare()
        assert actual == expected
        assert torch.get_default_dtype() == torch.float64
        assert torch.equal(torch.get_rng_state(), rng)
    finally:
        torch.set_default_dtype(dtype)


def test_runtime_restores_controls_after_failure_and_never_seeds_accelerators(
    monkeypatch,
):
    def forbidden(*args, **kwargs):
        raise AssertionError("torch.manual_seed would seed accelerator generators")

    monkeypatch.setattr(torch, "manual_seed", forbidden)
    compare()
    dtype = torch.get_default_dtype()
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    rng = torch.get_rng_state().clone()
    with pytest.raises(RuntimeError, match="injected"):
        with isolated_decoder_runtime():
            torch.rand(3)
            raise RuntimeError("injected")
    assert torch.get_default_dtype() == dtype
    assert torch.get_num_threads() == threads
    assert torch.are_deterministic_algorithms_enabled() == deterministic
    assert torch.is_deterministic_algorithms_warn_only_enabled() == warn_only
    assert torch.equal(torch.get_rng_state(), rng)


def test_admission_counts_evaluation_before_constructing_models():
    base = replace(config().gpt2, context_length=64, embedding_dim=64, layer_count=2)
    # Training alone is only 8192 work units, below the old 200000 limit.
    with pytest.raises(ValueError, match="training and evaluation"):
        require_comparison_budget(base, steps=1, batch_size=1, evaluation_windows=256)
    require_comparison_budget(
        config().gpt2, steps=2, batch_size=2, evaluation_windows=4
    )


def test_admission_bounds_peak_attention_independently():
    base = replace(config().gpt2, context_length=2048)
    with pytest.raises(ValueError, match="attention"):
        require_comparison_budget(
            base, steps=1, batch_size=1, evaluation_windows=1, variant_count=1
        )
