import pytest

from ai_journey.resource_accounting import (
    DecoderSpec,
    ResourceAccountingError,
    TrainingWorkload,
    attention_matmul_flops,
    forward_linear_flops,
    optimizer_update_flops,
    training_step_flops,
)


def small_spec(**changes) -> DecoderSpec:
    values = {
        "vocab_size": 100,
        "layer_count": 2,
        "hidden_size": 16,
        "attention_heads": 4,
        "kv_heads": 2,
        "feed_forward_size": 32,
    }
    values.update(changes)
    return DecoderSpec(**values)


def test_decoder_spec_validates_head_geometry_and_types():
    assert small_spec().head_size == 4
    assert small_spec().kv_size == 8
    with pytest.raises(ResourceAccountingError, match="divisible"):
        small_spec(hidden_size=15)
    with pytest.raises(ResourceAccountingError, match="kv_heads"):
        small_spec(kv_heads=3)
    with pytest.raises(TypeError, match="integer"):
        small_spec(layer_count=True)


def test_parameter_inventory_is_complete_and_non_overlapping():
    spec = small_spec()
    assert spec.parameter_groups() == {
        "token_embedding": 1_600,
        "attention": 1_536,
        "feed_forward": 3_072,
        "block_norms": 64,
        "final_norm": 16,
        "output_projection": 1_600,
    }
    assert spec.parameter_count == 7_888


def test_tied_output_and_bias_parameters_are_explicit():
    tied = small_spec(tie_embeddings=True)
    biased = small_spec(use_bias=True)
    assert tied.parameter_count == small_spec().parameter_count - 1_600
    assert biased.parameter_count == small_spec().parameter_count + 256


def test_training_workload_accounts_for_every_parallelism_axis():
    workload = TrainingWorkload(
        sequence_length=2_048,
        micro_batch_size=2,
        gradient_accumulation_steps=4,
        data_parallel_size=8,
        optimizer_steps=10,
    )
    assert workload.tokens_per_micro_batch == 4_096
    assert workload.tokens_per_optimizer_step == 131_072
    assert workload.total_training_tokens == 1_310_720
    with pytest.raises(ResourceAccountingError, match="optimizer_steps"):
        TrainingWorkload(2_048, 2, 4, 8, 0)


def test_forward_linear_flops_decompose_projection_work():
    flops = forward_linear_flops(small_spec(), batch_size=3, sequence_length=5)
    assert flops == {
        "attention_projections": 46_080,
        "feed_forward_projections": 92_160,
        "output_projection": 48_000,
    }
    assert sum(flops.values()) == 186_240


def test_attention_flops_scale_quadratically_with_sequence_length():
    short = attention_matmul_flops(small_spec(), batch_size=3, sequence_length=5)
    long = attention_matmul_flops(small_spec(), batch_size=3, sequence_length=10)
    assert short == {
        "query_key_scores": 4_800,
        "attention_value_mix": 4_800,
    }
    assert sum(long.values()) == 4 * sum(short.values())


def test_training_step_flops_include_accumulation_and_data_parallel_work():
    workload = TrainingWorkload(5, 3, 2, 4, 10)
    estimate = training_step_flops(small_spec(), workload)
    per_micro_batch = 186_240 + 9_600
    assert estimate == {
        "forward": per_micro_batch * 8,
        "backward": per_micro_batch * 16,
        "total": per_micro_batch * 24,
    }
    with pytest.raises(ResourceAccountingError, match="positive and finite"):
        training_step_flops(
            small_spec(), workload, backward_to_forward_ratio=float("nan")
        )


def test_optimizer_update_uses_an_explicit_per_parameter_budget():
    assert optimizer_update_flops(small_spec()) == 78_880
    assert optimizer_update_flops(small_spec(), operations_per_parameter=6) == 47_328
    with pytest.raises(ResourceAccountingError, match="operations_per_parameter"):
        optimizer_update_flops(small_spec(), operations_per_parameter=0)
