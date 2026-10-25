import pytest

from ai_journey.resource_accounting import (
    DecoderSpec,
    ResourceAccountingError,
    TrainingWorkload,
    activation_memory,
    attention_workspace_memory,
    attention_matmul_flops,
    bytes_per_element,
    checkpointed_activation_memory,
    compute_runtime,
    distributed_training_memory,
    forward_linear_flops,
    kv_cache_memory,
    memory_capacity,
    optimizer_update_flops,
    parameter_memory,
    training_state_memory,
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


def test_parameter_memory_preserves_group_accounting_across_dtypes():
    float32 = parameter_memory(small_spec(), dtype="float32")
    bfloat16 = parameter_memory(small_spec(), dtype="bfloat16")
    assert float32["total"] == 7_888 * 4
    assert bfloat16["total"] == float32["total"] // 2
    assert (
        sum(value for key, value in float32.items() if key != "total")
        == float32["total"]
    )
    assert bytes_per_element("int8") == 1
    with pytest.raises(ResourceAccountingError, match="unsupported dtype"):
        bytes_per_element("float24")


def test_training_state_memory_accounts_for_adam_and_master_weights():
    mixed = training_state_memory(
        small_spec(),
        parameter_dtype="bfloat16",
        gradient_dtype="bfloat16",
        optimizer_state_dtype="float32",
        master_weights=True,
    )
    assert mixed == {
        "parameters": 7_888 * 2,
        "gradients": 7_888 * 2,
        "optimizer_moments": 7_888 * 8,
        "master_parameters": 7_888 * 4,
        "total": 7_888 * 16,
    }
    without_master = training_state_memory(
        small_spec(),
        parameter_dtype="float32",
        gradient_dtype="float32",
        optimizer_state_dtype="float32",
        master_weights=False,
    )
    assert without_master["total"] == 7_888 * 16


def test_activation_memory_exposes_saved_tensor_families():
    memory = activation_memory(
        small_spec(), batch_size=3, sequence_length=5, dtype="bfloat16"
    )
    assert memory == {
        "residual_streams": 1_920,
        "attention_projections": 2_880,
        "feed_forward_intermediates": 3_840,
        "total": 8_640,
    }


def test_checkpointing_retains_boundaries_and_one_recompute_window():
    memory = checkpointed_activation_memory(
        small_spec(),
        batch_size=3,
        sequence_length=5,
        dtype="bfloat16",
        checkpoint_segments=2,
    )
    assert memory == {
        "checkpoint_boundaries": 1_440,
        "recompute_window": 4_320,
        "peak": 5_760,
        "saved_vs_uncheckpointed": 2_880,
    }
    with pytest.raises(ResourceAccountingError, match="cannot exceed"):
        checkpointed_activation_memory(
            small_spec(),
            batch_size=3,
            sequence_length=5,
            dtype="bfloat16",
            checkpoint_segments=3,
        )


def test_attention_workspace_exposes_quadratic_score_storage():
    saved = attention_workspace_memory(
        small_spec(),
        batch_size=3,
        sequence_length=5,
        dtype="bfloat16",
        save_probabilities=True,
    )
    assert saved == {"scores": 600, "probabilities": 600, "total": 1_200}
    recomputed = attention_workspace_memory(
        small_spec(),
        batch_size=3,
        sequence_length=5,
        dtype="bfloat16",
        save_probabilities=False,
    )
    assert recomputed["total"] == 600


def test_kv_cache_accounts_for_layers_batch_and_grouped_heads():
    cache = kv_cache_memory(
        small_spec(), batch_size=3, sequence_length=5, dtype="bfloat16"
    )
    assert cache == {"per_layer": 480, "per_sequence": 320, "total": 960}
    full_heads = kv_cache_memory(
        small_spec(kv_heads=4), batch_size=3, sequence_length=5, dtype="bfloat16"
    )
    assert full_heads["total"] == 2 * cache["total"]


def test_distributed_memory_distinguishes_replication_and_sharding_stages():
    common = {
        "model": small_spec(),
        "parameter_dtype": "bfloat16",
        "gradient_dtype": "bfloat16",
        "optimizer_state_dtype": "float32",
        "master_weights": True,
        "activation_bytes": 1_000,
        "data_parallel_size": 4,
    }
    assert distributed_training_memory(**common, strategy="ddp")["total"] == 127_208
    assert distributed_training_memory(**common, strategy="zero1")["total"] == 56_216
    assert distributed_training_memory(**common, strategy="zero2")["total"] == 44_384
    assert distributed_training_memory(**common, strategy="zero3")["total"] == 32_552
    with pytest.raises(ResourceAccountingError, match="strategy must be"):
        distributed_training_memory(**common, strategy="unknown")


def test_capacity_gate_reserves_memory_before_deciding_fit():
    fits = memory_capacity(80, device_capacity_bytes=100, reserve_fraction=0.1)
    assert fits == {
        "capacity": 100,
        "reserved": 10,
        "usable": 90,
        "required": 80,
        "headroom": 10,
        "fits": True,
    }
    assert not memory_capacity(91, device_capacity_bytes=100, reserve_fraction=0.1)[
        "fits"
    ]
    with pytest.raises(ResourceAccountingError, match=r"\[0, 1\)"):
        memory_capacity(80, device_capacity_bytes=100, reserve_fraction=1.0)


def test_compute_runtime_exposes_peak_and_mfu_assumptions():
    runtime = compute_runtime(
        1_000_000_000_000,
        accelerator_peak_flops=1_000_000_000_000,
        accelerator_count=2,
        model_flops_utilization=0.5,
    )
    assert runtime == {
        "total_flops": 1_000_000_000_000,
        "aggregate_peak_flops_per_second": 2_000_000_000_000,
        "effective_flops_per_second": 1_000_000_000_000.0,
        "seconds": 1.0,
    }
    with pytest.raises(ResourceAccountingError, match=r"\(0, 1\]"):
        compute_runtime(
            1,
            accelerator_peak_flops=1,
            accelerator_count=1,
            model_flops_utilization=0,
        )
