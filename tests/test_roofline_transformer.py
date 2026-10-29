import pytest

from ai_journey.resource_accounting import (
    DecoderSpec,
    ResourceAccountingError,
    forward_flops,
)
from ai_journey.roofline import RooflineHardware
from ai_journey.roofline_transformer import (
    analyze_decoder_microbatch,
    decoder_microbatch_workloads,
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


def test_decoder_roofline_matmuls_reconcile_with_resource_accounting():
    model = small_spec()
    workloads = decoder_microbatch_workloads(
        model,
        batch_size=3,
        sequence_length=5,
        dtype="bfloat16",
    )

    matmul_flops = sum(
        workload.flops for workload in workloads.values() if workload.kind == "matmul"
    )
    assert matmul_flops == sum(
        forward_flops(model, batch_size=3, sequence_length=5).values()
    )
    assert set(workloads) == {
        "attention_query",
        "attention_key_value",
        "attention_output",
        "attention_scores",
        "attention_value_mix",
        "feed_forward_gate_up",
        "feed_forward_down",
        "block_layer_norms",
        "final_layer_norm",
        "output_projection",
    }


def test_grouped_query_heads_reduce_key_value_projection_work_and_traffic():
    grouped = decoder_microbatch_workloads(
        small_spec(kv_heads=2), batch_size=3, sequence_length=5, dtype="bfloat16"
    )["attention_key_value"]
    full = decoder_microbatch_workloads(
        small_spec(kv_heads=4), batch_size=3, sequence_length=5, dtype="bfloat16"
    )["attention_key_value"]

    assert grouped.flops * 2 == full.flops
    assert grouped.bytes_written * 2 == full.bytes_written
    assert grouped.bytes_read < full.bytes_read


def test_decoder_analysis_classifies_matmul_and_layer_norm_separately():
    estimates = analyze_decoder_microbatch(
        small_spec(),
        batch_size=3,
        sequence_length=5,
        dtype="bfloat16",
        hardware=RooflineHardware(
            peak_flops_per_second=20,
            memory_bandwidth_bytes_per_second=10,
        ),
    )

    assert estimates["attention_query"].bottleneck == "compute"
    assert estimates["block_layer_norms"].bottleneck == "memory"
    assert estimates["attention_query"].workload.arithmetic_intensity > (
        estimates["block_layer_norms"].workload.arithmetic_intensity
    )


def test_decoder_workloads_validate_public_boundaries():
    with pytest.raises(TypeError, match="DecoderSpec"):
        decoder_microbatch_workloads(
            "model", batch_size=1, sequence_length=1, dtype="bfloat16"
        )
    with pytest.raises(ResourceAccountingError, match="batch_size must be positive"):
        decoder_microbatch_workloads(
            small_spec(), batch_size=0, sequence_length=1, dtype="bfloat16"
        )
    hardware = RooflineHardware(100, 10)
    with pytest.raises(TypeError, match="RooflineHardware"):
        analyze_decoder_microbatch(
            small_spec(),
            batch_size=1,
            sequence_length=1,
            dtype="bfloat16",
            hardware=hardware.__dict__,
        )
