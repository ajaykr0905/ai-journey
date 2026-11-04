from fractions import Fraction

from ai_journey.inference_kv import KVCacheLayout
from ai_journey.inference_work import (
    DecodeMemoryTraffic,
    DecodeStepWork,
    PrefillWork,
    estimate_decode_memory_traffic,
    estimate_decode_step_work,
    estimate_prefill_work,
)


def test_prefill_work_separates_dense_projections_from_quadratic_attention():
    work = estimate_prefill_work(
        parameter_count=7_000_000_000,
        layers=32,
        query_heads=32,
        head_dim=128,
        sequence_tokens=32_768,
        batch_size=16,
    )

    assert work.projection_flops == 7_340_032_000_000_000
    assert work.attention_flops == 9_007_199_254_740_992
    assert work.total_flops == 16_347_231_254_740_992


def test_prefill_work_preserves_exact_integer_arithmetic():
    assert estimate_prefill_work(
        parameter_count=10,
        layers=2,
        query_heads=2,
        head_dim=4,
        sequence_tokens=3,
        batch_size=2,
    ) == PrefillWork(projection_flops=120, attention_flops=1152)


def test_cached_decode_attention_is_linear_in_the_resident_context():
    first = estimate_decode_step_work(
        parameter_count=10,
        layers=2,
        query_heads=2,
        head_dim=4,
        context_tokens=3,
        batch_size=2,
    )
    second = estimate_decode_step_work(
        parameter_count=10,
        layers=2,
        query_heads=2,
        head_dim=4,
        context_tokens=6,
        batch_size=2,
    )

    assert first == DecodeStepWork(projection_flops=40, attention_flops=384)
    assert second.projection_flops == first.projection_flops
    assert second.attention_flops == 2 * first.attention_flops


def test_decode_memory_traffic_exposes_weight_and_kv_reads_separately():
    cache = KVCacheLayout(32, 8, 128, 32_768, 16)
    traffic = estimate_decode_memory_traffic(
        parameter_count=7_000_000_000,
        weight_bytes_per_parameter=2,
        cache=cache,
    )
    work = estimate_decode_step_work(
        parameter_count=7_000_000_000,
        layers=32,
        query_heads=32,
        head_dim=128,
        context_tokens=32_768,
        batch_size=16,
    )

    assert traffic == DecodeMemoryTraffic(
        weight_bytes=14_000_000_000,
        kv_read_bytes=64 * 1024**3,
    )
    assert traffic.arithmetic_intensity(work) == Fraction(
        work.total_flops, traffic.total_bytes
    )
