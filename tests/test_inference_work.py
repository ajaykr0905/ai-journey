from ai_journey.inference_work import PrefillWork, estimate_prefill_work


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
