from ai_journey.inference_protocol import InferenceProtocol
from ai_journey.inference_report import build_inference_report
from ai_journey.inference_speculative import SpeculativePlan


def test_report_exposes_day_61_memory_math_and_honest_boundaries():
    gib = 1024**3
    protocol = InferenceProtocol(
        "day-61-7b-inference",
        7_000_000_000,
        32,
        32,
        8,
        128,
        32_768,
        (32_768,) * 16,
        2,
        80 * gib,
        14 * gib,
        6 * gib,
        16,
        SpeculativePlan(4, 4, 5),
    )

    report = build_inference_report(protocol)

    assert report["kind"] == "analytical_inference_capacity_plan"
    assert report["measured"] is False
    assert report["kv_cache"]["mha_total_bytes"] == 256 * gib
    assert report["kv_cache"]["gqa_total_bytes"] == 64 * gib
    assert report["kv_cache"]["gqa_fraction_of_mha"] == "1/4"
    assert report["ragged_admission"]["admitted"] is False
    assert report["paged_allocation"]["internal_waste_fraction"] == "0/1"
    assert report["speculative"]["expected_output_tokens_per_target_pass"] == (
        "2101/625"
    )
    assert report["evidence_boundaries"]["gpu_executed"] is False
