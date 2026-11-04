from ai_journey.inference_cli import load_inference_protocol
from ai_journey.inference_report import build_inference_report


def test_pinned_day_61_protocol_reproduces_the_7b_32k_batch_16_calculation():
    protocol = load_inference_protocol("config/day-61-inference-plan.json")
    report = build_inference_report(protocol)

    assert protocol.name == "day-61-7b-inference"
    assert len(protocol.request_lengths) == 16
    assert report["kv_cache"]["mha_total_bytes"] == 256 * 1024**3
    assert report["kv_cache"]["gqa_total_bytes"] == 64 * 1024**3
    assert report["ragged_admission"]["capacity_bytes"] == 60 * 1024**3
    assert report["ragged_admission"]["admitted"] is False
