from pathlib import Path

from ai_journey.scaling_protocol import load_scaling_protocol
from ai_journey.scaling_report import build_scaling_report


ROOT = Path(__file__).resolve().parents[1]


def test_checked_in_day60_plan_sizes_the_100m_model_without_execution_claims():
    protocol = load_scaling_protocol(ROOT / "config/day-60-scaling-plan.json")
    report = build_scaling_report(protocol)

    assert protocol.name == "day-60-100m-chinchilla-plan"
    assert report["target"]["tokens"] == 2_000_000_000
    assert report["target"]["training_flops"] == 1_200_000_000_000_000_000
    assert report["schedule"]["optimizer_steps"] == 15_259
    assert report["schedule"]["overshoot_tokens"] == 27_648
    assert report["corpus"]["full_epochs"] == 28_724
    assert report["evidence_boundaries"]["training_executed"] is False
    assert report["evidence_boundaries"]["gpu_executed"] is False


def test_checked_in_day60_isoflop_grid_contains_the_declared_optimum():
    protocol = load_scaling_protocol(ROOT / "config/day-60-scaling-plan.json")
    report = build_scaling_report(protocol)

    assert report["selected_isoflop_parameter_count"] == 100_000_000
    optimal = report["isoflop_candidates"][1]
    assert optimal["token_count"] == 2_000_000_000
    assert optimal["tokens_per_parameter"] == "20"
    assert optimal["absolute_ratio_error"] == "0"
