from pathlib import Path

from ai_journey.roofline_cli import load_roofline_plan
from ai_journey.roofline_report import (
    build_roofline_report,
    seal_roofline_report,
    verify_roofline_report,
)


ROOT = Path(__file__).resolve().parents[1]


def test_pinned_day_55_plan_recomputes_the_documented_comparison():
    plan = load_roofline_plan(ROOT / "config/day-55-roofline-plan.json")
    report = seal_roofline_report(build_roofline_report(plan))

    assert verify_roofline_report(report) is report
    assert report["summary"] == {
        "operation_count": 10,
        "bottleneck_counts": {"compute": 8, "memory": 2, "ridge": 0},
        "total_flops": 29_265_974_263_808,
        "total_compulsory_bytes": 45_121_273_856,
        "serial_lower_bound_seconds": 0.29588767899648,
    }
    assert report["operations"]["block_layer_norms"]["bottleneck"] == "memory"
    assert report["operations"]["attention_query"]["bottleneck"] == "compute"


def test_roofline_runbook_preserves_analytical_evidence_boundaries():
    documentation = (ROOT / "docs/roofline-analysis.md").read_text()
    compact = " ".join(documentation.split())

    assert "ai-journey-roofline build" in documentation
    assert "ai-journey-roofline verify" in documentation
    assert "not an observed runtime" in compact
    assert "does not prove GPU availability" in compact
