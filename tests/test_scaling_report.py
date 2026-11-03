from ai_journey.scaling_protocol import ScalingProtocol
from ai_journey.scaling_report import build_scaling_report
from ai_journey.scaling_runtime import RuntimeAssumptions
from ai_journey.scaling_schedule import TrainingBatchGeometry


def make_protocol():
    return ScalingProtocol(
        name="day-60-100m-chinchilla-plan",
        parameter_count=100_000_000,
        tokens_per_parameter="20",
        training_flops_per_parameter_token="6",
        unique_train_tokens=69_626,
        batch=TrainingBatchGeometry(2_048, 2, 4, 8),
        maximum_overshoot_fraction=0.0001,
        isoflop_parameter_counts=(50_000_000, 100_000_000, 200_000_000),
        runtime=RuntimeAssumptions("100000000000000", 2, "3.50"),
    )


def test_report_reconciles_target_schedule_corpus_and_isoflop_rows():
    report = build_scaling_report(make_protocol())

    assert report["schema_version"] == 1
    assert report["kind"] == "analytical_scaling_plan"
    assert report["measured"] is False
    assert report["target"] == {
        "tokens": 2_000_000_000,
        "training_flops": 1_200_000_000_000_000_000,
        "actual_tokens_per_parameter": "20",
    }
    assert report["schedule"]["scheduled_tokens"] == 2_000_027_648
    assert report["corpus"]["full_epochs"] == 28_724
    assert report["selected_isoflop_parameter_count"] == 100_000_000
    assert [row["parameter_count"] for row in report["isoflop_candidates"]] == [
        50_000_000,
        100_000_000,
        200_000_000,
    ]
    assert report["runtime_estimate"]["seconds"] == "6000"
    assert report["runtime_estimate"]["monetary_cost"].startswith("11.666666666666")


def test_report_serializes_decimal_fields_without_binary_float_drift():
    report = build_scaling_report(make_protocol())

    assert report["corpus"]["total_epochs"].startswith("28724.901617")
    assert report["corpus"]["distinct_token_fraction"].startswith("0.000034813")
    assert all(
        isinstance(row["tokens_per_parameter"], str)
        for row in report["isoflop_candidates"]
    )


def test_report_keeps_execution_and_learning_claims_false():
    boundaries = build_scaling_report(make_protocol())["evidence_boundaries"]

    assert boundaries == {
        "analytical_only": True,
        "training_executed": False,
        "gpu_executed": False,
        "loss_measured": False,
        "learner_completion": False,
    }
