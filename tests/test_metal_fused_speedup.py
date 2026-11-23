import pytest

from ai_journey.kernel_metal_benchmark import DeviceTiming
from ai_journey.memory_access_benchmark import TIMER
from ai_journey.metal_fused_benchmark import side_rates
from ai_journey.metal_fused_plan import TORCH_SOFTMAX
from ai_journey.metal_fused_source import F1, F2, F3, STEP_VARIANTS, U0, MetalFusedError
from ai_journey.metal_fused_speedup import (
    BEST_RULE,
    FRACTION_RULE,
    HYPOTHESIS_RULE,
    RATIO_RULE,
    ratio_step,
    speedup,
    vendor_label,
)

COMPULSORY = 8 * 1000
MODELED = {U0: 16 * 1000, F1: 8 * 1000, F2: 8 * 1000, F3: 8 * 1000}


def side(name, samples):
    timing = DeviceTiming(name, 2, 1, len(samples), tuple(samples), TIMER)
    return side_rates(timing, COMPULSORY, MODELED.get(name))


def spread(median):
    return [median - 2, median - 1, median, median + 1, median + 2]


def point(medians, index=0):
    return {
        "index": index,
        "shape": {"rows": 10, "cols": 100},
        "sides": {name: side(name, spread(value)) for name, value in medians.items()},
    }


MEDIANS = {U0: 200, F1: 100, F2: 98, F3: 130, TORCH_SOFTMAX: 120}


def test_the_ratio_rule_needs_the_ratio_and_separated_quartiles():
    fast, slow = side(F1, spread(100)), side(U0, spread(200))
    close = side(F2, spread(101))

    assert ratio_step(fast, slow, 1.1) == {
        "first": F1,
        "second": U0,
        "ratio": 2.0,
        "iqr_separated": True,
        "verdict": "faster",
    }
    assert ratio_step(slow, fast, 1.1)["verdict"] == "slower"
    assert ratio_step(close, fast, 1.1)["verdict"] == "no_clear_difference"
    wide = side(F3, [10, 50, 105, 300, 400])
    assert ratio_step(fast, wide, 1.04)["iqr_separated"] is False
    assert ratio_step(fast, wide, 1.04)["verdict"] == "no_clear_difference"


def test_each_step_is_compared_with_its_base_with_u0_and_with_torch():
    section = speedup([point(MEDIANS)], 80.0, 1.1, "2.15.0")

    rows = {row["variant"]: row for row in section["points"][0]["rows"]}
    assert [row["base"] for row in rows.values()] == [TORCH_SOFTMAX, U0, F1, F2]
    assert rows[U0]["to_base"]["verdict"] == "slower"
    assert rows[U0]["to_u0"] is None
    assert rows[F1]["to_base"]["ratio"] == 2.0
    assert rows[F1]["to_base"]["verdict"] == "faster"
    assert rows[F2]["to_base"]["verdict"] == "no_clear_difference"
    assert rows[F2]["to_u0"]["ratio"] == 200 / 98
    assert rows[F3]["to_base"]["verdict"] == "slower"
    assert rows[F3]["to_torch"]["ratio"] == 120 / 130
    assert [rows[name]["held"] for name in STEP_VARIANTS] == [True, True, True, True]
    torch = section["points"][0]["torch"]
    assert torch["to_u0"]["verdict"] == "faster"
    assert section["points"][0]["best"]["variant"] == F2


def test_rates_report_both_fractions_of_the_ceiling():
    section = speedup([point(MEDIANS)], 80.0, 1.1, "2.15.0")

    rate = section["points"][0]["rows"][0]["rate"]
    effective = 2 * COMPULSORY / 200
    assert rate["effective_gb_per_second"] == effective
    assert rate["modeled_gb_per_second"] == 2 * MODELED[U0] / 200
    assert rate["effective_fraction_of_ceiling"] == effective / 80.0
    assert rate["modeled_fraction_of_ceiling"] == 2 * MODELED[U0] / 200 / 80.0
    torch_rate = section["points"][0]["torch"]["rate"]
    assert torch_rate["modeled_gb_per_second"] is None
    assert torch_rate["modeled_fraction_of_ceiling"] is None


def test_the_hypotheses_record_where_each_prediction_held():
    missed = {**MEDIANS, F1: 190}
    section = speedup([point(MEDIANS, 0), point(missed, 1)], 80.0, 1.1, "2.15.0")

    records = {record["variant"]: record for record in section["hypotheses"]}
    assert records[F1]["held_at"] == [0] and records[F1]["missed_at"] == [1]
    assert records[F2]["held_at"] == [0, 1] and records[F2]["missed_at"] == []
    assert records[U0]["held_at"] == [0, 1]
    assert section["vendor"] == vendor_label("2.15.0")
    assert (section["ratio_rule"], section["hypothesis_rule"]) == (
        RATIO_RULE,
        HYPOTHESIS_RULE,
    )
    assert (section["fraction_rule"], section["best_rule"]) == (
        FRACTION_RULE,
        BEST_RULE,
    )
    assert section["min_faster_ratio"] == 1.1 and section["ceiling_gb_per_second"] == 80


def test_a_step_that_did_not_run_gives_no_verdict_to_itself_or_the_next_step():
    medians = {name: value for name, value in MEDIANS.items() if name != F2}
    section = speedup([point(medians)], 80.0, 1.1, "2.15.0")

    rows = {row["variant"]: row for row in section["points"][0]["rows"]}
    assert rows[F2]["ran"] is False and rows[F2]["rate"] is None
    assert rows[F2]["held"] is None and rows[F3]["to_base"] is None
    assert rows[F3]["held"] is None and rows[F3]["to_u0"]["ratio"] == 200 / 130
    records = {record["variant"]: record for record in section["hypotheses"]}
    assert records[F2]["no_verdict_at"] == [0] and records[F3]["no_verdict_at"] == [0]


def test_a_point_without_torch_and_a_label_without_a_version_are_refused():
    with pytest.raises(MetalFusedError, match="torch.softmax side"):
        speedup([point({U0: 10})], 80.0, 1.1, "2.15.0")
    with pytest.raises(MetalFusedError, match="PyTorch version"):
        vendor_label("")
