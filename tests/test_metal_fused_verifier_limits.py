"""Pin the GPU-only error fields that the fused softmax verifier cannot recompute.

A report keeps only a digest of each GPU output, not its values. So the CPU
verifier cannot rebuild the error of an output whose bits differ from the
rounded reference. This test keeps the list true: a resealed report with any one of them
changed still verifies.
"""

import copy
import json

import pytest

from ai_journey.memory_access_report import report_digest
from ai_journey.metal_fused_config import config_from_mapping
from ai_journey.metal_fused_verify import verify_report
from test_metal_fused_ceiling import recorded_ceiling
from test_metal_fused_recorded import RECORDED_CONFIG, RECORDED_REPORT_JSON

CHECK_FIELDS = ("max_abs_error", "max_bound_ratio", "differing_from_rounded_reference")
OUTPUT_FIELDS = (*CHECK_FIELDS, "max_row_sum_error", "max_row_sum_ratio")


def _reseal(report):
    unsigned = {key: value for key, value in report.items() if key != "report_sha256"}
    return {**unsigned, "report_sha256": report_digest(unsigned)}


def _first_check(report):
    point = report["benchmark"]["points"][0]
    return point["checks"][sorted(point["checks"])[0]]


def _first_differing_output(report):
    for case in report["correctness"]["cases"]:
        for output in case["outputs"]:
            if output["differing_from_rounded_reference"] > 0:
                return output
    raise AssertionError("the recorded report has no output whose bits differ")


def _smaller(value):
    return 1 if isinstance(value, int) else value / 2


@pytest.fixture(name="recorded")
def _recorded(tmp_path):
    return (
        json.loads(RECORDED_REPORT_JSON),
        config_from_mapping(RECORDED_CONFIG),
        recorded_ceiling(tmp_path),
    )


@pytest.mark.parametrize(
    ("pick", "field"),
    [(_first_check, field) for field in CHECK_FIELDS]
    + [(_first_differing_output, field) for field in OUTPUT_FIELDS],
)
def test_a_resealed_gpu_only_error_change_still_verifies(recorded, pick, field):
    report, config, ceiling = recorded
    changed = copy.deepcopy(report)
    entry = pick(changed)
    assert entry[field] != _smaller(entry[field])
    entry[field] = _smaller(entry[field])

    assert verify_report(_reseal(changed), config=config, ceiling=ceiling)
