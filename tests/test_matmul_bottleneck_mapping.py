import pytest

from ai_journey.matmul_bottleneck_config import MappingRule, MatmulBottleneckError
from ai_journey.matmul_bottleneck_mapping import (
    NO_DIFFERENCE,
    NO_MODEL_REASON,
    NOT_SHOWN_BY_TIMING,
    WORSE_AS_MODELED,
    _verdict,
    compare_mappings,
    warp_model,
)
from ai_journey.matmul_metal_mapping import matmul_shape

RULE = MappingRule(slower_ratio_min=1.1)
SHAPE = matmul_shape(256, 256, 256)
GROUP = (32, 8)


def rates(columns, rows):
    return {"matmul_x_columns": columns, "matmul_x_rows": rows}


def test_x_over_rows_strides_a_and_c_while_x_over_columns_coalesces():
    columns = warp_model(SHAPE, GROUP, "x_columns")
    rows = warp_model(SHAPE, GROUP, "x_rows")
    k = SHAPE.k

    a, b, c = (columns["operands"][name] for name in "abc")
    assert a["classes"]["broadcast"] == k and a["sectors"] == k
    assert b["classes"]["coalesced"] == k and b["sectors"] == 4 * k
    assert c["classes"]["coalesced"] == 1 and c["sectors"] == 4
    a, b, c = (rows["operands"][name] for name in "abc")
    # Neighboring lanes read A one row, 4 * K bytes, apart: 32 sectors a step.
    assert a["classes"]["strided"] == k and a["sectors"] == 32 * k
    assert a["first_request"]["lane_stride_bytes"] == 4 * k
    assert b["classes"]["broadcast"] == k and b["sectors"] == k
    assert c["classes"]["strided"] == 1 and c["sectors"] == 32
    assert c["first_request"]["lane_stride_bytes"] == 4 * SHAPE.n
    assert (columns["sectors"], rows["sectors"]) == (5 * k + 4, 33 * k + 32)
    assert columns["active_lanes"] == rows["active_lanes"] == 32


def test_a_slower_x_rows_with_more_sectors_is_worse_as_modeled():
    comparison = compare_mappings(SHAPE, GROUP, 32, rates(300.0, 122.0), RULE)

    assert comparison["verdict"] == WORSE_AS_MODELED
    assert comparison["model_sector_ratio"] == (33 * 256 + 32) / (5 * 256 + 4)
    assert comparison["x_columns_to_x_rows_rate_ratio"] == 300.0 / 122.0
    assert comparison["model_predicts_x_rows_worse"] is True
    assert comparison["timing_shows_x_rows_slower"] is True
    assert comparison["probe_warp"] == {"block": [0, 0, 0], "warp": 0}


def test_timing_alone_never_claims_a_worse_mapping():
    # Equal rates: the model has a reason, but the timings show no effect.
    even = compare_mappings(SHAPE, GROUP, 32, rates(300.0, 290.0), RULE)
    assert even["verdict"] == NOT_SHOWN_BY_TIMING
    # Without a model reason, a slower x_rows claims nothing either.
    assert _verdict(False, True) == NO_MODEL_REASON
    assert _verdict(False, False) == NO_DIFFERENCE


def test_the_slower_ratio_is_inclusive():
    on_limit = compare_mappings(SHAPE, GROUP, 32, rates(110.0, 100.0), RULE)
    assert on_limit["timing_shows_x_rows_slower"] is True


def test_a_ragged_shape_still_models_a_full_first_warp():
    comparison = compare_mappings(
        matmul_shape(129, 65, 1000), GROUP, 32, rates(250.0, 180.0), RULE
    )
    warps = comparison["warps"]
    assert warps["x_columns"]["active_lanes"] == warps["x_rows"]["active_lanes"] == 32
    # B rows of 260 bytes do not align to sectors, so some steps take 5 sectors.
    assert warps["x_columns"]["operands"]["b"]["sectors"] > 4 * 1000
    assert comparison["verdict"] == WORSE_AS_MODELED


@pytest.mark.parametrize(
    ("group", "width", "message"),
    [
        ((16, 16), 32, "multiple of 32"),
        ((32, 8), 64, "SIMD width of 32"),
    ],
)
def test_the_model_refuses_geometry_it_does_not_describe(group, width, message):
    with pytest.raises(MatmulBottleneckError, match=message):
        compare_mappings(SHAPE, group, width, rates(1.0, 1.0), RULE)
