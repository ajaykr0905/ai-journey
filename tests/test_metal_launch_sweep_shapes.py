import pytest

from ai_journey.kernel_metal_geometry import (
    CheckedLaunch,
    KernelMetalError,
    PipelineLimits,
    vector_launch,
)
from ai_journey.metal_launch_sweep_shapes import (
    EXCEEDS_MAXIMUM,
    NOT_WHOLE_SIMD_GROUPS,
    MetalLaunchSweepError,
    ShapeAdmission,
    admit_shapes,
    checked_launch,
    group_shape,
    matmul_launch,
    shape_label,
    simd_footprint,
)

# The limits that every pipeline of the lab reported on the development host.
HOST = PipelineLimits(1024, 32, 0)
SWEPT = ((8, 8), (16, 16), (32, 32), (32, 8), (8, 32), (64, 4), (4, 64))


def test_the_swept_shapes_fit_the_host_and_split_into_whole_simd_groups():
    admissions = admit_shapes(SWEPT, HOST, requires_whole_simd_groups=True)

    assert all(admission.admitted for admission in admissions)
    table = {
        shape_label(item.group_size): (
            item.threads_per_group,
            item.footprint.simd_groups,
            item.footprint.rows,
            item.footprint.columns,
        )
        for item in admissions
    }
    # x runs over columns, so a shape narrower than the width of 32 puts
    # several output rows into one SIMD group.
    assert table == {
        "8x8": (64, 2, 4, 8),
        "16x16": (256, 8, 2, 16),
        "32x32": (1024, 32, 1, 32),
        "32x8": (256, 8, 1, 32),
        "8x32": (256, 8, 4, 8),
        "64x4": (256, 8, 1, 32),
        "4x64": (256, 8, 8, 4),
    }
    assert {item.footprint.idle_lanes for item in admissions} == {0}
    assert admissions[0].to_mapping() == {
        "group_size": [8, 8],
        "threads_per_group": 64,
        "admitted": True,
        "reason": None,
        "simd_groups": 2,
        "idle_lanes": 0,
        "rows_per_simd_group": 4,
        "columns_per_simd_group": 8,
    }


def test_a_pipeline_with_a_lower_limit_rejects_only_the_larger_shapes():
    admissions = admit_shapes(
        SWEPT, PipelineLimits(512, 32, 0), requires_whole_simd_groups=False
    )

    rejected = [item for item in admissions if not item.admitted]
    assert [(item.group_size, item.reason) for item in rejected] == [
        ((32, 32), EXCEEDS_MAXIMUM)
    ]
    assert rejected[0].to_mapping()["reason"] == EXCEEDS_MAXIMUM
    with pytest.raises(MetalLaunchSweepError, match="32x32 was rejected"):
        checked_launch(rejected[0], matmul_launch(64, 64, (32, 32)))


def test_whole_simd_groups_are_required_only_where_the_runtime_says_so():
    sizes = ((8,), (16,), (32,), (48,), (64,), (1024,))

    free = admit_shapes(sizes, HOST, requires_whole_simd_groups=False)
    strict = admit_shapes(sizes, HOST, requires_whole_simd_groups=True)

    assert all(item.admitted for item in free)
    assert [item.reason for item in strict] == [
        NOT_WHOLE_SIMD_GROUPS,
        NOT_WHOLE_SIMD_GROUPS,
        None,
        NOT_WHOLE_SIMD_GROUPS,
        None,
        None,
    ]
    # A group of 48 is one full SIMD group and one with 16 idle lanes.
    assert free[3].footprint.to_mapping() == {
        "simd_groups": 2,
        "idle_lanes": 16,
        "rows_per_simd_group": 1,
        "columns_per_simd_group": 32,
    }
    assert free[0].footprint.idle_lanes == 24
    # The maximum is checked first, so a shape that breaks both rules names it.
    both = ShapeAdmission((1000,), PipelineLimits(512, 32, 0), True)
    assert both.reason == EXCEEDS_MAXIMUM


def test_a_shape_that_does_not_divide_the_width_spans_rows_unevenly():
    footprint = simd_footprint((12, 4), 32)

    # SIMD group 0 holds rows 0, 1 and 2; SIMD group 1 holds the rest of
    # row 2 and all of row 3, with 16 idle lanes.
    assert footprint.to_mapping() == {
        "simd_groups": 2,
        "idle_lanes": 16,
        "rows_per_simd_group": 3,
        "columns_per_simd_group": 12,
    }
    assert simd_footprint((256,), 32).rows == 1
    assert simd_footprint((4, 4), 32).to_mapping() == {
        "simd_groups": 1,
        "idle_lanes": 16,
        "rows_per_simd_group": 4,
        "columns_per_simd_group": 4,
    }


def test_admitted_shapes_get_whole_threadgroup_grids_with_padding_threads():
    admission = ShapeAdmission((16, 16), HOST, False)

    launch = checked_launch(admission, matmul_launch(37, 53, (16, 16)))

    assert isinstance(launch, CheckedLaunch)
    geometry = launch.geometry
    # x covers the 53 columns and y the 37 rows.
    assert (geometry.extent, geometry.groups, geometry.threads) == (
        (53, 37),
        (4, 3),
        (64, 48),
    )
    assert geometry.padding_threads == 64 * 48 - 53 * 37
    assert geometry.launch_arguments() == {"threads": [64, 48], "group_size": [16, 16]}
    vector = checked_launch(ShapeAdmission((48,), HOST, False), vector_launch(1000, 48))
    assert (vector.geometry.threads, vector.idle_lanes_per_threadgroup) == ((1008,), 16)
    with pytest.raises(MetalLaunchSweepError, match="must use the admitted shape"):
        checked_launch(admission, matmul_launch(37, 53, (8, 32)))
    with pytest.raises(TypeError, match="ShapeAdmission"):
        checked_launch((16, 16), matmul_launch(37, 53, (16, 16)))


@pytest.mark.parametrize(
    ("value", "axes", "error", "message"),
    [
        ((0, 8), 2, KernelMetalError, "group_size x must be in"),
        ((8, True), 2, TypeError, "group_size y must be an integer"),
        ((8.0, 8), 2, TypeError, "must be an integer"),
        ([8, 8], 2, MetalLaunchSweepError, "tuple of 2 sizes"),
        ((8, 8, 1), None, MetalLaunchSweepError, "tuple of 1 or 2 sizes"),
        ((), None, MetalLaunchSweepError, "tuple of 1 or 2 sizes"),
        ((64, 32), 2, MetalLaunchSweepError, "2048 threads"),
        ((2048,), 1, KernelMetalError, "must be in"),
    ],
)
def test_shapes_must_be_strict_integer_tuples_within_the_metal_ceiling(
    value, axes, error, message
):
    with pytest.raises(error, match=message):
        group_shape(value, axes)


def test_admission_refuses_bad_sweeps_and_bad_parts():
    with pytest.raises(MetalLaunchSweepError, match="must not repeat"):
        admit_shapes(((8, 8), (8, 8)), HOST, requires_whole_simd_groups=False)
    with pytest.raises(MetalLaunchSweepError, match="1 to 16 shapes"):
        admit_shapes((), HOST, requires_whole_simd_groups=False)
    with pytest.raises(TypeError, match="PipelineLimits"):
        ShapeAdmission((8, 8), {"max_threads_per_threadgroup": 1024}, False)
    with pytest.raises(TypeError, match="must be a bool"):
        ShapeAdmission((8, 8), HOST, 1)
    with pytest.raises(KernelMetalError, match="m must be in"):
        matmul_launch(0, 8, (8, 8))
    with pytest.raises(KernelMetalError, match="n must be in"):
        matmul_launch(8, 4097, (8, 8))
    assert shape_label((256,)) == "256"
    assert issubclass(MetalLaunchSweepError, KernelMetalError)
