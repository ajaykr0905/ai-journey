import pytest

from ai_journey.kernel_metal_geometry import (
    MAX_GRID_THREADS_PER_AXIS,
    CheckedLaunch,
    KernelMetalError,
    LaunchGeometry,
    PipelineLimits,
    ceil_div,
    image_launch,
    vector_launch,
)

# What compile_shader reported for both kernels on the development host.
RECORDED_LIMITS = PipelineLimits(
    max_threads_per_threadgroup=1024,
    thread_execution_width=32,
    static_threadgroup_memory_bytes=0,
)


def test_ceil_div_rounds_up_only_when_a_remainder_is_left():
    counts = (1, 255, 256, 257, 512, 513)

    assert [ceil_div(n, 256) for n in counts] == [1, 1, 1, 2, 2, 3]


def test_a_vector_launch_rounds_the_grid_up_to_whole_threadgroups():
    geometry = vector_launch(1000, 256)

    assert geometry.groups == (4,)
    assert geometry.threads == (1024,)
    assert geometry.padding_threads == 24
    assert geometry.launch_arguments() == {"threads": 1024, "group_size": 256}
    assert geometry.to_mapping() == {
        "extent": [1000],
        "group_size": [256],
        "groups": [4],
        "threads": [1024],
        "threads_per_group": 256,
        "launched_threads": 1024,
        "active_threads": 1000,
        "padding_threads": 24,
    }
    exact = vector_launch(1024, 256)
    assert (exact.groups, exact.padding_threads) == ((4,), 0)
    single = vector_launch(1, 256)
    assert (single.threads, single.padding_threads) == ((256,), 255)


def test_an_image_launch_puts_columns_on_x_and_rows_on_y():
    geometry = image_launch(17, 33, (16, 16))

    assert geometry.extent == (33, 17)
    assert geometry.groups == (3, 2)
    assert geometry.threads == (48, 32)
    assert geometry.launched_threads == 1536
    assert geometry.active_threads == 561
    assert geometry.padding_threads == 975
    assert geometry.launch_arguments() == {"threads": [48, 32], "group_size": [16, 16]}
    wide = image_launch(16, 40, (32, 8))
    assert (wide.groups, wide.threads) == ((2, 2), (64, 16))


def test_a_checked_launch_splits_threadgroups_into_simd_groups():
    full = CheckedLaunch(image_launch(64, 64, (16, 16)), RECORDED_LIMITS)
    partial = CheckedLaunch(vector_launch(100, 48), RECORDED_LIMITS)

    assert full.simd_groups_per_threadgroup == 8
    assert full.idle_lanes_per_threadgroup == 0
    assert partial.simd_groups_per_threadgroup == 2
    assert partial.idle_lanes_per_threadgroup == 16
    assert partial.to_mapping()["idle_lanes_per_threadgroup"] == 16
    assert partial.to_mapping()["padding_threads"] == 44


def test_launches_beyond_the_pipeline_limit_fail_before_they_run():
    small = PipelineLimits(
        max_threads_per_threadgroup=512,
        thread_execution_width=32,
        static_threadgroup_memory_bytes=0,
    )

    with pytest.raises(KernelMetalError, match="this pipeline's limit of 512"):
        CheckedLaunch(vector_launch(4096, 1024), small)
    with pytest.raises(KernelMetalError, match="this pipeline's limit of 512"):
        CheckedLaunch(image_launch(64, 64, (32, 32)), small)
    admitted = CheckedLaunch(image_launch(64, 64, (32, 16)), small)
    assert admitted.geometry.groups == (2, 4)
    with pytest.raises(TypeError, match="LaunchGeometry"):
        CheckedLaunch((4096,), small)


@pytest.mark.parametrize(
    ("extent", "group_size", "error", "message"),
    [
        ((0,), (256,), KernelMetalError, r"extent\[0\]"),
        ((16,), (0,), KernelMetalError, r"group_size\[0\]"),
        ((16,), (2048,), KernelMetalError, r"group_size\[0\]"),
        ((64, 64), (64, 32), KernelMetalError, "1024-thread ceiling"),
        ((16, 16, 16), (4, 4, 4), KernelMetalError, "1 or 2 axes"),
        ((16, 16), (16,), KernelMetalError, "same 1 or 2 axes"),
        ([16], [16], KernelMetalError, "tuples"),
        ((16.0,), (16,), TypeError, "integer"),
        ((True,), (16,), TypeError, "integer"),
        ((MAX_GRID_THREADS_PER_AXIS,), (256,), KernelMetalError, "grid threads"),
    ],
)
def test_invalid_geometry_is_rejected(extent, group_size, error, message):
    with pytest.raises(error, match=message):
        LaunchGeometry(extent=extent, group_size=group_size)


def test_pipeline_limits_reject_values_the_api_cannot_report():
    for fields, message in (
        ((0, 32, 0), "max_threads_per_threadgroup"),
        ((2048, 32, 0), "max_threads_per_threadgroup"),
        ((256, 512, 0), "thread_execution_width"),
        ((1024, 32, -1), "static_threadgroup_memory_bytes"),
    ):
        with pytest.raises(KernelMetalError, match=message):
            PipelineLimits(*fields)
    with pytest.raises(TypeError, match="integer"):
        PipelineLimits(1024.0, 32, 0)
    assert RECORDED_LIMITS.to_mapping() == {
        "max_threads_per_threadgroup": 1024,
        "thread_execution_width": 32,
        "static_threadgroup_memory_bytes": 0,
    }
    with pytest.raises(KernelMetalError, match=r"\(x, y\) tuple"):
        image_launch(16, 16, [16, 16])
