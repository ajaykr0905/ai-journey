import pytest

from ai_journey.kernel_launch_buffer import (
    DeviceBuffer,
    OutOfBoundsAccess,
    ThreadRef,
)
from ai_journey.kernel_launch_config import Index3, launch_1d
from ai_journey.kernel_launch_kernels import (
    run_rgb_to_grayscale,
    run_vector_add,
    vector_add_unguarded_kernel,
)
from ai_journey.kernel_launch_reference import (
    reference_rgb_to_grayscale,
    reference_vector_add,
    rgb_image,
    vector_add_inputs,
)
from ai_journey.kernel_launch_simulator import LaunchOrder, launch
from ai_journey.simt_device import PINNED_ARCHITECTURES
from ai_journey.simt_indexing import coverage_1d, coverage_2d
from ai_journey.simt_launch import launch_for_elements, launch_for_image

AMPERE = PINNED_ARCHITECTURES["8.0"]
ORDERS = [LaunchOrder.canonical(), *(LaunchOrder.permuted(seed) for seed in (1, 2))]


def test_unguarded_vector_add_writes_past_a_ragged_extent():
    a, b = vector_add_inputs(73, 1000)
    run = run_vector_add(a, b, threads_per_block=256, guarded=False)
    log = run.result.log

    assert run.kernel == "vector_add_unguarded"
    # 4 blocks of 256 launch 1,024 threads for 1,000 elements.
    assert (log.out_of_bounds_writes, log.out_of_bounds_reads) == (24, 48)
    writes = [
        record for record in log.canonical_out_of_bounds if record.access == "write"
    ]
    assert [record.index for record in writes] == list(range(1000, 1024))
    assert writes[0] == OutOfBoundsAccess("out", "write", 1000, Index3(3), Index3(232))
    assert writes[-1] == OutOfBoundsAccess("out", "write", 1023, Index3(3), Index3(255))
    assert log.canonical_out_of_bounds[:3] == (
        OutOfBoundsAccess("a", "read", 1000, Index3(3), Index3(232)),
        OutOfBoundsAccess("b", "read", 1000, Index3(3), Index3(232)),
        writes[0],
    )
    # The dropped writes leave every in-bounds element equal to the reference.
    assert run.output == reference_vector_add(a, b)
    assert log.race_free


def test_unguarded_kernels_are_safe_only_on_block_multiples():
    a, b = vector_add_inputs(73, 1024)
    flat = run_vector_add(a, b, threads_per_block=256, guarded=False)
    image = rgb_image(73, 48, 32)
    square = run_rgb_to_grayscale(
        image, width=48, height=32, block=(16, 16), guarded=False
    )

    for run in (flat, square):
        assert run.result.log.memory_safe and run.result.log.race_free
    assert square.output == reference_rgb_to_grayscale(image, 48, 32)


def test_unguarded_grayscale_wraps_rows_and_writes_past_the_image():
    image = rgb_image(73, 40, 30)
    run = run_rgb_to_grayscale(
        image, width=40, height=30, block=(16, 16), guarded=False
    )
    log = run.result.log

    # The 3 x 2 grid covers 48 x 32 threads for 40 x 30 pixels.
    assert run.kernel == "rgb_to_grayscale_unguarded"
    assert log.out_of_bounds_writes == 104
    assert log.out_of_bounds_reads == 3 * 104
    # Row 30 starts at pixel 30 * 40 = 1200, one past the last pixel.
    first = next(r for r in log.canonical_out_of_bounds if r.access == "write")
    assert first == OutOfBoundsAccess(
        "gray", "write", 1200, Index3(0, 1), Index3(0, 14)
    )
    # Column 40 of row 0 computes pixel 40, which is column 0 of row 1.
    assert len(log.write_conflicts) == 232
    conflict = log.write_conflicts[0]
    assert (conflict.buffer, conflict.index) == ("gray", 40)
    assert conflict.writers == (
        ThreadRef(Index3(0, 0), Index3(0, 1)),
        ThreadRef(Index3(2, 0), Index3(8, 0)),
    )
    # Both writers read the same RGB pixel, so the race leaves correct bytes.
    assert run.output == reference_rgb_to_grayscale(image, 40, 30)
    assert log.read_write_conflicts == ()


@pytest.mark.parametrize(
    ("width", "height", "block"),
    [
        (40, 30, (16, 16)),
        (48, 32, (16, 16)),
        (5, 3, (16, 16)),
        (17, 9, (8, 4)),
        (1, 33, (16, 16)),
        (33, 1, (32, 1)),
    ],
)
def test_simulated_misuse_matches_the_simt_closed_forms(width, height, block):
    image = rgb_image(73, width, height)
    run = run_rgb_to_grayscale(
        image, width=width, height=height, block=block, guarded=False
    )
    coverage = coverage_2d(
        launch_for_image(
            AMPERE, width=width, height=height, block_x=block[0], block_y=block[1]
        ),
        width=width,
        height=height,
    )
    log = run.result.log

    assert log.out_of_bounds_writes == coverage.unguarded_out_of_bounds_writes
    extra_writes = sum(len(item.writers) - 1 for item in log.write_conflicts)
    assert extra_writes == coverage.unguarded_aliased_writes
    assert log.out_of_bounds_writes + extra_writes == coverage.masked_threads
    assert run.output == reference_rgb_to_grayscale(image, width, height)


@pytest.mark.parametrize("n", [1, 255, 257, 1000, 1024])
def test_one_dimensional_misuse_matches_the_simt_closed_form(n):
    a, b = vector_add_inputs(73, n)
    run = run_vector_add(a, b, threads_per_block=256, guarded=False)
    coverage = coverage_1d(
        launch_for_elements(AMPERE, elements=n, threads_per_block=256), elements=n
    )

    assert run.result.log.out_of_bounds_writes == (
        coverage.unguarded_out_of_bounds_writes
    )
    assert coverage.unguarded_aliased_writes == 0
    assert run.result.log.write_conflicts == ()


def test_misuse_records_do_not_depend_on_the_launch_order():
    image = rgb_image(73, 40, 30)
    logs = [
        run_rgb_to_grayscale(
            image, width=40, height=30, block=(16, 16), order=order, guarded=False
        ).result.log
        for order in ORDERS
    ]

    assert len({log.out_of_bounds for log in logs}) == len(ORDERS)
    assert len({log.canonical_out_of_bounds for log in logs}) == 1
    assert len({log.write_conflicts for log in logs}) == 1


def vector_add_beside_guard_kernel(t, a, b, out, guard, n):
    vector_add_unguarded_kernel(t, a, b, out, n)


def test_out_of_bounds_writes_never_reach_a_neighbouring_buffer():
    # guard is attached to the same launch, right after out, so a recorder
    # that let a write spill past the end of out would change it.
    config = launch_1d(10, 8)
    a = DeviceBuffer("a", "float32", [0.5] * 10)
    b = DeviceBuffer("b", "float32", [0.25] * 10)
    out = DeviceBuffer.zeros("out", "float32", 10)
    guard = DeviceBuffer("guard", "float32", [9.0] * 6)

    result = launch(vector_add_beside_guard_kernel, config, a, b, out, guard, 10)

    assert result.log.out_of_bounds_writes == 6
    assert {record.buffer for record in result.log.out_of_bounds} == {"a", "b", "out"}
    assert result.log.writer_threads == {"a": 0, "b": 0, "guard": 0, "out": 10}
    assert out.to_host() == (0.75,) * 10
    assert guard.to_host() == (9.0,) * 6


def test_the_guard_switch_takes_only_a_bool():
    with pytest.raises(TypeError, match="guarded must be a bool"):
        run_vector_add([0.5], [0.5], threads_per_block=32, guarded=0)
    with pytest.raises(TypeError, match="guarded must be a bool"):
        run_rgb_to_grayscale([0, 0, 0], width=1, height=1, block=(1, 1), guarded=None)
