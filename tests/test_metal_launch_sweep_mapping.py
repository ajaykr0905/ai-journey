import pytest

from ai_journey import metal_launch_sweep_mapping
from ai_journey.kernel_launch_buffer import to_float32, values_sha256
from ai_journey.kernel_metal_geometry import ceil_div
from ai_journey.metal_launch_sweep_mapping import (
    MAPPING_CASE,
    check_mapping,
    exact_product,
    mapping_inputs,
)
from ai_journey.metal_launch_sweep_shapes import MetalLaunchSweepError

SWEPT = ((8, 8), (16, 16), (32, 32), (32, 8), (8, 32), (64, 4), (4, 64))


def test_every_swept_shape_writes_each_output_once_and_stays_in_bounds():
    m, n, k = MAPPING_CASE
    expected = values_sha256("float32", exact_product())

    for group in SWEPT:
        check = check_mapping(group, order_seed=74)
        gx, gy = group
        launched = ceil_div(n, gx) * gx * ceil_div(m, gy) * gy
        assert check.to_mapping() == {
            "group_size": list(group),
            "order_seed": 74,
            "launched_threads": launched,
            "writer_threads": m * n,
            "reads": 2 * m * n * k,
            "writes": m * n,
            "out_of_bounds": 0,
            "write_conflicts": 0,
            "read_write_conflicts": 0,
            "output_sha256": expected,
            "exact": True,
        }, group
        # Prime extents leave padding threads on both axes of each shape.
        assert launched > m * n


def test_the_case_is_exact_in_float32_and_has_prime_extents():
    m, n, k = MAPPING_CASE
    a, b = mapping_inputs()

    assert (len(a), len(b)) == (m * k, k * n)
    assert all(value == to_float32(value) and abs(value) <= 3 for value in a + b)
    assert all(abs(value) <= 3 * 3 * k for value in exact_product())
    for extent in (m, n):
        assert all(extent % divisor for divisor in range(2, extent))


def unguarded_columns(t, a, b, c, m, n, k):
    column = t.block_idx.x * t.block_dim.x + t.thread_idx.x
    row = t.block_idx.y * t.block_dim.y + t.thread_idx.y
    if row < m:
        total = 0.0
        for i in range(k):
            total = to_float32(total + to_float32(a[row * k + i] * b[i * n + column]))
        c[row * n + column] = total


def x_over_rows(t, a, b, c, m, n, k):
    row = t.block_idx.x * t.block_dim.x + t.thread_idx.x
    column = t.block_idx.y * t.block_dim.y + t.thread_idx.y
    if row < n and column < m:
        total = 0.0
        for i in range(k):
            total = to_float32(total + to_float32(a[row * k + i] * b[i * n + column]))
        c[row * n + column] = total


@pytest.mark.parametrize("kernel", [unguarded_columns, x_over_rows])
def test_a_broken_guard_or_a_swapped_mapping_stops_the_check(monkeypatch, kernel):
    monkeypatch.setattr(metal_launch_sweep_mapping, "matmul_thread", kernel)

    # Dropping the column guard makes padding threads write into the next
    # row; swapping the axes reads and writes the wrong elements.
    with pytest.raises(MetalLaunchSweepError, match="broke its mapping"):
        check_mapping((16, 16), order_seed=74)


def test_the_check_refuses_bad_shapes_and_seeds():
    with pytest.raises(MetalLaunchSweepError, match="tuple of 2 sizes"):
        check_mapping((256,), order_seed=74)
    with pytest.raises(TypeError, match="order_seed must be an integer"):
        check_mapping((8, 8), order_seed=True)
