import pytest

from ai_journey.kernel_launch_buffer import DeviceBuffer, to_float32, values_sha256
from ai_journey.kernel_launch_config import Dim3, LaunchConfig, LaunchLimitError
from ai_journey.kernel_launch_reference import compare_exact
from ai_journey.kernel_launch_simulator import LaunchOrder, launch
from ai_journey.matmul_model_kernels import (
    KERNELS,
    MAPPINGS,
    X_COLUMNS,
    X_ROWS,
    matmul_launch,
    run_naive_matmul,
    thread_extent,
)
from ai_journey.matmul_model_reference import (
    MatmulModelError,
    matmul_inputs,
    reference_matmul,
)
from ai_journey.memory_access_tiling import MatmulShape

ORDERS = [LaunchOrder.canonical(), LaunchOrder.permuted(1), LaunchOrder.permuted(2)]
RAGGED = MatmulShape(m=37, n=45, k=29)
ALIGNED = MatmulShape(m=32, n=64, k=16)


def test_a_hand_checked_product_runs_one_guarded_thread_per_element():
    shape = MatmulShape(m=2, n=3, k=2)
    a = [1.0, 2.0, 3.0, 4.0]
    b = [5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

    for mapping in MAPPINGS:
        run = run_naive_matmul(a, b, shape, block=(32, 2), mapping=mapping)
        assert run.output == (21.0, 24.0, 27.0, 47.0, 54.0, 61.0)
        assert run.kernel == f"naive_matmul_{mapping}"
        assert run.result.config == LaunchConfig(
            Dim3(1, 2 if mapping == X_ROWS else 1), Dim3(32, 2)
        )
        assert (run.useful_threads, run.idle_threads) == (6, run.result.threads_run - 6)
        assert run.result.log.counts() == {
            "reads": 24,
            "writes": 6,
            "out_of_bounds_reads": 0,
            "out_of_bounds_writes": 0,
            "write_conflicts": 0,
            "read_write_conflicts": 0,
        }


def test_mappings_swap_the_axes_that_the_grid_covers():
    # x runs over the 45 columns or over the 37 rows; y covers the other axis.
    assert thread_extent(RAGGED, X_COLUMNS) == Dim3(45, 37)
    assert thread_extent(RAGGED, X_ROWS) == Dim3(37, 45)
    assert matmul_launch(RAGGED, (32, 8), X_COLUMNS) == LaunchConfig(
        Dim3(2, 5), Dim3(32, 8)
    )
    assert matmul_launch(RAGGED, (32, 8), X_ROWS) == LaunchConfig(
        Dim3(2, 6), Dim3(32, 8)
    )
    for mapping in MAPPINGS:
        config = matmul_launch(ALIGNED, (32, 8), mapping)
        assert config.launched_threads == ALIGNED.m * ALIGNED.n
        assert config.threads_inside(thread_extent(ALIGNED, mapping)) == 2048
    ragged = matmul_launch(RAGGED, (32, 8), X_ROWS)
    assert ragged.launched_threads - ragged.threads_inside(Dim3(37, 45)) == 3072 - 1665


@pytest.mark.parametrize("mapping", MAPPINGS)
@pytest.mark.parametrize(
    "shape",
    [
        RAGGED,
        ALIGNED,
        MatmulShape(1, 1, 1),
        MatmulShape(33, 1, 3),
        MatmulShape(1, 33, 5),
    ],
    ids=lambda shape: f"{shape.m}x{shape.n}x{shape.k}",
)
def test_both_mappings_match_the_reference_bit_for_bit_in_every_order(shape, mapping):
    a, b = matmul_inputs(74, shape)
    expected = reference_matmul(a, b, shape)
    elements = shape.m * shape.n
    digests = set()
    for order in ORDERS:
        run = run_naive_matmul(a, b, shape, block=(32, 8), mapping=mapping, order=order)
        log = run.result.log
        assert compare_exact("float32", expected, run.output).equal
        assert log.memory_safe and log.race_free
        assert run.useful_threads == elements
        assert run.idle_threads == run.result.config.launched_threads - elements
        assert log.counts()["reads"] == 2 * elements * shape.k
        assert log.counts()["writes"] == elements
        digests.add(run.output_sha256)
    assert digests == {values_sha256("float32", expected)}


def row_guard_only_kernel(t, a, b, c, m, n, k):
    row = t.block_idx.y * t.block_dim.y + t.thread_idx.y
    col = t.block_idx.x * t.block_dim.x + t.thread_idx.x
    if row < m:
        total = 0.0
        for i in range(k):
            total = to_float32(total + to_float32(a[row * k + i] * b[i * n + col]))
        c[row * n + col] = total


def col_guard_only_kernel(t, a, b, c, m, n, k):
    row = t.block_idx.y * t.block_dim.y + t.thread_idx.y
    col = t.block_idx.x * t.block_dim.x + t.thread_idx.x
    if col < n:
        total = 0.0
        for i in range(k):
            total = to_float32(total + to_float32(a[row * k + i] * b[i * n + col]))
        c[row * n + col] = total


def launch_with(kernel, shape):
    a, b = matmul_inputs(74, shape)
    c = DeviceBuffer.zeros("c", "float32", shape.m * shape.n)
    config = matmul_launch(shape, (32, 2), X_COLUMNS)
    buffers = DeviceBuffer("a", "float32", a), DeviceBuffer("b", "float32", b), c
    return launch(kernel, config, *buffers, shape.m, shape.n, shape.k).log


def test_a_guard_on_one_axis_misuses_memory_on_a_ragged_extent():
    # 5 x 35 elements in 32 x 2 blocks launch 64 columns and 6 rows.
    shape = MatmulShape(m=5, n=35, k=3)
    rows_only = launch_with(row_guard_only_kernel, shape)
    cols_only = launch_with(col_guard_only_kernel, shape)

    # Columns 35 to 63 of rows 0 to 4: the last B row reads past B, rows 0 to
    # 3 write into the next row of C, and row 4 writes past C.
    assert rows_only.counts() == {
        "reads": 2 * 5 * 64 * 3 - 145,
        "writes": 5 * 64 - 29,
        "out_of_bounds_reads": 145,
        "out_of_bounds_writes": 29,
        "write_conflicts": 4 * 29,
        "read_write_conflicts": 0,
    }
    assert {record.buffer for record in rows_only.out_of_bounds} == {"b", "c"}
    # Row 5 of columns 0 to 34 reads past A and writes past C.
    assert cols_only.counts() == {
        "reads": 2 * 6 * 35 * 3 - 105,
        "writes": 5 * 35,
        "out_of_bounds_reads": 105,
        "out_of_bounds_writes": 35,
        "write_conflicts": 0,
        "read_write_conflicts": 0,
    }
    assert {record.buffer for record in cols_only.out_of_bounds} == {"a", "c"}
    for kernel in (row_guard_only_kernel, col_guard_only_kernel):
        aligned = launch_with(kernel, MatmulShape(m=4, n=64, k=3))
        assert aligned.memory_safe and aligned.race_free


def test_kernels_and_run_helper_reject_bad_inputs():
    a, b = matmul_inputs(74, MatmulShape(2, 2, 2))

    assert set(KERNELS) == set(MAPPINGS) == {"x_columns", "x_rows"}
    with pytest.raises(MatmulModelError, match="m \\* k values"):
        run_naive_matmul(a[:3], b, MatmulShape(2, 2, 2), block=(32, 1))
    with pytest.raises(MatmulModelError, match="mapping must be one of"):
        run_naive_matmul(a, b, MatmulShape(2, 2, 2), block=(32, 1), mapping="x")
    with pytest.raises(MatmulModelError, match="block_x, block_y"):
        run_naive_matmul(a, b, MatmulShape(2, 2, 2), block=[32, 1])
    with pytest.raises(LaunchLimitError, match="max_threads_per_block"):
        run_naive_matmul(a, b, MatmulShape(2, 2, 2), block=(64, 32))
    with pytest.raises(TypeError, match="MatmulShape"):
        thread_extent((2, 2, 2), X_COLUMNS)
