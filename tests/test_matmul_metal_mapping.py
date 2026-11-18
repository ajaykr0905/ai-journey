import itertools

import pytest

from ai_journey.kernel_metal_geometry import KernelMetalError
from ai_journey.matmul_metal_mapping import (
    MAPPINGS,
    MAX_DIMENSION,
    X_COLUMNS,
    X_ROWS,
    AccessSteps,
    MatmulMetalError,
    access_steps,
    grid_extent,
    matmul_launch,
    matmul_shape,
    output_element,
)
from ai_journey.memory_access_tiling import MatmulShape


def a_index(row, column, step, shape):
    return row * shape.k + step


def b_index(row, column, step, shape):
    return step * shape.n + column


def c_index(row, column, step, shape):
    return row * shape.n + column


def test_shapes_are_bounded_true_integers():
    shape = matmul_shape(17, 33, 9)

    assert (shape.m, shape.n, shape.k, shape.flops) == (17, 33, 9, 2 * 17 * 33 * 9)
    assert matmul_shape(MAX_DIMENSION, MAX_DIMENSION, MAX_DIMENSION).m == 8192
    for bad in ((0, 1, 1), (1, MAX_DIMENSION + 1, 1), (1, 1, -3)):
        with pytest.raises(KernelMetalError, match="must be in"):
            matmul_shape(*bad)
    for bad in ((True, 1, 1), (1, 2.0, 1), (1, 1, "4")):
        with pytest.raises(TypeError, match="must be an integer"):
            matmul_shape(*bad)


def test_x_walks_columns_or_rows_and_the_same_group_serves_both():
    shape = matmul_shape(17, 33, 9)

    assert grid_extent(X_COLUMNS, shape) == (33, 17)
    assert grid_extent(X_ROWS, shape) == (17, 33)
    columns = matmul_launch(X_COLUMNS, shape, (32, 8))
    rows = matmul_launch(X_ROWS, shape, (32, 8))
    assert (columns.groups, columns.threads, columns.padding_threads) == (
        (2, 3),
        (64, 24),
        64 * 24 - 17 * 33,
    )
    assert (rows.groups, rows.threads, rows.padding_threads) == (
        (1, 5),
        (32, 40),
        32 * 40 - 17 * 33,
    )
    assert columns.active_threads == rows.active_threads == 17 * 33
    assert columns.launch_arguments() == {"threads": [64, 24], "group_size": [32, 8]}


@pytest.mark.parametrize("mapping", MAPPINGS)
def test_each_active_thread_owns_one_distinct_output(mapping):
    shape = matmul_shape(5, 7, 3)
    width, height = grid_extent(mapping, shape)

    owned = [
        output_element(mapping, x, y)
        for x, y in itertools.product(range(width), range(height))
    ]

    assert sorted(owned) == sorted(itertools.product(range(5), range(7)))


@pytest.mark.parametrize("mapping", MAPPINGS)
def test_access_steps_follow_from_the_kernel_index_formulas(mapping):
    shape = matmul_shape(6, 10, 4)
    steps = access_steps(mapping, shape)

    for formula, (x_step, y_step) in (
        (a_index, steps.a),
        (b_index, steps.b),
        (c_index, steps.c),
    ):
        for x, y, step in itertools.product(range(3), range(3), range(2)):
            here = formula(*output_element(mapping, x, y), step, shape)
            right = formula(*output_element(mapping, x + 1, y), step, shape)
            below = formula(*output_element(mapping, x, y + 1), step, shape)
            assert (right - here, below - here) == (x_step, y_step)


def test_x_over_columns_reads_consecutive_b_and_x_over_rows_strided_a():
    shape = matmul_shape(4096, 1024, 512)

    assert access_steps(X_COLUMNS, shape) == AccessSteps(
        X_COLUMNS, a=(0, 512), b=(1, 0), c=(1, 1024)
    )
    assert access_steps(X_ROWS, shape) == AccessSteps(
        X_ROWS, a=(512, 0), b=(0, 1), c=(1024, 1)
    )
    assert access_steps(X_ROWS, shape).to_mapping() == {
        "mapping": "x_rows",
        "a_step_bytes": {"x": 2048, "y": 0},
        "b_step_bytes": {"x": 0, "y": 4},
        "c_step_bytes": {"x": 4096, "y": 4},
    }


def test_unknown_mappings_shapes_and_groups_are_refused():
    shape = matmul_shape(4, 4, 4)

    for call in (
        lambda: grid_extent("x_diagonal", shape),
        lambda: output_element("x_diagonal", 0, 0),
        lambda: access_steps("X_ROWS", shape),
    ):
        with pytest.raises(MatmulMetalError, match="mapping must be one of"):
            call()
    with pytest.raises(MatmulMetalError, match=r"an \(x, y\) tuple"):
        matmul_launch(X_COLUMNS, shape, [32, 8])
    with pytest.raises(TypeError, match="MatmulShape"):
        grid_extent(X_COLUMNS, (4, 4, 4))
    with pytest.raises(KernelMetalError, match="must be in"):
        grid_extent(X_COLUMNS, MatmulShape(4, MAX_DIMENSION + 1, 4))
    with pytest.raises(KernelMetalError, match="1024-thread ceiling"):
        matmul_launch(X_ROWS, shape, (64, 32))
