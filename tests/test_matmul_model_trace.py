import pytest

from ai_journey import matmul_model_trace
from ai_journey.kernel_launch_buffer import to_float32
from ai_journey.kernel_launch_config import Index3
from ai_journey.matmul_model_access import (
    MatmulLaunch,
    WarpRef,
    warp_of_element,
    warp_refs,
)
from ai_journey.matmul_model_kernels import KERNELS, MAPPINGS, X_COLUMNS, X_ROWS
from ai_journey.matmul_model_reference import (
    MatmulModelError,
    matmul_inputs,
    reference_matmul,
)
from ai_journey.matmul_model_trace import TraceBuffer, trace_matches_model, trace_warp
from ai_journey.memory_access_tiling import MatmulShape

RAGGED = MatmulShape(m=37, n=45, k=29)
ALIGNED = MatmulShape(m=32, n=64, k=16)


def operands(shape):
    a, b = matmul_inputs(74, shape)
    return a, b, reference_matmul(a, b, shape)


@pytest.mark.parametrize("mapping", MAPPINGS)
@pytest.mark.parametrize("shape", [RAGGED, ALIGNED], ids=["ragged", "aligned"])
def test_the_kernel_code_reproduces_the_model_for_every_warp(shape, mapping):
    launch = MatmulLaunch(shape, (32, 8), mapping)
    a, b, c = operands(shape)

    assert all(trace_matches_model(launch, ref, a, b, c) for ref in warp_refs(launch))


def test_a_lane_loads_its_row_of_a_and_column_of_b_in_step_order():
    launch = MatmulLaunch(RAGGED, (32, 8), X_COLUMNS)
    a, b, c = operands(RAGGED)
    lanes = trace_warp(launch, warp_of_element(launch, 36, 44), a, b)

    # Row 36, column 32 + 3: A elements 36 * 29 + i, B elements i * 45 + 35.
    lane = lanes[3]
    assert lane.thread == Index3(3, 4)
    assert lane.a_loads == tuple(36 * 29 + i for i in range(29))
    assert lane.b_loads == tuple(i * 45 + 35 for i in range(29))
    assert lane.c_stores == ((36 * 45 + 35, c[36 * 45 + 35]),)
    # Lanes 13 to 31 hold columns 45 to 63: the guard stops them.
    assert all(not (x.a_loads or x.b_loads or x.c_stores) for x in lanes[13:])
    rows = MatmulLaunch(RAGGED, (32, 8), X_ROWS)
    first = trace_warp(rows, WarpRef(Index3(0, 0), 0), a, b)
    assert [lane.c_stores[0][0] for lane in first] == [row * 45 for row in range(32)]


def reversed_kernel(t, a, b, c, m, n, k):
    row = t.block_idx.y * t.block_dim.y + t.thread_idx.y
    col = t.block_idx.x * t.block_dim.x + t.thread_idx.x
    if row < m and col < n:
        total = 0.0
        for i in reversed(range(k)):
            total = to_float32(total + to_float32(a[row * k + i] * b[i * n + col]))
        c[row * n + col] = total


def transposed_b_kernel(t, a, b, c, m, n, k):
    row = t.block_idx.y * t.block_dim.y + t.thread_idx.y
    col = t.block_idx.x * t.block_dim.x + t.thread_idx.x
    if row < m and col < n:
        total = 0.0
        for i in range(k):
            total = to_float32(total + to_float32(a[row * k + i] * b[col * k + i]))
        c[row * n + col] = total


@pytest.mark.parametrize("kernel", [reversed_kernel, transposed_b_kernel])
def test_a_kernel_that_drifts_from_the_model_fails_its_trace(kernel, monkeypatch):
    shape = MatmulShape(m=4, n=40, k=3)
    launch = MatmulLaunch(shape, (32, 2), X_COLUMNS)
    a, b, c = operands(shape)
    ref = WarpRef(Index3(0, 0), 0)
    assert trace_matches_model(launch, ref, a, b, c)

    monkeypatch.setattr(matmul_model_trace, "KERNELS", {**KERNELS, X_COLUMNS: kernel})
    assert not trace_matches_model(launch, ref, a, b, c)


def test_trace_buffers_record_indices_and_reject_other_keys():
    buffer = TraceBuffer([0.5, 0.25])

    assert (buffer[1], buffer[7]) == (0.25, 0.0)
    buffer[0] = 1.0
    assert (buffer.loads, buffer.stores) == ([1, 7], [(0, 1.0)])
    with pytest.raises(TypeError, match="integer"):
        buffer[0.0]
    with pytest.raises(TypeError, match="integer"):
        buffer[True] = 1.0
    with pytest.raises(MatmulModelError, match="outside the launch"):
        trace_warp(
            MatmulLaunch(ALIGNED, (32, 8), X_ROWS), WarpRef(Index3(1, 0), 0), [], []
        )
