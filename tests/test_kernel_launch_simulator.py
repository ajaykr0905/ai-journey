import itertools
from hashlib import sha256

import pytest

from ai_journey.kernel_launch_buffer import DeviceBuffer
from ai_journey.kernel_launch_config import (
    Dim3,
    Index3,
    KernelLaunchError,
    LaunchConfig,
    launch_1d,
)
from ai_journey.kernel_launch_simulator import (
    MAX_SIMULATED_THREADS,
    LaunchOrder,
    ThreadContext,
    launch,
    order_sha256,
    thread_order,
)

# A 2 x 2 grid of 4 x 2 blocks: 32 threads.
SMALL = LaunchConfig(Dim3(2, 2), Dim3(4, 2))
PERMUTED = [LaunchOrder.permuted(seed) for seed in (1, 2, 3, 73)]


def visit(config, order=None):
    visited = []

    def kernel(t):
        visited.append(t)

    result = launch(kernel, config, order=order)
    return visited, result


def test_canonical_order_runs_blocks_then_threads_x_fastest():
    visited, result = visit(SMALL)

    expected = [
        ThreadContext(Index3(bx, by), Index3(tx, ty), SMALL.block, SMALL.grid)
        for by, bx in itertools.product(range(2), range(2))
        for ty, tx in itertools.product(range(2), range(4))
    ]
    assert visited == expected
    assert result.threads_run == 32
    assert result.order == LaunchOrder.canonical()
    assert result.order_sha256 == order_sha256(range(32))
    assert all(t.block_dim == Dim3(4, 2) and t.grid_dim == Dim3(2, 2) for t in visited)


@pytest.mark.parametrize("order", PERMUTED, ids=lambda order: f"seed{order.seed}")
def test_every_permuted_order_runs_each_thread_exactly_once(order):
    visited, result = visit(SMALL, order)
    canonical, _ = visit(SMALL)

    def linear_id(t):
        return SMALL.thread_id(t.block_idx, t.thread_idx)

    assert len(visited) == 32
    assert sorted(visited, key=linear_id) == canonical
    assert visited != canonical
    assert result.order == order
    assert result.order_sha256 == order_sha256(thread_order(SMALL, order))


def test_permutations_are_pinned_seeded_and_distinct():
    # Pinned so a change to the SHA-256 stream cannot pass unnoticed.
    assert thread_order(SMALL, LaunchOrder.permuted(1)) == (
        1, 18, 9, 21, 10, 23, 4, 31, 17, 26, 12, 29, 15, 30, 19, 3,
        16, 8, 22, 13, 6, 25, 2, 14, 24, 11, 7, 27, 5, 28, 20, 0,
    )  # fmt: skip
    orders = {thread_order(SMALL, order) for order in PERMUTED}
    assert len(orders) == len(PERMUTED)
    for order in PERMUTED:
        assert thread_order(SMALL, order) == thread_order(SMALL, order)
        assert sorted(thread_order(SMALL, order)) == list(range(32))
    assert thread_order(LaunchConfig(Dim3(1), Dim3(1)), PERMUTED[0]) == (0,)
    assert (
        order_sha256([1, 2])
        == sha256((1).to_bytes(8, "big") + (2).to_bytes(8, "big")).hexdigest()
    )


def copy_kernel(t, source, out, n):
    i = t.block_idx.x * t.block_dim.x + t.thread_idx.x
    if i < n:
        out[i] = source[i]


def test_a_race_free_kernel_gives_identical_results_in_every_order():
    config = launch_1d(50, 16)
    outputs = set()
    for order in (LaunchOrder.canonical(), *PERMUTED):
        source = DeviceBuffer("source", "uint8", [(7 * i) % 256 for i in range(50)])
        out = DeviceBuffer.zeros("out", "uint8", 50)
        result = launch(copy_kernel, config, source, out, 50, order=order)
        assert result.log.race_free and result.log.memory_safe
        assert result.log.writer_threads["out"] == 50
        outputs.add(out.to_host())
    assert outputs == {tuple((7 * i) % 256 for i in range(50))}


def last_writer_kernel(t, out):
    out[0] = t.block_idx.x * t.block_dim.x + t.thread_idx.x


def test_a_write_race_is_recorded_and_its_result_depends_on_the_order():
    config = launch_1d(64, 32)
    finals = {}
    for order in (LaunchOrder.canonical(), *PERMUTED):
        out = DeviceBuffer.zeros("out", "uint8", 1)
        result = launch(last_writer_kernel, config, out, order=order)
        finals[order] = out.to_host()[0]
        (conflict,) = result.log.write_conflicts
        assert conflict.index == 0
        assert len(conflict.writers) == 64
        assert conflict.writers[0].thread == Index3(0)
        assert not result.log.race_free
    assert finals[LaunchOrder.canonical()] == 63
    assert len(set(finals.values())) > 1


def shift_kernel(t, x, n):
    i = t.block_idx.x * t.block_dim.x + t.thread_idx.x
    if i < n:
        x[i] = x[i + 1] + 1


def test_a_read_write_race_is_recorded_and_changes_the_result():
    config = launch_1d(8, 8)
    results = {}
    for order in (LaunchOrder.canonical(), *PERMUTED):
        x = DeviceBuffer.zeros("x", "uint8", 9)
        result = launch(shift_kernel, config, x, 8, order=order)
        results[order] = x.to_host()
        assert result.log.write_conflicts == ()
        assert [item.index for item in result.log.read_write_conflicts] == list(
            range(1, 8)
        )
    assert results[LaunchOrder.canonical()] == (1,) * 8 + (0,)
    assert len(set(results.values())) > 1


def test_launch_refuses_unsafe_arguments_and_return_values():
    out = DeviceBuffer.zeros("out", "uint8", 1)

    with pytest.raises(TypeError, match="callable"):
        launch("kernel", SMALL)
    with pytest.raises(TypeError, match="LaunchConfig"):
        launch(copy_kernel, (1, 1))
    for argument in ([0], True, "n", None):
        with pytest.raises(TypeError, match="DeviceBuffer values or int or float"):
            launch(copy_kernel, SMALL, argument)
    with pytest.raises(TypeError, match="LaunchOrder"):
        launch(copy_kernel, SMALL, order="canonical")
    with pytest.raises(KernelLaunchError, match="returns nothing"):
        launch(lambda t, out: 1, SMALL, out)
    assert out.to_host() == (0,)


def test_a_failing_kernel_detaches_every_buffer():
    out = DeviceBuffer.zeros("out", "uint8", 4)

    def failing(t, out):
        out[0] = 1
        raise RuntimeError("kernel bug")

    with pytest.raises(RuntimeError, match="kernel bug"):
        launch(failing, SMALL, out)
    assert out.to_host() == (1, 0, 0, 0)
    launch(copy_kernel, SMALL, out, out, 4)


def test_oversized_launches_are_refused_before_any_thread_runs():
    calls = []
    config = launch_1d(MAX_SIMULATED_THREADS + 1, 1024)

    with pytest.raises(KernelLaunchError, match="simulator runs at most 1048576"):
        launch(lambda t: calls.append(t), config)
    assert calls == []


@pytest.mark.parametrize(
    ("kind", "seed", "error", "message"),
    [
        ("canonical", 1, KernelLaunchError, "takes no seed"),
        ("permuted", None, TypeError, "order seed must be an integer"),
        ("permuted", -1, KernelLaunchError, "order seed must be between 0"),
        ("permuted", True, TypeError, "order seed must be an integer"),
        ("reversed", None, KernelLaunchError, "order kind must be"),
    ],
)
def test_orders_reject_ambiguous_definitions(kind, seed, error, message):
    with pytest.raises(error, match=message):
        LaunchOrder(kind, seed)


def test_orders_and_contexts_round_trip_and_stay_typed():
    for order in (LaunchOrder.canonical(), *PERMUTED):
        assert LaunchOrder.from_mapping(order.to_mapping()) == order
    with pytest.raises(KernelLaunchError, match="exactly kind and seed"):
        LaunchOrder.from_mapping({"kind": "canonical"})
    with pytest.raises(TypeError, match="two Index3 and two Dim3"):
        ThreadContext((0, 0, 0), Index3(0), Dim3(1), Dim3(1))
