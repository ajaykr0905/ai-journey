"""CPU tests of the Day 78 CUDA schedule simulator. No test compiles or runs CUDA.

``metal_r1``, ``metal_r2``, ``metal_r3`` and ``metal_combine`` follow the MSL
text of ``gpu_reduce_source`` thread by thread, in plain Python, apart from
the vectorized simulator. They run on float32 values and on symbolic values,
so a test compares the bits and the tree of additions.
"""

import json
from pathlib import Path

import numpy
import pytest

from ai_journey.gpu_reduce_bound import benchmark_values, case_values
from ai_journey.gpu_reduce_config import load_config
from ai_journey.gpu_reduce_cuda import (
    COMBINE_BARRIERS,
    PINNED_KERNELS,
    R1_SHARED,
    WARP_SHUFFLE,
    GpuReduceCudaError,
)
from ai_journey.gpu_reduce_cuda_sim import (
    MAX_TREE_ELEMENTS,
    MAX_TREE_PARTIALS,
    combine_tree,
    predict,
    simulate,
    summation_tree,
)
from ai_journey.gpu_reduce_source import R1, R2, R3
from test_gpu_reduce_recorded import RECORDED_CONFIG, RECORDED_REPORT_JSON

ROOT = Path(__file__).resolve().parents[1]
SMALL_SIZES = (1, 31, 33, 255, 256, 257, 511, 512, 513, 1025, 4099)


class Sym:
    """A symbolic value: a sum is the pair of the two trees."""

    def __init__(self, tree):
        self.tree = tree

    def __add__(self, other):
        return Sym((self.tree, other.tree))


def metal_combine(partials, zero):
    """reduce_combine: four accumulators per thread, the join, then the tree."""

    count, shared = len(partials), []
    for t in range(1024):
        a0 = a1 = a2 = a3 = zero
        e = t
        while e + 3072 < count:
            a0 = a0 + partials[e]
            a1 = a1 + partials[e + 1024]
            a2 = a2 + partials[e + 2048]
            a3 = a3 + partials[e + 3072]
            e += 4096
        while e < count:
            a0 = a0 + partials[e]
            e += 1024
        shared.append((a0 + a1) + (a2 + a3))
    return tree(shared, 1)[0]


def tree(shared, stop):
    """Sequential addressing: threads t < s add shared[t + s], s halves to stop."""

    s = len(shared) // 2
    while s >= stop:
        for t in range(s):
            shared[t] = shared[t] + shared[t + s]
        s //= 2
    return shared


def metal_r1(x, zero):
    partials = []
    for group in range(-(-len(x) // 256)):
        index = [group * 256 + t for t in range(256)]
        shared = [x[i] if i < len(x) else zero for i in index]
        partials.append(tree(shared, 1)[0])
    return metal_combine(partials, zero)


def first_add(x, zero, group):
    """The R2 load: each thread adds two elements 256 apart."""

    shared = []
    for t in range(256):
        i = group * 512 + t
        first = x[i] if i < len(x) else zero
        second = x[i + 256] if i + 256 < len(x) else zero
        shared.append(first + second)
    return shared


def metal_r2(x, zero):
    groups = range(-(-len(x) // 512))
    partials = [tree(first_add(x, zero, group), 1)[0] for group in groups]
    return metal_combine(partials, zero)


def metal_r3(x, zero):
    partials = []
    for group in range(-(-len(x) // 512)):
        w = tree(first_add(x, zero, group), 32)[:32]
        for delta in (16, 8, 4, 2, 1):
            w = [
                w[lane] + w[lane + delta if lane + delta < 32 else lane]
                for lane in range(32)
            ]
        partials.append(w[0])
    return metal_combine(partials, zero)


def symbols(count):
    return [Sym(index) for index in range(count)], Sym(None)


def leaves(node):
    if type(node) is tuple:
        return leaves(node[0]) + leaves(node[1])
    return [] if node is None else [node]


def recorded_inputs():
    """Yield the inputs and the Metal outputs of the recorded measured report."""

    report = json.loads(RECORDED_REPORT_JSON)
    for case in report["correctness"]["cases"]:
        pattern, n = RECORDED_CONFIG["correctness"]["cases"][case["index"]]
        outputs = {item["kernel"]: item["output"] for item in case["outputs"]}
        yield case_values(78, case["index"], pattern, n), outputs
    timing = benchmark_values(78, max(RECORDED_CONFIG["benchmark"]["sizes"]))
    for point in report["benchmark"]["points"]:
        outputs = {side: item["output"] for side, item in point["checks"].items()}
        yield timing[: point["n"]], outputs


def test_the_simulated_bits_are_the_bits_of_the_recorded_metal_run():
    differ = 0
    for values, outputs in recorded_inputs():
        r1 = simulate(values, kernel=R1_SHARED).output
        shuffle = simulate(values, kernel=WARP_SHUFFLE).output

        assert r1 == outputs[R1], values.size
        assert shuffle == outputs[R2] == outputs[R3], values.size
        differ += r1 != shuffle
    # The two orders give other bits on 2 of the 8 inputs, so the match is
    # no accident of a well-conditioned sum.
    assert differ == 2


@pytest.mark.parametrize("n", SMALL_SIZES)
def test_each_tree_is_the_declared_order_of_its_metal_twin(n):
    values, zero = symbols(n)

    assert summation_tree(R1_SHARED, n) == metal_r1(values, zero).tree
    shuffle = summation_tree(WARP_SHUFFLE, n)
    assert shuffle == metal_r3(values, zero).tree == metal_r2(values, zero).tree
    for kernel in PINNED_KERNELS:
        # Each element enters each tree once.
        assert sorted(leaves(summation_tree(kernel, n))) == list(range(n))
    if n > 1:
        assert summation_tree(R1_SHARED, n) != shuffle


@pytest.mark.parametrize("count", [1, 2, 1024, 1025, 3072, 3073, 4096, 4097, 9000])
def test_the_combine_tree_is_the_metal_combine_order(count):
    values, zero = symbols(count)

    assert combine_tree(count) == metal_combine(values, zero).tree


@pytest.mark.parametrize("pattern", ["normal", "mixed"])
@pytest.mark.parametrize("n", [1, 33, 513, 4099])
def test_the_float_bits_follow_the_metal_order_on_small_inputs(pattern, n):
    values = case_values(78, 9, pattern, n)
    floats, zero = list(values), numpy.float32(0)

    assert simulate(values, kernel=R1_SHARED).output == float(metal_r1(floats, zero))
    shuffle = simulate(values, kernel=WARP_SHUFFLE).output
    assert shuffle == float(metal_r3(floats, zero)) == float(metal_r2(floats, zero))


def test_every_pinned_input_stays_inside_the_bound_of_its_order():
    config = load_config(ROOT / "config/day-78-gpu-reduce.json")
    predictions = predict(config)
    records = [prediction.to_mapping() for prediction in predictions]

    assert len(records) == 2 * (13 + 5) and all(item.passed for item in predictions)
    assert [record["input"] for record in records[:4]] == [
        "case-0",
        "case-0",
        "case-1",
        "case-1",
    ]
    exact = {record["input"] for record in records if record["exact_result"]}
    assert exact == {"case-0", "case-11", "case-12"}
    largest = max(records, key=lambda record: record["bound_ratio"])
    assert (largest["kernel"], largest["input"], largest["n"]) == (
        "reduce_r1_shared",
        "case-6",
        511,
    )
    assert round(largest["bound_ratio"], 3) == 0.124
    # The depths are the Metal depths of the same orders at 2^26.
    depths = {record["kernel"]: record["depth"] for record in records[-4:-2]}
    assert depths == {"reduce_r1_shared": 273, "reduce_warp_shuffle": 146}


@pytest.mark.parametrize("kernel", PINNED_KERNELS, ids=lambda item: item.name)
def test_the_counts_follow_the_text_and_no_access_races(kernel):
    for n in (1, 255, 256, 257, 511, 513, 786433, 1572865):
        run = simulate(numpy.ones(n, dtype=numpy.float32), kernel=kernel)
        blocks = kernel.launch(n).grid.x

        assert run.matches_model, n
        assert run.launch == kernel.launch(n) and run.partial_writes == blocks
        assert run.loads == n == run.output
        assert run.zero_fills == blocks * kernel.elements_per_block - n
        assert run.combine_reads == blocks
        assert run.hazards.race_free and run.combine_hazards.race_free
        assert run.hazards.barriers == kernel.barriers_per_block
        assert run.combine_hazards.barriers == COMBINE_BARRIERS
    assert run.to_mapping()["output_bits"] == run.output_bits


@pytest.mark.parametrize(
    ("kernel", "hazards"),
    [
        (R1_SHARED, (128, 64, 32, 16, 8, 4, 2, 1, 0)),
        (WARP_SHUFFLE, (128, 64, 32, 0)),
    ],
)
def test_each_dropped_barrier_leaves_the_races_of_its_level(kernel, hazards):
    values = case_values(78, 4, "mixed", 1000)
    kept = simulate(values, kernel=kernel)

    for barrier, expected in enumerate(hazards):
        run = simulate(values, kernel=kernel, dropped_barriers=(barrier,))
        # The next level reads s / 2 slots that other threads wrote.
        assert run.hazards.read_after_write == expected, barrier
        assert run.hazards.write_after_read == 0
        assert run.output_bits == kept.output_bits
        assert run.matches_model
    every = tuple(range(kernel.barriers_per_block))
    assert simulate(
        values, kernel=kernel, dropped_barriers=every
    ).hazards.to_mapping() == {
        "barriers": 0,
        "read_after_write": sum(hazards),
        "write_after_read": 0,
        "race_free": False,
    }


@pytest.mark.parametrize("dropped", [[0], (1, 0), (0, 0), (9,), (-1,), (True,), "0"])
def test_dropped_barriers_must_be_sorted_barrier_numbers(dropped):
    with pytest.raises(GpuReduceCudaError, match="dropped_barriers must be"):
        simulate(
            numpy.ones(4, numpy.float32), kernel=R1_SHARED, dropped_barriers=dropped
        )


@pytest.mark.parametrize(
    "values",
    [
        numpy.ones(4, dtype=numpy.float64),
        numpy.ones((2, 2), dtype=numpy.float32),
        [1.0, 2.0],
    ],
)
def test_simulate_needs_a_1d_float32_array(values):
    with pytest.raises(GpuReduceCudaError, match="1D float32 array"):
        simulate(values, kernel=WARP_SHUFFLE)


def test_the_symbolic_runs_refuse_large_inputs():
    with pytest.raises(GpuReduceCudaError, match="between 1 and"):
        simulate(numpy.ones(0, numpy.float32), kernel=R1_SHARED)
    with pytest.raises(GpuReduceCudaError, match=f"at most {MAX_TREE_ELEMENTS}"):
        summation_tree(R1_SHARED, MAX_TREE_ELEMENTS + 1)
    for count in (0, MAX_TREE_PARTIALS + 1):
        with pytest.raises(GpuReduceCudaError, match="count must be between"):
            combine_tree(count)
    with pytest.raises(TypeError, match="count must be an integer"):
        combine_tree(2.0)
    with pytest.raises(TypeError, match="config must be GpuReduceConfig"):
        predict(RECORDED_CONFIG)
