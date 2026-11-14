import itertools
import math
import time
import tracemalloc

import numpy
import pytest

from ai_journey.memory_hierarchy_matmul import (
    check_matmul,
    matmul_allocation_bytes,
    measure_matmul,
    rounding_tolerance,
    validate_matmul,
)
from ai_journey.memory_hierarchy_stream import MemoryHierarchyError


def measure(**changes):
    arguments = {
        "m": 64,
        "n": 48,
        "k": 32,
        "dtype": "float64",
        "seed": 71,
        "warmup": 1,
        "repeats": 3,
        "clock": itertools.count(0, 4_000).__next__,
        **changes,
    }
    return measure_matmul(**arguments)


def operands(dtype, m=40, n=24, k=32):
    generator = numpy.random.default_rng(5)
    left = generator.standard_normal((m, k), dtype=numpy.dtype(dtype))
    right = generator.standard_normal((k, n), dtype=numpy.dtype(dtype))
    return left, right


def test_flops_and_traffic_come_from_the_roofline_matmul_workload():
    result = measure()
    mapping = result.to_mapping()

    assert mapping["flops_per_call"] == 2 * 64 * 48 * 32
    assert mapping["compulsory_bytes_per_call"] == (64 * 32 + 32 * 48 + 64 * 48) * 8
    assert mapping["calls_per_sample"] == 1
    assert mapping["samples_ns"] == [4000, 4000, 4000]
    assert mapping["median_gflops_per_second"] == 2 * 64 * 48 * 32 / 4000
    assert mapping["best_gflops_per_second"] == mapping["median_gflops_per_second"]
    assert mapping["call"] == "numpy.matmul(left, right, out=product)"
    correctness = mapping["correctness"]
    assert correctness["reference"] == "float64"
    assert correctness["checked_elements"] == 64 * 48
    assert correctness["unit_roundoff"] == 2.0**-53
    assert correctness["tolerance"] == rounding_tolerance(32, "float64")
    assert 0.0 <= correctness["max_bound_ratio"] <= 1.0


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_a_correct_product_passes_and_a_wrong_one_fails(dtype):
    left, right = operands(dtype, m=150)
    product = numpy.matmul(left, right)

    ratio = check_matmul(left, right, product)

    assert 0.0 <= ratio < 1.0
    product[149, 3] += 1e-3
    with pytest.raises(MemoryHierarchyError, match="rounding bound"):
        check_matmul(left, right, product)
    product[149, 3] = numpy.nan
    with pytest.raises(MemoryHierarchyError, match="not finite"):
        check_matmul(left, right, product)
    with pytest.raises(MemoryHierarchyError, match="conformable"):
        check_matmul(left, right, product.T)
    with pytest.raises(MemoryHierarchyError, match="conformable"):
        check_matmul(left, right.astype(numpy.float16), product)


def test_tolerance_covers_both_the_product_and_reference_roundings():
    single = 32 * 2.0**-24
    double = 32 * 2.0**-53

    assert rounding_tolerance(32, "float32") == single / (1 - single) + double / (
        1 - double
    )
    assert rounding_tolerance(32, "float64") == 2 * (double / (1 - double))
    with pytest.raises(MemoryHierarchyError, match="dtype"):
        rounding_tolerance(32, "float16")


def test_a_matmul_that_writes_nothing_fails_its_check(monkeypatch):
    monkeypatch.setattr(numpy, "matmul", lambda left, right, out: out)

    with pytest.raises(MemoryHierarchyError, match="not finite"):
        measure()


@pytest.mark.parametrize(
    ("dimensions", "error", "message"),
    [
        ((15, 64, 64), MemoryHierarchyError, "m must be in"),
        ((64, 2049, 64), MemoryHierarchyError, "n must be in"),
        ((64, 64, 64.0), TypeError, "k must be an integer"),
        ((64, True, 64), TypeError, "n must be an integer"),
    ],
)
def test_matmul_shapes_are_bounded(dimensions, error, message):
    with pytest.raises(error, match=message):
        validate_matmul(*dimensions)


def test_allocation_counts_operands_product_and_check_temporaries():
    assert matmul_allocation_bytes(64, 48, 32, "float32") == (
        (64 * 32 + 32 * 48 + 64 * 48) * 4 + 2 * 32 * 48 * 8 + 64 * (2 * 32 + 4 * 48) * 8
    )
    assert matmul_allocation_bytes(16, 16, 16, "float64") == (
        3 * 256 * 8 + 2 * 256 * 8 + 16 * (2 * 16 + 4 * 16) * 8
    )
    with pytest.raises(MemoryHierarchyError, match="dtype"):
        matmul_allocation_bytes(64, 48, 32, "int8")


@pytest.mark.parametrize(
    ("m", "n", "k", "dtype"), [(256, 1024, 16, "float32"), (200, 2048, 16, "float64")]
)
def test_a_measurement_stays_within_its_planned_allocation(m, n, k, dtype):
    # Several row blocks and a wide, shallow product: a block's temporaries
    # outliving it, or masked copies of a block, would exceed the plan.
    tracemalloc.start()
    try:
        measure(m=m, n=n, k=k, dtype=dtype)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()

    # 64 KiB covers Python objects and iterator buffers outside the model.
    assert peak <= matmul_allocation_bytes(m, n, k, dtype) + (1 << 16)


def test_host_timer_matmul_reports_a_positive_finite_rate():
    result = measure(dtype="float32", clock=time.perf_counter_ns)

    assert result.timing.timer == "time.perf_counter_ns"
    assert result.median_gflops_per_second > 0
    assert math.isfinite(result.to_mapping()["best_gflops_per_second"])
