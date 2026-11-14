import itertools
import math
import time
import tracemalloc

import numpy
import pytest

from ai_journey.memory_hierarchy_access import (
    ACCESS_PATTERNS,
    PATTERN_NAMES,
    AccessBuffers,
    AccessMeasurement,
    access_contrast,
    access_pattern,
    measure_access_patterns,
    validate_access,
)
from ai_journey.memory_hierarchy_stream import MemoryHierarchyError
from ai_journey.memory_hierarchy_timing import BatchTiming


def measure(**changes):
    arguments = {
        "elements": 2048,
        "stride_elements": 16,
        "dtype": "float64",
        "seed": 71,
        "warmup": 1,
        "repeats": 3,
        "target_bytes_per_sample": 32768,
        "clock": itertools.count(0, 1_000).__next__,
        **changes,
    }
    return measure_access_patterns(**arguments)


def test_orders_visit_every_element_once_and_differ_only_in_order():
    buffers = AccessBuffers(1024, 4, "float64", seed=71)

    assert buffers.indices("copy") is None
    sequential = buffers.indices("sequential")
    strided = buffers.indices("strided")
    random = buffers.indices("random")
    assert numpy.array_equal(sequential, numpy.arange(1024))
    assert strided.tolist()[:6] == [0, 4, 8, 12, 16, 20]
    assert strided.tolist()[256:259] == [1, 5, 9]
    assert strided.tolist() == [j + 4 * i for j in range(4) for i in range(256)]
    for order in (sequential, strided, random):
        assert order.dtype == numpy.int64 and order.flags.c_contiguous
        assert numpy.array_equal(numpy.sort(order), sequential)
    assert numpy.array_equal(random, buffers.indices("random"))
    assert not numpy.array_equal(
        random, AccessBuffers(1024, 4, "float64", 72).indices("random")
    )
    assert not numpy.array_equal(random, sequential)


def test_gathers_account_for_the_same_bytes_and_copy_skips_the_index():
    sizes = {
        name: pattern.bytes_per_call(1000, "float64")
        for name, pattern in ACCESS_PATTERNS.items()
    }

    assert PATTERN_NAMES == ("copy", "sequential", "strided", "random")
    assert sizes == {
        "copy": 16_000,
        "sequential": 24_000,
        "strided": 24_000,
        "random": 24_000,
    }
    assert access_pattern("strided").bytes_per_call(1000, "float32") == 16_000
    assert access_pattern("random").to_mapping() == {
        "name": "random",
        "call": "source.take(random, out=output, mode='clip')",
        "order": "numpy.random.default_rng((seed, 1)).permutation(n)",
        "index_bytes": 8,
    }
    assert access_pattern("copy").to_mapping()["index_bytes"] == 0


@pytest.mark.parametrize("name", PATTERN_NAMES)
def test_each_pattern_reads_the_source_in_its_order_without_temporaries(name):
    buffers = AccessBuffers(1 << 16, 16, "float64", seed=71)
    indices = buffers.indices(name)
    call = buffers.bind(name, indices)
    call()
    tracemalloc.start()
    try:
        before = tracemalloc.get_traced_memory()[0]
        call()
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()

    expected = buffers.source if indices is None else buffers.source[indices]
    assert numpy.array_equal(buffers.output, expected)
    buffers.check(name, indices)
    # A buffered output would be 512 KiB; mode="clip" writes in place.
    assert peak - before < 4096


def test_checks_reject_wrong_skipped_or_mismatched_reads():
    buffers = AccessBuffers(2048, 16, "float32", seed=71)
    random = buffers.indices("random")
    buffers.bind("random", random)()
    buffers.check("random", random)

    with pytest.raises(MemoryHierarchyError, match="random output"):
        buffers.check("random", buffers.indices("strided"))
    buffers.output[7] = 0.5
    with pytest.raises(MemoryHierarchyError, match="random output"):
        buffers.check("random", random)
    buffers.reset_output()
    with pytest.raises(MemoryHierarchyError, match="copy output"):
        buffers.check("copy", None)
    with pytest.raises(MemoryHierarchyError, match="int64 index"):
        buffers.bind("random", random.astype(numpy.int32))
    with pytest.raises(MemoryHierarchyError, match="int64 index"):
        buffers.bind("sequential", None)
    with pytest.raises(MemoryHierarchyError, match="without an index"):
        buffers.bind("copy", random)
    with pytest.raises(MemoryHierarchyError, match="pattern must be one of"):
        buffers.indices("diagonal")


def test_patterns_are_measured_in_order_with_exact_accounting():
    results = measure()

    assert [item.pattern for item in results] == list(PATTERN_NAMES)
    copy, sequential = results[0].to_mapping(), results[1].to_mapping()
    assert copy["calls_per_sample"] == 1
    assert copy["bytes_per_call"] == 2048 * 16
    assert sequential["bytes_per_call"] == 2048 * 24
    assert sequential == {
        "name": "sequential",
        "call": "source.take(sequential, out=output, mode='clip')",
        "order": "element i at position i",
        "index_bytes": 8,
        "bytes_per_call": 49152,
        "calls_per_sample": 1,
        "warmup": 1,
        "repeats": 3,
        "samples_ns": [1000, 1000, 1000],
        "timing": {
            "count": 3,
            "min_ns": 1000,
            "p25_ns": 1000,
            "median_ns": 1000,
            "p75_ns": 1000,
            "max_ns": 1000,
            "iqr_ns": 0,
        },
        "bytes_per_sample": 49152,
        "elements_per_sample": 2048,
        "median_ns_per_element": 1000 / 2048,
        "median_elements_per_ns": 2.048,
        "median_gb_per_second": 49.152,
        "best_gb_per_second": 49.152,
    }
    small = measure(elements=1024, stride_elements=2, target_bytes_per_sample=65536)
    assert [item.timing.calls_per_sample for item in small] == [4, 3, 3, 3]


def test_contrast_divides_time_per_element_by_the_sequential_gather():
    def timed(pattern, median):
        timing = BatchTiming(1, 0, 3, (median, median, median), "injected")
        return AccessMeasurement(pattern, 1024, "float64", timing)

    results = tuple(
        timed(name, median)
        for name, median in zip(PATTERN_NAMES, (200, 2000, 7000, 11000))
    )

    assert access_contrast(results) == {
        "sequential_to_copy_time_ratio": (2000 / 1024) / (200 / 1024),
        "strided_to_sequential_time_ratio": (7000 / 1024) / (2000 / 1024),
        "random_to_sequential_time_ratio": (11000 / 1024) / (2000 / 1024),
        "random_to_strided_time_ratio": (11000 / 1024) / (7000 / 1024),
    }
    with pytest.raises(MemoryHierarchyError, match="every pattern once"):
        access_contrast(results[1:])
    with pytest.raises(MemoryHierarchyError, match="every pattern once"):
        access_contrast(results + results[:1])


@pytest.mark.parametrize(
    ("elements", "stride", "error", "message"),
    [
        (1000, 8, MemoryHierarchyError, "access elements"),
        (4096, 3, MemoryHierarchyError, "multiple of stride_elements"),
        (4096, 4096, MemoryHierarchyError, "two reads per pass"),
        (4096, 1, MemoryHierarchyError, "stride_elements"),
        (4096, 8192, MemoryHierarchyError, "stride_elements"),
        (4096.0, 8, TypeError, "access elements"),
        (4096, True, TypeError, "stride_elements"),
    ],
)
def test_access_controls_need_whole_stride_passes(elements, stride, error, message):
    with pytest.raises(error, match=message):
        validate_access(elements, stride)


def test_host_timer_contrast_reports_positive_finite_rates():
    results = measure(clock=time.perf_counter_ns)

    assert {item.timing.timer for item in results} == {"time.perf_counter_ns"}
    for item in results:
        assert item.median_elements_per_ns > 0
        assert math.isfinite(item.median_ns_per_element)
    assert all(value > 0 for value in access_contrast(results).values())
