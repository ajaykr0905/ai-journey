import tracemalloc

import numpy
import pytest

from ai_journey.memory_hierarchy_stream import (
    KERNEL_NAMES,
    STREAM_KERNELS,
    MemoryHierarchyError,
    StreamBuffers,
    element_bytes,
    require_integer,
    stream_kernel,
)


def test_accounting_counts_every_numpy_call_operand_once():
    table = {
        name: (
            kernel.reads_per_element,
            kernel.writes_per_element,
            kernel.flops_per_element,
            kernel.arrays,
        )
        for name, kernel in STREAM_KERNELS.items()
    }

    assert KERNEL_NAMES == ("copy", "scale", "add", "triad")
    assert table == {
        "copy": (1, 1, 0, ("a", "c")),
        "scale": (1, 1, 1, ("a", "c")),
        "add": (2, 1, 1, ("a", "b", "c")),
        "triad": (3, 2, 2, ("a", "b", "c")),
    }
    triad = stream_kernel("triad")
    assert triad.bytes_per_call(1000, "float64") == 5 * 1000 * 8
    assert triad.bytes_per_call(1000, "float32") == 5 * 1000 * 4
    assert triad.working_set_bytes(1000, "float64") == 3 * 1000 * 8
    assert stream_kernel("copy").working_set_bytes(1000, "float64") == 2 * 1000 * 8
    assert triad.flops_per_call(1000) == 2000
    assert stream_kernel("copy").flops_per_call(1000) == 0
    assert triad.to_mapping() == {
        "name": "triad",
        "calls": ["numpy.multiply(q, b, out=c)", "numpy.add(c, a, out=c)"],
        "arrays": ["a", "b", "c"],
        "reads_per_element": 3,
        "writes_per_element": 2,
        "flops_per_element": 2,
    }
    assert stream_kernel("copy").to_mapping()["calls"] == ["numpy.copyto(c, a)"]


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("name", KERNEL_NAMES)
def test_kernels_compute_stream_definitions_on_the_prefix_only(name, dtype):
    buffers = StreamBuffers(300, dtype, seed=71)
    a, b, q = buffers.a[:256], buffers.b[:256], buffers.q
    expected = {"copy": a, "scale": q * a, "add": a + b, "triad": a + q * b}[name]

    buffers.bind(name, 256)()

    assert buffers.c.dtype == numpy.dtype(dtype)
    assert numpy.array_equal(buffers.c[:256], expected)
    assert numpy.isnan(buffers.c[256:]).all()
    buffers.check(name, 256)


@pytest.mark.parametrize("name", KERNEL_NAMES)
def test_kernels_allocate_no_temporary_arrays(name):
    buffers = StreamBuffers(1 << 16, "float64", seed=71)
    kernel = buffers.bind(name, 1 << 16)
    kernel()
    tracemalloc.start()
    try:
        before = tracemalloc.get_traced_memory()[0]
        kernel()
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()

    # One temporary would be 512 KiB; call bookkeeping stays far below that.
    assert peak - before < 4096


def test_check_rejects_wrong_values_and_skipped_writes():
    buffers = StreamBuffers(4096, "float64", seed=71)
    buffers.bind("triad", 4096)()
    buffers.check("triad", 4096)

    with pytest.raises(MemoryHierarchyError, match="add output"):
        buffers.check("add", 4096)
    buffers.c[4000] += 1.0
    with pytest.raises(MemoryHierarchyError, match="triad output"):
        buffers.check("triad", 4096)
    buffers.reset_output(4096)
    with pytest.raises(MemoryHierarchyError, match="triad output"):
        buffers.check("triad", 4096)
    buffers.bind("triad", 2048)()
    buffers.check("triad", 2048)
    with pytest.raises(MemoryHierarchyError, match="triad output"):
        buffers.check("triad", 4096)


def test_inputs_are_seeded_and_reproducible():
    first = StreamBuffers(512, "float32", seed=7)
    again = StreamBuffers(512, "float32", seed=7)
    other = StreamBuffers(512, "float32", seed=8)

    assert numpy.array_equal(first.a, again.a)
    assert numpy.array_equal(first.b, again.b)
    assert not numpy.array_equal(first.a, other.a)
    assert not numpy.array_equal(first.a, first.b)
    assert first.q == numpy.float32(3.0) and first.q.dtype == numpy.float32


@pytest.mark.parametrize(
    ("call", "error", "message"),
    [
        (lambda: StreamBuffers(0, "float64", 1), MemoryHierarchyError, "elements"),
        (lambda: StreamBuffers(True, "float64", 1), TypeError, "elements"),
        (lambda: StreamBuffers(64, "float16", 1), MemoryHierarchyError, "dtype"),
        (lambda: StreamBuffers(64, "float64", -1), MemoryHierarchyError, "seed"),
        (
            lambda: StreamBuffers(64, "float64", 1).bind("copy", 65),
            MemoryHierarchyError,
            "elements",
        ),
        (
            lambda: StreamBuffers(64, "float64", 1).bind("fma", 8),
            MemoryHierarchyError,
            "kernel",
        ),
        (
            lambda: stream_kernel("copy").bytes_per_call(8, "int8"),
            MemoryHierarchyError,
            "dtype",
        ),
        (
            lambda: stream_kernel("copy").bytes_per_call(8.0, "float64"),
            TypeError,
            "elements",
        ),
    ],
)
def test_invalid_buffers_kernels_and_counts_are_rejected(call, error, message):
    with pytest.raises(error, match=message):
        call()


def test_integer_and_dtype_validators_reject_lookalikes():
    assert require_integer(5, "count", 1, 5) == 5
    assert element_bytes("float32") == 4 and element_bytes("float64") == 8
    with pytest.raises(TypeError, match="count"):
        require_integer(False, "count", 0, 5)
    with pytest.raises(MemoryHierarchyError, match=r"count must be in \[1, 5\]"):
        require_integer(6, "count", 1, 5)
    with pytest.raises(MemoryHierarchyError, match="dtype"):
        element_bytes(numpy.float64)
