"""STREAM-style NumPy kernels whose operand traffic is known exactly.

The four kernels follow J. D. McCalpin's STREAM benchmark: copy ``c = a``,
scale ``c = q * a``, add ``c = a + b`` and triad ``c = a + q * b`` with
``q = 3``. Each kernel is a fixed list of NumPy calls that write through an
output operand (ufuncs take a positional ``out``; ``numpy.copyto`` writes its
destination in place), so no call allocates a temporary array and each call
reads every input array once and writes its output once. The bytes moved per
element therefore follow from the call list alone, counted the way STREAM
counts them: one element per array read and one per array written.
Write-allocate traffic, cache hits and prefetching are not observed, so the
memory system can move more or fewer bytes than the accounting states.

NumPy has no fused multiply-add ufunc, so triad runs as two calls, ``c = q *
b`` and then ``c = c + a``: five element transfers and two FLOPs per element,
where STREAM's single-loop triad counts three transfers.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial

import numpy


class MemoryHierarchyError(ValueError):
    """Raised when a memory-hierarchy control, measurement or report is invalid."""


DTYPE_BYTES = {"float32": 4, "float64": 8}
STREAM_SCALAR = 3.0
ARRAY_NAMES = ("a", "b", "c")
MAX_STREAM_ELEMENTS = 1 << 24
MAX_SEED = 2**32 - 1
CHECK_CHUNK_ELEMENTS = 1 << 20
_FUNCTIONS = {"copyto": numpy.copyto, "multiply": numpy.multiply, "add": numpy.add}


def require_integer(value: object, name: str, minimum: int, maximum: int) -> int:
    """Return a true integer inside ``[minimum, maximum]``.

    Every count and bound in the lab passes through here, so JSON ``true`` or
    ``2.0`` can never stand in for a count.
    """

    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if not minimum <= value <= maximum:
        raise MemoryHierarchyError(f"{name} must be in [{minimum}, {maximum}]")
    return value


def element_bytes(dtype: object) -> int:
    """Return the element width of a supported dtype for byte accounting."""

    if not isinstance(dtype, str) or dtype not in DTYPE_BYTES:
        raise MemoryHierarchyError("dtype must be float32 or float64")
    return DTYPE_BYTES[dtype]


@dataclass(frozen=True)
class KernelCall:
    """One NumPy call of a kernel: the operands it reads and the array it writes."""

    function: str
    inputs: tuple[str, ...]
    output: str

    @property
    def reads(self) -> int:
        """Array elements read per output element; the scalar ``q`` is free."""

        return sum(1 for name in self.inputs if name in ARRAY_NAMES)

    @property
    def flops(self) -> int:
        """FLOPs per element: one for a ufunc, none for a copy."""

        return 0 if self.function == "copyto" else 1

    def expression(self) -> str:
        """Render the call as source text, so a report states what was timed."""

        if self.function == "copyto":
            return f"numpy.copyto({self.output}, {self.inputs[0]})"
        return f"numpy.{self.function}({', '.join(self.inputs)}, out={self.output})"


@dataclass(frozen=True)
class StreamKernel:
    """A STREAM kernel as its ordered NumPy calls, with per-element accounting."""

    name: str
    calls: tuple[KernelCall, ...]

    @property
    def arrays(self) -> tuple[str, ...]:
        """Distinct arrays the calls touch, which together form the working set."""

        touched = {name for call in self.calls for name in (*call.inputs, call.output)}
        return tuple(name for name in ARRAY_NAMES if name in touched)

    @property
    def reads_per_element(self) -> int:
        """Array elements read per element over all calls of the kernel."""

        return sum(call.reads for call in self.calls)

    @property
    def writes_per_element(self) -> int:
        """Elements written per element: every call writes its output once."""

        return len(self.calls)

    @property
    def flops_per_element(self) -> int:
        """FLOPs per element over all calls of the kernel."""

        return sum(call.flops for call in self.calls)

    def bytes_per_call(self, elements: int, dtype: str) -> int:
        """Return the bytes one kernel call reads plus writes, by STREAM counting.

        This is the numerator of every bandwidth the lab reports.
        """

        count = require_integer(elements, "elements", 1, MAX_STREAM_ELEMENTS)
        transfers = self.reads_per_element + self.writes_per_element
        return transfers * count * element_bytes(dtype)

    def flops_per_call(self, elements: int) -> int:
        """Return the FLOPs one call performs, for roofline placement."""

        count = require_integer(elements, "elements", 1, MAX_STREAM_ELEMENTS)
        return self.flops_per_element * count

    def working_set_bytes(self, elements: int, dtype: str) -> int:
        """Return the bytes of the distinct arrays one call touches.

        This is the sweep's x-axis: what has to fit in a cache for a repeated
        call to be served from it.
        """

        count = require_integer(elements, "elements", 1, MAX_STREAM_ELEMENTS)
        return len(self.arrays) * count * element_bytes(dtype)

    def to_mapping(self) -> dict[str, object]:
        """Return the accounting a report records, so a verifier can recount it."""

        return {
            "name": self.name,
            "calls": [call.expression() for call in self.calls],
            "arrays": list(self.arrays),
            "reads_per_element": self.reads_per_element,
            "writes_per_element": self.writes_per_element,
            "flops_per_element": self.flops_per_element,
        }


STREAM_KERNELS = {
    kernel.name: kernel
    for kernel in (
        StreamKernel("copy", (KernelCall("copyto", ("a",), "c"),)),
        StreamKernel("scale", (KernelCall("multiply", ("q", "a"), "c"),)),
        StreamKernel("add", (KernelCall("add", ("a", "b"), "c"),)),
        StreamKernel(
            "triad",
            (
                KernelCall("multiply", ("q", "b"), "c"),
                KernelCall("add", ("c", "a"), "c"),
            ),
        ),
    )
}
KERNEL_NAMES = tuple(STREAM_KERNELS)

# STREAM's definitions as plain expressions, independent of the call lists
# above, so a wrong call list cannot also supply its own expected answer.
_EXPECTED = {
    "copy": lambda a, b, q: a,
    "scale": lambda a, b, q: q * a,
    "add": lambda a, b, q: a + b,
    "triad": lambda a, b, q: a + q * b,
}


def stream_kernel(name: object) -> StreamKernel:
    """Look up a kernel by name, rejecting anything that is not a STREAM kernel."""

    if not isinstance(name, str) or name not in STREAM_KERNELS:
        raise MemoryHierarchyError(f"kernel must be one of {', '.join(KERNEL_NAMES)}")
    return STREAM_KERNELS[name]


def _bind(call: KernelCall, operands: Mapping[str, object]) -> Callable[[], object]:
    function = _FUNCTIONS[call.function]
    inputs = [operands[name] for name in call.inputs]
    if call.function == "copyto":
        return partial(function, operands[call.output], *inputs)
    return partial(function, *inputs, operands[call.output])


class StreamBuffers:
    """Seeded inputs ``a`` and ``b`` and an output ``c``, allocated once.

    Every sweep point binds its kernels to the first ``elements`` entries, so
    one allocation and its first-touch page faults serve the whole sweep and
    no point pays for them inside a timed sample.
    """

    def __init__(self, elements: int, dtype: str, seed: int) -> None:
        self.elements = require_integer(elements, "elements", 1, MAX_STREAM_ELEMENTS)
        element_bytes(dtype)
        kind = numpy.dtype(dtype)
        generator = numpy.random.default_rng(require_integer(seed, "seed", 0, MAX_SEED))
        self.dtype = dtype
        self.a = generator.standard_normal(self.elements, dtype=kind)
        self.b = generator.standard_normal(self.elements, dtype=kind)
        self.c = numpy.full(self.elements, numpy.nan, dtype=kind)
        self.q = kind.type(STREAM_SCALAR)

    def _prefix(self, elements: int) -> int:
        return require_integer(elements, "elements", 1, self.elements)

    def bind(self, name: str, elements: int) -> Callable[[], object]:
        """Return a no-argument callable that runs one kernel on a prefix.

        Operands are bound up front with ``functools.partial`` and a positional
        ``out``, which keeps per-call Python overhead, the dominant cost at
        cache-sized points, as low as NumPy allows.
        """

        kernel = stream_kernel(name)
        count = self._prefix(elements)
        operands = {
            "a": self.a[:count],
            "b": self.b[:count],
            "c": self.c[:count],
            "q": self.q,
        }
        bound = tuple(_bind(call, operands) for call in kernel.calls)
        if len(bound) == 1:
            return bound[0]

        def run() -> None:
            for call in bound:
                call()

        return run

    def reset_output(self, elements: int) -> None:
        """Fill the output prefix with NaN so a kernel that skips writes fails."""

        self.c[: self._prefix(elements)] = numpy.nan

    def check(self, name: str, elements: int) -> None:
        """Require the output prefix to equal the kernel's STREAM definition.

        The comparison is exact: the definitions perform the same IEEE
        operations, and chunking keeps the check's temporaries small next to
        the arrays being measured.
        """

        expected = _EXPECTED[stream_kernel(name).name]
        count = self._prefix(elements)
        for start in range(0, count, CHECK_CHUNK_ELEMENTS):
            stop = min(count, start + CHECK_CHUNK_ELEMENTS)
            want = expected(self.a[start:stop], self.b[start:stop], self.q)
            if not numpy.array_equal(self.c[start:stop], want):
                raise MemoryHierarchyError(
                    f"{name} output does not match its STREAM definition"
                )
