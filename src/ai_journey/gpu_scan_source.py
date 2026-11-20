"""Metal Shading Language source of the Day 78 float32 inclusive scans.

Curriculum Day 78 follows GPU MODE Lecture 20, "Scan Algorithm". Each kernel
here gives one threadgroup one block of ``BLOCK`` = 1024 elements, writes the
inclusive prefix sums of that block and writes the block total to ``sums``:

- ``scan_s0_hillis_steele``: 1024 threads, one element each, run the 10
  Hillis-Steele steps (Kogge-Stone in the PMPP book). Each step reads one
  threadgroup buffer and writes the other, so one barrier for each step keeps
  every read ahead of the next write to the same buffer.
- ``scan_s1_blelloch``: 512 threads, two elements each, run Blelloch's
  up-sweep and down-sweep (the PMPP book calls a close variant Brent-Kung)
  in one threadgroup buffer. Element ``j`` then takes the exclusive sum of ``j + 1``,
  and the last element takes the block total, so no extra add rounds it.
- ``scan_s2_simd``: 1024 threads scan each run of 32 elements with
  ``simd_prefix_inclusive_sum``. SIMD group 0 scans the 32 run totals the
  same way and shifts them up one lane with ``simd_shuffle_up``, and each
  thread adds the total of the runs before its own. SIMD-group functions are
  Metal's counterpart of CUDA warp shuffles, not the same instruction, and
  Metal does not say in which order a prefix sum adds.

A longer input runs as a multi-block scan: scan each block, scan the block
sums with the same kernel (in blocks again while they fill more than one),
then ``scan_add_offsets`` adds the scanned sum of the earlier blocks to each
element of a later block. These rules hold in every kernel:

- Every thread loads its value or the identity 0, so a ragged last block
  scans like a full one, and only the stores are guarded.
- Every barrier sits in the kernel body or in a loop whose bounds are the
  same for every thread, never under a condition on the thread, and no
  kernel returns early. ``barrier_heads`` reads the text to check this.
- Scalars are 32-bit ``constant uint&`` values, passed with ``arg_casts``
  as in the Day 73 to 77 Metal labs.

Each kernel's digest covers its own text, so a new kernel moves no other
digest. The library is the Day 76 header and the four kernels in order.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from math import prod

from .kernel_metal_geometry import KernelMetalError, ceil_div, require_integer
from .matmul_tiled_source import BARRIER, HEADER
from .matmul_tiled_variants import text_sha256

BLOCK = 1024
LOG2_BLOCK = 10
SIMD_WIDTH = 32
ADD_GROUP = 256
FLOAT32_BYTES = 4
# Metal guarantees at least this much static threadgroup memory per group.
MAX_THREADGROUP_BYTES = 32 * 1024
MAX_LENGTH = 1 << 24
MAX_DEPTH = 1 << 12

S0_HILLIS_STEELE = "scan_s0_hillis_steele"
S1_BLELLOCH = "scan_s1_blelloch"
S2_SIMD = "scan_s2_simd"
ADD_OFFSETS = "scan_add_offsets"
SCAN_ARG_CASTS = {3: "int32"}
ADD_ARG_CASTS = {2: "int32"}

# Every loop that holds a barrier, with its trip count: each runs once for
# each power of two below BLOCK, the same number of times in every thread.
UNIFORM_LOOPS = {
    "for (uint offset = 1u; offset < 1024u; offset <<= 1u) {": LOG2_BLOCK,
    "for (uint active = 512u; active > 0u; active >>= 1u) {": LOG2_BLOCK,
    "for (uint active = 1u; active <= 512u; active <<= 1u) {": LOG2_BLOCK,
}
KERNEL_BODY = "kernel"
_NAME = re.compile(r"scan_[a-z0-9_]{1,48}")


class GpuScanError(KernelMetalError):
    """Raised when a scan kernel, control, launch, check or report is invalid."""


def require_count(value: object, name: str, minimum: int, maximum: int) -> int:
    """Bound an integer by the Metal lab's rule, as this lab's error.

    ``require_integer`` refuses ``true`` and ``2.0`` with ``TypeError``; a
    value out of range becomes ``GpuScanError``.
    """

    try:
        return require_integer(value, name, minimum, maximum)
    except GpuScanError:
        raise
    except KernelMetalError as error:
        raise GpuScanError(str(error)) from error


def require_length(value: object, name: str = "length") -> int:
    """Return a scan length of 1 to ``MAX_LENGTH``."""

    return require_count(value, name, 1, MAX_LENGTH)


def block_count(length: int) -> int:
    """Blocks of one level: ``ceil(length / BLOCK)``."""

    return ceil_div(require_length(length), BLOCK)


def level_lengths(length: int) -> tuple[int, ...]:
    """The length that each level scans: the input, then each level's block sums."""

    lengths = [require_length(length)]
    while lengths[-1] > BLOCK:
        lengths.append(block_count(lengths[-1]))
    return tuple(lengths)


def dispatches(length: int) -> int:
    """Kernel calls of one scan: one block scan per level, one add per lower level."""

    return 2 * len(level_lengths(length)) - 1


def guard_elements(length: int) -> int:
    """The sentinel tail of an output: spare slots of the last block, one block more.

    No launched thread stores at or past ``length``, so a store that escaped
    its guard, or an address one block too far, shows in this tail.
    """

    return block_count(length) * BLOCK - length + BLOCK


def barrier_heads(source: str) -> tuple[tuple[str, ...], ...]:
    """Return, for each barrier of one kernel text, the heads of its blocks.

    The first opened block is the kernel body, named ``KERNEL_BODY``; every
    later head is the stripped line that opened its block.
    """

    stack: list[str] = []
    found = []
    for raw in source.splitlines():
        line = raw.strip()
        if line.startswith("//"):
            continue
        if line.startswith("}"):
            if not stack:
                raise GpuScanError("a kernel text closes a block it never opened")
            stack.pop()
            line = line[1:].strip()
        if BARRIER in line:
            found.append(tuple(stack))
        if line.endswith("{"):
            stack.append(KERNEL_BODY if not stack else line)
    if stack:
        raise GpuScanError("a kernel text leaves a block open")
    return tuple(found)


def uniform_barriers(source: str) -> int:
    """Count the barriers one group waits at, after checking that each is uniform.

    A barrier may sit only in the kernel body or in a ``UNIFORM_LOOPS`` loop,
    and a kernel with a barrier may not return early, so every thread of a
    group reaches every barrier the same number of times.
    """

    heads = barrier_heads(source)
    if heads and re.search(r"\breturn\b", source):
        raise GpuScanError("a kernel with a barrier must not return early")
    total = 0
    for stack in heads:
        if not stack or stack[0] != KERNEL_BODY:
            raise GpuScanError("a barrier must sit inside the kernel body")
        for head in stack[1:]:
            if head not in UNIFORM_LOOPS:
                raise GpuScanError(
                    f"a barrier sits under {head!r}, which is not a loop with the "
                    "same bounds in every thread"
                )
        total += prod(UNIFORM_LOOPS[head] for head in stack[1:])
    return total


def _signature(name: str, parameters: tuple[str, ...]) -> str:
    """Align each parameter under the first, as the Day 73 source does."""

    head = f"kernel void {name}("
    return head + f",\n{' ' * len(head)}".join(parameters) + ") {"


_SCAN_BUFFERS = (
    "device float* out [[buffer(0)]]",
    "device const float* in [[buffer(1)]]",
    "device float* sums [[buffer(2)]]",
    "constant uint& n [[buffer(3)]]",
    "uint group [[threadgroup_position_in_grid]]",
)


def _hillis_steele() -> str:
    signature = _signature(
        S0_HILLIS_STEELE, (*_SCAN_BUFFERS, "uint t [[thread_index_in_threadgroup]]")
    )
    return f"""\
// Hillis-Steele scan of one block of {BLOCK} elements by {BLOCK} threads, one
// element each. Step k adds the value 2^k slots back. It reads one buffer and
// writes the other, so one barrier for each step keeps reads ahead of writes.
{signature}
    threadgroup float buffers[2][{BLOCK}];
    uint i = group * {BLOCK}u + t;
    buffers[0][t] = i < n ? in[i] : 0.0f;
    {BARRIER}
    uint source = 0u;
    for (uint offset = 1u; offset < {BLOCK}u; offset <<= 1u) {{
        float value = buffers[source][t];
        if (t >= offset) {{
            value = buffers[source][t - offset] + value;
        }}
        buffers[source ^ 1u][t] = value;
        source ^= 1u;
        {BARRIER}
    }}
    float result = buffers[source][t];
    if (i < n) {{
        out[i] = result;
    }}
    if (t == {BLOCK - 1}u) {{
        sums[group] = result;
    }}
}}
"""


def _blelloch() -> str:
    half = BLOCK // 2
    signature = _signature(
        S1_BLELLOCH, (*_SCAN_BUFFERS, "uint t [[thread_index_in_threadgroup]]")
    )
    return f"""\
// Blelloch scan of one block of {BLOCK} elements by {half} threads, two
// elements each, in one buffer. The up-sweep leaves the sum of each aligned
// run of 2^(d+1) elements in the last slot of the run. The down-sweep turns
// the tree into exclusive sums, and element j takes the sum of j + 1.
{signature}
    threadgroup float tree[{BLOCK}];
    uint first = group * {BLOCK}u + t;
    uint second = first + {half}u;
    tree[t] = first < n ? in[first] : 0.0f;
    tree[t + {half}u] = second < n ? in[second] : 0.0f;
    uint stride = 1u;
    for (uint active = {half}u; active > 0u; active >>= 1u) {{
        {BARRIER}
        if (t < active) {{
            uint right = (2u * t + 2u) * stride - 1u;
            tree[right] = tree[right - stride] + tree[right];
        }}
        stride <<= 1u;
    }}
    {BARRIER}
    float total = 0.0f;
    if (t == {half - 1}u) {{
        total = tree[{BLOCK - 1}];
        tree[{BLOCK - 1}] = 0.0f;
    }}
    for (uint active = 1u; active <= {half}u; active <<= 1u) {{
        stride >>= 1u;
        {BARRIER}
        if (t < active) {{
            uint right = (2u * t + 2u) * stride - 1u;
            float carry = tree[right - stride];
            tree[right - stride] = tree[right];
            tree[right] = carry + tree[right];
        }}
    }}
    {BARRIER}
    float low = tree[t + 1u];
    float high = t == {half - 1}u ? total : tree[t + {half + 1}u];
    if (first < n) {{
        out[first] = low;
    }}
    if (second < n) {{
        out[second] = high;
    }}
    if (t == {half - 1}u) {{
        sums[group] = total;
    }}
}}
"""


def _simd() -> str:
    signature = _signature(
        S2_SIMD,
        (
            *_SCAN_BUFFERS,
            "uint lane [[thread_index_in_simdgroup]]",
            "uint simd [[simdgroup_index_in_threadgroup]]",
        ),
    )
    groups = BLOCK // SIMD_WIDTH
    return f"""\
// SIMD-group scan of one block of {BLOCK} elements by {groups} SIMD groups of {SIMD_WIDTH}
// lanes. Each SIMD group scans its own {SIMD_WIDTH} elements. SIMD group 0 scans the
// {groups} totals and moves them up one lane, so each thread adds the total of
// the SIMD groups before its own.
{signature}
    threadgroup float totals[{groups}];
    uint local = simd * {SIMD_WIDTH}u + lane;
    uint i = group * {BLOCK}u + local;
    float value = i < n ? in[i] : 0.0f;
    float prefix = simd_prefix_inclusive_sum(value);
    if (lane == {SIMD_WIDTH - 1}u) {{
        totals[simd] = prefix;
    }}
    {BARRIER}
    if (simd == 0u) {{
        float running = simd_prefix_inclusive_sum(totals[lane]);
        float earlier = simd_shuffle_up(running, 1u);
        totals[lane] = lane == 0u ? 0.0f : earlier;
    }}
    {BARRIER}
    float result = prefix + totals[simd];
    if (i < n) {{
        out[i] = result;
    }}
    if (local == {BLOCK - 1}u) {{
        sums[group] = result;
    }}
}}
"""


def _add() -> str:
    signature = _signature(
        ADD_OFFSETS,
        (
            "device float* data [[buffer(0)]]",
            "device const float* offsets [[buffer(1)]]",
            "constant uint& n [[buffer(2)]]",
            "uint i [[thread_position_in_grid]]",
        ),
    )
    return f"""\
// Add the scanned sum of every earlier block to each element of a later
// block, one thread for each element. Block 0 needs no add.
{signature}
    uint block = i / {BLOCK}u;
    if (i < n && block > 0u) {{
        data[i] = data[i] + offsets[block - 1u];
    }}
}}
"""


ADD_SOURCE = _add()


@dataclass(frozen=True)
class ScanKernel:
    """One block scan: its group, memory, declared order and text.

    ``block_depth`` declares the most float32 additions on any path from an
    input to an output of one block. ``replay`` is true when the text fixes
    every addition, so the CPU can replay the order bit for bit.
    """

    name: str
    algorithm: str
    threads_per_group: int
    elements_per_thread: int
    threadgroup_bytes: int
    simd_width: int | None
    block_depth: int
    replay: bool
    source: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or _NAME.fullmatch(self.name) is None:
            raise GpuScanError("a kernel name must be scan_ and a short slug")
        if not isinstance(self.algorithm, str) or not self.algorithm.isascii():
            raise GpuScanError("an algorithm must be an ASCII label")
        require_count(self.threads_per_group, "threads_per_group", 1, BLOCK)
        require_count(self.elements_per_thread, "elements_per_thread", 1, BLOCK)
        if self.threads_per_group * self.elements_per_thread != BLOCK:
            raise GpuScanError(f"one group must scan exactly {BLOCK} elements")
        require_count(
            self.threadgroup_bytes, "threadgroup_bytes", 0, MAX_THREADGROUP_BYTES
        )
        if self.simd_width is not None and self.simd_width != SIMD_WIDTH:
            raise GpuScanError(f"a SIMD-group kernel needs {SIMD_WIDTH} lanes")
        require_count(self.block_depth, "block_depth", LOG2_BLOCK, MAX_DEPTH)
        if type(self.replay) is not bool:
            raise TypeError("replay must be a bool")
        if (
            not isinstance(self.source, str)
            or not self.source.isascii()
            or f"kernel void {self.name}(" not in self.source
        ):
            raise GpuScanError(f"{self.name} needs ASCII source that defines it")
        uniform_barriers(self.source)

    @property
    def block(self) -> int:
        """Elements one group scans: threads times elements for each thread."""

        return self.threads_per_group * self.elements_per_thread

    @property
    def barriers_per_block(self) -> int:
        """Barriers that every thread of one group waits at, loops counted."""

        return uniform_barriers(self.source)

    @property
    def source_sha256(self) -> str:
        """The digest of this kernel's text alone."""

        return text_sha256(self.source)

    @property
    def arg_casts(self) -> dict[int, str]:
        """Pass the length as 32 bits, the width of ``constant uint&``."""

        return dict(SCAN_ARG_CASTS)

    def to_mapping(self) -> dict[str, object]:
        """Return what a report records about the kernel, without its text."""

        return {
            "name": self.name,
            "algorithm": self.algorithm,
            "block": self.block,
            "threads_per_group": self.threads_per_group,
            "elements_per_thread": self.elements_per_thread,
            "threadgroup_bytes": self.threadgroup_bytes,
            "simd_width": self.simd_width,
            "barriers_per_block": self.barriers_per_block,
            "block_depth": self.block_depth,
            "replay": self.replay,
            "source_sha256": self.source_sha256,
        }


KERNELS: dict[str, ScanKernel] = {
    kernel.name: kernel
    for kernel in (
        ScanKernel(
            S0_HILLIS_STEELE,
            "hillis_steele",
            BLOCK,
            1,
            2 * BLOCK * FLOAT32_BYTES,
            None,
            # Element t takes one add at each step k with t >= 2^k.
            LOG2_BLOCK,
            True,
            _hillis_steele(),
        ),
        ScanKernel(
            S1_BLELLOCH,
            "blelloch",
            BLOCK // 2,
            2,
            BLOCK * FLOAT32_BYTES,
            None,
            # A subtree sum of 2^9 leaves holds 9 adds, and the down-sweep adds
            # one more at each of the 10 levels: 2 * 10 - 1.
            2 * LOG2_BLOCK - 1,
            True,
            _blelloch(),
        ),
        ScanKernel(
            S2_SIMD,
            "simd_prefix",
            BLOCK,
            1,
            BLOCK // SIMD_WIDTH * FLOAT32_BYTES,
            SIMD_WIDTH,
            # Any order of 32 lanes holds at most 31 adds on a path, any order
            # of the 31 earlier totals at most 30, and the last add 1.
            (SIMD_WIDTH - 1) + (BLOCK // SIMD_WIDTH - 2) + 1,
            False,
            _simd(),
        ),
    )
}


def add_mapping() -> dict[str, object]:
    """Return what a report records about the add kernel."""

    return {
        "name": ADD_OFFSETS,
        "group_size": ADD_GROUP,
        "threadgroup_bytes": 0,
        "source_sha256": text_sha256(ADD_SOURCE),
    }


def library_source() -> str:
    """Join the header, the three block scans and the add kernel into one text."""

    return HEADER + "".join(
        "\n" + text
        for text in (*(kernel.source for kernel in KERNELS.values()), ADD_SOURCE)
    )


def library_sha256() -> str:
    """The digest of the exact library text that ``compile_shader`` receives."""

    return text_sha256(library_source())


def rounding_depth(name: str, length: int) -> int:
    """The most float32 additions on a path from an input to an output.

    Each level adds its block depth, and each add of a scanned block sum adds
    one more, so ``L`` levels give ``L * block_depth + L - 1``.
    """

    if name not in KERNELS:
        raise GpuScanError(f"{name!r} is not a registered scan kernel")
    count = len(level_lengths(length))
    return count * KERNELS[name].block_depth + count - 1
