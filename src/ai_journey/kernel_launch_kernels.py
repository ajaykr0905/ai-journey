"""Lecture-style per-thread kernels for the launch simulator.

Each kernel computes its global index from the CUDA built-ins, as GPU MODE
Lecture 3 does, and guards that index before it touches memory: ``if i < n``
in 1D, and a test on both axes in 2D. The run helpers size the grid with
``ceil_div``, copy the inputs into device buffers, launch the kernel in one
declared order and copy the output back to the host.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction

from .kernel_launch_buffer import DeviceBuffer, to_float32
from .kernel_launch_config import (
    Dim3,
    KernelLaunchError,
    launch_1d,
    launch_2d,
    require_int,
)
from .kernel_launch_reference import GRAYSCALE_WEIGHTS
from .kernel_launch_simulator import LaunchOrder, LaunchResult, ThreadContext, launch

GRAYSCALE_WEIGHTS_PER_10000 = tuple(
    int(Fraction(weight) * 10_000) for weight in GRAYSCALE_WEIGHTS
)
_RED, _GREEN, _BLUE = GRAYSCALE_WEIGHTS_PER_10000


def luma_u8(red: int, green: int, blue: int) -> int:
    """Return the gray value of one pixel by the stated rule, in integers.

    The weights are exact decimals, so ``2989 r + 5870 g + 1140 b`` is the
    exact weighted sum times 10,000. Adding 5,000 before the floor division
    rounds half up, and ``min`` clamps to 255. The CUDA source uses the same
    unsigned integer expression, so no float rounding can move a tie.
    """

    total = _RED * red + _GREEN * green + _BLUE * blue
    return min(255, (total + 5_000) // 10_000)


def vector_add_kernel(
    t: ThreadContext, a: DeviceBuffer, b: DeviceBuffer, out: DeviceBuffer, n: int
) -> None:
    """Run one thread of ``out[i] = a[i] + b[i]``, guarded by ``if i < n``."""

    i = t.block_idx.x * t.block_dim.x + t.thread_idx.x
    if i < n:
        out[i] = to_float32(a[i] + b[i])


def rgb_to_grayscale_kernel(
    t: ThreadContext, rgb: DeviceBuffer, gray: DeviceBuffer, width: int, height: int
) -> None:
    """Run one thread of the HxWx3 to gray conversion, guarded on both axes."""

    col = t.block_idx.x * t.block_dim.x + t.thread_idx.x
    row = t.block_idx.y * t.block_dim.y + t.thread_idx.y
    if col < width and row < height:
        pixel = row * width + col
        gray[pixel] = luma_u8(rgb[3 * pixel], rgb[3 * pixel + 1], rgb[3 * pixel + 2])


@dataclass(frozen=True)
class KernelRun:
    """One simulated launch of a lab kernel, with its output and its log."""

    kernel: str
    extent: Dim3
    output_buffer: str
    result: LaunchResult
    output: tuple[int | float, ...]
    output_sha256: str

    @property
    def useful_threads(self) -> int:
        """Count threads that wrote an output element: those the guard admitted."""

        return self.result.log.writer_threads[self.output_buffer]

    @property
    def idle_threads(self) -> int:
        return self.result.threads_run - self.useful_threads


def run_vector_add(
    a: Sequence[float],
    b: Sequence[float],
    *,
    threads_per_block: int,
    order: LaunchOrder | None = None,
) -> KernelRun:
    """Launch the vector add with one thread per element in ``order``."""

    if len(a) != len(b):
        raise KernelLaunchError("vector add operands need one length")
    n = len(a)
    config = launch_1d(n, threads_per_block)
    out = DeviceBuffer.zeros("out", "float32", n)
    result = launch(
        vector_add_kernel,
        config,
        DeviceBuffer("a", "float32", a),
        DeviceBuffer("b", "float32", b),
        out,
        n,
        order=order,
    )
    return KernelRun(
        kernel="vector_add",
        extent=Dim3(n),
        output_buffer="out",
        result=result,
        output=out.to_host(),
        output_sha256=out.sha256(),
    )


def run_rgb_to_grayscale(
    rgb: Sequence[int],
    *,
    width: int,
    height: int,
    block: tuple[int, int],
    order: LaunchOrder | None = None,
) -> KernelRun:
    """Launch the grayscale kernel with 2D blocks over image columns and rows."""

    require_int(width, "width", minimum=1, maximum=2**31 - 1)
    require_int(height, "height", minimum=1, maximum=2**31 - 1)
    if type(block) is not tuple or len(block) != 2:
        raise KernelLaunchError("block must be a (block_x, block_y) tuple")
    if len(rgb) != 3 * width * height:
        raise KernelLaunchError("an HxWx3 image needs 3 * width * height values")
    config = launch_2d(width, height, *block)
    gray = DeviceBuffer.zeros("gray", "uint8", width * height)
    result = launch(
        rgb_to_grayscale_kernel,
        config,
        DeviceBuffer("rgb", "uint8", rgb),
        gray,
        width,
        height,
        order=order,
    )
    return KernelRun(
        kernel="rgb_to_grayscale",
        extent=Dim3(width, height),
        output_buffer="gray",
        result=result,
        output=gray.to_host(),
        output_sha256=gray.sha256(),
    )
