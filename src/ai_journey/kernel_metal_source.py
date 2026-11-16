"""Metal Shading Language source of the vector-add and grayscale kernels.

``torch.mps.compile_shader`` compiles this text at run time, the Metal
counterpart of CUDA's ``load_inline``. Each kernel takes one thread per
element or pixel from ``thread_position_in_grid``, which equals CUDA's
``blockIdx * blockDim + threadIdx``, and guards it against the data extent,
because the grid is rounded up to whole threadgroups. Scalars are 32-bit
``constant uint&`` parameters, and every buffer binding is explicit.

The grayscale weights are formatted from ``GRAY_WEIGHTS``, so the kernel and
the CPU reference cannot drift apart. The source also holds a contraction
probe: one ``x[0] * x[1] + x[2]`` whose float32 result is 0 when the compiler
rounds the product and the sum separately, and 2**-24 when it fuses them
into one fused multiply-add. The probe shows whether this host's compiler
fuses; it cannot show which operations it fused inside another kernel.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

from .kernel_metal_reference import GRAY_WEIGHTS

VECTOR_ADD = "vector_add"
RGB_TO_GRAY = "rgb_to_gray"
CONTRACTION_PROBE = "contraction_probe"
PROBE_EXPRESSION = "x[0] * x[1] + x[2]"
# Each input is exact in float32. The exact product is 1 + 2**-11 + 2**-24;
# rounded to float32 it loses the 2**-24 tie to even, so only a fused
# multiply-add keeps it.
PROBE_INPUTS = (1.0 + 2.0**-12, 1.0 + 2.0**-12, -(1.0 + 2.0**-11))
PROBE_UNFUSED_RESULT = 0.0
PROBE_FUSED_RESULT = 2.0**-24


# 0.2989f * r + 0.5870f * g + 0.1140f * b, the reference's operation order.
GRAY_EXPRESSION = " + ".join(
    f"{weight:.4f}f * {channel}" for weight, channel in zip(GRAY_WEIGHTS, "rgb")
)

SHADER_SOURCE = f"""#include <metal_stdlib>
using namespace metal;

// One thread per element. thread_position_in_grid is CUDA's
// blockIdx.x * blockDim.x + threadIdx.x. The grid is whole threadgroups,
// so the guard stops the padding threads past n.
kernel void {VECTOR_ADD}(device float* out [[buffer(0)]],
                       device const float* a [[buffer(1)]],
                       device const float* b [[buffer(2)]],
                       constant uint& n [[buffer(3)]],
                       uint i [[thread_position_in_grid]]) {{
    if (i < n) {{
        out[i] = a[i] + b[i];
    }}
}}

// One thread per pixel: x is the column and y the row, as in a CUDA 2D
// launch. Pixels are row-major RGB triples; out holds one level per pixel.
// Round half up, then clamp to 0..255.
kernel void {RGB_TO_GRAY}(device uchar* out [[buffer(0)]],
                        device const uchar* rgb [[buffer(1)]],
                        constant uint& width [[buffer(2)]],
                        constant uint& height [[buffer(3)]],
                        uint2 position [[thread_position_in_grid]]) {{
    uint column = position.x;
    uint row = position.y;
    if (column < width && row < height) {{
        uint pixel = row * width + column;
        float r = float(rgb[3 * pixel]);
        float g = float(rgb[3 * pixel + 1]);
        float b = float(rgb[3 * pixel + 2]);
        float y = {GRAY_EXPRESSION};
        out[pixel] = uchar(clamp(floor(y + 0.5f), 0.0f, 255.0f));
    }}
}}

// Fused multiply-add probe: 0 if the product is rounded before the sum,
// 2**-24 if the compiler fuses them.
kernel void {CONTRACTION_PROBE}(device float* out [[buffer(0)]],
                              device const float* x [[buffer(1)]],
                              uint i [[thread_position_in_grid]]) {{
    if (i == 0) {{
        out[0] = {PROBE_EXPRESSION};
    }}
}}
"""


@dataclass(frozen=True)
class KernelSpec:
    """One kernel's axes, buffer bindings, scalar bindings and compulsory bytes.

    ``bytes_per_item`` counts what one element or pixel must read and write,
    the numerator of every effective bandwidth the lab reports.
    """

    name: str
    axes: int
    buffers: tuple[str, ...]
    scalar_buffers: tuple[int, ...]
    bytes_per_item: int

    @property
    def arg_casts(self) -> dict[int, str]:
        """Pass each scalar as 32 bits, the width of ``constant uint&``."""

        return {index: "int32" for index in self.scalar_buffers}

    def to_mapping(self) -> dict[str, object]:
        """Return the bindings and byte accounting a report records."""

        return {
            "name": self.name,
            "axes": self.axes,
            "buffers": list(self.buffers),
            "scalar_buffers": list(self.scalar_buffers),
            "bytes_per_item": self.bytes_per_item,
        }


KERNELS = {
    VECTOR_ADD: KernelSpec(
        name=VECTOR_ADD,
        axes=1,
        buffers=("out", "a", "b", "n"),
        scalar_buffers=(3,),
        # Two float32 reads and one float32 write per element.
        bytes_per_item=12,
    ),
    RGB_TO_GRAY: KernelSpec(
        name=RGB_TO_GRAY,
        axes=2,
        buffers=("out", "rgb", "width", "height"),
        scalar_buffers=(2, 3),
        # Three uint8 reads and one uint8 write per pixel.
        bytes_per_item=4,
    ),
}


def source_sha256() -> str:
    """Hash the exact source text that ``compile_shader`` receives."""

    return sha256(SHADER_SOURCE.encode("ascii")).hexdigest()
