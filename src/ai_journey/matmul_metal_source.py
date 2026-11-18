"""Metal Shading Language source of the two naive float32 matmul kernels.

``torch.mps.compile_shader`` compiles this text at run time. Each kernel
gives one thread one element of ``C = A @ B`` from ``thread_position_in_grid``
and guards both axes, because the grid is rounded up to whole threadgroups on
each axis. The kernels differ only in the two lines that name the output
coordinates: ``matmul_x_columns`` takes the column of C from x, and
``matmul_x_rows`` takes the row from x. The guard, the loop and the store
come from one template, so the two kernels cannot drift apart.

The loop applies an explicit ``fma`` for each k in increasing order. The
Metal Shading Language defines ``fma(a, b, c)`` as ``a * b + c`` with a single
rounding, so the source fixes every rounding of each output, and a CPU can
replay a kernel bit for bit. The Day 73 probe showed that this host's
compiler fuses ``x * y + z`` on its own; the explicit call takes that choice
away from the compiler. The correctness cases check the replay on every run.

Scalars are 32-bit ``constant uint&`` parameters, which the runtime passes
through ``arg_casts``, and every buffer binding is explicit.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

from .matmul_metal_mapping import X_COLUMNS, X_ROWS, require_mapping

MATMUL_X_COLUMNS = "matmul_x_columns"
MATMUL_X_ROWS = "matmul_x_rows"
BUFFERS = ("c", "a", "b", "m", "n", "k")
SCALAR_BUFFERS = (3, 4, 5)
LOOP_STATEMENT = "sum = fma(a[row * k + i], b[i * n + column], sum);"

# The guard, the K loop and the store that both kernels share, byte for byte.
GUARDED_LOOP = f"""    if (row < m && column < n) {{
        float sum = 0.0f;
        for (uint i = 0; i < k; ++i) {{
            {LOOP_STATEMENT}
        }}
        c[row * n + column] = sum;
    }}"""


def _kernel(name: str, coordinates: str, comment: str) -> str:
    signature = f"kernel void {name}("
    indent = " " * len(signature)
    parameters = f",\n{indent}".join(
        (
            "device float* c [[buffer(0)]]",
            "device const float* a [[buffer(1)]]",
            "device const float* b [[buffer(2)]]",
            "constant uint& m [[buffer(3)]]",
            "constant uint& n [[buffer(4)]]",
            "constant uint& k [[buffer(5)]]",
            "uint2 position [[thread_position_in_grid]]",
        )
    )
    body = f"{coordinates}\n{GUARDED_LOOP}\n}}\n"
    return f"{comment}\n{signature}{parameters}) {{\n{body}"


SHADER_SOURCE = (
    "#include <metal_stdlib>\nusing namespace metal;\n\n"
    + _kernel(
        MATMUL_X_COLUMNS,
        "    uint column = position.x;\n    uint row = position.y;",
        "// One thread per element of C = A @ B; all three are row-major float32.\n"
        "// x is the column of C and y its row, so threads that are consecutive\n"
        "// in x read consecutive elements of B. The grid is whole threadgroups,\n"
        "// so the guard stops the padding threads past n and past m.",
    )
    + "\n"
    + _kernel(
        MATMUL_X_ROWS,
        "    uint row = position.x;\n    uint column = position.y;",
        "// The same guard, loop and store with x as the row of C and y as its\n"
        "// column, so threads that are consecutive in x read elements of A\n"
        "// that lie k floats apart.",
    )
)


@dataclass(frozen=True)
class MatmulKernelSpec:
    """One kernel's mapping, buffer bindings and 32-bit scalar bindings."""

    name: str
    mapping: str
    buffers: tuple[str, ...] = BUFFERS
    scalar_buffers: tuple[int, ...] = SCALAR_BUFFERS

    def __post_init__(self) -> None:
        require_mapping(self.mapping)

    @property
    def arg_casts(self) -> dict[int, str]:
        """Pass each scalar as 32 bits, the width of ``constant uint&``."""

        return {index: "int32" for index in self.scalar_buffers}

    def to_mapping(self) -> dict[str, object]:
        """Return the bindings a report records."""

        return {
            "name": self.name,
            "mapping": self.mapping,
            "axes": 2,
            "buffers": list(self.buffers),
            "scalar_buffers": list(self.scalar_buffers),
        }


KERNELS = {
    MATMUL_X_COLUMNS: MatmulKernelSpec(MATMUL_X_COLUMNS, X_COLUMNS),
    MATMUL_X_ROWS: MatmulKernelSpec(MATMUL_X_ROWS, X_ROWS),
}
KERNEL_FOR_MAPPING = {spec.mapping: name for name, spec in KERNELS.items()}


def source_sha256() -> str:
    """Hash the exact source text that ``compile_shader`` receives."""

    return sha256(SHADER_SOURCE.encode("ascii")).hexdigest()
