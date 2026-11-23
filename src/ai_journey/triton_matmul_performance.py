"""The performance half of the Day 79 comparison, stated without a GPU number.

This host has no NVIDIA or AMD GPU, so no Triton kernel ran on a GPU here,
and the lab gives no Triton GPU rate. The CPU interpreter runs each program as
Python and NumPy code, so its time says nothing about a GPU, and the lab does
not report it. The Day 76 and Day 77 CUDA texts and the cuBLAS call text were
never compiled, so they have no rate either.

The only matmul rates of this host come from two recorded runs on its Apple
GPU, through Metal: the Day 76 tiled run and the Day 77 checklist run.
``METAL_RATES`` copies their medians and IQR ranges at the four shapes that
both runs timed, and ``METAL_REPORTS`` names the digest, the guide and the
load averages of each report. Each rate is a host-specific Apple GPU (Metal)
measurement from one run. No rate here predicts a Triton or CUDA rate on
another GPU.

``LATER_GPU_RUN`` lists what a run on a GPU host must do to give the missing
numbers.
"""

from __future__ import annotations

from dataclasses import dataclass

from .memory_access_tiling import MatmulShape
from .triton_matmul_kernel import TritonMatmulError

TRITON_GPU = "not measured: no NVIDIA or AMD GPU on this host"
CUDA_TEXT = (
    "not measured: no NVIDIA GPU, CUDA toolkit or nvcc on this host, and the "
    "Day 76 and Day 77 CUDA texts were never compiled"
)
CUBLAS = "not measured: no NVIDIA GPU or cuBLAS on this host"
INTERPRETER_TIME = (
    "not reported: the Triton CPU interpreter runs each program as Python and "
    "NumPy code on the CPU, so its time is not a performance number"
)
BOUNDARY = (
    "Each Metal rate is a host-specific Apple GPU (Metal) measurement from one "
    "recorded run at the load averages of its report. The lab measured no rate: "
    "it copies the recorded rates, and no rate here measures or predicts Triton, "
    "CUDA, cuBLAS, NVIDIA or AMD performance."
)
PERFORMANCE_SHAPES = (
    MatmulShape(1024, 1024, 1024),
    MatmulShape(2048, 2048, 2048),
    MatmulShape(4096, 4096, 4096),
    MatmulShape(2000, 3000, 1000),
)
# Why each recorded Metal kernel stands in this table.
CITED_KERNELS = {
    ("day-76", "matmul_tiled_32"): (
        "the tiling of the Day 76 CUDA text at tile 32: one output for each "
        "thread, 32 x 32 tiles"
    ),
    ("day-76", "torch_matmul"): "torch.matmul on the MPS device in the Day 76 run",
    ("day-77", "matmul_s3_block_4x4"): (
        "the blocking of the Day 77 CUDA text blocked_matmul_m64_n64_k16_t4x4: "
        "64 x 64 x 16 tiles, 4 x 4 outputs for each thread"
    ),
    ("day-77", "matmul_s7_interior"): "the fastest Metal kernel of the Day 77 run",
    ("day-77", "torch_matmul"): "torch.matmul on the MPS device in the Day 77 run",
}
LATER_GPU_RUN = (
    "Use a Linux host with an NVIDIA or AMD GPU. Install triton 3.9.0 and a "
    "torch build for that GPU, leave TRITON_INTERPRET unset, and record the GPU "
    "name, the driver and the CUDA or ROCm version.",
    "Check that TRITON_SOURCE still has its pinned SHA-256 digest. Run the "
    "correctness cases of config/day-79-triton-matmul.json on the GPU: each "
    "normal output must keep the Day 74 bound, and each exact output must hold "
    "its exact bits.",
    "Wrap matmul_kernel in triton.autotune with AUTOTUNE_CONFIGS and the key "
    "M, N, K. Record the config that the tuner picks at each shape and each "
    "config that fails to compile. The tuning times are not results.",
    "Time the four shapes of this table with the Day 74 benchmark operands: "
    "1024 x 1024 x 1024, 2048 x 2048 x 2048, 4096 x 4096 x 4096 and "
    "2000 x 3000 x 1000.",
    "Give each side 2 calls that are not timed and then 11 timed samples, with "
    "CUDA events or triton.testing.do_bench. Move the first side by one at "
    "each repeat, as the Day 76 Metal lab does.",
    "On an NVIDIA GPU, time the tuned Triton kernel, the Day 76 CUDA text at "
    "tile 32, the five Day 77 CUDA texts and sgemm_row_major with cuBLAS on "
    "the same device operands, and check the Day 76 TF32 assumption. On an AMD "
    "GPU, record that the CUDA texts and cuBLAS do not run there.",
    "Record the load averages before and after the timing. Divide "
    "2 * M * N * K by each median time to get GFLOP/s, and apply the Day 76 "
    "ratio rule: a ratio is faster only when it is 1.1 or more and the IQR "
    "ranges do not overlap.",
    "Seal the times, the host records and the correctness results in a report "
    "that a verifier checks. Compare a GPU rate with a Metal rate of this "
    "table only as a rate of other hardware.",
)


@dataclass(frozen=True)
class MetalReport:
    """One recorded Metal run: its digest, guide, time and load averages."""

    lab: str
    report_sha256: str
    run: str
    guide: str
    load_before: tuple[float, float, float]
    load_after: tuple[float, float, float]

    def to_mapping(self) -> dict[str, object]:
        return {
            "lab": self.lab,
            "report_sha256": self.report_sha256,
            "run": self.run,
            "guide": self.guide,
            "load_average_before": list(self.load_before),
            "load_average_after": list(self.load_after),
            "measurement": "host-specific Apple GPU (Metal) measurement",
        }


@dataclass(frozen=True)
class MetalRate:
    """One recorded median and IQR range in GFLOP/s, at one shape."""

    lab: str
    kernel: str
    shape: MatmulShape
    median: float
    low: float
    high: float

    def __post_init__(self) -> None:
        if (self.lab, self.kernel) not in CITED_KERNELS:
            raise TritonMatmulError(f"{self.kernel} is not a cited Metal kernel")
        if self.shape not in PERFORMANCE_SHAPES:
            raise TritonMatmulError("a Metal rate must be at a performance shape")
        if not 0 < self.low <= self.median <= self.high:
            raise TritonMatmulError("a median must lie inside its IQR range")

    def to_mapping(self) -> dict[str, object]:
        return {
            "lab": self.lab,
            "kernel": self.kernel,
            "stands_for": CITED_KERNELS[(self.lab, self.kernel)],
            "median_gflops_per_second": self.median,
            "iqr_gflops_per_second": [self.low, self.high],
        }


METAL_REPORTS = (
    MetalReport(
        "day-76",
        "371d65363ce5c53813c34a6edfc9db4725445bf4ffea605b7c81e0076fc03427",
        "2026-11-19 11:43 IST",
        "docs/matmul-tiled-lab.md",
        (9.93, 11.62, 14.06),
        (9.70, 11.40, 13.90),
    ),
    MetalReport(
        "day-77",
        "130322a7b46db6bc7e9e9bf135a39225e2c139097cd167b74ceb22fd54b2e062",
        "2026-11-20 11:29 IST",
        "docs/matmul-checklist-lab.md",
        (19.21, 21.17, 23.01),
        (16.63, 20.39, 22.64),
    ),
)
_CUBE_1024, _CUBE_2048, _CUBE_4096, _RAGGED = PERFORMANCE_SHAPES
METAL_RATES = (
    MetalRate("day-76", "matmul_tiled_32", _CUBE_1024, 338.5, 332.6, 342.8),
    MetalRate("day-76", "torch_matmul", _CUBE_1024, 2294.1, 2002.0, 2380.0),
    MetalRate("day-77", "matmul_s3_block_4x4", _CUBE_1024, 920.3, 571.7, 1049.5),
    MetalRate("day-77", "matmul_s7_interior", _CUBE_1024, 1312.0, 1145.2, 1469.4),
    MetalRate("day-77", "torch_matmul", _CUBE_1024, 2246.2, 1840.6, 2534.7),
    MetalRate("day-76", "matmul_tiled_32", _CUBE_2048, 339.6, 337.1, 348.3),
    MetalRate("day-76", "torch_matmul", _CUBE_2048, 3608.8, 3464.3, 3656.4),
    MetalRate("day-77", "matmul_s3_block_4x4", _CUBE_2048, 1177.9, 1116.9, 1222.9),
    MetalRate("day-77", "matmul_s7_interior", _CUBE_2048, 1774.7, 1668.4, 1819.4),
    MetalRate("day-77", "torch_matmul", _CUBE_2048, 3390.3, 2710.8, 3505.9),
    MetalRate("day-76", "matmul_tiled_32", _CUBE_4096, 353.5, 333.1, 359.1),
    MetalRate("day-76", "torch_matmul", _CUBE_4096, 3940.1, 3703.5, 4010.8),
    MetalRate("day-77", "matmul_s3_block_4x4", _CUBE_4096, 1160.2, 1153.2, 1195.8),
    MetalRate("day-77", "matmul_s7_interior", _CUBE_4096, 1825.8, 1815.1, 1832.2),
    MetalRate("day-77", "torch_matmul", _CUBE_4096, 3521.4, 3250.9, 3698.8),
    MetalRate("day-76", "matmul_tiled_32", _RAGGED, 342.3, 339.8, 345.4),
    MetalRate("day-76", "torch_matmul", _RAGGED, 3448.9, 3269.8, 3483.6),
    MetalRate("day-77", "matmul_s3_block_4x4", _RAGGED, 1334.0, 1293.8, 1368.5),
    MetalRate("day-77", "matmul_s7_interior", _RAGGED, 1665.8, 1505.8, 1737.9),
    MetalRate("day-77", "torch_matmul", _RAGGED, 3042.8, 2966.5, 3291.3),
)


def performance_mapping() -> dict[str, object]:
    """Return the recorded Metal rates and the unmeasured sides, by shape."""

    return {
        "boundary": BOUNDARY,
        "metal_reports": [report.to_mapping() for report in METAL_REPORTS],
        "shapes": [
            {
                "shape": shape.to_mapping(),
                "metal": [
                    rate.to_mapping() for rate in METAL_RATES if rate.shape == shape
                ],
                "triton_gpu": TRITON_GPU,
                "cuda_text": CUDA_TEXT,
                "cublas": CUBLAS,
            }
            for shape in PERFORMANCE_SHAPES
        ],
        "interpreter_time": INTERPRETER_TIME,
        "later_gpu_run": list(LATER_GPU_RUN),
    }
