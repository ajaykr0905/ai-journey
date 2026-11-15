"""Torch-free records of the memory-access lab's CPU matmul baselines.

The plan, report and verifier modules import these names, so analyzing or
verifying a report never loads torch or NumPy; only the timing code in
``memory_access_baseline`` does, when ``benchmark`` runs.
"""

from __future__ import annotations

from dataclasses import dataclass

from .memory_access_benchmark import (
    TimingSummary,
    gflops_per_second,
    summarize_timings,
)
from .memory_access_pattern import MemoryAccessError
from .memory_access_tiling import MatmulShape

BACKENDS = ("numpy", "torch")
BASELINE_DTYPE = "float32"
MAX_BASELINE_DIMENSION = 1024
MAX_BASELINE_SEED = 2**32 - 1
# The keys ``memory_access_baseline.cpu_environment()`` records.
CPU_ENVIRONMENT_FIELDS = frozenset(
    {
        "python_version",
        "python_implementation",
        "numpy_version",
        "torch_version",
        "platform_system",
        "platform_machine",
        "cpu_count",
        "torch_threads",
        "torch_float32_matmul_precision",
    }
)


@dataclass(frozen=True)
class MatmulCase:
    """One bounded float32 CPU matmul to measure with one backend."""

    backend: str
    shape: MatmulShape

    def __post_init__(self) -> None:
        if self.backend not in BACKENDS:
            raise MemoryAccessError(f"backend must be one of {', '.join(BACKENDS)}")
        if not isinstance(self.shape, MatmulShape):
            raise TypeError("shape must be a MatmulShape")
        if max(self.shape.m, self.shape.n, self.shape.k) > MAX_BASELINE_DIMENSION:
            raise MemoryAccessError(
                f"baseline dimensions must not exceed {MAX_BASELINE_DIMENSION}"
            )


@dataclass(frozen=True)
class CorrectnessCheck:
    """Worst observed error of every checked result against float64."""

    tolerance: float
    checked_results: int
    max_abs_error: float
    max_bound_ratio: float

    def to_mapping(self) -> dict[str, object]:
        return {
            "reference": "float64",
            "bound": "(gamma_K(2**-24) + gamma_K(2**-53)) * (|A| @ |B|)",
            "tolerance": self.tolerance,
            "checked_results": self.checked_results,
            "max_abs_error": self.max_abs_error,
            "max_bound_ratio": self.max_bound_ratio,
        }


@dataclass(frozen=True)
class MatmulMeasurement:
    """Timed samples of one checked CPU matmul case and the timer behind them."""

    case: MatmulCase
    warmup: int
    repeats: int
    seed: int
    input_sha256: str
    samples_ns: tuple[int, ...]
    correctness: CorrectnessCheck
    timer: str

    @property
    def flops(self) -> int:
        return self.case.shape.flops

    @property
    def timing(self) -> TimingSummary:
        return summarize_timings(self.samples_ns)

    @property
    def median_gflops_per_second(self) -> float:
        return gflops_per_second(self.flops, self.timing.median_ns)

    @property
    def best_gflops_per_second(self) -> float:
        return gflops_per_second(self.flops, self.timing.min_ns)

    def to_mapping(self) -> dict[str, object]:
        return {
            "backend": self.case.backend,
            "device": "cpu",
            "dtype": BASELINE_DTYPE,
            **self.case.shape.to_mapping(),
            "warmup": self.warmup,
            "repeats": self.repeats,
            "seed": self.seed,
            "input_sha256": self.input_sha256,
            "samples_ns": list(self.samples_ns),
            "timing": self.timing.to_mapping(),
            "flops": self.flops,
            "median_gflops_per_second": self.median_gflops_per_second,
            "best_gflops_per_second": self.best_gflops_per_second,
            "correctness": self.correctness.to_mapping(),
        }
