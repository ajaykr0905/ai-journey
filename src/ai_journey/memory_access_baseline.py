"""Real float32 CPU matmul baselines for NumPy and PyTorch on this host.

Every duration produced with the default host timer is a wall-clock
measurement of a CPU matmul on the machine that ran it; a measurement records
which timer produced its samples, so durations from an injected clock stay
labeled synthetic. Nothing is a GPU measurement or prediction. Inputs
are seeded standard-normal float32 matrices. After its clock stops, every
warmup and measured result is checked against a float64 reference: each
element must lie within ``(gamma_K(2**-24) + gamma_K(2**-53)) * (|A| @ |B|)``,
where ``gamma_K(u) = K * u / (1 - K * u)`` is the classical inner-product
rounding bound (N. J. Higham, Accuracy and Stability of Numerical Algorithms,
2nd ed., SIAM, 2002, section 3.1). The bound holds for every summation order,
so a correct float32 matmul cannot fail it, while reduced-precision math or a
wrong result does.
"""

from __future__ import annotations

import os
import platform
import time
from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256

import numpy
import torch

from .memory_access_benchmark import (
    TimingSummary,
    gflops_per_second,
    summarize_timings,
    time_repeats,
    timer_name,
)
from .memory_access_pattern import MemoryAccessError
from .memory_access_tiling import MatmulShape

BACKENDS = ("numpy", "torch")
BASELINE_DTYPE = "float32"
MAX_BASELINE_DIMENSION = 1024
MAX_BASELINE_SEED = 2**32 - 1
FLOAT32_UNIT_ROUNDOFF = 2.0**-24
FLOAT64_UNIT_ROUNDOFF = 2.0**-53


def _gamma(inner: int, unit_roundoff: float) -> float:
    product = inner * unit_roundoff
    return product / (1 - product)


def rounding_tolerance(inner: int) -> float:
    """Return the relative bound for one float32 dot product of length ``inner``."""

    if type(inner) is not int:
        raise TypeError("inner must be an integer")
    if not 1 <= inner <= MAX_BASELINE_DIMENSION:
        raise MemoryAccessError(f"inner must be in [1, {MAX_BASELINE_DIMENSION}]")
    return _gamma(inner, FLOAT32_UNIT_ROUNDOFF) + _gamma(inner, FLOAT64_UNIT_ROUNDOFF)


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


def _seed(seed: object) -> int:
    if type(seed) is not int:
        raise TypeError("seed must be an integer")
    if not 0 <= seed <= MAX_BASELINE_SEED:
        raise MemoryAccessError(f"seed must be in [0, {MAX_BASELINE_SEED}]")
    return seed


def matmul_inputs(shape: MatmulShape, seed: int) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Draw seeded standard-normal float32 operands A[m, k] and B[k, n]."""

    if not isinstance(shape, MatmulShape):
        raise TypeError("shape must be a MatmulShape")
    generator = numpy.random.default_rng(_seed(seed))
    left = generator.standard_normal((shape.m, shape.k), dtype=numpy.float32)
    right = generator.standard_normal((shape.k, shape.n), dtype=numpy.float32)
    return left, right


def inputs_sha256(left: numpy.ndarray, right: numpy.ndarray) -> str:
    """Fingerprint both operands as shaped little-endian float32 bytes."""

    digest = sha256()
    for array in (left, right):
        if not isinstance(array, numpy.ndarray) or array.dtype != numpy.float32:
            raise TypeError("operands must be float32 NumPy arrays")
        digest.update(f"{array.shape}".encode())
        digest.update(numpy.ascontiguousarray(array, dtype="<f4").tobytes())
    return digest.hexdigest()


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


class MatmulChecker:
    """Check float32 results against one float64 reference and record maxima."""

    def __init__(self, left: numpy.ndarray, right: numpy.ndarray) -> None:
        if (
            not isinstance(left, numpy.ndarray)
            or not isinstance(right, numpy.ndarray)
            or left.ndim != 2
            or right.ndim != 2
            or left.shape[1] != right.shape[0]
        ):
            raise MemoryAccessError("operands must be conformable 2D arrays")
        left64 = left.astype(numpy.float64)
        right64 = right.astype(numpy.float64)
        self.shape = (left.shape[0], right.shape[1])
        self.tolerance = rounding_tolerance(left.shape[1])
        self.reference = left64 @ right64
        self.bound = self.tolerance * (numpy.abs(left64) @ numpy.abs(right64))
        self.checked_results = 0
        self.max_abs_error = 0.0
        self.max_bound_ratio = 0.0

    def _array(self, result: object) -> numpy.ndarray:
        if isinstance(result, torch.Tensor):
            if result.device.type != "cpu" or result.dtype != torch.float32:
                raise MemoryAccessError("torch result must be a CPU float32 tensor")
            result = result.detach().numpy()
        if not isinstance(result, numpy.ndarray) or result.dtype != numpy.float32:
            raise MemoryAccessError("matmul result must be a float32 array")
        if result.shape != self.shape:
            raise MemoryAccessError("matmul result has the wrong shape")
        return result

    def __call__(self, result: object) -> None:
        array = self._array(result)
        if not numpy.isfinite(array).all():
            raise MemoryAccessError("matmul result is not finite")
        error = numpy.abs(array.astype(numpy.float64) - self.reference)
        if bool((error > self.bound).any()):
            raise MemoryAccessError(
                "matmul result exceeds the float32 rounding bound of its reference"
            )
        positive = self.bound > 0
        ratio = float((error[positive] / self.bound[positive]).max(initial=0.0))
        self.checked_results += 1
        self.max_abs_error = max(self.max_abs_error, float(error.max()))
        self.max_bound_ratio = max(self.max_bound_ratio, ratio)

    def result(self) -> CorrectnessCheck:
        if not self.checked_results:
            raise MemoryAccessError("no matmul result was checked")
        return CorrectnessCheck(
            tolerance=self.tolerance,
            checked_results=self.checked_results,
            max_abs_error=self.max_abs_error,
            max_bound_ratio=self.max_bound_ratio,
        )


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


def measure_matmul(
    case: MatmulCase,
    *,
    warmup: int,
    repeats: int,
    seed: int,
    clock: Callable[[], int] = time.perf_counter_ns,
) -> MatmulMeasurement:
    """Time one backend's float32 matmul and check every result it returns."""

    if not isinstance(case, MatmulCase):
        raise TypeError("case must be a MatmulCase")
    left, right = matmul_inputs(case.shape, seed)
    checker = MatmulChecker(left, right)
    if case.backend == "numpy":

        def operation() -> object:
            return numpy.matmul(left, right)

    else:
        left_tensor = torch.from_numpy(left)
        right_tensor = torch.from_numpy(right)

        def operation() -> object:
            return torch.matmul(left_tensor, right_tensor)

    samples = time_repeats(
        operation, warmup=warmup, repeats=repeats, clock=clock, inspect=checker
    )
    return MatmulMeasurement(
        case=case,
        warmup=warmup,
        repeats=repeats,
        seed=seed,
        input_sha256=inputs_sha256(left, right),
        samples_ns=samples,
        correctness=checker.result(),
        timer=timer_name(clock),
    )


def cpu_environment() -> dict[str, object]:
    """Describe the runtime without host names, user names, paths, or serials."""

    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "numpy_version": numpy.__version__,
        "torch_version": str(torch.__version__),
        "platform_system": platform.system() or "unknown",
        "platform_machine": platform.machine() or "unknown",
        "cpu_count": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "torch_float32_matmul_precision": torch.get_float32_matmul_precision(),
    }
