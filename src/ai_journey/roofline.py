"""Analytical kernel traffic for roofline planning.

The estimates model compulsory tensor traffic. They do not measure a kernel,
cache behavior, fusion, allocator traffic, or accelerator performance.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from .resource_accounting import ResourceAccountingError, bytes_per_element


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ResourceAccountingError(f"{name} must be positive")
    return value


def _positive_finite(value: float, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ResourceAccountingError(f"{name} must be positive and finite")
    return float(value)


@dataclass(frozen=True)
class KernelWorkload:
    """A reproducible FLOP and compulsory-byte estimate for one kernel family."""

    kind: str
    flops: int
    bytes_read: int
    bytes_written: int

    def __post_init__(self) -> None:
        if not isinstance(self.kind, str) or not self.kind:
            raise ResourceAccountingError("kind must be a nonempty string")
        _positive_integer(self.flops, "flops")
        _positive_integer(self.bytes_read, "bytes_read")
        _positive_integer(self.bytes_written, "bytes_written")

    @property
    def total_bytes(self) -> int:
        return self.bytes_read + self.bytes_written

    @property
    def arithmetic_intensity(self) -> float:
        """Return modeled FLOPs per compulsory byte transferred."""

        return self.flops / self.total_bytes


@dataclass(frozen=True)
class RooflineHardware:
    """Declared peak compute and memory bandwidth for analytical planning."""

    peak_flops_per_second: float
    memory_bandwidth_bytes_per_second: float

    def __post_init__(self) -> None:
        _positive_finite(self.peak_flops_per_second, "peak_flops_per_second")
        _positive_finite(
            self.memory_bandwidth_bytes_per_second,
            "memory_bandwidth_bytes_per_second",
        )

    @property
    def ridge_point(self) -> float:
        """Return the arithmetic intensity where the two ceilings intersect."""

        return self.peak_flops_per_second / self.memory_bandwidth_bytes_per_second


@dataclass(frozen=True)
class RooflineEstimate:
    """Lower-bound timing and bottleneck classification for one workload."""

    workload: KernelWorkload
    hardware: RooflineHardware
    compute_seconds: float
    memory_seconds: float
    lower_bound_seconds: float
    attainable_flops_per_second: float
    bottleneck: str


def analyze_roofline(
    workload: KernelWorkload, hardware: RooflineHardware
) -> RooflineEstimate:
    """Classify one workload against declared roofline ceilings.

    Times are analytical lower bounds. Equality at the ridge is reported
    explicitly instead of being assigned to either limiting resource.
    """

    if not isinstance(workload, KernelWorkload):
        raise TypeError("workload must be a KernelWorkload")
    if not isinstance(hardware, RooflineHardware):
        raise TypeError("hardware must be RooflineHardware")
    compute_seconds = workload.flops / hardware.peak_flops_per_second
    memory_seconds = workload.total_bytes / hardware.memory_bandwidth_bytes_per_second
    if compute_seconds > memory_seconds:
        bottleneck = "compute"
    elif memory_seconds > compute_seconds:
        bottleneck = "memory"
    else:
        bottleneck = "ridge"
    return RooflineEstimate(
        workload=workload,
        hardware=hardware,
        compute_seconds=compute_seconds,
        memory_seconds=memory_seconds,
        lower_bound_seconds=max(compute_seconds, memory_seconds),
        attainable_flops_per_second=min(
            hardware.peak_flops_per_second,
            hardware.memory_bandwidth_bytes_per_second * workload.arithmetic_intensity,
        ),
        bottleneck=bottleneck,
    )


def matmul_workload(
    *,
    rows: int,
    inner: int,
    columns: int,
    batch_count: int = 1,
    dtype: str,
) -> KernelWorkload:
    """Estimate dense matmul FLOPs and compulsory input/output traffic.

    Every batch member owns its left and right operands. The model counts one
    multiply and one add per scalar product term and one output write. It does
    not count accumulator spills or reuse beyond the compulsory tensor reads.
    """

    m = _positive_integer(rows, "rows")
    k = _positive_integer(inner, "inner")
    n = _positive_integer(columns, "columns")
    batches = _positive_integer(batch_count, "batch_count")
    width = bytes_per_element(dtype)
    return KernelWorkload(
        kind="matmul",
        flops=2 * batches * m * k * n,
        bytes_read=batches * (m * k + k * n) * width,
        bytes_written=batches * m * n * width,
    )


def layer_norm_workload(
    *,
    tokens: int,
    hidden_size: int,
    dtype: str,
    operations_per_element: float = 8.0,
    input_passes: int = 2,
) -> KernelWorkload:
    """Estimate a non-fused LayerNorm with explicit work and pass budgets.

    ``input_passes`` counts full input reads. One output tensor is always
    written. Scale and bias traffic, cache reuse, and fused residual traffic are
    excluded so callers can see exactly which simplified model is being used.
    """

    token_count = _positive_integer(tokens, "tokens")
    hidden = _positive_integer(hidden_size, "hidden_size")
    passes = _positive_integer(input_passes, "input_passes")
    operations = _positive_finite(operations_per_element, "operations_per_element")
    width = bytes_per_element(dtype)
    elements = token_count * hidden
    flops = math.ceil(elements * operations)
    return KernelWorkload(
        kind="layer_norm",
        flops=flops,
        bytes_read=elements * width * passes,
        bytes_written=elements * width,
    )
