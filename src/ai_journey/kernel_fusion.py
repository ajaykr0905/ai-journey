"""Analytical traffic comparison for fused elementwise kernel pipelines.

The model counts compulsory tensor traffic for a linear chain of elementwise
stages. It does not execute Triton, CUDA, or any accelerator kernel and does
not model cache hits, launch overhead, occupancy, or compiler decisions.
"""

from __future__ import annotations

from dataclasses import dataclass

from .resource_accounting import ResourceAccountingError, bytes_per_element
from .roofline import (
    KernelWorkload,
    RooflineEstimate,
    RooflineHardware,
    analyze_roofline,
)


def _positive_integer(value: int, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ResourceAccountingError(f"{name} must be positive")
    return value


@dataclass(frozen=True)
class ElementwiseStage:
    """One stage in a single-input, single-output elementwise pipeline."""

    name: str
    flops_per_element: int
    auxiliary_reads_per_element: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ResourceAccountingError("stage name must be nonempty")
        _positive_integer(self.flops_per_element, "flops_per_element")
        if type(self.auxiliary_reads_per_element) is not int:
            raise TypeError("auxiliary_reads_per_element must be an integer")
        if self.auxiliary_reads_per_element < 0:
            raise ResourceAccountingError(
                "auxiliary_reads_per_element must be nonnegative"
            )


@dataclass(frozen=True)
class FusionComparison:
    """Roofline lower bounds for materialized and fused implementations."""

    elements: int
    dtype: str
    stages: tuple[ElementwiseStage, ...]
    unfused: RooflineEstimate
    fused: RooflineEstimate

    @property
    def intermediate_bytes_eliminated(self) -> int:
        return self.unfused.workload.total_bytes - self.fused.workload.total_bytes

    @property
    def traffic_reduction_fraction(self) -> float:
        return self.intermediate_bytes_eliminated / self.unfused.workload.total_bytes

    @property
    def roofline_lower_bound_ratio(self) -> float:
        """Return unfused/fused analytical lower-bound time, not measured speedup."""

        return self.unfused.lower_bound_seconds / self.fused.lower_bound_seconds


def compare_elementwise_fusion(
    *,
    elements: int,
    dtype: str,
    stages: tuple[ElementwiseStage, ...],
    hardware: RooflineHardware,
) -> FusionComparison:
    """Compare materialized intermediates with one fused kernel boundary.

    Each stage reads one primary tensor and writes one result tensor. Auxiliary
    reads represent per-element inputs such as scale, bias, or a residual. The
    fused form reads the original primary tensor once, preserves all auxiliary
    reads, and writes only the final result.
    """

    count = _positive_integer(elements, "elements")
    if not isinstance(stages, tuple) or not stages:
        raise ResourceAccountingError("stages must be a nonempty tuple")
    if not isinstance(hardware, RooflineHardware):
        raise TypeError("hardware must be RooflineHardware")
    if any(not isinstance(stage, ElementwiseStage) for stage in stages):
        raise TypeError("every stage must be an ElementwiseStage")
    names = [stage.name for stage in stages]
    if len(set(names)) != len(names):
        raise ResourceAccountingError("stage names must be unique")

    width = bytes_per_element(dtype)
    primary_tensor_bytes = count * width
    auxiliary_bytes = (
        count * width * sum(stage.auxiliary_reads_per_element for stage in stages)
    )
    flops = count * sum(stage.flops_per_element for stage in stages)
    unfused_workload = KernelWorkload(
        kind="unfused_elementwise_pipeline",
        flops=flops,
        bytes_read=len(stages) * primary_tensor_bytes + auxiliary_bytes,
        bytes_written=len(stages) * primary_tensor_bytes,
    )
    fused_workload = KernelWorkload(
        kind="fused_elementwise_pipeline",
        flops=flops,
        bytes_read=primary_tensor_bytes + auxiliary_bytes,
        bytes_written=primary_tensor_bytes,
    )
    return FusionComparison(
        count,
        dtype,
        stages,
        analyze_roofline(unfused_workload, hardware),
        analyze_roofline(fused_workload, hardware),
    )
