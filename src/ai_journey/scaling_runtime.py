"""Translate analytical FLOPs into declared runtime and monetary estimates."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from .scaling_laws import _positive_decimal, _positive_integer


@dataclass(frozen=True)
class RuntimeAssumptions:
    sustained_flops_per_second: int | float | str
    accelerator_count: int
    hourly_cost_per_accelerator: int | float | str

    def __post_init__(self) -> None:
        _positive_decimal(
            self.sustained_flops_per_second,
            "sustained_flops_per_second",
        )
        _positive_integer(self.accelerator_count, "accelerator_count")
        _positive_decimal(
            self.hourly_cost_per_accelerator,
            "hourly_cost_per_accelerator",
        )


@dataclass(frozen=True)
class RuntimeEstimate:
    training_flops: int
    assumptions: RuntimeAssumptions
    seconds: Decimal
    accelerator_hours: Decimal
    monetary_cost: Decimal


def estimate_runtime_and_cost(
    training_flops: int, assumptions: RuntimeAssumptions
) -> RuntimeEstimate:
    """Estimate idealized wall time and cost from declared sustained throughput."""

    work = _positive_integer(training_flops, "training_flops")
    if not isinstance(assumptions, RuntimeAssumptions):
        raise TypeError("assumptions must be RuntimeAssumptions")
    per_accelerator = _positive_decimal(
        assumptions.sustained_flops_per_second,
        "sustained_flops_per_second",
    )
    seconds = Decimal(work) / (per_accelerator * Decimal(assumptions.accelerator_count))
    accelerator_hours = (
        seconds * Decimal(assumptions.accelerator_count) / Decimal(3_600)
    )
    cost = accelerator_hours * _positive_decimal(
        assumptions.hourly_cost_per_accelerator,
        "hourly_cost_per_accelerator",
    )
    return RuntimeEstimate(
        training_flops=work,
        assumptions=assumptions,
        seconds=seconds,
        accelerator_hours=accelerator_hours,
        monetary_cost=cost,
    )
