"""Deterministic compute-optimal training targets for dense language models.

These formulas are analytical planning inputs. They do not claim that a model
was trained, that a loss curve was measured, or that a particular scaling law
holds outside the declared scenario.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_CEILING
import math


class ScalingLawError(ValueError):
    """Raised when an analytical scaling-law contract is invalid."""


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ScalingLawError(f"{name} must be positive")
    return value


def _positive_decimal(value: int | float | str, name: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise TypeError(f"{name} must be a number or decimal string")
    if isinstance(value, float) and not math.isfinite(value):
        raise ScalingLawError(f"{name} must be positive and finite")
    try:
        result = Decimal(str(value))
    except InvalidOperation as error:
        raise ScalingLawError(f"{name} must be a decimal number") from error
    if not result.is_finite() or result <= 0:
        raise ScalingLawError(f"{name} must be positive and finite")
    return result


@dataclass(frozen=True)
class ComputeOptimalTarget:
    """A declared token/parameter ratio and training-compute convention."""

    parameter_count: int
    tokens_per_parameter: int | float | str = "20"
    training_flops_per_parameter_token: int | float | str = "6"

    def __post_init__(self) -> None:
        _positive_integer(self.parameter_count, "parameter_count")
        _positive_decimal(self.tokens_per_parameter, "tokens_per_parameter")
        _positive_decimal(
            self.training_flops_per_parameter_token,
            "training_flops_per_parameter_token",
        )

    @property
    def token_target(self) -> int:
        """Return the declared compute-optimal token target, rounded upward."""

        target = Decimal(self.parameter_count) * _positive_decimal(
            self.tokens_per_parameter, "tokens_per_parameter"
        )
        return int(target.to_integral_value(rounding=ROUND_CEILING))

    @property
    def training_flops(self) -> int:
        """Return analytical training FLOPs under the declared convention."""

        work = (
            Decimal(self.parameter_count)
            * Decimal(self.token_target)
            * _positive_decimal(
                self.training_flops_per_parameter_token,
                "training_flops_per_parameter_token",
            )
        )
        return int(work.to_integral_value(rounding=ROUND_CEILING))

    @property
    def actual_tokens_per_parameter(self) -> Decimal:
        """Return the realized ratio once the token target is rounded up."""

        return Decimal(self.token_target) / Decimal(self.parameter_count)
