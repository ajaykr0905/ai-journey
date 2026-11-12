"""Group-relative advantage normalization used by GRPO-style objectives."""

from __future__ import annotations

import math
from collections.abc import Iterable

from .verifiable_reward_contract import VerifiableRewardError


def group_relative_advantages(
    rewards: Iterable[float], *, epsilon: float = 1e-8
) -> tuple[float, ...]:
    values = tuple(float(value) for value in rewards)
    if len(values) < 2 or not all(math.isfinite(value) for value in values):
        raise VerifiableRewardError("group rewards require at least two finite values")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise VerifiableRewardError("epsilon must be finite and positive")
    mean = math.fsum(values) / len(values)
    variance = math.fsum((value - mean) ** 2 for value in values) / len(values)
    if variance == 0:
        return tuple(0.0 for _ in values)
    scale = math.sqrt(variance + epsilon)
    return tuple((value - mean) / scale for value in values)
