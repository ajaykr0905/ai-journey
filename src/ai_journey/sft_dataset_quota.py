"""Deterministic largest-remainder quotas for balanced SFT task selection."""

import math

from .sft_dataset_record import SFTDatasetError


def category_quotas(counts: dict[str, int], target: int) -> dict[str, int]:
    """Apportion target examples proportionally while retaining every category."""
    if (
        not isinstance(counts, dict)
        or not counts
        or any(
            not isinstance(category, str)
            or not category
            or type(count) is not int
            or count < 1
            for category, count in counts.items()
        )
    ):
        raise SFTDatasetError("category counts must be nonempty positive integers")
    if type(target) is not int or not len(counts) <= target <= sum(counts.values()):
        raise SFTDatasetError(
            "selection target cannot preserve every available category"
        )
    total = sum(counts.values())
    ideals = {category: target * count / total for category, count in counts.items()}
    quotas = {
        category: min(counts[category], max(1, math.floor(ideal)))
        for category, ideal in ideals.items()
    }
    while sum(quotas.values()) < target:
        candidates = [
            category for category in counts if quotas[category] < counts[category]
        ]
        if not candidates:
            raise SFTDatasetError("category capacities cannot satisfy selection target")
        selected = min(
            candidates,
            key=lambda category: (
                -(ideals[category] - math.floor(ideals[category])),
                -counts[category],
                category,
            ),
        )
        quotas[selected] += 1
        ideals[selected] = math.floor(ideals[selected])
    while sum(quotas.values()) > target:
        candidates = [category for category in counts if quotas[category] > 1]
        if not candidates:
            raise SFTDatasetError("minimum category coverage exceeds selection target")
        selected = min(
            candidates,
            key=lambda category: (
                ideals[category] - math.floor(ideals[category]),
                counts[category],
                category,
            ),
        )
        quotas[selected] -= 1
        ideals[selected] = math.ceil(ideals[selected])
    return dict(sorted(quotas.items()))
