"""Exposure-aware descriptive comparisons of aligned MoE expert histograms."""

import math

DRIFT_SCOPE = (
    "descriptive distribution drift over the same expert identity/order; "
    "not evidence of specialization, causality or architecture superiority"
)


def _histogram(counts):
    if (
        type(counts) not in {list, tuple}
        or not 2 <= len(counts) <= 64
        or any(
            type(value) is not int or not 0 <= value <= 2**31 - 1 for value in counts
        )
    ):
        raise ValueError("expert histogram requires two to 64 bounded integer counts")
    total = sum(counts)
    if not total:
        return {
            "counts": list(counts),
            "assignments": 0,
            "fractions": None,
            "entropy_nats": None,
            "normalized_entropy": None,
            "maximum_fraction": None,
            "effective_experts": None,
        }
    fractions = [count / total for count in counts]
    entropy = -math.fsum(value * math.log(value) for value in fractions if value)
    return {
        "counts": list(counts),
        "assignments": total,
        "fractions": fractions,
        "entropy_nats": entropy,
        "normalized_entropy": entropy / math.log(len(counts)),
        "maximum_fraction": max(fractions),
        "effective_experts": math.exp(entropy),
    }


def compare_moe_route_distributions(reference_counts, observed_counts):
    """Retain real denominators; missing exposure is never a uniform distribution.

    Counts must refer to the same expert order and assignment meaning, such as
    attempted training versus held-out assignments for one trained model. Do
    not align experts by array index across independently initialized models.
    Changing exposure alone does not imply changed routing probabilities.
    """

    reference, observed = _histogram(reference_counts), _histogram(observed_counts)
    if len(reference["counts"]) != len(observed["counts"]):
        raise ValueError("route distributions require matching expert identities")
    left, right = reference["fractions"], observed["fractions"]
    divergence = distance = None
    if left is not None and right is not None:
        midpoint = [(a + b) / 2 for a, b in zip(left, right, strict=True)]
        divergence = 0.5 * math.fsum(
            value * math.log(value / middle)
            for distribution in (left, right)
            for value, middle in zip(distribution, midpoint, strict=True)
            if value
        )
        distance = 0.5 * math.fsum(abs(a - b) for a, b in zip(left, right, strict=True))
    return {
        "scope": DRIFT_SCOPE,
        "reference": reference,
        "observed": observed,
        "jensen_shannon_nats": divergence,
        "total_variation": distance,
    }
