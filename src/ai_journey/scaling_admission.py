"""Admit or block scaling plans against explicit corpus-sufficiency limits."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from .scaling_corpus import CorpusReusePlan
from .scaling_laws import _positive_decimal


@dataclass(frozen=True)
class CorpusAdmissionPolicy:
    """Declared limits on corpus epochs and distinct-token share for a plan.

    Both must be positive decimals and the fraction at most one; invalid
    values raise TypeError, ScalingLawError or ValueError.
    """

    maximum_epochs: int | float | str
    minimum_distinct_token_fraction: int | float | str

    def __post_init__(self) -> None:
        _positive_decimal(self.maximum_epochs, "maximum_epochs")
        fraction = _positive_decimal(
            self.minimum_distinct_token_fraction,
            "minimum_distinct_token_fraction",
        )
        if fraction > 1:
            raise ValueError("minimum_distinct_token_fraction must not exceed 1")


@dataclass(frozen=True)
class CorpusAdmission:
    """Corpus verdict with the reasons for any failed limit.

    It records the reuse it judged and never changes the token target.
    """

    admitted: bool
    reasons: tuple[str, ...]
    total_epochs: Decimal
    distinct_token_fraction: Decimal
    policy: CorpusAdmissionPolicy


def assess_corpus_admission(
    reuse: CorpusReusePlan, policy: CorpusAdmissionPolicy
) -> CorpusAdmission:
    """Evaluate reuse without mutating or reinterpreting the token target."""

    if not isinstance(reuse, CorpusReusePlan):
        raise TypeError("reuse must be CorpusReusePlan")
    if not isinstance(policy, CorpusAdmissionPolicy):
        raise TypeError("policy must be CorpusAdmissionPolicy")
    reasons = []
    if reuse.total_epochs > _positive_decimal(policy.maximum_epochs, "maximum_epochs"):
        reasons.append("corpus epochs exceed the declared limit")
    if reuse.distinct_token_fraction < _positive_decimal(
        policy.minimum_distinct_token_fraction,
        "minimum_distinct_token_fraction",
    ):
        reasons.append("distinct token fraction is below the declared minimum")
    return CorpusAdmission(
        admitted=not reasons,
        reasons=tuple(reasons),
        total_epochs=reuse.total_epochs,
        distinct_token_fraction=reuse.distinct_token_fraction,
        policy=policy,
    )
