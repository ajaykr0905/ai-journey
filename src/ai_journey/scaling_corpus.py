"""Translate token targets into explicit public-corpus reuse requirements."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from .scaling_laws import _positive_integer


@dataclass(frozen=True)
class CorpusReusePlan:
    target_tokens: int
    unique_train_tokens: int
    full_epochs: int
    final_epoch_tokens: int
    total_epochs: Decimal

    @property
    def repeats_data(self) -> bool:
        return self.target_tokens > self.unique_train_tokens

    @property
    def distinct_token_fraction(self) -> Decimal:
        return min(
            Decimal(1),
            Decimal(self.unique_train_tokens) / Decimal(self.target_tokens),
        )


def plan_corpus_reuse(
    *, target_tokens: int, unique_train_tokens: int
) -> CorpusReusePlan:
    """Expose the exact number of full and partial corpus passes required."""

    target = _positive_integer(target_tokens, "target_tokens")
    unique = _positive_integer(unique_train_tokens, "unique_train_tokens")
    full_epochs, final_epoch_tokens = divmod(target, unique)
    return CorpusReusePlan(
        target_tokens=target,
        unique_train_tokens=unique,
        full_epochs=full_epochs,
        final_epoch_tokens=final_epoch_tokens,
        total_epochs=Decimal(target) / Decimal(unique),
    )
