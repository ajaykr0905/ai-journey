from decimal import Decimal

import pytest

from ai_journey.scaling_admission import (
    CorpusAdmissionPolicy,
    assess_corpus_admission,
)
from ai_journey.scaling_corpus import plan_corpus_reuse


def test_corpus_admission_blocks_extreme_fixture_reuse():
    reuse = plan_corpus_reuse(
        target_tokens=2_000_000_000,
        unique_train_tokens=69_626,
    )
    policy = CorpusAdmissionPolicy(
        maximum_epochs="2",
        minimum_distinct_token_fraction="0.5",
    )

    result = assess_corpus_admission(reuse, policy)

    assert result.admitted is False
    assert result.reasons == (
        "corpus epochs exceed the declared limit",
        "distinct token fraction is below the declared minimum",
    )
    assert result.total_epochs > Decimal("28000")


def test_corpus_admission_accepts_a_target_within_both_limits():
    reuse = plan_corpus_reuse(target_tokens=150, unique_train_tokens=100)
    policy = CorpusAdmissionPolicy("2", "0.5")

    result = assess_corpus_admission(reuse, policy)

    assert result.admitted is True
    assert result.reasons == ()
    assert result.distinct_token_fraction == Decimal(2) / Decimal(3)


@pytest.mark.parametrize(
    "values,message",
    [
        (("0", "0.5"), "maximum_epochs"),
        (("2", "0"), "minimum_distinct_token_fraction"),
        (("2", "1.1"), "must not exceed 1"),
    ],
)
def test_corpus_admission_policy_rejects_invalid_limits(values, message):
    with pytest.raises(ValueError, match=message):
        CorpusAdmissionPolicy(*values)
