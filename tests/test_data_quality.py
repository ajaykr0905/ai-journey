import pytest

from ai_journey.bpe import TokenizerError
from ai_journey.data_cleaning import clean_document
from ai_journey.data_documents import DataDocument
from ai_journey.data_quality import QualityPolicy, assess_quality


def _clean(text):
    return clean_document(DataDocument("id", text, "public://snapshot"))


def test_quality_accepts_text_that_meets_every_declared_bound():
    decision = assess_quality(
        _clean("A useful public sentence with enough letters."), QualityPolicy()
    )

    assert decision.accepted is True
    assert decision.reasons == ()


def test_quality_reports_every_rejection_reason_without_short_circuiting():
    policy = QualityPolicy(
        min_characters=1,
        max_characters=12,
        min_alphanumeric_fraction=0.8,
        max_repeated_run=3,
    )

    decision = assess_quality(_clean("!!!!!!!!!!!!"), policy)

    assert decision.accepted is False
    assert decision.reasons == (
        "low_alphanumeric_fraction",
        "repeated_character_run",
    )


def test_quality_distinguishes_short_and_long_documents():
    policy = QualityPolicy(min_characters=5, max_characters=8)

    assert assess_quality(_clean("abc"), policy).reasons == ("too_short",)
    assert assess_quality(_clean("abcdefghij"), policy).reasons == ("too_long",)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_characters": 0},
        {"min_characters": 5, "max_characters": 4},
        {"min_alphanumeric_fraction": 1.1},
        {"max_repeated_run": 0},
    ],
)
def test_quality_policy_rejects_invalid_bounds(kwargs):
    with pytest.raises(TokenizerError, match="bounds"):
        QualityPolicy(**kwargs)
