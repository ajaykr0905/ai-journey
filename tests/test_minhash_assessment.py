import pytest

from ai_journey.bpe import TokenizerError
from ai_journey.minhash_assessment import CandidateAssessment, assess_candidates


def test_candidate_assessment_preserves_false_positives_as_evidence():
    assessments = assess_candidates(
        (("b", "c"), ("a", "b")),
        {"a": ("x", "y"), "b": ("x", "y"), "c": ("x", "z")},
        {"a": (1, 2), "b": (1, 2), "c": (1, 9)},
        threshold=0.75,
    )

    assert assessments == (
        CandidateAssessment("a", "b", 1.0, 1.0, True),
        CandidateAssessment("b", "c", 0.5, 1 / 3, False),
    )


@pytest.mark.parametrize("threshold", [True, 0, -0.1, 1.1, float("nan")])
def test_candidate_assessment_requires_a_bounded_threshold(threshold):
    with pytest.raises(TokenizerError, match="threshold"):
        assess_candidates((), {}, {}, threshold=threshold)


def test_candidate_assessment_rejects_duplicate_or_reversed_pairs():
    shingles = {"a": ("x",), "b": ("x",)}
    signatures = {"a": (1,), "b": (1,)}
    with pytest.raises(TokenizerError, match="unique"):
        assess_candidates((("a", "b"), ("a", "b")), shingles, signatures, threshold=0.9)
    with pytest.raises(TokenizerError, match="pair is invalid"):
        assess_candidates((("b", "a"),), shingles, signatures, threshold=0.9)


def test_candidate_assessment_requires_matching_document_coverage():
    with pytest.raises(TokenizerError, match="coverage"):
        assess_candidates((), {"a": ("x",)}, {}, threshold=0.9)
