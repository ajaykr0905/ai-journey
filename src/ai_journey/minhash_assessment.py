"""Exact acceptance evidence for LSH-generated document pairs."""

from dataclasses import dataclass
import math

from .bpe import TokenizerError
from .minhash_signature import signature_similarity
from .minhash_similarity import exact_jaccard


@dataclass(frozen=True)
class CandidateAssessment:
    """One LSH candidate pair with its MinHash estimate and exact shingle Jaccard.

    duplicate compares the exact Jaccard with the threshold, never the estimate,
    so LSH only proposes pairs and cannot remove a document by itself.
    """

    left_id: str
    right_id: str
    minhash_similarity: float
    jaccard_similarity: float
    duplicate: bool


def assess_candidates(
    pairs: tuple[tuple[str, str], ...],
    shingles: dict[str, tuple[str, ...]],
    signatures: dict[str, tuple[int, ...]],
    *,
    threshold: float,
) -> tuple[CandidateAssessment, ...]:
    """Measure every candidate exactly and mark threshold-qualified duplicates."""
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(threshold)
        or not 0 < threshold <= 1
    ):
        raise TokenizerError("duplicate threshold must be finite above zero to one")
    if (
        not isinstance(pairs, tuple)
        or not isinstance(shingles, dict)
        or not isinstance(signatures, dict)
    ):
        raise TokenizerError("candidate assessment inputs are invalid")
    if set(shingles) != set(signatures):
        raise TokenizerError("candidate assessment document coverage differs")
    ordered_pairs = tuple(sorted(pairs))
    if len(set(ordered_pairs)) != len(ordered_pairs):
        raise TokenizerError("candidate assessment pairs must be unique")
    assessments = []
    for pair in ordered_pairs:
        if (
            not isinstance(pair, tuple)
            or len(pair) != 2
            or pair[0] >= pair[1]
            or any(document_id not in shingles for document_id in pair)
        ):
            raise TokenizerError("candidate assessment pair is invalid")
        exact = exact_jaccard(shingles[pair[0]], shingles[pair[1]])
        estimate = signature_similarity(signatures[pair[0]], signatures[pair[1]])
        assessments.append(
            CandidateAssessment(pair[0], pair[1], estimate, exact, exact >= threshold)
        )
    return tuple(assessments)
