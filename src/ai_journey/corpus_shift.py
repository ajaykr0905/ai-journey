"""Compare character-bigram behavior across two corpora."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from itertools import pairwise
from math import exp, isfinite, log

import numpy as np

from ai_journey.bigram_lm import (
    BOUNDARY_TOKEN,
    Vocabulary,
    count_bigrams,
    iter_bigrams,
    normalize_corpus,
)


class CorpusShiftError(ValueError):
    """Raised when corpus-shift inputs are invalid."""


@dataclass(frozen=True)
class Evaluation:
    trained_on: str
    evaluated_on: str
    transitions: int
    mean_nll: float
    perplexity: float


@dataclass(frozen=True)
class ShiftReport:
    baseline_records: int
    shifted_records: int
    vocabulary_size: int
    smoothing: float
    baseline_only_characters: tuple[str, ...]
    shifted_only_characters: tuple[str, ...]
    character_jaccard: float
    transition_js_divergence: float
    baseline_boundary_leaks: int
    shifted_boundary_leaks: int
    evaluations: tuple[Evaluation, ...]


def _validate_smoothing(smoothing: float) -> float:
    if isinstance(smoothing, bool) or not isinstance(smoothing, (int, float)):
        raise TypeError("smoothing must be a real number")
    value = float(smoothing)
    if not isfinite(value) or value <= 0:
        raise CorpusShiftError("smoothing must be finite and positive")
    return value


def build_shared_vocabulary(
    baseline: Iterable[str], shifted: Iterable[str]
) -> Vocabulary:
    """Build a stable vocabulary covering both corpora."""

    words = (*normalize_corpus(baseline), *normalize_corpus(shifted))
    characters = sorted({character for word in words for character in word})
    return Vocabulary((BOUNDARY_TOKEN, *characters))


def smoothed_probabilities(
    words: Iterable[str], vocabulary: Vocabulary, *, smoothing: float = 1.0
) -> np.ndarray:
    """Estimate a row-normalized bigram matrix with additive smoothing."""

    alpha = _validate_smoothing(smoothing)
    counts = count_bigrams(normalize_corpus(words), vocabulary).astype(np.float64)
    counts += alpha
    return counts / counts.sum(axis=1, keepdims=True)


def _transition_distribution(
    words: Iterable[str], vocabulary: Vocabulary, *, smoothing: float
) -> np.ndarray:
    counts = count_bigrams(normalize_corpus(words), vocabulary).astype(np.float64)
    counts += smoothing
    return counts.reshape(-1) / counts.sum()


def evaluate(
    words: Iterable[str],
    vocabulary: Vocabulary,
    probabilities: np.ndarray,
    *,
    trained_on: str,
    evaluated_on: str,
) -> Evaluation:
    """Compute boundary-aware mean NLL and perplexity."""

    expected_shape = (vocabulary.size, vocabulary.size)
    if probabilities.shape != expected_shape:
        raise CorpusShiftError(
            f"probabilities must have shape {expected_shape}, got {probabilities.shape}"
        )
    if (
        probabilities.dtype.kind != "f"
        or not np.all(np.isfinite(probabilities))
        or np.any(probabilities <= 0)
        or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-12)
    ):
        raise CorpusShiftError("probabilities must contain positive unit-sum rows")

    log_probabilities = [
        log(probabilities[vocabulary.encode(previous), vocabulary.encode(next_token)])
        for word in normalize_corpus(words)
        for previous, next_token in iter_bigrams(word)
    ]
    mean_nll = -sum(log_probabilities) / len(log_probabilities)
    return Evaluation(
        trained_on=trained_on,
        evaluated_on=evaluated_on,
        transitions=len(log_probabilities),
        mean_nll=mean_nll,
        perplexity=exp(mean_nll),
    )


def jensen_shannon(left: np.ndarray, right: np.ndarray) -> float:
    """Compute natural-log Jensen-Shannon divergence."""

    if left.ndim != 1 or left.shape != right.shape:
        raise CorpusShiftError("distributions must be aligned vectors")
    if (
        left.dtype.kind != "f"
        or right.dtype.kind != "f"
        or not np.all(np.isfinite(left))
        or not np.all(np.isfinite(right))
        or np.any(left < 0)
        or np.any(right < 0)
        or left.sum() <= 0
        or right.sum() <= 0
    ):
        raise CorpusShiftError("distributions must be finite and non-negative")

    left = left / left.sum()
    right = right / right.sum()
    midpoint = 0.5 * (left + right)

    def kl_divergence(distribution: np.ndarray) -> float:
        nonzero = distribution > 0
        return float(
            np.sum(
                distribution[nonzero]
                * np.log(distribution[nonzero] / midpoint[nonzero])
            )
        )

    return 0.5 * (kl_divergence(left) + kl_divergence(right))


def boundary_leaks(words: Iterable[str]) -> tuple[tuple[str, str], ...]:
    """Return transitions created by concatenating independent records."""

    corpus = normalize_corpus(words)
    return tuple((left[-1], right[0]) for left, right in pairwise(corpus))


def analyze_corpus_shift(
    baseline: Iterable[str],
    shifted: Iterable[str],
    *,
    smoothing: float = 1.0,
) -> ShiftReport:
    """Train on each corpus and evaluate the complete 2x2 comparison."""

    baseline = normalize_corpus(baseline)
    shifted = normalize_corpus(shifted)
    alpha = _validate_smoothing(smoothing)
    vocabulary = build_shared_vocabulary(baseline, shifted)
    baseline_probabilities = smoothed_probabilities(
        baseline, vocabulary, smoothing=alpha
    )
    shifted_probabilities = smoothed_probabilities(shifted, vocabulary, smoothing=alpha)

    evaluations = tuple(
        evaluate(
            corpus,
            vocabulary,
            probabilities,
            trained_on=trained_on,
            evaluated_on=evaluated_on,
        )
        for trained_on, probabilities in (
            ("baseline", baseline_probabilities),
            ("shifted", shifted_probabilities),
        )
        for evaluated_on, corpus in (("baseline", baseline), ("shifted", shifted))
    )

    baseline_characters = set("".join(baseline))
    shifted_characters = set("".join(shifted))
    union = baseline_characters | shifted_characters

    return ShiftReport(
        baseline_records=len(baseline),
        shifted_records=len(shifted),
        vocabulary_size=vocabulary.size,
        smoothing=alpha,
        baseline_only_characters=tuple(
            sorted(baseline_characters - shifted_characters)
        ),
        shifted_only_characters=tuple(sorted(shifted_characters - baseline_characters)),
        character_jaccard=len(baseline_characters & shifted_characters) / len(union),
        transition_js_divergence=jensen_shannon(
            _transition_distribution(baseline, vocabulary, smoothing=alpha),
            _transition_distribution(shifted, vocabulary, smoothing=alpha),
        ),
        baseline_boundary_leaks=len(boundary_leaks(baseline)),
        shifted_boundary_leaks=len(boundary_leaks(shifted)),
        evaluations=evaluations,
    )


def report_payload(report: ShiftReport) -> dict[str, object]:
    """Return a compact JSON report without raw corpora or probability matrices."""

    if not isinstance(report, ShiftReport):
        raise TypeError("report must be ShiftReport")
    return asdict(report)
