"""Day 21: deterministic corpus-shift and boundary-integrity analysis.

The experiment reuses the inspectable Day 19 character bigram model while
adding a shared vocabulary, additive smoothing, cross-corpus evaluation, and a
structural audit for transitions introduced by incorrectly concatenating
independent records without boundary tokens.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
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


class CorpusShiftValidationError(ValueError):
    """Raised when corpus-shift inputs or experiment invariants are invalid."""


@dataclass(frozen=True)
class CorpusEvaluation:
    """Mean loss and perplexity for one model/corpus pairing."""

    trained_on: str
    evaluated_on: str
    transition_count: int
    mean_nll: float
    perplexity: float


@dataclass(frozen=True)
class BoundaryAudit:
    """Structural evidence for correct and incorrectly packed record streams."""

    record_count: int
    character_count: int
    boundary_aware_transitions: int
    packed_transitions: int
    cross_record_events: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class CorpusShiftExperiment:
    """All deterministic evidence for the Day 21 dataset swap."""

    baseline_corpus: tuple[str, ...]
    shifted_corpus: tuple[str, ...]
    vocabulary: Vocabulary
    smoothing: float
    baseline_probabilities: np.ndarray
    shifted_probabilities: np.ndarray
    evaluations: tuple[CorpusEvaluation, ...]
    baseline_only_characters: tuple[str, ...]
    shifted_only_characters: tuple[str, ...]
    character_jaccard: float
    transition_js_divergence: float
    baseline_boundary_audit: BoundaryAudit
    shifted_boundary_audit: BoundaryAudit


def validate_smoothing(smoothing: float) -> float:
    """Return additive smoothing as a finite, strictly positive float."""

    if isinstance(smoothing, bool) or not isinstance(smoothing, (int, float)):
        raise TypeError("smoothing must be a real number")
    value = float(smoothing)
    if not isfinite(value) or value <= 0.0:
        raise CorpusShiftValidationError(
            "smoothing must be finite and strictly positive"
        )
    return value


def character_set(words: Iterable[str]) -> frozenset[str]:
    """Return the normalized corpus character set without the boundary token."""

    corpus = normalize_corpus(words)
    return frozenset(character for word in corpus for character in word)


def unseen_characters(
    reference_words: Iterable[str], candidate_words: Iterable[str]
) -> tuple[str, ...]:
    """Return candidate characters absent from the reference corpus."""

    reference = character_set(reference_words)
    candidate = character_set(candidate_words)
    return tuple(sorted(candidate - reference))


def character_jaccard_similarity(
    baseline_words: Iterable[str], shifted_words: Iterable[str]
) -> float:
    """Return intersection-over-union for two non-empty character sets."""

    baseline = character_set(baseline_words)
    shifted = character_set(shifted_words)
    union = baseline | shifted
    if not union:
        raise CorpusShiftValidationError("character union must not be empty")
    return len(baseline & shifted) / len(union)


def build_shared_vocabulary(
    baseline_words: Iterable[str], shifted_words: Iterable[str]
) -> Vocabulary:
    """Build one stable vocabulary covering both corpora."""

    baseline = normalize_corpus(baseline_words)
    shifted = normalize_corpus(shifted_words)
    characters = tuple(
        sorted({character for word in (*baseline, *shifted) for character in word})
    )
    return Vocabulary((BOUNDARY_TOKEN, *characters))


def smoothed_bigram_probabilities(
    words: Iterable[str], vocabulary: Vocabulary, *, smoothing: float = 1.0
) -> np.ndarray:
    """Return additively smoothed probabilities in a shared vocabulary."""

    corpus = normalize_corpus(words)
    if not isinstance(vocabulary, Vocabulary):
        raise TypeError("vocabulary must be Vocabulary")
    alpha = validate_smoothing(smoothing)
    counts = count_bigrams(corpus, vocabulary).astype(np.float64)
    smoothed = counts + alpha
    probabilities = smoothed / smoothed.sum(axis=1, keepdims=True)
    validate_probability_matrix(probabilities, vocabulary.size)
    return probabilities


def validate_probability_matrix(
    probabilities: np.ndarray, vocabulary_size: int
) -> None:
    """Require a finite, positive square probability matrix with unit rows."""

    if not isinstance(probabilities, np.ndarray):
        raise TypeError("probabilities must be a NumPy array")
    expected = (vocabulary_size, vocabulary_size)
    if probabilities.shape != expected:
        raise CorpusShiftValidationError(
            f"probabilities must have shape {expected}, got {probabilities.shape}"
        )
    if probabilities.dtype.kind != "f":
        raise TypeError("probabilities must use a floating-point dtype")
    if not np.all(np.isfinite(probabilities)):
        raise CorpusShiftValidationError("probabilities contain a non-finite value")
    if np.any(probabilities <= 0.0):
        raise CorpusShiftValidationError("probabilities must be strictly positive")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-12):
        raise CorpusShiftValidationError("probability rows must sum to one")


def evaluate_corpus(
    words: Iterable[str],
    vocabulary: Vocabulary,
    probabilities: np.ndarray,
    *,
    trained_on: str,
    evaluated_on: str,
) -> CorpusEvaluation:
    """Evaluate boundary-aware mean NLL and perplexity for one corpus."""

    corpus = normalize_corpus(words)
    validate_probability_matrix(probabilities, vocabulary.size)
    selected: list[float] = []
    for word in corpus:
        for previous, next_token in iter_bigrams(word):
            selected.append(
                float(
                    probabilities[
                        vocabulary.encode(previous), vocabulary.encode(next_token)
                    ]
                )
            )
    if not selected:
        raise CorpusShiftValidationError("evaluation must contain transitions")
    mean_nll = -sum(log(probability) for probability in selected) / len(selected)
    perplexity = exp(mean_nll)
    if not isfinite(mean_nll) or not isfinite(perplexity):
        raise CorpusShiftValidationError("evaluation metrics must be finite")
    return CorpusEvaluation(
        trained_on=trained_on,
        evaluated_on=evaluated_on,
        transition_count=len(selected),
        mean_nll=mean_nll,
        perplexity=perplexity,
    )


def smoothed_joint_distribution(
    words: Iterable[str], vocabulary: Vocabulary, *, smoothing: float = 1.0
) -> np.ndarray:
    """Return one flattened, globally normalized transition distribution."""

    corpus = normalize_corpus(words)
    alpha = validate_smoothing(smoothing)
    counts = count_bigrams(corpus, vocabulary).astype(np.float64) + alpha
    return (counts / counts.sum()).reshape(-1)


def jensen_shannon_divergence(left: np.ndarray, right: np.ndarray) -> float:
    """Return the natural-log Jensen-Shannon divergence of two vectors."""

    if not isinstance(left, np.ndarray) or not isinstance(right, np.ndarray):
        raise TypeError("distributions must be NumPy arrays")
    if left.ndim != 1 or right.ndim != 1 or left.shape != right.shape:
        raise CorpusShiftValidationError(
            "distributions must be aligned one-dimensional vectors"
        )
    if left.dtype.kind != "f" or right.dtype.kind != "f":
        raise TypeError("distributions must use floating-point dtypes")
    if (
        not np.all(np.isfinite(left))
        or not np.all(np.isfinite(right))
        or np.any(left < 0.0)
        or np.any(right < 0.0)
    ):
        raise CorpusShiftValidationError(
            "distributions must be finite and non-negative"
        )
    left_total = float(left.sum())
    right_total = float(right.sum())
    if left_total <= 0.0 or right_total <= 0.0:
        raise CorpusShiftValidationError("distribution totals must be positive")
    p = left / left_total
    q = right / right_total
    midpoint = 0.5 * (p + q)

    def _kl(source: np.ndarray) -> float:
        positive = source > 0.0
        return float(
            np.sum(source[positive] * np.log(source[positive] / midpoint[positive]))
        )

    value = 0.5 * (_kl(p) + _kl(q))
    if value < -1e-12 or value > log(2.0) + 1e-12:
        raise CorpusShiftValidationError("Jensen-Shannon divergence is out of range")
    return max(0.0, value)


def cross_record_transitions(words: Iterable[str]) -> tuple[tuple[str, str], ...]:
    """Return the spurious last-to-first pairs created by naive concatenation."""

    corpus = normalize_corpus(words)
    return tuple((left[-1], right[0]) for left, right in pairwise(corpus))


def packed_stream_bigrams(words: Iterable[str]) -> tuple[tuple[str, str], ...]:
    """Return bigrams from the intentionally incorrect concatenated stream."""

    corpus = normalize_corpus(words)
    stream = BOUNDARY_TOKEN + "".join(corpus) + BOUNDARY_TOKEN
    return tuple(pairwise(stream))


def audit_boundaries(words: Iterable[str]) -> BoundaryAudit:
    """Compare per-record boundaries with a deliberately packed stream."""

    corpus = normalize_corpus(words)
    character_count = sum(len(word) for word in corpus)
    aware_count = sum(len(iter_bigrams(word)) for word in corpus)
    packed_count = len(packed_stream_bigrams(corpus))
    audit = BoundaryAudit(
        record_count=len(corpus),
        character_count=character_count,
        boundary_aware_transitions=aware_count,
        packed_transitions=packed_count,
        cross_record_events=cross_record_transitions(corpus),
    )
    validate_boundary_audit(audit)
    return audit


def validate_boundary_audit(audit: BoundaryAudit) -> None:
    """Check the exact transition counts implied by both stream layouts."""

    if not isinstance(audit, BoundaryAudit):
        raise TypeError("audit must be BoundaryAudit")
    if audit.record_count <= 0 or audit.character_count <= 0:
        raise CorpusShiftValidationError("boundary audit counts must be positive")
    if audit.boundary_aware_transitions != audit.character_count + audit.record_count:
        raise CorpusShiftValidationError("boundary-aware transition count is invalid")
    if audit.packed_transitions != audit.character_count + 1:
        raise CorpusShiftValidationError("packed transition count is invalid")
    if len(audit.cross_record_events) != audit.record_count - 1:
        raise CorpusShiftValidationError("cross-record event count is invalid")


def run_corpus_shift_experiment(
    baseline_words: Iterable[str],
    shifted_words: Iterable[str],
    *,
    smoothing: float = 1.0,
) -> CorpusShiftExperiment:
    """Build and validate the complete deterministic Day 21 experiment."""

    baseline = normalize_corpus(baseline_words)
    shifted = normalize_corpus(shifted_words)
    alpha = validate_smoothing(smoothing)
    vocabulary = build_shared_vocabulary(baseline, shifted)
    baseline_probabilities = smoothed_bigram_probabilities(
        baseline, vocabulary, smoothing=alpha
    )
    shifted_probabilities = smoothed_bigram_probabilities(
        shifted, vocabulary, smoothing=alpha
    )
    evaluations = (
        evaluate_corpus(
            baseline,
            vocabulary,
            baseline_probabilities,
            trained_on="baseline",
            evaluated_on="baseline",
        ),
        evaluate_corpus(
            shifted,
            vocabulary,
            baseline_probabilities,
            trained_on="baseline",
            evaluated_on="shifted",
        ),
        evaluate_corpus(
            baseline,
            vocabulary,
            shifted_probabilities,
            trained_on="shifted",
            evaluated_on="baseline",
        ),
        evaluate_corpus(
            shifted,
            vocabulary,
            shifted_probabilities,
            trained_on="shifted",
            evaluated_on="shifted",
        ),
    )
    baseline_joint = smoothed_joint_distribution(baseline, vocabulary, smoothing=alpha)
    shifted_joint = smoothed_joint_distribution(shifted, vocabulary, smoothing=alpha)
    experiment = CorpusShiftExperiment(
        baseline_corpus=baseline,
        shifted_corpus=shifted,
        vocabulary=vocabulary,
        smoothing=alpha,
        baseline_probabilities=baseline_probabilities,
        shifted_probabilities=shifted_probabilities,
        evaluations=evaluations,
        baseline_only_characters=unseen_characters(shifted, baseline),
        shifted_only_characters=unseen_characters(baseline, shifted),
        character_jaccard=character_jaccard_similarity(baseline, shifted),
        transition_js_divergence=jensen_shannon_divergence(
            baseline_joint, shifted_joint
        ),
        baseline_boundary_audit=audit_boundaries(baseline),
        shifted_boundary_audit=audit_boundaries(shifted),
    )
    validate_experiment(experiment)
    return experiment


def validate_experiment(experiment: CorpusShiftExperiment) -> None:
    """Validate vocabulary, metrics, evaluations, and boundary evidence."""

    if not isinstance(experiment, CorpusShiftExperiment):
        raise TypeError("experiment must be CorpusShiftExperiment")
    validate_smoothing(experiment.smoothing)
    expected_vocabulary = build_shared_vocabulary(
        experiment.baseline_corpus, experiment.shifted_corpus
    )
    if experiment.vocabulary != expected_vocabulary:
        raise CorpusShiftValidationError("experiment vocabulary is not shared")
    validate_probability_matrix(
        experiment.baseline_probabilities, experiment.vocabulary.size
    )
    validate_probability_matrix(
        experiment.shifted_probabilities, experiment.vocabulary.size
    )
    expected_baseline_probabilities = smoothed_bigram_probabilities(
        experiment.baseline_corpus,
        experiment.vocabulary,
        smoothing=experiment.smoothing,
    )
    expected_shifted_probabilities = smoothed_bigram_probabilities(
        experiment.shifted_corpus,
        experiment.vocabulary,
        smoothing=experiment.smoothing,
    )
    if not np.array_equal(
        experiment.baseline_probabilities, expected_baseline_probabilities
    ):
        raise CorpusShiftValidationError("baseline probabilities are inconsistent")
    if not np.array_equal(
        experiment.shifted_probabilities, expected_shifted_probabilities
    ):
        raise CorpusShiftValidationError("shifted probabilities are inconsistent")
    expected_evaluations = (
        evaluate_corpus(
            experiment.baseline_corpus,
            experiment.vocabulary,
            experiment.baseline_probabilities,
            trained_on="baseline",
            evaluated_on="baseline",
        ),
        evaluate_corpus(
            experiment.shifted_corpus,
            experiment.vocabulary,
            experiment.baseline_probabilities,
            trained_on="baseline",
            evaluated_on="shifted",
        ),
        evaluate_corpus(
            experiment.baseline_corpus,
            experiment.vocabulary,
            experiment.shifted_probabilities,
            trained_on="shifted",
            evaluated_on="baseline",
        ),
        evaluate_corpus(
            experiment.shifted_corpus,
            experiment.vocabulary,
            experiment.shifted_probabilities,
            trained_on="shifted",
            evaluated_on="shifted",
        ),
    )
    if experiment.evaluations != expected_evaluations:
        raise CorpusShiftValidationError("experiment evaluations are inconsistent")
    for evaluation in experiment.evaluations:
        if evaluation.transition_count <= 0:
            raise CorpusShiftValidationError(
                "evaluation transition count must be positive"
            )
        if (
            not isfinite(evaluation.mean_nll)
            or evaluation.mean_nll < 0.0
            or not isfinite(evaluation.perplexity)
            or evaluation.perplexity < 1.0
        ):
            raise CorpusShiftValidationError("evaluation metrics are invalid")
    if not 0.0 <= experiment.character_jaccard <= 1.0:
        raise CorpusShiftValidationError("character Jaccard similarity is invalid")
    if experiment.baseline_only_characters != unseen_characters(
        experiment.shifted_corpus, experiment.baseline_corpus
    ):
        raise CorpusShiftValidationError("baseline-only character set is invalid")
    if experiment.shifted_only_characters != unseen_characters(
        experiment.baseline_corpus, experiment.shifted_corpus
    ):
        raise CorpusShiftValidationError("shifted-only character set is invalid")
    expected_jaccard = character_jaccard_similarity(
        experiment.baseline_corpus, experiment.shifted_corpus
    )
    if experiment.character_jaccard != expected_jaccard:
        raise CorpusShiftValidationError("character Jaccard similarity is inconsistent")
    if not 0.0 <= experiment.transition_js_divergence <= log(2.0) + 1e-12:
        raise CorpusShiftValidationError("transition divergence is invalid")
    expected_divergence = jensen_shannon_divergence(
        smoothed_joint_distribution(
            experiment.baseline_corpus,
            experiment.vocabulary,
            smoothing=experiment.smoothing,
        ),
        smoothed_joint_distribution(
            experiment.shifted_corpus,
            experiment.vocabulary,
            smoothing=experiment.smoothing,
        ),
    )
    if experiment.transition_js_divergence != expected_divergence:
        raise CorpusShiftValidationError("transition divergence is inconsistent")
    validate_boundary_audit(experiment.baseline_boundary_audit)
    validate_boundary_audit(experiment.shifted_boundary_audit)
    if experiment.baseline_boundary_audit != audit_boundaries(
        experiment.baseline_corpus
    ):
        raise CorpusShiftValidationError("baseline boundary audit is inconsistent")
    if experiment.shifted_boundary_audit != audit_boundaries(experiment.shifted_corpus):
        raise CorpusShiftValidationError("shifted boundary audit is inconsistent")


def experiment_metrics(
    experiment: CorpusShiftExperiment,
) -> dict[str, int | float | str]:
    """Return compact JSON-safe Day 21 metrics."""

    validate_experiment(experiment)
    evaluations = {
        f"{item.trained_on}_to_{item.evaluated_on}": item
        for item in experiment.evaluations
    }
    return {
        "baseline_records": len(experiment.baseline_corpus),
        "shifted_records": len(experiment.shifted_corpus),
        "vocabulary_size": experiment.vocabulary.size,
        "smoothing": experiment.smoothing,
        "character_jaccard": experiment.character_jaccard,
        "baseline_only_character_count": len(experiment.baseline_only_characters),
        "shifted_only_character_count": len(experiment.shifted_only_characters),
        "transition_js_divergence": experiment.transition_js_divergence,
        "baseline_self_nll": evaluations["baseline_to_baseline"].mean_nll,
        "baseline_to_shifted_nll": evaluations["baseline_to_shifted"].mean_nll,
        "shifted_to_baseline_nll": evaluations["shifted_to_baseline"].mean_nll,
        "shifted_self_nll": evaluations["shifted_to_shifted"].mean_nll,
        "baseline_cross_record_events": len(
            experiment.baseline_boundary_audit.cross_record_events
        ),
        "shifted_cross_record_events": len(
            experiment.shifted_boundary_audit.cross_record_events
        ),
        "boundary_policy": "one start/end boundary per independent record",
    }


def experiment_payload(experiment: CorpusShiftExperiment) -> dict[str, object]:
    """Convert the complete experiment into JSON-compatible values."""

    validate_experiment(experiment)
    return {
        "metrics": experiment_metrics(experiment),
        "baseline_corpus": list(experiment.baseline_corpus),
        "shifted_corpus": list(experiment.shifted_corpus),
        "vocabulary": list(experiment.vocabulary.tokens),
        "baseline_only_characters": list(experiment.baseline_only_characters),
        "shifted_only_characters": list(experiment.shifted_only_characters),
        "evaluations": [
            {
                "trained_on": item.trained_on,
                "evaluated_on": item.evaluated_on,
                "transition_count": item.transition_count,
                "mean_nll": item.mean_nll,
                "perplexity": item.perplexity,
            }
            for item in experiment.evaluations
        ],
        "boundary_audits": {
            "baseline": _boundary_payload(experiment.baseline_boundary_audit),
            "shifted": _boundary_payload(experiment.shifted_boundary_audit),
        },
    }


def _boundary_payload(audit: BoundaryAudit) -> dict[str, object]:
    return {
        "record_count": audit.record_count,
        "character_count": audit.character_count,
        "boundary_aware_transitions": audit.boundary_aware_transitions,
        "packed_transitions": audit.packed_transitions,
        "cross_record_events": [list(pair) for pair in audit.cross_record_events],
    }


def render_boundary_mermaid(experiment: CorpusShiftExperiment) -> str:
    """Render the correct and incorrect corpus layouts as a small diagram."""

    validate_experiment(experiment)
    records = len(experiment.shifted_corpus)
    leaks = len(experiment.shifted_boundary_audit.cross_record_events)
    return "\n".join(
        (
            "flowchart TD",
            f'    records["{records} independent city records"] --> aware["boundary-aware encoding"]',
            '    aware --> isolated["start/end token per record"]',
            f'    records --> packed["naive concatenation: {leaks} cross-record events"]',
            '    packed --> contaminated["last character can transition to next record"]',
            '    isolated --> compare["cross-corpus NLL and divergence"]',
            "    contaminated --> compare",
        )
    )


def render_corpus_shift_markdown(experiment: CorpusShiftExperiment) -> str:
    """Render a factual Day 21 report without claiming learner completion."""

    validate_experiment(experiment)
    metrics = experiment_metrics(experiment)
    lines = [
        "# Day 21 corpus shift and boundary integrity",
        "",
        "This deterministic report is repository evidence. It does not prove that the learner completed the blank-file rebuild or watched the scheduled lesson.",
        "",
        "## Dataset swap",
        "",
        f"- Baseline records: {metrics['baseline_records']}",
        f"- Indian-city records: {metrics['shifted_records']}",
        f"- Shared vocabulary size: {metrics['vocabulary_size']}",
        f"- Character-set Jaccard similarity: {float(metrics['character_jaccard']):.6f}",
        f"- Baseline-only characters: `{''.join(experiment.baseline_only_characters) or 'none'}`",
        f"- Shifted-only characters: `{''.join(experiment.shifted_only_characters) or 'none'}`",
        f"- Smoothed transition Jensen-Shannon divergence: {float(metrics['transition_js_divergence']):.9f}",
        "",
        "## Cross-corpus evaluation",
        "",
        "| Trained on | Evaluated on | Transitions | Mean NLL | Perplexity |",
        "|---|---|---:|---:|---:|",
    ]
    for evaluation in experiment.evaluations:
        lines.append(
            f"| {evaluation.trained_on} | {evaluation.evaluated_on} | "
            f"{evaluation.transition_count} | {evaluation.mean_nll:.9f} | "
            f"{evaluation.perplexity:.6f} |"
        )
    lines.extend(
        (
            "",
            "## Boundary audit",
            "",
            "Each word is an independent record. Correct encoding adds a start and end boundary to every record. The negative control concatenates all records and therefore creates one last-to-first transition between every neighboring pair.",
            "",
            "| Corpus | Records | Boundary-aware transitions | Packed transitions | Cross-record events |",
            "|---|---:|---:|---:|---:|",
            f"| baseline | {experiment.baseline_boundary_audit.record_count} | {experiment.baseline_boundary_audit.boundary_aware_transitions} | {experiment.baseline_boundary_audit.packed_transitions} | {len(experiment.baseline_boundary_audit.cross_record_events)} |",
            f"| shifted | {experiment.shifted_boundary_audit.record_count} | {experiment.shifted_boundary_audit.boundary_aware_transitions} | {experiment.shifted_boundary_audit.packed_transitions} | {len(experiment.shifted_boundary_audit.cross_record_events)} |",
            "",
            "```mermaid",
            render_boundary_mermaid(experiment),
            "```",
            "",
            "## Scope",
            "",
            "The experiment demonstrates dataset shift and record-boundary semantics in a character bigram model. It is not a transformer-attention implementation, a fix for an upstream project, or evidence about production model quality.",
            "",
        )
    )
    return "\n".join(lines)
