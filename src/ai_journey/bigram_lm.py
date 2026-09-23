"""Day 19: a deterministic count-based character bigram language model.

The implementation is intentionally small and inspectable.  It builds a
vocabulary, counts adjacent character pairs including a boundary token,
normalizes each count row with explicit NumPy broadcasting, and samples from
the resulting categorical distributions with a seeded generator.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from hashlib import sha256
from itertools import pairwise
from math import isfinite
from pathlib import Path

import numpy as np

BOUNDARY_TOKEN = "."
DEFAULT_CORPUS = (
    "adara",
    "alani",
    "amira",
    "anika",
    "aravi",
    "avani",
    "devin",
    "elara",
    "ishan",
    "kaira",
    "leora",
    "malin",
    "navin",
    "orin",
    "priya",
    "reyan",
    "samira",
    "tavin",
    "viara",
    "zarin",
)


class BigramValidationError(ValueError):
    """Raised when corpus, model, sampling, or experiment invariants fail."""


@dataclass(frozen=True)
class Vocabulary:
    """Stable token ordering with the boundary token fixed at index zero."""

    tokens: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.tokens or self.tokens[0] != BOUNDARY_TOKEN:
            raise BigramValidationError("boundary token must be vocabulary index 0")
        if len(set(self.tokens)) != len(self.tokens):
            raise BigramValidationError("vocabulary tokens must be unique")
        if any(len(token) != 1 for token in self.tokens):
            raise BigramValidationError("vocabulary tokens must be single characters")

    @property
    def size(self) -> int:
        return len(self.tokens)

    def encode(self, token: str) -> int:
        """Map one known token to its stable integer id."""

        if not isinstance(token, str) or len(token) != 1:
            raise TypeError("token must be a single-character string")
        try:
            return self.tokens.index(token)
        except ValueError as exc:
            raise BigramValidationError(f"unknown token: {token!r}") from exc

    def decode(self, token_id: int) -> str:
        """Map a non-boolean integer id back to its token."""

        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise TypeError("token id must be an integer")
        if not 0 <= token_id < self.size:
            raise BigramValidationError(f"token id out of range: {token_id}")
        return self.tokens[token_id]


@dataclass(frozen=True)
class CorpusManifest:
    """Reproducibility metadata for the exact normalized corpus bytes."""

    source: str
    provenance: str
    license: str
    encoding: str
    sha256: str
    word_count: int
    character_count: int
    unique_word_count: int
    vocabulary: tuple[str, ...]


@dataclass(frozen=True)
class BigramModel:
    """Integer transition counts and row-normalized probabilities."""

    vocabulary: Vocabulary
    counts: np.ndarray
    probabilities: np.ndarray


@dataclass(frozen=True)
class RankedTransition:
    """One observed transition ranked by count and lexical tie-breakers."""

    previous: str
    next: str
    count: int
    probability: float


@dataclass(frozen=True)
class SampleTrace:
    """Generated text plus every sampled token id and model probability."""

    text: str
    token_ids: tuple[int, ...]
    transition_probabilities: tuple[float, ...]
    terminated: bool


@dataclass(frozen=True)
class BigramExperiment:
    """The Day 19 corpus manifest, model, oracle comparison, and samples."""

    corpus: tuple[str, ...]
    manifest: CorpusManifest
    model: BigramModel
    oracle_max_error: float
    samples: tuple[SampleTrace, ...]
    seed: int


def normalize_corpus(words: Iterable[str]) -> tuple[str, ...]:
    """Return non-empty lowercase ASCII words with surrounding space removed."""

    if isinstance(words, (str, bytes)):
        raise TypeError("corpus must be an iterable of words, not one string")
    normalized: list[str] = []
    for index, word in enumerate(words):
        if not isinstance(word, str):
            raise TypeError(f"corpus item {index} must be a string")
        clean = word.strip()
        if not clean:
            raise BigramValidationError(f"corpus item {index} is empty")
        if BOUNDARY_TOKEN in clean:
            raise BigramValidationError(
                "corpus words must not contain boundary token '.'"
            )
        if not clean.isascii() or not clean.isalpha() or clean != clean.lower():
            raise BigramValidationError(
                "corpus words must contain lowercase ASCII letters only"
            )
        normalized.append(clean)
    if not normalized:
        raise BigramValidationError("corpus must not be empty")
    return tuple(normalized)


def load_corpus(path: Path) -> tuple[str, ...]:
    """Load and validate a UTF-8 corpus containing one word per line."""

    if not isinstance(path, Path):
        raise TypeError("path must be pathlib.Path")
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise BigramValidationError("corpus must be valid UTF-8") from exc
    return normalize_corpus(text.splitlines())


def canonical_corpus_bytes(words: Iterable[str]) -> bytes:
    """Serialize normalized words with LF endings and one final newline."""

    corpus = normalize_corpus(words)
    return ("\n".join(corpus) + "\n").encode("utf-8")


def build_vocabulary(words: Iterable[str]) -> Vocabulary:
    """Build a boundary-first, lexically sorted character vocabulary."""

    corpus = normalize_corpus(words)
    characters = tuple(sorted({character for word in corpus for character in word}))
    return Vocabulary((BOUNDARY_TOKEN, *characters))


def iter_bigrams(word: str) -> tuple[tuple[str, str], ...]:
    """Return adjacent pairs after adding start and end boundary tokens."""

    (normalized,) = normalize_corpus((word,))
    augmented = BOUNDARY_TOKEN + normalized + BOUNDARY_TOKEN
    return tuple(pairwise(augmented))


def count_bigrams(words: Iterable[str], vocabulary: Vocabulary) -> np.ndarray:
    """Count every boundary-aware transition into a square integer matrix."""

    corpus = normalize_corpus(words)
    if not isinstance(vocabulary, Vocabulary):
        raise TypeError("vocabulary must be Vocabulary")
    counts = np.zeros((vocabulary.size, vocabulary.size), dtype=np.int64)
    for word in corpus:
        for previous, next_token in iter_bigrams(word):
            counts[vocabulary.encode(previous), vocabulary.encode(next_token)] += 1
    return counts


def normalize_counts(counts: np.ndarray) -> np.ndarray:
    """Normalize rows using ``(vocab, 1)`` totals broadcast over ``(vocab, vocab)``."""

    if not isinstance(counts, np.ndarray):
        raise TypeError("counts must be a NumPy array")
    if counts.ndim != 2 or counts.shape[0] != counts.shape[1]:
        raise BigramValidationError("counts must be a square matrix")
    if counts.dtype.kind not in {"i", "u"}:
        raise TypeError("counts must use an integer dtype")
    if np.any(counts < 0):
        raise BigramValidationError("counts must be non-negative")
    row_totals = counts.sum(axis=1, keepdims=True)
    if np.any(row_totals == 0):
        raise BigramValidationError(
            "every vocabulary token must have an outgoing count"
        )
    return counts.astype(np.float64) / row_totals


def loop_probability_oracle(counts: np.ndarray) -> np.ndarray:
    """Independently normalize counts with explicit Python loops."""

    if not isinstance(counts, np.ndarray) or counts.ndim != 2:
        raise TypeError("counts must be a two-dimensional NumPy array")
    rows, columns = counts.shape
    if rows != columns:
        raise BigramValidationError("counts must be a square matrix")
    probabilities = np.zeros((rows, columns), dtype=np.float64)
    for row in range(rows):
        total = int(sum(int(counts[row, column]) for column in range(columns)))
        if total <= 0:
            raise BigramValidationError("every count row must have a positive total")
        for column in range(columns):
            probabilities[row, column] = int(counts[row, column]) / total
    return probabilities


def build_bigram_model(words: Iterable[str]) -> BigramModel:
    """Build and validate a count-based bigram model from normalized words."""

    corpus = normalize_corpus(words)
    vocabulary = build_vocabulary(corpus)
    counts = count_bigrams(corpus, vocabulary)
    model = BigramModel(vocabulary, counts, normalize_counts(counts))
    validate_model(model)
    return model


def validate_model(model: BigramModel, *, tolerance: float = 1e-12) -> None:
    """Check shapes, dtypes, counts, finiteness, and probability row sums."""

    if not isinstance(model, BigramModel):
        raise TypeError("model must be BigramModel")
    expected_shape = (model.vocabulary.size, model.vocabulary.size)
    if model.counts.shape != expected_shape:
        raise BigramValidationError("count matrix shape does not match vocabulary")
    if model.probabilities.shape != expected_shape:
        raise BigramValidationError(
            "probability matrix shape does not match vocabulary"
        )
    if model.counts.dtype.kind not in {"i", "u"}:
        raise BigramValidationError("count matrix must use an integer dtype")
    if np.any(model.counts < 0):
        raise BigramValidationError("count matrix contains a negative value")
    if not np.all(np.isfinite(model.probabilities)):
        raise BigramValidationError("probability matrix contains a non-finite value")
    if np.any(model.probabilities < 0.0):
        raise BigramValidationError("probability matrix contains a negative value")
    if not np.allclose(model.probabilities.sum(axis=1), 1.0, atol=tolerance):
        raise BigramValidationError("probability rows must sum to one")


def build_manifest(
    words: Iterable[str],
    *,
    source: str = "data/day-19-demo-names.txt",
) -> CorpusManifest:
    """Build a public-safe corpus manifest with a content hash and basic counts."""

    corpus = normalize_corpus(words)
    vocabulary = build_vocabulary(corpus)
    return CorpusManifest(
        source=source,
        provenance="Independently selected synthetic name-like tokens for education",
        license="CC0-1.0",
        encoding="UTF-8, one lowercase ASCII token per line, LF line endings",
        sha256=sha256(canonical_corpus_bytes(corpus)).hexdigest(),
        word_count=len(corpus),
        character_count=sum(map(len, corpus)),
        unique_word_count=len(set(corpus)),
        vocabulary=vocabulary.tokens,
    )


def transition_probability(model: BigramModel, previous: str, next_token: str) -> float:
    """Look up one conditional transition probability by token."""

    validate_model(model)
    row = model.vocabulary.encode(previous)
    column = model.vocabulary.encode(next_token)
    return float(model.probabilities[row, column])


def top_transitions(
    model: BigramModel, limit: int = 10
) -> tuple[RankedTransition, ...]:
    """Rank observed transitions by count, then by token for deterministic ties."""

    validate_model(model)
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise TypeError("limit must be an integer")
    if limit <= 0:
        raise BigramValidationError("limit must be positive")
    transitions: list[RankedTransition] = []
    for row, previous in enumerate(model.vocabulary.tokens):
        for column, next_token in enumerate(model.vocabulary.tokens):
            count = int(model.counts[row, column])
            if count:
                transitions.append(
                    RankedTransition(
                        previous,
                        next_token,
                        count,
                        float(model.probabilities[row, column]),
                    )
                )
    transitions.sort(key=lambda item: (-item.count, item.previous, item.next))
    return tuple(transitions[:limit])


def sample_word(
    model: BigramModel,
    rng: np.random.Generator,
    *,
    max_length: int = 32,
) -> SampleTrace:
    """Sample one word autoregressively from the boundary token."""

    validate_model(model)
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be numpy.random.Generator")
    if isinstance(max_length, bool) or not isinstance(max_length, int):
        raise TypeError("max_length must be an integer")
    if max_length <= 0:
        raise BigramValidationError("max_length must be positive")

    current = 0
    token_ids = [current]
    probabilities: list[float] = []
    characters: list[str] = []
    terminated = False
    for _ in range(max_length):
        next_id = int(rng.choice(model.vocabulary.size, p=model.probabilities[current]))
        probability = float(model.probabilities[current, next_id])
        probabilities.append(probability)
        token_ids.append(next_id)
        if next_id == 0:
            terminated = True
            break
        characters.append(model.vocabulary.decode(next_id))
        current = next_id
    return SampleTrace(
        "".join(characters), tuple(token_ids), tuple(probabilities), terminated
    )


def sample_words(
    model: BigramModel,
    *,
    count: int = 10,
    seed: int = 1909,
    max_length: int = 32,
) -> tuple[SampleTrace, ...]:
    """Generate a deterministic sequence of sample traces from one seed."""

    if isinstance(count, bool) or not isinstance(count, int):
        raise TypeError("count must be an integer")
    if count <= 0:
        raise BigramValidationError("count must be positive")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    rng = np.random.default_rng(seed)
    return tuple(sample_word(model, rng, max_length=max_length) for _ in range(count))


def run_bigram_experiment(
    words: Iterable[str] = DEFAULT_CORPUS,
    *,
    sample_count: int = 10,
    seed: int = 1909,
    source: str = "data/day-19-demo-names.txt",
) -> BigramExperiment:
    """Run the fixed Day 19 model build, oracle comparison, and smoke samples."""

    corpus = normalize_corpus(words)
    model = build_bigram_model(corpus)
    oracle = loop_probability_oracle(model.counts)
    experiment = BigramExperiment(
        corpus=corpus,
        manifest=build_manifest(corpus, source=source),
        model=model,
        oracle_max_error=float(np.max(np.abs(model.probabilities - oracle))),
        samples=sample_words(model, count=sample_count, seed=seed),
        seed=seed,
    )
    validate_experiment(experiment)
    return experiment


def validate_experiment(experiment: BigramExperiment) -> None:
    """Validate manifest, transition totals, oracle agreement, and sample traces."""

    if not isinstance(experiment, BigramExperiment):
        raise TypeError("experiment must be BigramExperiment")
    validate_model(experiment.model)
    expected_manifest = build_manifest(
        experiment.corpus, source=experiment.manifest.source
    )
    if experiment.manifest != expected_manifest:
        raise BigramValidationError("corpus manifest does not match normalized corpus")
    expected_transitions = sum(len(word) + 1 for word in experiment.corpus)
    if int(experiment.model.counts.sum()) != expected_transitions:
        raise BigramValidationError("count total does not match boundary-aware corpus")
    if not isfinite(experiment.oracle_max_error) or experiment.oracle_max_error > 1e-12:
        raise BigramValidationError("broadcast probabilities disagree with loop oracle")
    if not experiment.samples:
        raise BigramValidationError("experiment must contain at least one sample")
    vocabulary_size = experiment.model.vocabulary.size
    for sample in experiment.samples:
        if len(sample.token_ids) != len(sample.transition_probabilities) + 1:
            raise BigramValidationError("sample trace lengths do not reconcile")
        if sample.token_ids[0] != 0:
            raise BigramValidationError("sample trace must start at boundary token")
        if any(not 0 <= token_id < vocabulary_size for token_id in sample.token_ids):
            raise BigramValidationError("sample trace contains an invalid token id")
        for previous, next_id, probability in zip(
            sample.token_ids, sample.token_ids[1:], sample.transition_probabilities
        ):
            expected = float(experiment.model.probabilities[previous, next_id])
            if probability != expected or probability <= 0.0:
                raise BigramValidationError("sample probability does not match model")
        if sample.terminated != (sample.token_ids[-1] == 0):
            raise BigramValidationError("sample termination flag is inconsistent")


def experiment_metrics(experiment: BigramExperiment) -> dict[str, int | float | str]:
    """Return compact, JSON-safe metrics for CI and reports."""

    validate_experiment(experiment)
    return {
        "corpus_sha256": experiment.manifest.sha256,
        "word_count": experiment.manifest.word_count,
        "character_count": experiment.manifest.character_count,
        "vocabulary_size": experiment.model.vocabulary.size,
        "observed_transitions": int(experiment.model.counts.sum()),
        "nonzero_bigram_types": int(np.count_nonzero(experiment.model.counts)),
        "probability_shape": f"{experiment.model.probabilities.shape[0]}x"
        f"{experiment.model.probabilities.shape[1]}",
        "row_total_shape": f"{experiment.model.counts.sum(axis=1, keepdims=True).shape[0]}x1",
        "oracle_max_error": experiment.oracle_max_error,
        "sample_count": len(experiment.samples),
        "terminated_samples": sum(sample.terminated for sample in experiment.samples),
        "seed": experiment.seed,
    }


def manifest_payload(manifest: CorpusManifest) -> dict[str, object]:
    """Convert a manifest to a stable JSON-compatible mapping."""

    if not isinstance(manifest, CorpusManifest):
        raise TypeError("manifest must be CorpusManifest")
    return {
        "source": manifest.source,
        "provenance": manifest.provenance,
        "license": manifest.license,
        "encoding": manifest.encoding,
        "sha256": manifest.sha256,
        "word_count": manifest.word_count,
        "character_count": manifest.character_count,
        "unique_word_count": manifest.unique_word_count,
        "vocabulary": list(manifest.vocabulary),
    }


def experiment_payload(experiment: BigramExperiment) -> dict[str, object]:
    """Convert the complete deterministic experiment to JSON-safe data."""

    validate_experiment(experiment)
    return {
        "metrics": experiment_metrics(experiment),
        "manifest": manifest_payload(experiment.manifest),
        "vocabulary": list(experiment.model.vocabulary.tokens),
        "counts": experiment.model.counts.tolist(),
        "probabilities": experiment.model.probabilities.tolist(),
        "samples": [
            {
                "text": sample.text,
                "token_ids": list(sample.token_ids),
                "transition_probabilities": list(sample.transition_probabilities),
                "terminated": sample.terminated,
            }
            for sample in experiment.samples
        ],
    }


def render_bigram_mermaid(model: BigramModel, *, limit: int = 12) -> str:
    """Render the most frequent observed transitions as a Mermaid graph."""

    transitions = top_transitions(model, limit)
    lines = ["flowchart LR"]
    for transition in transitions:
        previous = (
            "START_END"
            if transition.previous == BOUNDARY_TOKEN
            else transition.previous
        )
        next_token = (
            "START_END" if transition.next == BOUNDARY_TOKEN else transition.next
        )
        lines.append(
            f'    {previous}["{transition.previous}"] -->|"n={transition.count}, '
            f'p={transition.probability:.3f}"| {next_token}["{transition.next}"]'
        )
    return "\n".join(lines)


def render_bigram_markdown(experiment: BigramExperiment) -> str:
    """Render an inspectable Day 19 report without claiming learner completion."""

    validate_experiment(experiment)
    metrics = experiment_metrics(experiment)
    lines = [
        "# Day 19 count-based character bigram model",
        "",
        "This deterministic report is repository evidence only. It does not prove that ",
        "the scheduled lectures were watched or that the learner completed the type-along.",
        "",
        "## Data manifest",
        "",
        f"- Source: `{experiment.manifest.source}`",
        f"- Provenance: {experiment.manifest.provenance}",
        f"- License: {experiment.manifest.license}",
        f"- SHA-256: `{experiment.manifest.sha256}`",
        f"- Words: {experiment.manifest.word_count}",
        f"- Characters: {experiment.manifest.character_count}",
        f"- Vocabulary: `{' '.join(experiment.manifest.vocabulary)}`",
        "",
        "## Model smoke test",
        "",
        f"- Count matrix: `{metrics['probability_shape']}`",
        f"- Broadcast row totals: `{metrics['row_total_shape']}`",
        f"- Observed transitions: {metrics['observed_transitions']}",
        f"- Non-zero bigram types: {metrics['nonzero_bigram_types']}",
        f"- Maximum broadcast/oracle error: {float(metrics['oracle_max_error']):.3e}",
        f"- Seed: {experiment.seed}",
        f"- Terminated samples: {metrics['terminated_samples']}/{metrics['sample_count']}",
        "",
        "## Deterministic samples",
        "",
    ]
    for index, sample in enumerate(experiment.samples, start=1):
        state = "terminated" if sample.terminated else "max length reached"
        lines.append(f"{index}. `{sample.text}` ({state})")
    lines.extend(
        [
            "",
            "## Frequent transitions",
            "",
            "| Previous | Next | Count | Conditional probability |",
            "|:--:|:--:|--:|--:|",
        ]
    )
    for transition in top_transitions(experiment.model):
        lines.append(
            f"| `{transition.previous}` | `{transition.next}` | {transition.count} | "
            f"{transition.probability:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Transition diagram",
            "",
            "```mermaid",
            render_bigram_mermaid(experiment.model),
            "```",
            "",
        ]
    )
    return "\n".join(lines)
