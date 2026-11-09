"""Deterministic MinHash/LSH deduplication for cleaned public corpora."""

from dataclasses import dataclass
import math

from .bpe import TokenizerError
from .data_cleaning import CleanedDocument
from .minhash_assessment import CandidateAssessment, assess_candidates
from .minhash_candidates import candidate_pairs
from .minhash_clusters import duplicate_clusters
from .minhash_family import MinHashFamily
from .minhash_hash import hash_shingles
from .minhash_lsh import LSHBanding
from .minhash_selection import ClusterDecision, select_representatives
from .minhash_signature import build_minhash_signature
from .minhash_text import word_shingles


@dataclass(frozen=True)
class MinHashDedupPolicy:
    shingle_width: int
    num_hashes: int
    seed: int
    bands: int
    duplicate_threshold: float

    def __post_init__(self) -> None:
        if type(self.shingle_width) is not int or not 1 <= self.shingle_width <= 32:
            raise TokenizerError("dedup policy shingle width must be from 1 to 32")
        MinHashFamily(self.num_hashes, self.seed)
        LSHBanding(self.num_hashes, self.bands)
        if (
            isinstance(self.duplicate_threshold, bool)
            or not isinstance(self.duplicate_threshold, (int, float))
            or not math.isfinite(self.duplicate_threshold)
            or not 0 < self.duplicate_threshold <= 1
        ):
            raise TokenizerError("dedup policy threshold must be above zero to one")


@dataclass(frozen=True)
class MinHashDedupMetrics:
    input_documents: int
    candidate_pairs: int
    duplicate_pairs: int
    duplicate_clusters: int
    kept_documents: int
    removed_documents: int
    removed_fraction: float


@dataclass(frozen=True)
class MinHashDedupResult:
    documents: tuple[CleanedDocument, ...]
    assessments: tuple[CandidateAssessment, ...]
    decisions: tuple[ClusterDecision, ...]
    metrics: MinHashDedupMetrics


def deduplicate_with_minhash(
    documents: tuple[CleanedDocument, ...], policy: MinHashDedupPolicy
) -> MinHashDedupResult:
    """Run MinHash candidate discovery with exact threshold verification."""
    if (
        not isinstance(documents, tuple)
        or not documents
        or not all(isinstance(document, CleanedDocument) for document in documents)
    ):
        raise TokenizerError("MinHash deduplication requires cleaned documents")
    if not isinstance(policy, MinHashDedupPolicy):
        raise TypeError("MinHash deduplication requires a policy")
    ordered = tuple(sorted(documents, key=lambda row: row.document_id))
    if len({row.document_id for row in ordered}) != len(ordered):
        raise TokenizerError("MinHash deduplication requires unique document ids")
    shingles = {
        row.document_id: word_shingles(row.text, width=policy.shingle_width)
        for row in ordered
    }
    family = MinHashFamily(policy.num_hashes, policy.seed)
    signatures = {
        identity: build_minhash_signature(hash_shingles(values), family)
        for identity, values in shingles.items()
    }
    banding = LSHBanding(policy.num_hashes, policy.bands)
    pairs = candidate_pairs(signatures, banding)
    assessments = assess_candidates(
        pairs,
        shingles,
        signatures,
        threshold=policy.duplicate_threshold,
    )
    clusters = duplicate_clusters(tuple(shingles), assessments)
    kept_ids, decisions = select_representatives(tuple(shingles), clusters)
    kept = tuple(row for row in ordered if row.document_id in set(kept_ids))
    removed = len(ordered) - len(kept)
    metrics = MinHashDedupMetrics(
        input_documents=len(ordered),
        candidate_pairs=len(assessments),
        duplicate_pairs=sum(row.duplicate for row in assessments),
        duplicate_clusters=len(decisions),
        kept_documents=len(kept),
        removed_documents=removed,
        removed_fraction=removed / len(ordered),
    )
    return MinHashDedupResult(kept, assessments, decisions, metrics)
