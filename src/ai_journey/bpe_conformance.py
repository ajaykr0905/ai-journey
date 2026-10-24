"""Source-backed conformance checks for learned byte-pair merge tables."""

from collections import Counter
from dataclasses import asdict, dataclass
from hashlib import sha256
import json

from .bpe import (
    ByteTokenizer,
    RegexTokenizer,
    TokenizerError,
    _merge_weighted_streams,
    _weighted_pair_counts,
)
from .tokenizer_training import training_segments
from .tokenizer_io import tokenizer_fingerprint


@dataclass(frozen=True)
class MergeObservation:
    """The canonical pair selected at one training rank."""

    rank: int
    pair: tuple[int, int]
    frequency: int
    applied_count: int
    tokens_before: int
    tokens_after: int
    byte_length: int
    dependency_depth: int
    token_hex: str


def _weighted_token_count(streams: Counter) -> int:
    return sum(len(stream) * weight for stream, weight in streams.items())


def _replay_merge_prefix(
    documents: tuple[str, ...],
    tokenizer: ByteTokenizer,
    *,
    min_frequency: int = 2,
):
    if not isinstance(tokenizer, ByteTokenizer):
        raise TypeError("merge replay requires a byte tokenizer")
    ByteTokenizer.train("", min_frequency=min_frequency)
    pattern = tokenizer.pattern if isinstance(tokenizer, RegexTokenizer) else None
    streams = Counter(
        tuple(segment.encode("utf-8"))
        for segment in training_segments(documents, pattern=pattern)
    )
    observations = []
    depths = [0] * 256
    for rank, stored_pair in enumerate(tokenizer.merges):
        counts = _weighted_pair_counts(streams)
        if not counts:
            raise TokenizerError("stored merge continues after corpus convergence")
        selected = min(counts, key=lambda pair: (-counts[pair], pair))
        frequency = counts[selected]
        if frequency < min_frequency:
            raise TokenizerError("stored merge falls below the frequency policy")
        if stored_pair != selected:
            raise TokenizerError("stored merge is not the canonical corpus pair")
        tokens_before = _weighted_token_count(streams)
        merged = _merge_weighted_streams(streams, selected, 256 + rank)
        tokens_after = _weighted_token_count(merged)
        applied_count = tokens_before - tokens_after
        if not 0 < applied_count <= frequency:
            raise TokenizerError("merge application count contradicts pair frequency")
        token_id = 256 + rank
        dependency_depth = 1 + max(depths[selected[0]], depths[selected[1]])
        token_bytes = tokenizer.token_bytes(token_id)
        depths.append(dependency_depth)
        observations.append(
            MergeObservation(
                rank,
                selected,
                frequency,
                applied_count,
                tokens_before,
                tokens_after,
                len(token_bytes),
                dependency_depth,
                token_bytes.hex(),
            )
        )
        streams = merged
    return tuple(observations), streams


def replay_merge_prefix(
    documents: tuple[str, ...],
    tokenizer: ByteTokenizer,
    *,
    min_frequency: int = 2,
) -> tuple[MergeObservation, ...]:
    """Prove that every stored merge is the canonical next corpus merge."""
    observations, _ = _replay_merge_prefix(
        documents, tokenizer, min_frequency=min_frequency
    )
    return observations


def corpus_fingerprint(documents: tuple[str, ...]) -> str:
    """Bind document order and boundaries without normalizing source text."""
    training_segments(documents)
    return sha256(
        json.dumps(
            documents,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def audit_fingerprint(report: dict) -> str:
    if not isinstance(report, dict):
        raise TypeError("audit report must be an object")
    semantic = {key: value for key, value in report.items() if key != "audit_sha256"}
    return sha256(
        json.dumps(
            semantic,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def build_training_audit(
    documents: tuple[str, ...],
    tokenizer: ByteTokenizer,
    *,
    min_frequency: int = 2,
) -> dict:
    """Build a JSON-safe audit of corpus identity and every learned merge."""
    observations, final_streams = _replay_merge_prefix(
        documents, tokenizer, min_frequency=min_frequency
    )
    utf8_bytes = sum(len(document.encode("utf-8")) for document in documents)
    encoded = tuple(tokenizer.encode(document) for document in documents)
    if any(
        tokenizer.decode(ids) != document for ids, document in zip(encoded, documents)
    ):
        raise TokenizerError("tokenizer failed corpus round-trip during audit")
    final_tokens = sum(map(len, encoded))
    expected_tokens = observations[-1].tokens_after if observations else utf8_bytes
    if final_tokens != expected_tokens:
        raise TokenizerError("merge replay and encoder token counts disagree")
    token_counts = Counter(token for sequence in encoded for token in sequence)
    token_byte_lengths = Counter()
    for token_id, count in token_counts.items():
        token_byte_lengths[len(tokenizer.token_bytes(token_id))] += count
    if (
        sum(length * count for length, count in token_byte_lengths.items())
        != utf8_bytes
    ):
        raise TokenizerError("token byte-length histogram does not cover the corpus")
    unused_learned = [
        token_id
        for token_id in range(256, tokenizer.vocab_size)
        if token_id not in token_counts
    ]
    remaining = _weighted_pair_counts(final_streams)
    next_pair = (
        min(remaining, key=lambda pair: (-remaining[pair], pair)) if remaining else None
    )
    next_frequency = remaining[next_pair] if next_pair is not None else 0
    stop_reason = (
        "no_pairs"
        if next_pair is None
        else "min_frequency"
        if next_frequency < min_frequency
        else "vocabulary_limit"
    )
    report = {
        "schema_version": 1,
        "corpus_sha256": corpus_fingerprint(documents),
        "tokenizer_sha256": tokenizer_fingerprint(tokenizer),
        "document_count": len(documents),
        "utf8_bytes": utf8_bytes,
        "initial_tokens": utf8_bytes,
        "final_tokens": final_tokens,
        "compression_ratio": final_tokens / utf8_bytes,
        "used_vocabulary": len(token_counts),
        "byte_fallback_tokens": sum(
            count for token_id, count in token_counts.items() if token_id < 256
        ),
        "unused_learned_token_ids": unused_learned,
        "token_frequencies": [
            {"token_id": token_id, "count": count}
            for token_id, count in sorted(token_counts.items())
        ],
        "token_byte_length_histogram": [
            {"bytes": length, "count": count}
            for length, count in sorted(token_byte_lengths.items())
        ],
        "max_token_bytes": max(token_byte_lengths),
        "min_frequency": min_frequency,
        "pattern": tokenizer.pattern if isinstance(tokenizer, RegexTokenizer) else None,
        "stop_reason": stop_reason,
        "next_pair": next_pair,
        "next_frequency": next_frequency,
        "merges": [asdict(observation) for observation in observations],
    }
    report["audit_sha256"] = audit_fingerprint(report)
    return report


def verify_training_audit(
    report: dict,
    documents: tuple[str, ...],
    tokenizer: ByteTokenizer,
) -> dict:
    """Verify both the report seal and a fresh source-backed replay."""
    if not isinstance(report, dict) or type(report.get("audit_sha256")) is not str:
        raise TokenizerError("training audit fields are invalid")
    if audit_fingerprint(report) != report["audit_sha256"]:
        raise TokenizerError("training audit fingerprint mismatch")
    try:
        expected = build_training_audit(
            documents, tokenizer, min_frequency=report["min_frequency"]
        )
        observed_json = json.dumps(
            report, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        expected_json = json.dumps(
            expected, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
    except (KeyError, TypeError, ValueError, RecursionError) as exc:
        raise TokenizerError("training audit fields are invalid") from exc
    if observed_json != expected_json:
        raise TokenizerError("training audit does not match exact source replay")
    return {
        "audit_sha256": report["audit_sha256"],
        "verified_merges": len(report["merges"]),
        "final_tokens": report["final_tokens"],
    }
