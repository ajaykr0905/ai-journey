from hashlib import sha256

import pytest

from ai_journey.bpe import TokenizerError
from ai_journey.data_cleaning import clean_documents
from ai_journey.data_documents import DataDocument
from ai_journey.minhash_corpus import Project1Corpus
from ai_journey.minhash_dedup import MinHashDedupPolicy, deduplicate_with_minhash
from ai_journey.minhash_report import (
    build_minhash_report,
    canonical_json_bytes,
    deduplicated_corpus_payload,
)


def _evidence():
    documents = clean_documents(
        (
            DataDocument("b", "same public text", "public://b"),
            DataDocument("a", "same public text", "public://a"),
        )
    )
    corpus = Project1Corpus(documents, "a" * 64, ("b" * 64,))
    policy = MinHashDedupPolicy(2, 64, 66, 32, 0.8)
    result = deduplicate_with_minhash(documents, policy)
    return corpus, policy, result


def test_report_seals_removal_metrics_and_output_corpus_identity():
    corpus, policy, result = _evidence()

    report = build_minhash_report(corpus, policy, result)
    payload = deduplicated_corpus_payload(result)

    assert report["metrics"]["removed_documents"] == 1
    assert report["metrics"]["removed_fraction"] == 0.5
    assert report["corpus"]["artifact_bytes"] == len(payload)
    assert report["corpus"]["sha256"] == sha256(payload).hexdigest()
    seal = report.pop("report_sha256")
    assert seal == sha256(canonical_json_bytes(report)).hexdigest()


def test_report_is_deterministic_for_identical_evidence():
    corpus, policy, result = _evidence()
    assert build_minhash_report(corpus, policy, result) == build_minhash_report(
        corpus, policy, result
    )


def test_canonical_json_rejects_nonfinite_values():
    with pytest.raises(TokenizerError, match="canonical JSON"):
        canonical_json_bytes({"bad": float("nan")})


def test_corpus_payload_requires_a_minhash_result():
    with pytest.raises(TypeError, match="MinHash result"):
        deduplicated_corpus_payload(object())
