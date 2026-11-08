from dataclasses import replace
import json

import pytest

from ai_journey.bpe import TokenizerError
from ai_journey.data_cleaning import clean_documents
from ai_journey.data_documents import DataDocument
from ai_journey.minhash_corpus import Project1Corpus
from ai_journey.minhash_dedup import MinHashDedupPolicy, deduplicate_with_minhash
from ai_journey.minhash_io import publish_minhash_artifacts
from ai_journey.minhash_verify import verify_minhash_artifacts


def _publish(tmp_path):
    documents = clean_documents(
        (
            DataDocument("a", "same public text", "public://a"),
            DataDocument("b", "same public text", "public://b"),
            DataDocument("c", "different public record", "public://c"),
        )
    )
    documents = tuple(replace(row, source_sha256="1" * 64) for row in documents)
    corpus = Project1Corpus(documents, "a" * 64, ("b" * 64,))
    policy = MinHashDedupPolicy(2, 64, 66, 32, 0.8)
    result = deduplicate_with_minhash(documents, policy)
    output = tmp_path / "minhash"
    publish_minhash_artifacts(output, corpus, policy, result)
    return output


def test_verifier_accepts_complete_sealed_minhash_evidence(tmp_path):
    output = _publish(tmp_path)

    report = verify_minhash_artifacts(output)

    assert report["metrics"]["input_documents"] == 3
    assert report["metrics"]["removed_documents"] == 1
    assert report["metrics"]["kept_documents"] == 2


def test_verifier_rejects_manifest_tampering(tmp_path):
    output = _publish(tmp_path)
    path = output / "manifest.json"
    report = json.loads(path.read_text())
    report["metrics"]["removed_documents"] = 0
    path.write_text(json.dumps(report))

    with pytest.raises(TokenizerError, match="seal mismatch"):
        verify_minhash_artifacts(output)


def test_verifier_rejects_corpus_tampering(tmp_path):
    output = _publish(tmp_path)
    with (output / "deduplicated-corpus.jsonl").open("ab") as stream:
        stream.write(b"changed")

    with pytest.raises(TokenizerError, match="byte budget|identity"):
        verify_minhash_artifacts(output)


def test_verifier_rejects_extra_inventory(tmp_path):
    output = _publish(tmp_path)
    (output / "extra.txt").write_text("unexpected")
    with pytest.raises(TokenizerError, match="inventory"):
        verify_minhash_artifacts(output)


def test_verifier_requires_pathlib_root():
    with pytest.raises(TypeError, match="pathlib.Path"):
        verify_minhash_artifacts("minhash")
