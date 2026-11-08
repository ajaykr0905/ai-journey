from dataclasses import replace
from hashlib import sha256
import json

import pytest

from ai_journey.bpe import TokenizerError
from ai_journey.data_cleaning import clean_documents
from ai_journey.data_documents import DataDocument
from ai_journey.minhash_corpus import Project1Corpus, load_project1_corpus
from ai_journey.minhash_dedup import MinHashDedupPolicy, deduplicate_with_minhash
from ai_journey.minhash_io import publish_minhash_artifacts
from ai_journey.minhash_report import canonical_json_bytes
from ai_journey.minhash_verify import replay_minhash_artifacts, verify_minhash_artifacts


def _dataset(root, *, suffix=""):
    root.mkdir()
    rows = (
        ("a", "same public text" + suffix),
        ("b", "same public text"),
        ("c", "different public record"),
    )
    partitions = {}
    for partition, selected in (("train", rows[:2]), ("validation", rows[2:])):
        name = f"{partition}-00000.jsonl"
        payload = b"".join(
            canonical_json_bytes(
                {
                    "document_id": identity,
                    "source": "https://example.org/public",
                    "source_sha256": "1" * 64,
                    "text": text,
                }
            )
            for identity, text in selected
        )
        (root / name).write_bytes(payload)
        partitions[partition] = {
            "documents": len(selected),
            "shards": [
                {
                    "path": name,
                    "documents": len(selected),
                    "utf8_text_bytes": sum(len(text.encode()) for _, text in selected),
                    "artifact_bytes": len(payload),
                    "sha256": sha256(payload).hexdigest(),
                }
            ],
        }
    report = {
        "schema_version": 1,
        "source": {
            "metadata": {},
            "source_url": "https://example.org/public",
            "license_url": "https://example.org/license",
            "attribution": "Public test fixture",
        },
        "config": {},
        "counts": {},
        "quality_rejections": [],
        "duplicates": [],
        "partitions": partitions,
        "tokenizer": {},
        "evidence_boundaries": {},
    }
    report["report_sha256"] = sha256(canonical_json_bytes(report)).hexdigest()
    (root / "manifest.json").write_bytes(canonical_json_bytes(report))
    return root


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


def test_replay_recomputes_evidence_from_verified_source_shards(tmp_path):
    dataset = _dataset(tmp_path / "dataset")
    corpus = load_project1_corpus(dataset)
    policy = MinHashDedupPolicy(2, 64, 66, 32, 0.8)
    output = tmp_path / "minhash"
    publish_minhash_artifacts(
        output,
        corpus,
        policy,
        deduplicate_with_minhash(corpus.documents, policy),
    )

    replayed = replay_minhash_artifacts(dataset, output)

    assert replayed["metrics"]["removed_documents"] == 1


def test_replay_rejects_a_different_verified_source_dataset(tmp_path):
    dataset = _dataset(tmp_path / "dataset")
    corpus = load_project1_corpus(dataset)
    policy = MinHashDedupPolicy(2, 64, 66, 32, 0.8)
    output = tmp_path / "minhash"
    publish_minhash_artifacts(
        output,
        corpus,
        policy,
        deduplicate_with_minhash(corpus.documents, policy),
    )
    changed = _dataset(tmp_path / "changed", suffix=" changed")

    with pytest.raises(TokenizerError, match="source identity"):
        replay_minhash_artifacts(changed, output)
