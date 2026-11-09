from hashlib import sha256

import pytest

from ai_journey.bpe import TokenizerError
from ai_journey.data_pipeline_report import canonical_json_bytes
from ai_journey.minhash_corpus import load_project1_corpus


def _publish_dataset(root):
    root.mkdir()
    source_sha = "1" * 64
    rows = {
        "train-00000.jsonl": {
            "document_id": "z",
            "source": "https://example.org/public",
            "source_sha256": source_sha,
            "text": "alpha beta gamma",
        },
        "validation-00000.jsonl": {
            "document_id": "a",
            "source": "https://example.org/public",
            "source_sha256": source_sha,
            "text": "alpha beta gamma delta",
        },
    }
    partitions = {}
    for partition, name in (
        ("train", "train-00000.jsonl"),
        ("validation", "validation-00000.jsonl"),
    ):
        payload = canonical_json_bytes(rows[name])
        (root / name).write_bytes(payload)
        partitions[partition] = {
            "documents": 1,
            "shards": [
                {
                    "path": name,
                    "documents": 1,
                    "utf8_text_bytes": len(rows[name]["text"].encode()),
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
    return report


def test_project1_corpus_loads_verified_shards_and_identities(tmp_path):
    root = tmp_path / "dataset"
    report = _publish_dataset(root)

    corpus = load_project1_corpus(root)

    assert [row.document_id for row in corpus.documents] == ["a", "z"]
    assert corpus.dataset_report_sha256 == report["report_sha256"]
    assert corpus.shard_sha256s == tuple(
        report["partitions"][name]["shards"][0]["sha256"]
        for name in ("train", "validation")
    )


def test_project1_corpus_rejects_changed_shard_bytes(tmp_path):
    root = tmp_path / "dataset"
    _publish_dataset(root)
    with (root / "train-00000.jsonl").open("ab") as stream:
        stream.write(b"changed")

    with pytest.raises(TokenizerError, match="byte budget|content identity"):
        load_project1_corpus(root)


def test_project1_corpus_requires_a_path():
    with pytest.raises(TypeError, match="pathlib.Path"):
        load_project1_corpus("dataset")
