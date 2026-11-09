import json

import pytest

from ai_journey.bpe import TokenizerError
from ai_journey.data_cleaning import clean_documents
from ai_journey.data_documents import DataDocument
from ai_journey.minhash_corpus import Project1Corpus
from ai_journey.minhash_dedup import MinHashDedupPolicy, deduplicate_with_minhash
from ai_journey.minhash_io import publish_minhash_artifacts


def _evidence():
    documents = clean_documents(
        (
            DataDocument("a", "same public text", "public://a"),
            DataDocument("b", "same public text", "public://b"),
        )
    )
    corpus = Project1Corpus(documents, "a" * 64, ("b" * 64,))
    policy = MinHashDedupPolicy(2, 64, 66, 32, 0.8)
    return corpus, policy, deduplicate_with_minhash(documents, policy)


def test_publisher_commits_complete_artifact_directory(tmp_path):
    corpus, policy, result = _evidence()
    output = tmp_path / "minhash"

    report = publish_minhash_artifacts(output, corpus, policy, result)

    assert {path.name for path in output.iterdir()} == {
        "manifest.json",
        "deduplicated-corpus.jsonl",
    }
    assert json.loads((output / "manifest.json").read_text()) == report
    assert list(tmp_path.glob(".minhash.*")) == []


def test_publisher_refuses_to_replace_existing_output(tmp_path):
    corpus, policy, result = _evidence()
    output = tmp_path / "minhash"
    output.mkdir()
    marker = output / "owned.txt"
    marker.write_text("preserve")

    with pytest.raises(TokenizerError, match="already exists"):
        publish_minhash_artifacts(output, corpus, policy, result)

    assert marker.read_text() == "preserve"


def test_publisher_requires_pathlib_output():
    corpus, policy, result = _evidence()
    with pytest.raises(TypeError, match="pathlib.Path"):
        publish_minhash_artifacts("output", corpus, policy, result)
