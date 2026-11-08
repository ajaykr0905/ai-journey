from hashlib import sha256
import json

import pytest

from ai_journey.bpe import TokenizerError
from ai_journey.web_corpus import ExtractionPolicy, extract_web_corpus
from ai_journey.web_extraction_io import publish_extraction_artifacts
from ai_journey.web_source import WebPage


def _inputs():
    page = WebPage(
        "https://example.com/page",
        "text/html",
        b"<article>Enough useful public text to accept this document.</article>",
    )
    policy = ExtractionPolicy(minimum_pages=1, minimum_text_characters=10)
    return extract_web_corpus((page,), policy), policy


def test_publisher_atomically_writes_complete_extraction_directory(tmp_path):
    result, policy = _inputs()
    output = tmp_path / "evidence"

    report = publish_extraction_artifacts(
        output,
        result,
        policy,
        snapshot_sha256=sha256(b"snapshot").hexdigest(),
    )

    assert {path.name for path in output.iterdir()} == {
        "corpus.jsonl",
        "manifest.json",
    }
    assert json.loads((output / "manifest.json").read_bytes()) == report
    assert (
        sha256((output / "corpus.jsonl").read_bytes()).hexdigest()
        == report["corpus"]["sha256"]
    )


def test_publisher_refuses_to_replace_prior_evidence(tmp_path):
    result, policy = _inputs()
    output = tmp_path / "evidence"
    output.mkdir()

    with pytest.raises(TokenizerError, match="already exists"):
        publish_extraction_artifacts(
            output,
            result,
            policy,
            snapshot_sha256=sha256(b"snapshot").hexdigest(),
        )


def test_publisher_cleans_temporary_directory_after_write_failure(
    tmp_path, monkeypatch
):
    result, policy = _inputs()
    output = tmp_path / "evidence"

    def fail_write(path, payload):
        raise OSError("injected write failure")

    monkeypatch.setattr("ai_journey.web_extraction_io._write_durable", fail_write)
    with pytest.raises(OSError, match="injected"):
        publish_extraction_artifacts(
            output,
            result,
            policy,
            snapshot_sha256=sha256(b"snapshot").hexdigest(),
        )

    assert not output.exists()
    assert list(tmp_path.iterdir()) == []
