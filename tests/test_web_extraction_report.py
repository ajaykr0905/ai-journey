from hashlib import sha256
import json

from ai_journey.web_corpus import ExtractionPolicy, extract_web_corpus
from ai_journey.web_extraction_report import (
    build_extraction_report,
    canonical_json_bytes,
    corpus_payload,
)
from ai_journey.web_source import WebPage


def _result():
    html = b"<article><p>Useful public text for a deterministic corpus.</p></article>"
    pages = (WebPage("https://example.com/page", "text/html", html),)
    policy = ExtractionPolicy(minimum_pages=1, minimum_text_characters=10)
    return extract_web_corpus(pages, policy), policy


def test_report_seals_snapshot_policy_outcomes_and_corpus_bytes():
    result, policy = _result()
    snapshot_digest = sha256(b"snapshot").hexdigest()

    report = build_extraction_report(result, policy, snapshot_sha256=snapshot_digest)
    unsealed = {key: value for key, value in report.items() if key != "report_sha256"}

    assert report["snapshot_sha256"] == snapshot_digest
    assert report["counts"] == {
        "input_pages": 1,
        "accepted_pages": 1,
        "rejected_pages": 0,
        "extracted_characters": 46,
    }
    assert report["corpus"]["sha256"] == sha256(corpus_payload(result)).hexdigest()
    assert report["report_sha256"] == sha256(canonical_json_bytes(unsealed)).hexdigest()
    assert "no claim" in report["evidence_boundary"]


def test_corpus_payload_is_canonical_newline_terminated_jsonl():
    result, _ = _result()

    payload = corpus_payload(result)
    row = json.loads(payload)

    assert payload.endswith(b"\n")
    assert row["document_id"] == result.documents[0].document_id
    assert row["text_sha256"] == sha256(row["text"].encode()).hexdigest()
