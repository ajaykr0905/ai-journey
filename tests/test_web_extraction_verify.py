from hashlib import sha256
import json

import pytest

from ai_journey.bpe import TokenizerError
from ai_journey.web_corpus import ExtractionPolicy, extract_web_corpus
from ai_journey.web_extraction_io import publish_extraction_artifacts
from ai_journey.web_extraction_verify import verify_extraction_artifacts
from ai_journey.web_source import WebPage


def _publish(tmp_path):
    page = WebPage(
        "https://example.com/page",
        "text/html",
        b"<article>Enough useful public text to accept this document.</article>",
    )
    policy = ExtractionPolicy(minimum_pages=1, minimum_text_characters=10)
    result = extract_web_corpus((page,), policy)
    output = tmp_path / "evidence"
    publish_extraction_artifacts(
        output,
        result,
        policy,
        snapshot_sha256=sha256(b"snapshot").hexdigest(),
    )
    return output


def _reseal(manifest):
    from ai_journey.web_extraction_report import canonical_json_bytes

    unsealed = {key: value for key, value in manifest.items() if key != "report_sha256"}
    manifest["report_sha256"] = sha256(canonical_json_bytes(unsealed)).hexdigest()


def test_verifier_recomputes_complete_extraction_evidence(tmp_path):
    output = _publish(tmp_path)

    report = verify_extraction_artifacts(output)

    assert report["counts"]["accepted_pages"] == 1
    assert report["counts"]["rejected_pages"] == 0


def test_verifier_rejects_corpus_tampering(tmp_path):
    output = _publish(tmp_path)
    path = output / "corpus.jsonl"
    path.write_bytes(path.read_bytes().replace(b"public", b"secret"))

    with pytest.raises(TokenizerError, match="identity mismatch"):
        verify_extraction_artifacts(output)


def test_verifier_rejects_resealed_count_contradictions(tmp_path):
    output = _publish(tmp_path)
    path = output / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["counts"]["accepted_pages"] = 0
    _reseal(manifest)
    path.write_text(json.dumps(manifest))

    with pytest.raises(TokenizerError, match="counts"):
        verify_extraction_artifacts(output)


def test_verifier_rejects_unexpected_inventory_and_symlink_roots(tmp_path):
    output = _publish(tmp_path)
    (output / "notes.txt").write_text("not evidence")
    with pytest.raises(TokenizerError, match="inventory"):
        verify_extraction_artifacts(output)

    (output / "notes.txt").unlink()
    link = tmp_path / "linked"
    link.symlink_to(output, target_is_directory=True)
    with pytest.raises(TokenizerError, match="directory"):
        verify_extraction_artifacts(link)
