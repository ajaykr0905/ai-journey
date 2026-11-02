from dataclasses import replace
from hashlib import sha256

import pytest

from ai_journey.bpe import TokenizerError
from ai_journey.data_pipeline_source import load_pipeline_source
from ai_journey.tokenizer_sources import ENGLISH


def _source_bytes():
    return b"# sent_id = a\n# text = A useful public sentence.\n\n# sent_id = b\n# text = Another public sentence."


def test_source_adapter_binds_validated_bytes_license_and_document_provenance(tmp_path):
    payload = _source_bytes()
    snapshot = tmp_path / "english.conllu"
    snapshot.write_bytes(payload)
    source = replace(
        ENGLISH, size_bytes=len(payload), sha256=sha256(payload).hexdigest()
    )

    evidence = load_pipeline_source(snapshot, source)

    assert evidence.metadata["revision"] == source.revision
    assert evidence.metadata["sha256"] == source.sha256
    assert evidence.source_url == source.url
    assert evidence.license_url == source.license_url
    assert [row.document_id for row in evidence.documents] == ["a", "b"]
    assert all(row.source == source.url for row in evidence.documents)


def test_source_adapter_rejects_changed_public_bytes(tmp_path):
    payload = _source_bytes()
    snapshot = tmp_path / "english.conllu"
    snapshot.write_bytes(payload + b"changed")
    source = replace(
        ENGLISH, size_bytes=len(payload), sha256=sha256(payload).hexdigest()
    )

    with pytest.raises(TokenizerError, match="snapshot identity"):
        load_pipeline_source(snapshot, source)
