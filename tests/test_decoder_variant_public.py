from dataclasses import replace
from hashlib import sha256

import pytest

from ai_journey.decoder_variant_public import (
    public_decoder_partitions,
    run_public_decoder_comparison,
    verify_public_decoder_report,
)
from ai_journey.tokenizer_sources import ENGLISH
from test_decoder_variant_plan import plan


def snapshot(tmp_path):
    payload = b"\n".join(
        f"# sent_id = s{index}\n# text = authored public fixture sentence number {index}\n".encode()
        for index in range(12)
    )
    path = tmp_path / "source.conllu"
    path.write_bytes(payload)
    source = replace(
        ENGLISH, size_bytes=len(payload), sha256=sha256(payload).hexdigest()
    )
    return path, source


def test_public_partitions_verify_snapshot_and_preserve_disjoint_source_evidence(
    tmp_path,
):
    path, source = snapshot(tmp_path)
    first = public_decoder_partitions(
        path, source, train_documents=4, heldout_documents=2, split_seed=7
    )
    assert first == public_decoder_partitions(
        path, source, train_documents=4, heldout_documents=2, split_seed=7
    )
    train, heldout, evidence = first
    assert set(train.splitlines()).isdisjoint(heldout.splitlines())
    assert source.revision in evidence["source_url"]
    assert evidence["source"]["license"] == source.license
    assert set(evidence["train_document_sha256"]).isdisjoint(
        evidence["heldout_document_sha256"]
    )
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="identity"):
        public_decoder_partitions(path, source, train_documents=4, heldout_documents=2)


def test_public_comparison_replays_pins_partitions_and_measured_code(tmp_path):
    path, source = snapshot(tmp_path)
    report = run_public_decoder_comparison(
        plan(), path, source, train_documents=4, heldout_documents=2
    )
    verify_public_decoder_report(report, path, source)
    assert str(path) not in str(report)
    with pytest.raises(ValueError, match="pinned source"):
        verify_public_decoder_report(report, path, replace(source, revision="0" * 40))
