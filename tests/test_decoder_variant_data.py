import pytest
import torch

from ai_journey.decoder_variant_data import (
    document_schedule_fingerprint,
    document_training_batch,
    document_training_schedule,
    prepare_decoder_documents,
)


def test_partitions_reject_overlapping_and_repeated_documents():
    with pytest.raises(ValueError, match="content-disjoint"):
        prepare_decoder_documents(
            b"first long document\nsecond document",
            b"first long document\n",
            context_length=4,
        )
    with pytest.raises(ValueError, match="distinct"):
        prepare_decoder_documents(b"repeated\nrepeated", b"heldout", context_length=4)


def test_matched_schedule_never_crosses_document_boundaries():
    documents = prepare_decoder_documents(
        b"aaaaaaaa\nbbbbbbbb", b"cccc\ndddd", context_length=4
    )
    first = document_training_schedule(documents, seed=7, steps=16, batch_size=8)
    second = document_training_schedule(documents, seed=7, steps=16, batch_size=8)
    assert first == second
    assert document_schedule_fingerprint(first) == document_schedule_fingerprint(second)
    seen = set()
    for rows in first:
        inputs, targets = document_training_batch(documents, rows)
        assert inputs.shape == targets.shape == (8, 4)
        assert torch.equal(inputs[:, 1:], targets[:, :-1])
        for inputs_row, targets_row in zip(inputs.tolist(), targets.tolist()):
            content = set(inputs_row + targets_row) - {10}
            assert content in ({ord("a")}, {ord("b")})
            seen |= content
    assert seen == {ord("a"), ord("b")}
    evidence = documents.evidence()
    assert evidence["heldout_target_tokens"] == 8
    assert evidence["train_window_count"] == 10
    assert all(len(digest) == 64 for digest in evidence["train_documents"])


def test_short_training_documents_are_not_sampled():
    documents = prepare_decoder_documents(
        b"x\nlonger document", b"heldout", context_length=4
    )
    schedule = document_training_schedule(documents, seed=0, steps=2, batch_size=3)
    assert all(index == 1 for batch in schedule for index, start in batch)
    with pytest.raises(ValueError, match="complete context"):
        prepare_decoder_documents(b"x", b"heldout", context_length=4)
