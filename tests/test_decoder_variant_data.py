import pytest
import torch

from ai_journey.decoder_variant_data import (
    DecoderDocuments,
    document_schedule_fingerprint,
    document_training_batch,
    document_training_schedule,
    prepare_decoder_documents,
)
from ai_journey.ddp_evaluation import Document


@pytest.mark.parametrize("context_length", [True, 0, -1, 65, 1.5, "4"])
def test_direct_documents_reject_invalid_context(context_length):
    with pytest.raises(ValueError, match="context_length"):
        DecoderDocuments(
            train=(Document("train", b"a" * 67),),
            heldout=(Document("heldout", b"b\n"),),
            context_length=context_length,
        )


@pytest.mark.parametrize("partition", ["train", "heldout"])
@pytest.mark.parametrize(
    "invalid",
    [(), [], [Document("row", b"valid\n")], iter((Document("row", b"row\n"),))],
)
def test_direct_documents_require_nonempty_immutable_partitions(partition, invalid):
    values = {
        "train": (Document("train", b"training\n"),),
        "heldout": (Document("heldout", b"heldout\n"),),
    }
    values[partition] = invalid
    with pytest.raises(ValueError, match="nonempty tuples"):
        DecoderDocuments(**values, context_length=4)


@pytest.mark.parametrize("partition", ["train", "heldout"])
@pytest.mark.parametrize("invalid", [None, b"raw bytes\n", "raw text"])
def test_direct_documents_require_document_entries(partition, invalid):
    values = {
        "train": (Document("train", b"training\n"),),
        "heldout": (Document("heldout", b"heldout\n"),),
    }
    values[partition] = (invalid,)
    with pytest.raises(ValueError, match="Document"):
        DecoderDocuments(**values, context_length=4)


def test_direct_documents_bound_the_combined_document_count():
    train = tuple(Document("row", f"train-{index}\n".encode()) for index in range(1024))
    heldout = tuple(
        Document("row", f"heldout-{index}\n".encode()) for index in range(1025)
    )
    with pytest.raises(ValueError, match="2048"):
        DecoderDocuments(train, heldout, context_length=4)
    admitted = DecoderDocuments(train, heldout[:-1], context_length=4)
    assert len(admitted.train) + len(admitted.heldout) == 2048


@pytest.mark.parametrize("partition", ["train", "heldout"])
@pytest.mark.parametrize("lengths", [(1024 * 1024 + 2,), (512 * 1024 + 1,) * 2])
def test_direct_documents_bound_total_bytes_in_each_partition(partition, lengths):
    values = {
        "train": (Document("train", b"training\n"),),
        "heldout": (Document("heldout", b"heldout\n"),),
    }
    values[partition] = tuple(
        Document("row", bytes([97 + index]) * length)
        for index, length in enumerate(lengths)
    )
    with pytest.raises(ValueError, match="one MiB"):
        DecoderDocuments(**values, context_length=4)


@pytest.mark.parametrize("partition", ["train", "heldout"])
def test_direct_documents_reject_repeated_content_in_either_partition(partition):
    values = {
        "train": (Document("train", b"training\n"),),
        "heldout": (Document("heldout", b"heldout\n"),),
    }
    values[partition] = (
        Document("first", b"repeated\n"),
        Document("alias", b"repeated\n"),
    )
    with pytest.raises(ValueError, match="content-distinct"):
        DecoderDocuments(**values, context_length=4)


def test_direct_documents_reject_cross_partition_content_leakage():
    with pytest.raises(ValueError, match="content-disjoint"):
        DecoderDocuments(
            train=(Document("training-id", b"shared document\n"),),
            heldout=(Document("heldout-id", b"shared document\n"),),
            context_length=4,
        )


def test_direct_documents_require_at_least_one_complete_training_window():
    with pytest.raises(ValueError, match="complete context"):
        DecoderDocuments(
            train=(Document("train", b"x\n"),),
            heldout=(Document("heldout", b"heldout\n"),),
            context_length=4,
        )


@pytest.mark.parametrize(
    "partition,key,expected",
    [
        ("train", "train_window_count", 1048573),
        ("heldout", "heldout_target_tokens", 1048576),
    ],
)
def test_factory_preserves_one_mib_source_without_final_newline(
    partition, key, expected
):
    sources = {"train_bytes": b"training\n", "heldout_bytes": b"heldout\n"}
    sources[f"{partition}_bytes"] = b"a" * (1024 * 1024)
    documents = prepare_decoder_documents(**sources, context_length=4)
    assert len(getattr(documents, partition)[0].content) == 1048577
    assert documents.evidence()[key] == expected
    rows = document_training_schedule(documents, seed=0, steps=1, batch_size=1)[0]
    inputs, targets = document_training_batch(documents, rows)
    assert inputs.shape == targets.shape == (1, 4)


@pytest.mark.parametrize("partition", ["train", "heldout"])
def test_factory_keeps_raw_source_limit_before_normalization(partition):
    sources = {"train_bytes": b"training\n", "heldout_bytes": b"heldout\n"}
    sources[f"{partition}_bytes"] = b"a" * (1024 * 1024 + 1)
    with pytest.raises(ValueError, match="document source exceeds one MiB"):
        prepare_decoder_documents(**sources, context_length=4)


def test_direct_documents_keep_short_rows_and_repeated_identifiers_at_context_limit():
    documents = DecoderDocuments(
        train=(Document("same-id", b"x\n"), Document("same-id", b"a" * 65)),
        heldout=(Document("same-id", b"b\n"),),
        context_length=64,
    )
    assert documents.evidence()["train_window_count"] == 1
    schedule = document_training_schedule(documents, seed=0, steps=2, batch_size=3)
    assert schedule == (((1, 0), (1, 0), (1, 0)),) * 2
    inputs, targets = document_training_batch(documents, schedule[0])
    assert inputs.shape == targets.shape == (3, 64)
    assert inputs.tolist() == targets.tolist() == [[97] * 64] * 3


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
