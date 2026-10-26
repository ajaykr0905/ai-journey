import pytest
import torch

from ai_journey.decoder_variant_data import (
    document_training_batch,
    document_training_schedule,
    prepare_decoder_documents,
)


class UnreadableRows:
    def __iter__(self):
        raise AssertionError("unsupported schedule must not be consumed")


@pytest.fixture
def documents():
    return prepare_decoder_documents(
        b"aaaaaaaa\nbbbbbbbb", b"heldout", context_length=4
    )


@pytest.mark.parametrize(
    "rows",
    [
        (),
        ((-1, 0),),
        ((2, 0),),
        ((True, 0),),
        ((0.0, 0),),
        ((0, -1),),
        ((0, True),),
        ((0, 0.0),),
        ((0, 5),),
        ((0,),),
        ((0, 0, 0),),
        (None,),
        ((0, 0),) * 33,
        "00",
        UnreadableRows(),
    ],
)
def test_invalid_schedule_is_rejected_before_tensor_allocation(
    documents, rows, monkeypatch
):
    def forbidden(*args, **kwargs):
        raise AssertionError("invalid schedule allocated a tensor")

    monkeypatch.setattr(torch, "tensor", forbidden)
    with pytest.raises(ValueError, match="document training"):
        document_training_batch(documents, rows)


def test_valid_last_window_and_original_matched_schedule_still_work(documents):
    inputs, targets = document_training_batch(documents, ((0, 4), (1, 4)))
    assert inputs.tolist() == [[97] * 4, [98] * 4]
    assert targets.tolist() == [[97, 97, 97, 10], [98, 98, 98, 10]]
    schedule = document_training_schedule(documents, seed=53, steps=2, batch_size=32)
    for rows in schedule:
        inputs, targets = document_training_batch(documents, rows)
        assert inputs.shape == targets.shape == (32, 4)
        assert torch.equal(inputs[:, 1:], targets[:, :-1])


def test_valid_list_rows_remain_supported(documents):
    inputs, targets = document_training_batch(documents, [[0, 0]])
    assert inputs.shape == targets.shape == (1, 4)
