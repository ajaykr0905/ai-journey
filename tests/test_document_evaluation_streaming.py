import pytest
import torch

from ai_journey.ddp_evaluation import Document, evaluation_batches


def test_first_batch_does_not_consume_later_documents():
    def documents():
        yield Document("first", b"abcde")
        raise AssertionError("read beyond the requested batch")

    inputs, targets, mask = next(
        evaluation_batches(documents(), context_length=4, batch_size=1)
    )
    assert inputs.tolist() == [[97, 98, 99, 100]]
    assert targets.tolist() == [[98, 99, 100, 101]]
    assert mask.all()


def test_first_batch_does_not_slice_every_window_in_a_long_document():
    class GuardedBytes(bytes):
        slices = 0

        def __getitem__(self, key):
            if isinstance(key, slice):
                self.slices += 1
                if self.slices > 1:
                    raise AssertionError("materialized future windows")
            return super().__getitem__(key)

    content = GuardedBytes(b"abcdefghij" * 100)
    batch = next(
        evaluation_batches((Document("long", content),), context_length=4, batch_size=1)
    )
    assert batch[0].shape == (1, 4)
    assert content.slices == 1


@pytest.mark.parametrize("context_length", [1, 4, 128])
@pytest.mark.parametrize("batch_size", [1, 3, 8])
def test_streaming_preserves_order_padding_and_each_target_once(
    context_length, batch_size
):
    documents = (Document("first", b"abcdefghij"), Document("second", b"XYZ"))
    observed = []
    for inputs, targets, mask in evaluation_batches(
        iter(documents), context_length=context_length, batch_size=batch_size
    ):
        assert inputs.shape == targets.shape == mask.shape
        assert len(inputs) <= batch_size
        assert inputs.dtype == targets.dtype == torch.long
        assert mask.dtype == torch.bool
        for row in range(len(inputs)):
            length = int(mask[row].sum())
            assert mask[row, :length].all()
            assert not mask[row, length:].any()
            observed.extend(targets[row, mask[row]].tolist())
    assert observed == list(b"bcdefghijYZ")
