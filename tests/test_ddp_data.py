from ai_journey.ddp_contracts import Topology
from ai_journey.ddp_data import shard_indices
from ai_journey.ddp_data import byte_batches, BatchPlan
import pytest


def test_shards_are_disjoint_balanced_and_repeatable():
    shards = [shard_indices(13, Topology(rank, 3), seed=48) for rank in range(3)]
    assert [len(shard) for shard in shards] == [4, 4, 4]
    assert len(set().union(*map(set, shards))) == 12
    assert shards[1] == shard_indices(13, Topology(1, 3), seed=48)
    assert shards[1] != shard_indices(13, Topology(1, 3), seed=48, epoch=1)


def test_shuffling_does_not_mutate_global_random_state():
    import random

    before = random.getstate()
    shard_indices(10, Topology(0, 2), seed=48)
    assert random.getstate() == before


def test_byte_batches_account_for_the_exact_global_tail():
    batches, plan = byte_batches(
        bytes(range(100)),
        sequence_length=4,
        topology=Topology(0, 2),
        micro_batch_size=2,
        accumulation_steps=2,
        seed=48,
    )
    assert len(batches) == 6
    assert plan.updates_per_epoch == 3
    assert plan.discarded_samples == 0
    for inputs, targets in batches:
        assert inputs.shape == targets.shape == (2, 4)
        assert (targets == inputs + 1).all()
    assert BatchPlan(27, 2, 2, 2).discarded_samples == 3


def test_too_small_corpus_is_rejected_instead_of_hanging_a_rank():
    with pytest.raises(ValueError, match="complete global update"):
        byte_batches(
            b"short",
            sequence_length=4,
            topology=Topology(0, 2),
            micro_batch_size=2,
            accumulation_steps=2,
            seed=48,
        )
