from ai_journey.ddp_contracts import Topology
from ai_journey.ddp_data import shard_indices


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
