import pytest
import torch
from ai_journey.ddp_cache import KVCache


def test_cache_accounts_storage_and_clones_without_aliases():
    tensor = torch.ones(1, 2, 3, 4, dtype=torch.float64)
    cache = KVCache("a" * 64, (tensor,), (tensor.clone(),))
    assert cache.length == 3
    assert cache.storage_bytes == 384
    cloned = cache.clone()
    cloned.keys[0].zero_()
    assert cache.keys[0].sum() == 24


def test_cache_rejects_shape_mismatch_and_nonfinite_storage():
    with pytest.raises(ValueError):
        KVCache("a" * 64, (torch.ones(1, 2, 3, 4),), (torch.ones(1, 2, 2, 4),))
    with pytest.raises(ValueError):
        KVCache(
            "a" * 64,
            (torch.full((1, 2, 3, 4), float("nan")),),
            (torch.ones(1, 2, 3, 4),),
        )
