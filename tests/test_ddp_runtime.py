import pytest
import torch.distributed as dist

from ai_journey.ddp_contracts import Topology
from ai_journey.ddp_runtime import process_group
from ai_journey.ddp_runtime import reduce_token_totals


def test_owned_group_is_destroyed_when_work_raises(tmp_path):
    with pytest.raises(RuntimeError, match="work failed"):
        with process_group(Topology(0, 1), rendezvous=(tmp_path / "rdzv").as_uri()):
            assert dist.get_world_size() == 1
            raise RuntimeError("work failed")
    assert not dist.is_initialized()


def test_nested_group_cannot_destroy_its_caller(tmp_path):
    with process_group(Topology(0, 1), rendezvous=(tmp_path / "rdzv").as_uri()):
        with pytest.raises(RuntimeError, match="caller-owned"):
            with process_group(
                Topology(0, 1), rendezvous=(tmp_path / "other").as_uri()
            ):
                pass
        assert dist.is_initialized()


def test_collective_reduction_preserves_token_accounting(tmp_path):
    with process_group(Topology(0, 1), rendezvous=(tmp_path / "rdzv").as_uri()):
        assert reduce_token_totals(12.0, 3) == (12.0, 3, 4.0)
