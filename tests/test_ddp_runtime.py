import pytest
import torch.distributed as dist

from ai_journey.ddp_contracts import Topology
from ai_journey.ddp_runtime import process_group
from ai_journey.ddp_runtime import reduce_token_totals
from ai_journey.ddp_runtime import require_all_ranks
from ai_journey.ddp_runtime import require_model_consensus
from ai_journey.gpt2_model import GPT2LanguageModel
from ai_journey.gpt2_config import GPT2Config


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
        assert reduce_token_totals(1.0, 2**53 + 1)[1] == 2**53 + 1


def test_quorum_failure_is_raised_before_the_next_phase(tmp_path):
    with process_group(Topology(0, 1), rendezvous=(tmp_path / "rdzv").as_uri()):
        require_all_ranks(True, phase="input validation")
        with pytest.raises(RuntimeError, match="input validation"):
            require_all_ranks(False, phase="input validation")
        require_all_ranks(True, phase="recovered collective")


def test_consensus_returns_the_measured_model_fingerprint(tmp_path):
    model = GPT2LanguageModel(GPT2Config(8, 4, 8, 2, 1))
    with process_group(Topology(0, 1), rendezvous=(tmp_path / "rdzv").as_uri()):
        fingerprint = require_model_consensus(model)
        assert len(fingerprint) == 64
        assert require_model_consensus(model) == fingerprint
