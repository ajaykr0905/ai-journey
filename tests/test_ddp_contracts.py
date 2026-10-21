import pytest

from ai_journey.ddp_contracts import Topology


def test_torchrun_topology_identifies_only_rank_zero_as_writer():
    topology = Topology.from_environment(
        {"RANK": "1", "WORLD_SIZE": "2", "LOCAL_RANK": "1"}
    )
    assert topology == Topology(1, 2, 1)
    assert not topology.is_writer
    assert Topology(0, 2).is_writer


@pytest.mark.parametrize(
    "environment",
    [
        {},
        {"RANK": "-1", "WORLD_SIZE": "2", "LOCAL_RANK": "0"},
        {"RANK": "2", "WORLD_SIZE": "2", "LOCAL_RANK": "0"},
        {"RANK": "0", "WORLD_SIZE": "0", "LOCAL_RANK": "0"},
    ],
)
def test_invalid_torchrun_environment_fails_before_collectives(environment):
    with pytest.raises(ValueError):
        Topology.from_environment(environment)


def test_boolean_rank_is_not_an_integer_rank():
    with pytest.raises(ValueError):
        Topology(True, 2)
