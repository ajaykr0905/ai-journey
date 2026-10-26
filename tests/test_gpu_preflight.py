"""NVIDIA requests must identify unambiguous local devices and worker ranks."""

import pytest

from ai_journey.gpu_preflight import GpuTopology, validate_rank_environment


def test_explicit_rank_device_mapping_round_trips():
    topology = GpuTopology((3, 1))
    assert topology.device_for_rank(0) == 3
    assert topology.device_for_rank(1) == 1
    assert GpuTopology.from_mapping(topology.to_mapping()) == topology
    assert (
        validate_rank_environment(
            {
                "RANK": "1",
                "LOCAL_RANK": "1",
                "WORLD_SIZE": "2",
                "LOCAL_WORLD_SIZE": "2",
            },
            topology,
        )["device_ordinal"]
        == 1
    )


@pytest.mark.parametrize("devices", [(), (0, 0), (True,), (-1,), (64,), [0], ("0",)])
def test_ambiguous_or_invalid_devices_are_rejected(devices):
    with pytest.raises(ValueError, match="devices"):
        GpuTopology(devices)


@pytest.mark.parametrize("field,value", [("backend", "gloo"), ("dtype", "int8")])
def test_unsupported_backend_or_dtype_is_rejected(field, value):
    with pytest.raises(ValueError):
        GpuTopology((0,), **{field: value})


@pytest.mark.parametrize(
    "name,value",
    [
        ("RANK", "2"),
        ("LOCAL_RANK", "-1"),
        ("WORLD_SIZE", "4"),
        ("LOCAL_WORLD_SIZE", "1"),
        ("RANK", "True"),
        ("RANK", "١"),
    ],
)
def test_rank_mismatch_rejects_before_any_device_probe(name, value):
    environment = {
        "RANK": "0",
        "LOCAL_RANK": "0",
        "WORLD_SIZE": "2",
        "LOCAL_WORLD_SIZE": "2",
    }
    environment[name] = value
    with pytest.raises(ValueError):
        validate_rank_environment(environment, GpuTopology((0, 1)))


def test_missing_rank_environment_cannot_claim_torchrun_execution():
    with pytest.raises(ValueError, match="RANK"):
        validate_rank_environment({}, GpuTopology((0,)))
