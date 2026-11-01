"""Activation handoffs admit only canonical adjacent-stage transfers."""

import json
from hashlib import sha256

import pytest

from ai_journey.activation_handoff import (
    MAX_ACTIVATION_PAYLOAD_BYTES,
    ActivationHandoffContract,
    activation_handoff_digest,
)
from ai_journey.hybrid_topology import HybridParallelTopology
from ai_journey.pipeline_partition import partition_transformer_layers


def handoff_fixture():
    topology = HybridParallelTopology(4, 1, 1, 2, 2)
    partition = partition_transformer_layers(topology, layer_count=6)
    payload = bytes(range(48))
    return topology, partition, payload


def test_adjacent_stage_handoff_round_trips_canonical_bytes():
    topology, partition, payload = handoff_fixture()

    contract = ActivationHandoffContract.admit(
        topology=topology,
        partition=partition,
        source_rank=0,
        destination_rank=2,
        tensor_shape=(2, 3, 2),
        dtype="float32",
        payload=payload,
    )

    assert contract.payload_length == 48
    assert contract.payload_sha256 == sha256(payload).hexdigest()
    expected_handoff_sha256 = activation_handoff_digest(
        source_rank=0,
        destination_rank=2,
        tensor_shape=(2, 3, 2),
        dtype="float32",
        payload=payload,
    )
    assert contract.handoff_sha256 == expected_handoff_sha256
    assert contract.to_mapping() == {
        "schema": "ai-journey.activation-handoff.v1",
        "source_rank": 0,
        "destination_rank": 2,
        "tensor_shape": [2, 3, 2],
        "dtype": "float32",
        "payload_length": 48,
        "payload_sha256": sha256(payload).hexdigest(),
        "handoff_sha256": expected_handoff_sha256,
    }
    raw = contract.to_json_bytes()
    assert raw == json.dumps(
        contract.to_mapping(), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    assert (
        ActivationHandoffContract.from_json_bytes(
            raw,
            topology=topology,
            partition=partition,
            payload=payload,
        )
        == contract
    )


@pytest.mark.parametrize(
    ("source_rank", "destination_rank", "match"),
    [
        (-1, 2, "source_rank is outside"),
        (0, 4, "destination_rank is outside"),
        (0, 1, "different pipeline stages"),
        (0, 3, "non-pipeline coordinates"),
        (2, 0, "next adjacent pipeline stage"),
    ],
)
def test_handoff_rejects_invalid_rank_and_stage_routes(
    source_rank, destination_rank, match
):
    topology, partition, payload = handoff_fixture()

    with pytest.raises(ValueError, match=match):
        ActivationHandoffContract.admit(
            topology=topology,
            partition=partition,
            source_rank=source_rank,
            destination_rank=destination_rank,
            tensor_shape=(2, 3, 2),
            dtype="float32",
            payload=payload,
        )


def test_handoff_rejects_partition_that_does_not_own_canonical_stage_boundary():
    topology, _, payload = handoff_fixture()
    incompatible = partition_transformer_layers(topology, layer_count=5)
    mutated_stage = type(incompatible.stages[0])(0, (0, 1, 3))
    noncanonical = type(incompatible)(
        layer_count=5,
        stages=(mutated_stage, incompatible.stages[1]),
    )

    with pytest.raises(ValueError, match="canonical stage ownership"):
        ActivationHandoffContract.admit(
            topology=topology,
            partition=noncanonical,
            source_rank=0,
            destination_rank=2,
            tensor_shape=(2, 3, 2),
            dtype="float32",
            payload=payload,
        )


@pytest.mark.parametrize(
    ("shape", "dtype", "payload", "match"),
    [
        ((), "float32", b"", "non-empty"),
        ((2, 0), "float32", b"", "positive integers"),
        ((2, True), "float32", b"", "positive integers"),
        ((2, 3), "int64", b"", "unsupported activation dtype"),
        ((2, 3), "float32", b"short", "exactly 24 bytes"),
    ],
)
def test_handoff_rejects_malformed_tensor_metadata(shape, dtype, payload, match):
    topology, partition, _ = handoff_fixture()

    with pytest.raises((TypeError, ValueError), match=match):
        ActivationHandoffContract.admit(
            topology=topology,
            partition=partition,
            source_rank=0,
            destination_rank=2,
            tensor_shape=shape,
            dtype=dtype,
            payload=payload,
        )


def test_handoff_rejects_payload_over_the_configured_and_system_bounds():
    topology, partition, _ = handoff_fixture()

    with pytest.raises(ValueError, match="maximum payload bound"):
        ActivationHandoffContract.admit(
            topology=topology,
            partition=partition,
            source_rank=0,
            destination_rank=2,
            tensor_shape=(7,),
            dtype="float32",
            payload=bytes(28),
            maximum_payload_bytes=24,
        )
    with pytest.raises(ValueError, match="cannot exceed system bound"):
        ActivationHandoffContract.admit(
            topology=topology,
            partition=partition,
            source_rank=0,
            destination_rank=2,
            tensor_shape=(1,),
            dtype="float32",
            payload=bytes(4),
            maximum_payload_bytes=MAX_ACTIVATION_PAYLOAD_BYTES + 1,
        )


def test_handoff_recomputes_payload_identity_during_mapping_admission():
    topology, partition, payload = handoff_fixture()
    contract = ActivationHandoffContract.admit(
        topology=topology,
        partition=partition,
        source_rank=0,
        destination_rank=2,
        tensor_shape=(2, 3, 2),
        dtype="float32",
        payload=payload,
    )
    mapping = contract.to_mapping()
    mapping["payload_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="payload digest differs"):
        ActivationHandoffContract.from_mapping(
            mapping,
            topology=topology,
            partition=partition,
            payload=payload,
        )

    metadata_mutation = contract.to_mapping()
    metadata_mutation["tensor_shape"] = [1, 12]
    with pytest.raises(ValueError, match="handoff digest differs"):
        ActivationHandoffContract.from_mapping(
            metadata_mutation,
            topology=topology,
            partition=partition,
            payload=payload,
        )


def test_handoff_rejects_noncanonical_and_duplicate_serialization():
    topology, partition, payload = handoff_fixture()
    contract = ActivationHandoffContract.admit(
        topology=topology,
        partition=partition,
        source_rank=0,
        destination_rank=2,
        tensor_shape=(2, 3, 2),
        dtype="float32",
        payload=payload,
    )
    mapping = contract.to_mapping()
    mapping["unexpected"] = 1
    with pytest.raises(ValueError, match="canonical schema"):
        ActivationHandoffContract.from_mapping(
            mapping,
            topology=topology,
            partition=partition,
            payload=payload,
        )

    duplicate = contract.to_json_bytes()[:-1] + b',"source_rank":0}'
    with pytest.raises(ValueError, match="duplicate field: source_rank"):
        ActivationHandoffContract.from_json_bytes(
            duplicate,
            topology=topology,
            partition=partition,
            payload=payload,
        )

    noncanonical = json.dumps(contract.to_mapping(), indent=2).encode("utf-8")
    with pytest.raises(ValueError, match="not canonical JSON"):
        ActivationHandoffContract.from_json_bytes(
            noncanonical,
            topology=topology,
            partition=partition,
            payload=payload,
        )
