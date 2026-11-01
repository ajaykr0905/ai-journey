"""A bounded CPU/Gloo probe verifies one admitted pipeline handoff."""

from __future__ import annotations

import subprocess

import pytest

from ai_journey.activation_handoff import ActivationHandoffContract
from ai_journey.hybrid_topology import HybridParallelTopology
from ai_journey.pipeline_handoff_probe import (
    PipelineHandoffTimeout,
    PipelineHandoffWorkerFailure,
    run_pipeline_handoff_probe,
)
from ai_journey.pipeline_partition import partition_transformer_layers


def admitted_handoff():
    topology = HybridParallelTopology(2, 1, 1, 2, 1)
    partition = partition_transformer_layers(topology, layer_count=4)
    payload = bytes(range(16))
    contract = ActivationHandoffContract.admit(
        topology=topology,
        partition=partition,
        source_rank=0,
        destination_rank=1,
        tensor_shape=(4,),
        dtype="float32",
        payload=payload,
    )
    return topology, partition, contract, payload


def test_two_rank_gloo_probe_verifies_payload_metadata_and_one_outcome():
    topology, partition, contract, payload = admitted_handoff()

    result = run_pipeline_handoff_probe(
        topology=topology,
        partition=partition,
        contract=contract,
        payload=payload,
    )

    assert result.to_mapping() == {
        "schema": "ai-journey.pipeline-handoff-probe.v1",
        "executed": True,
        "performance_measured": False,
        "backend": "gloo",
        "device": "cpu",
        "world_size": 2,
        "outcome": "success",
        "contract": contract.to_mapping(),
        "rank_outcomes": [
            {"probe_rank": 0, "role": "source", "outcome": "success"},
            {"probe_rank": 1, "role": "destination", "outcome": "success"},
        ],
        "limitations": [
            "CPU/Gloo transfer correctness only; no NVIDIA GPU or NCCL was used.",
            "No latency, bandwidth, throughput, utilization, memory, or cost was measured.",
        ],
    }


def test_malformed_caller_payload_rejects_before_spawning(monkeypatch):
    topology, partition, contract, _ = admitted_handoff()

    def unexpected_spawn(*args, **kwargs):
        raise AssertionError("malformed input must reject before spawning")

    monkeypatch.setattr(subprocess, "Popen", unexpected_spawn)
    with pytest.raises(ValueError, match="payload digest differs"):
        run_pipeline_handoff_probe(
            topology=topology,
            partition=partition,
            contract=contract,
            payload=bytes(reversed(range(16))),
        )


@pytest.mark.parametrize("failure_mode", ["malformed_payload", "worker_exit"])
def test_failed_handoff_reaps_every_owned_worker(monkeypatch, failure_mode):
    topology, partition, contract, payload = admitted_handoff()
    owned = []
    spawn = subprocess.Popen

    def tracked_spawn(*args, **kwargs):
        process = spawn(*args, **kwargs)
        owned.append(process)
        return process

    monkeypatch.setattr(subprocess, "Popen", tracked_spawn)
    with pytest.raises(PipelineHandoffWorkerFailure) as failed:
        run_pipeline_handoff_probe(
            topology=topology,
            partition=partition,
            contract=contract,
            payload=payload,
            failure_mode=failure_mode,
        )

    assert len(owned) == 2
    assert all(process.poll() is not None for process in owned)
    assert failed.value.supervision["outcome"] == "failure"
    assert failed.value.supervision["all_workers_reaped"] is True
    if failure_mode == "malformed_payload":
        assert failed.value.supervision["rank_outcomes"] == [
            {"probe_rank": 0, "role": "source", "outcome": "failure"},
            {"probe_rank": 1, "role": "destination", "outcome": "failure"},
        ]


@pytest.mark.parametrize(
    ("failure_mode", "phase"),
    [("startup_timeout", "startup"), ("completion_timeout", "completion")],
)
def test_probe_deadlines_reap_every_owned_worker(monkeypatch, failure_mode, phase):
    topology, partition, contract, payload = admitted_handoff()
    owned = []
    spawn = subprocess.Popen

    def tracked_spawn(*args, **kwargs):
        process = spawn(*args, **kwargs)
        owned.append(process)
        return process

    monkeypatch.setattr(subprocess, "Popen", tracked_spawn)
    with pytest.raises(PipelineHandoffTimeout, match=phase) as failed:
        run_pipeline_handoff_probe(
            topology=topology,
            partition=partition,
            contract=contract,
            payload=payload,
            startup_timeout_seconds=3,
            completion_timeout_seconds=3,
            failure_mode=failure_mode,
        )

    assert len(owned) == 2
    assert all(process.poll() is not None for process in owned)
    assert failed.value.supervision["phase"] == phase
    assert failed.value.supervision["all_workers_reaped"] is True
