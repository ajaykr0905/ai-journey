import pytest

from ai_journey.distributed_capacity import ClusterTopology, LinkProfile
from ai_journey.distributed_capacity_report import (
    DistributedCapacityPlan,
    DistributedCapacityReportError,
    build_distributed_capacity_report,
    seal_distributed_capacity_report,
    verify_distributed_capacity_report,
)
from ai_journey.resource_accounting import DecoderSpec


def _plan() -> DistributedCapacityPlan:
    return DistributedCapacityPlan(
        model=DecoderSpec(32, 2, 16, 4, 2, 32),
        topology=ClusterTopology(
            world_size=4,
            workers_per_node=2,
            intra_node=LinkProfile(1_000_000, 0.00001),
            inter_node=LinkProfile(100_000, 0.001),
        ),
        parameter_dtype="bfloat16",
        gradient_dtype="float32",
        optimizer_dtype="float32",
        optimizer_slots=2,
        master_parameter_dtype="float32",
        gradient_bucket_bytes=1024,
        gradient_sync_algorithm="hierarchical_ring",
        reshard_after_forward=True,
        checkpoint_sharded=True,
        checkpoint_rank_local_bytes=64,
        checkpoint_metadata_bytes=128,
        checkpoint_rank_bandwidth_bytes_per_second=100_000,
        checkpoint_shared_bandwidth_bytes_per_second=200_000,
        checkpoint_storage_budget_bytes=10_000_000,
    )


def test_distributed_capacity_report_cross_checks_model_state_and_checkpoint():
    plan = _plan()
    report = build_distributed_capacity_report(plan)
    parameter_count = plan.model.parameter_count

    assert report["measured"] is False
    assert report["kind"] == "analytical_distributed_capacity"
    assert sum(item["payload_bytes"] for item in report["parameter_shards"]) == (
        parameter_count * 2
    )
    assert sum(item["total_bytes"] for item in report["training_state"]) == (
        parameter_count * 18
    )
    assert report["gradient_sync"]["bucket_count"] > 1
    assert report["checkpoint"]["image"]["sharded"] is True


def test_distributed_capacity_report_is_deterministic_for_same_plan():
    assert build_distributed_capacity_report(
        _plan()
    ) == build_distributed_capacity_report(_plan())


def test_distributed_capacity_report_seal_detects_tampering():
    sealed = seal_distributed_capacity_report(
        build_distributed_capacity_report(_plan())
    )
    assert verify_distributed_capacity_report(sealed) is sealed

    tampered = {**sealed, "measured": True}
    with pytest.raises(DistributedCapacityReportError, match="unmeasured"):
        verify_distributed_capacity_report(tampered)


def test_distributed_capacity_report_recomputes_derived_fields():
    sealed = seal_distributed_capacity_report(
        build_distributed_capacity_report(_plan())
    )
    drifted = {
        **sealed,
        "gradient_sync": {**sealed["gradient_sync"], "bucket_count": 999},
    }
    drifted = seal_distributed_capacity_report(drifted)
    with pytest.raises(DistributedCapacityReportError, match="derived fields"):
        verify_distributed_capacity_report(drifted)
