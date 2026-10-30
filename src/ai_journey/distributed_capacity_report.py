"""Replayable reports for analytical distributed training capacity plans."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

from .checkpoint_planning import (
    checkpoint_image,
    checkpoint_retention,
    checkpoint_write_estimate,
)
from .distributed_capacity import (
    ClusterTopology,
    LinkProfile,
    bucketize_gradients,
    gradient_sync_schedule,
)
from .fsdp_planning import (
    decoder_parameter_shards,
    forward_materialization,
    sharded_training_state,
)
from .resource_accounting import DecoderSpec, bytes_per_element


class DistributedCapacityReportError(ValueError):
    """Raised when distributed capacity evidence cannot be verified."""


@dataclass(frozen=True)
class DistributedCapacityPlan:
    """Declared assumptions for one analytical distributed capacity report."""

    model: DecoderSpec
    topology: ClusterTopology
    parameter_dtype: str
    gradient_dtype: str
    optimizer_dtype: str
    optimizer_slots: int
    master_parameter_dtype: str | None
    gradient_bucket_bytes: int
    gradient_sync_algorithm: str
    reshard_after_forward: bool
    checkpoint_sharded: bool
    checkpoint_rank_local_bytes: int
    checkpoint_metadata_bytes: int
    checkpoint_rank_bandwidth_bytes_per_second: float
    checkpoint_shared_bandwidth_bytes_per_second: float
    checkpoint_storage_budget_bytes: int


def build_distributed_capacity_report(
    plan: DistributedCapacityPlan,
) -> dict[str, object]:
    """Build a JSON-safe report without probing hardware or running collectives."""

    if not isinstance(plan, DistributedCapacityPlan):
        raise TypeError("plan must be a DistributedCapacityPlan")
    model = plan.model
    topology = plan.topology
    parameter_shards = decoder_parameter_shards(
        model, dtype=plan.parameter_dtype, world_size=topology.world_size
    )
    training_state = sharded_training_state(
        model,
        parameter_dtype=plan.parameter_dtype,
        gradient_dtype=plan.gradient_dtype,
        optimizer_dtype=plan.optimizer_dtype,
        optimizer_slots=plan.optimizer_slots,
        master_parameter_dtype=plan.master_parameter_dtype,
        world_size=topology.world_size,
    )
    parameter_groups = {
        name: count * bytes_per_element(plan.parameter_dtype)
        for name, count in model.parameter_groups().items()
    }
    materialization = forward_materialization(
        parameter_groups,
        parameter_shards,
        reshard_after_forward=plan.reshard_after_forward,
    )
    gradient_groups = {
        name: count * bytes_per_element(plan.gradient_dtype)
        for name, count in model.parameter_groups().items()
    }
    gradient_buckets = bucketize_gradients(
        gradient_groups, bucket_bytes=plan.gradient_bucket_bytes
    )
    sync = gradient_sync_schedule(
        gradient_buckets,
        topology,
        algorithm=plan.gradient_sync_algorithm,
    )
    image = checkpoint_image(
        model,
        parameter_dtype=plan.parameter_dtype,
        optimizer_dtype=plan.optimizer_dtype,
        optimizer_slots=plan.optimizer_slots,
        master_parameter_dtype=plan.master_parameter_dtype,
        world_size=topology.world_size,
        sharded=plan.checkpoint_sharded,
        rank_local_bytes=plan.checkpoint_rank_local_bytes,
        metadata_bytes=plan.checkpoint_metadata_bytes,
    )
    write = checkpoint_write_estimate(
        image,
        rank_bandwidth_bytes_per_second=(
            plan.checkpoint_rank_bandwidth_bytes_per_second
        ),
        shared_bandwidth_bytes_per_second=(
            plan.checkpoint_shared_bandwidth_bytes_per_second
        ),
    )
    retention = checkpoint_retention(
        image, storage_budget_bytes=plan.checkpoint_storage_budget_bytes
    )
    report = {
        "schema_version": 1,
        "kind": "analytical_distributed_capacity",
        "measured": False,
        "model": asdict(model),
        "topology": asdict(topology),
        "assumptions": {
            key: value
            for key, value in asdict(plan).items()
            if key not in {"model", "topology"}
        },
        "parameter_shards": [asdict(shard) for shard in parameter_shards],
        "training_state": [
            {**asdict(state), "total_bytes": state.total_bytes}
            for state in training_state
        ],
        "forward_materialization": [asdict(item) for item in materialization],
        "gradient_sync": {
            "algorithm": sync.algorithm,
            "bucket_count": len(sync.buckets),
            "wire_bytes_per_rank": sync.wire_bytes_per_rank,
            "seconds": sync.seconds,
            "buckets": [asdict(bucket) for bucket in sync.buckets],
        },
        "checkpoint": {
            "image": {**asdict(image), "total_bytes": image.total_bytes},
            "write": asdict(write),
            "retention": asdict(retention),
        },
        "limitations": [
            "Analytical estimate only; no collective, storage, or GPU was measured.",
            "Bandwidth and latency are declared assumptions, not host observations.",
            "Allocator fragmentation and kernel workspaces are excluded.",
        ],
    }
    return json.loads(json.dumps(report, allow_nan=False))


def _canonical_report_bytes(report: dict[str, object]) -> bytes:
    unsigned = {key: value for key, value in report.items() if key != "integrity"}
    return json.dumps(
        unsigned,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def seal_distributed_capacity_report(
    report: dict[str, object],
) -> dict[str, object]:
    """Attach a deterministic SHA-256 digest without mutating the report."""

    if not isinstance(report, dict):
        raise TypeError("report must be a dictionary")
    sealed = {key: value for key, value in report.items() if key != "integrity"}
    sealed["integrity"] = {
        "algorithm": "sha256",
        "digest": hashlib.sha256(_canonical_report_bytes(sealed)).hexdigest(),
    }
    return sealed


def verify_distributed_capacity_report(
    report: dict[str, object],
) -> dict[str, object]:
    """Rebuild every derived field and reject integrity or assumption drift."""

    if not isinstance(report, dict):
        raise TypeError("report must be a dictionary")
    if report.get("schema_version") != 1:
        raise DistributedCapacityReportError("unsupported report schema")
    if report.get("kind") != "analytical_distributed_capacity":
        raise DistributedCapacityReportError("unexpected report kind")
    if report.get("measured") is not False:
        raise DistributedCapacityReportError("capacity report must remain unmeasured")
    try:
        expected_integrity = seal_distributed_capacity_report(report)["integrity"]
    except (TypeError, ValueError, OverflowError) as error:
        raise DistributedCapacityReportError(
            "capacity report is not canonical"
        ) from error
    if report.get("integrity") != expected_integrity:
        raise DistributedCapacityReportError("capacity report integrity mismatch")
    try:
        model_payload = report["model"]
        topology_payload = report["topology"]
        assumptions = report["assumptions"]
        if not all(
            isinstance(item, dict)
            for item in (model_payload, topology_payload, assumptions)
        ):
            raise TypeError("plan sections must be objects")
        topology_payload = dict(topology_payload)
        topology = ClusterTopology(
            world_size=topology_payload["world_size"],
            workers_per_node=topology_payload["workers_per_node"],
            intra_node=LinkProfile(**topology_payload["intra_node"]),
            inter_node=LinkProfile(**topology_payload["inter_node"]),
        )
        plan = DistributedCapacityPlan(
            model=DecoderSpec(**model_payload),
            topology=topology,
            **assumptions,
        )
        expected = seal_distributed_capacity_report(
            build_distributed_capacity_report(plan)
        )
    except (KeyError, TypeError, ValueError) as error:
        raise DistributedCapacityReportError(
            "capacity report cannot be rebuilt"
        ) from error
    if report != expected:
        raise DistributedCapacityReportError("capacity report derived fields drifted")
    return report
