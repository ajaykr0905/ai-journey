"""Analytical NVIDIA memory admission bound to existing resource accounting."""

from hashlib import sha256
import json

from .gpu_preflight import GpuTopology
from .gpu_preflight_probe import validate_gpu_probe
from .resource_accounting import memory_capacity
from .resource_report import verify_resource_report


def gpu_record_digest(payload: dict) -> str:
    return sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def admit_gpu_resources(
    resource_report: dict, topology: GpuTopology, probe: dict | None = None
) -> dict:
    """Compare modeled per-device bytes with declared and observed free memory.

    Admission covers only the requested preflight. The resource report remains
    an analytical estimate, including any declared sharding assumptions, and no
    training or CUDA allocation is performed by this function.
    """
    verify_resource_report(resource_report)
    if not isinstance(topology, GpuTopology):
        raise TypeError("topology must be GpuTopology")
    workload, assumptions = resource_report["workload"], resource_report["assumptions"]
    if workload["data_parallel_size"] != topology.world_size:
        raise ValueError(
            "resource data parallel size does not match requested topology"
        )
    if assumptions["parameter_dtype"] != topology.dtype:
        raise ValueError("resource parameter dtype does not match requested topology")
    if probe is not None:
        validate_gpu_probe(probe)
    blockers = []
    if probe is None:
        blockers.append("host_not_probed")
    elif not probe["available"]:
        blockers.append(probe["reason"])
    required = resource_report["memory"]["per_device"]["total"]
    devices = []
    for ordinal in topology.devices:
        observed = None
        if probe is not None and probe["available"]:
            if ordinal >= len(probe["devices"]):
                blockers.append(f"device_{ordinal}_not_visible")
            else:
                observed = probe["devices"][ordinal]
                if (
                    topology.dtype == "bfloat16"
                    and observed["compute_capability"][0] < 8
                ):
                    blockers.append(f"device_{ordinal}_lacks_native_bfloat16")
        capacity = assumptions["device_capacity_bytes"]
        if observed is not None:
            capacity = min(capacity, observed["total_bytes"])
        budget = memory_capacity(
            required,
            device_capacity_bytes=capacity,
            reserve_fraction=assumptions["reserve_fraction"],
        )
        usable = (
            budget["usable"]
            if observed is None
            else max(0, min(capacity, observed["free_bytes"]) - budget["reserved"])
        )
        fits = required <= usable
        if not fits:
            blockers.append(f"device_{ordinal}_memory_budget_exceeded")
        devices.append(
            {
                "ordinal": ordinal,
                "capacity": budget,
                "observed_free_bytes": observed["free_bytes"] if observed else None,
                "usable_bytes_now": usable,
                "fits_current_observation": fits,
            }
        )
    return {
        "schema_version": 1,
        "kind": "nvidia_analytical_admission",
        "resource_report_sha256": resource_report["integrity"]["digest"],
        "probe_sha256": gpu_record_digest(probe) if probe is not None else None,
        "topology": topology.to_mapping(),
        "analytical_only": True,
        "training_executed": False,
        "global_batch_size": workload["micro_batch_size"]
        * workload["gradient_accumulation_steps"]
        * topology.world_size,
        "tokens_per_optimizer_step": workload["micro_batch_size"]
        * workload["sequence_length"]
        * workload["gradient_accumulation_steps"]
        * topology.world_size,
        "required_bytes_per_device": required,
        "devices": devices,
        "preflight_ready": not blockers,
        "blockers": blockers,
    }
