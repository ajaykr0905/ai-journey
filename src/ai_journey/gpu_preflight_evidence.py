"""Integrity and provenance checks for hardware observations from one worker."""

from uuid import UUID

from .gpu_plan import admit_gpu_resources, gpu_record_digest
from .gpu_preflight import GpuTopology, validate_rank_environment
from .gpu_preflight_probe import validate_gpu_probe
from .resource_report import verify_resource_report


def validate_run_id(run_id: str) -> str:
    if not isinstance(run_id, str) or str(UUID(run_id)) != run_id:
        raise ValueError("preflight run_id must be a canonical UUID")
    return run_id


def verify_gpu_rank_evidence(payload: dict) -> dict:
    fields = {
        "schema_version",
        "kind",
        "run_id",
        "process_id",
        "rank",
        "topology",
        "resource_report",
        "probe",
        "admission",
        "training_executed",
        "gpu_kernel_executed",
        "collective_executed",
        "evidence_sha256",
    }
    if (
        not isinstance(payload, dict)
        or set(payload) != fields
        or type(payload["schema_version"]) is not int
        or payload["schema_version"] != 1
        or payload["kind"] != "nvidia_rank_preflight"
    ):
        raise ValueError("rank preflight evidence schema does not match")
    unsigned = {
        key: value for key, value in payload.items() if key != "evidence_sha256"
    }
    if gpu_record_digest(unsigned) != payload["evidence_sha256"]:
        raise ValueError("rank preflight evidence digest does not match")
    validate_run_id(payload["run_id"])
    if type(payload["process_id"]) is not int or payload["process_id"] <= 0:
        raise ValueError("rank preflight requires a real process identity")
    if any(
        payload[name] is not False
        for name in ("training_executed", "gpu_kernel_executed", "collective_executed")
    ):
        raise ValueError("preflight observations cannot claim GPU execution")
    topology = GpuTopology.from_mapping(payload["topology"])
    rank = payload["rank"]
    if (
        not isinstance(rank, dict)
        or set(rank) != {"rank", "local_rank", "world_size", "device_ordinal"}
        or any(type(value) is not int for value in rank.values())
    ):
        raise ValueError("rank evidence fields do not match")
    expected = validate_rank_environment(
        {
            "RANK": str(rank["rank"]),
            "LOCAL_RANK": str(rank["local_rank"]),
            "WORLD_SIZE": str(rank["world_size"]),
            "LOCAL_WORLD_SIZE": str(rank["world_size"]),
        },
        topology,
    )
    if rank != expected:
        raise ValueError("rank device mapping does not match topology")
    validate_gpu_probe(payload["probe"])
    expected_admission = admit_gpu_resources(
        payload["resource_report"], topology, payload["probe"]
    )
    if gpu_record_digest(payload["admission"]) != gpu_record_digest(expected_admission):
        raise ValueError("rank admission does not match measured provenance")
    return payload


def build_gpu_run_evidence(
    records: list[dict], *, run_id: str, topology: GpuTopology, resource_report: dict
) -> dict:
    """Require the complete requested rank set from one consistent preflight run."""
    validate_run_id(run_id)
    verify_resource_report(resource_report)
    if not isinstance(records, list) or len(records) != topology.world_size:
        raise ValueError("preflight evidence requires every requested rank")
    for record in records:
        verify_gpu_rank_evidence(record)
        if (
            record["run_id"] != run_id
            or record["topology"] != topology.to_mapping()
            or record["resource_report"]["integrity"] != resource_report["integrity"]
        ):
            raise ValueError(
                "rank evidence belongs to a different run or resource contract"
            )
    if {record["rank"]["rank"] for record in records} != set(
        range(topology.world_size)
    ) or len({record["process_id"] for record in records}) != topology.world_size:
        raise ValueError("rank or process identities are incomplete or duplicated")
    runtime_fields = (
        "system",
        "machine",
        "torch",
        "cuda_runtime",
        "nccl_available",
        "visible_devices_sha256",
    )
    reference = records[0]["probe"]
    for record in records[1:]:
        probe = record["probe"]
        if any(probe[name] != reference[name] for name in runtime_fields):
            raise ValueError("rank runtime or visible-device provenance disagrees")
        if probe["available"] and reference["available"]:

            def inventory(observed):
                return [
                    {key: value for key, value in device.items() if key != "free_bytes"}
                    for device in observed["devices"]
                ]

            if inventory(probe) != inventory(reference):
                raise ValueError("rank device inventories disagree")
    payload = {
        "schema_version": 1,
        "kind": "nvidia_preflight_run",
        "run_id": run_id,
        "topology": topology.to_mapping(),
        "resource_report": resource_report,
        "ranks": sorted(records, key=lambda record: record["rank"]["rank"]),
        "preflight_passed": all(
            record["admission"]["preflight_ready"] for record in records
        ),
        "training_executed": False,
        "gpu_kernel_executed": False,
        "collective_executed": False,
    }
    payload["evidence_sha256"] = gpu_record_digest(payload)
    return payload


def verify_gpu_run_evidence(payload: dict) -> dict:
    fields = {
        "schema_version",
        "kind",
        "run_id",
        "topology",
        "resource_report",
        "ranks",
        "preflight_passed",
        "training_executed",
        "gpu_kernel_executed",
        "collective_executed",
        "evidence_sha256",
    }
    if (
        not isinstance(payload, dict)
        or set(payload) != fields
        or type(payload["schema_version"]) is not int
        or payload["schema_version"] != 1
        or payload["kind"] != "nvidia_preflight_run"
    ):
        raise ValueError("run preflight evidence schema does not match")
    expected = build_gpu_run_evidence(
        payload["ranks"],
        run_id=payload["run_id"],
        topology=GpuTopology.from_mapping(payload["topology"]),
        resource_report=payload["resource_report"],
    )
    if gpu_record_digest(payload) != gpu_record_digest(expected):
        raise ValueError("run preflight evidence does not match its rank observations")
    return payload
