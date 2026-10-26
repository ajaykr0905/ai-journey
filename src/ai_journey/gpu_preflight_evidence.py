"""Integrity and provenance checks for hardware observations from one worker."""

from uuid import UUID

from .gpu_plan import admit_gpu_resources, gpu_record_digest
from .gpu_preflight import GpuTopology, validate_rank_environment
from .gpu_preflight_probe import validate_gpu_probe


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
    if payload["admission"] != expected_admission:
        raise ValueError("rank admission does not match measured provenance")
    return payload
