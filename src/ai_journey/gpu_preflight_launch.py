"""Prepare argv-only torchrun instructions for the implemented preflight worker."""

from pathlib import Path
import sys

from .gpu_plan import admit_gpu_resources
from .gpu_preflight import GpuTopology
from .gpu_preflight_evidence import validate_run_id
from .gpu_preflight_io import read_gpu_json


def build_gpu_launch_plan(
    *,
    resource_report_path: Path,
    topology: GpuTopology,
    output_dir: Path,
    run_id: str,
    probe: dict | None = None,
) -> dict:
    """Return a reviewable command only; never execute hardware or training work."""
    validate_run_id(run_id)
    resource = read_gpu_json(resource_report_path)
    admission = admit_gpu_resources(resource, topology, probe)
    resource_path, rank_dir = resource_report_path.resolve(), output_dir.resolve()
    rank_paths = [
        rank_dir / run_id / f"rank-{rank}.json" for rank in range(topology.world_size)
    ]
    if resource_path in rank_paths:
        raise ValueError("rank evidence must not overwrite its resource input")
    argv = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--nproc-per-node={topology.world_size}",
        "--max-restarts=0",
        "--module",
        "ai_journey.gpu_preflight_worker",
        "--devices",
        *(str(device) for device in topology.devices),
        "--dtype",
        topology.dtype,
        "--resource-report",
        str(resource_path),
        "--output-dir",
        str(rank_dir),
        "--run-id",
        run_id,
    ]
    return {
        "schema_version": 1,
        "kind": "nvidia_preflight_launch_plan",
        "run_id": run_id,
        "topology": topology.to_mapping(),
        "resource_report_sha256": resource["integrity"]["digest"],
        "admission": admission,
        "argv": argv,
        "rank_record_paths": [str(path) for path in rank_paths],
        "command_executed": False,
        "training_executed": False,
        "scope": "single_host_hardware_preflight_only",
    }
