"""A torchrun worker that checks hardware and budgets without running training."""

import argparse
import os
from pathlib import Path

from .gpu_plan import admit_gpu_resources, gpu_record_digest
from .gpu_preflight import GpuTopology, validate_rank_environment
from .gpu_preflight_evidence import validate_run_id, verify_gpu_rank_evidence
from .gpu_preflight_io import read_gpu_json, write_gpu_json
from .gpu_preflight_probe import probe_gpu_host


def run_rank_preflight(
    *, topology: GpuTopology, resource_report: dict, run_id: str
) -> dict:
    """Observe this actual process only after validating its torchrun rank mapping."""
    validate_run_id(run_id)
    rank = validate_rank_environment(os.environ, topology)
    probe = probe_gpu_host()
    payload = {
        "schema_version": 1,
        "kind": "nvidia_rank_preflight",
        "run_id": run_id,
        "process_id": os.getpid(),
        "rank": rank,
        "topology": topology.to_mapping(),
        "resource_report": resource_report,
        "probe": probe,
        "admission": admit_gpu_resources(resource_report, topology, probe),
        "training_executed": False,
        "gpu_kernel_executed": False,
        "collective_executed": False,
    }
    payload["evidence_sha256"] = gpu_record_digest(payload)
    return verify_gpu_rank_evidence(payload)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", type=int, nargs="+", required=True)
    parser.add_argument(
        "--dtype", choices=("float32", "float16", "bfloat16"), required=True
    )
    parser.add_argument("--resource-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    try:
        topology = GpuTopology(tuple(args.devices), dtype=args.dtype)
        record = run_rank_preflight(
            topology=topology,
            resource_report=read_gpu_json(args.resource_report),
            run_id=args.run_id,
        )
        output = args.output_dir / args.run_id / f"rank-{record['rank']['rank']}.json"
        if output.resolve() == args.resource_report.resolve():
            raise ValueError("preflight output must not overwrite its resource input")
        write_gpu_json(output, record)
    except (OSError, TypeError, ValueError, KeyError) as error:
        parser.error(str(error))
    print(
        f"rank={record['rank']['rank']} preflight_ready={str(record['admission']['preflight_ready']).lower()} training_executed=false"
    )
    return 0 if record["admission"]["preflight_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
