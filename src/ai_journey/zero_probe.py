"""Execute CPU/Gloo collective conformance for ZeRO communication phases."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Sequence

import torch
import torch.distributed as dist


def _canonical_bytes(report: dict[str, object]) -> bytes:
    return json.dumps(
        report,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _write_atomic(path: Path, report: dict[str, object]) -> None:
    payload = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def run_probe(output: Path) -> dict[str, object] | None:
    """Run equivalent collectives and publish rank-zero conformance evidence."""

    if not isinstance(output, Path):
        raise TypeError("output must be pathlib.Path")
    dist.init_process_group("gloo")
    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        chunk_elements = 4
        base = torch.arange(1, world_size * chunk_elements + 1, dtype=torch.float64)
        local = base + rank * 10
        expected = base * world_size + 10 * sum(range(world_size))

        all_reduce = local.clone()
        dist.all_reduce(all_reduce, op=dist.ReduceOp.SUM)

        reduce_scatter = torch.empty(chunk_elements, dtype=torch.float64)
        dist.reduce_scatter_tensor(reduce_scatter, local, op=dist.ReduceOp.SUM)
        all_gather = torch.empty_like(local)
        dist.all_gather_into_tensor(all_gather, reduce_scatter)

        if not torch.equal(all_reduce, expected):
            raise RuntimeError("all-reduce output did not match the expected sum")
        if not torch.equal(all_gather, expected):
            raise RuntimeError(
                "reduce-scatter plus all-gather did not reconstruct the expected sum"
            )

        local_result = {
            "rank": rank,
            "input": local.tolist(),
            "all_reduce": all_reduce.tolist(),
            "reduce_scatter": reduce_scatter.tolist(),
            "all_gather": all_gather.tolist(),
        }
        rank_results: list[dict[str, object] | None] = [None] * world_size
        dist.all_gather_object(rank_results, local_result)
        report = None
        if rank == 0:
            unsigned = {
                "schema_version": 1,
                "kind": "cpu_gloo_zero_collective_conformance",
                "executed": True,
                "performance_measured": False,
                "backend": dist.get_backend(),
                "device": "cpu",
                "dtype": "float64",
                "world_size": world_size,
                "chunk_elements": chunk_elements,
                "expected_sum": expected.tolist(),
                "ranks": rank_results,
                "torch_version": torch.__version__,
                "limitations": [
                    "CPU/Gloo correctness evidence only; no NVIDIA GPU or NCCL was used.",
                    "No latency, bandwidth, utilization, memory, or cost was measured.",
                ],
            }
            report = {
                **unsigned,
                "integrity": {
                    "algorithm": "sha256",
                    "digest": hashlib.sha256(_canonical_bytes(unsigned)).hexdigest(),
                },
            }
            _write_atomic(output, report)
            print(
                f"zero_probe={report['integrity']['digest']} backend=gloo "
                f"world_size={world_size} performance_measured=false"
            )
        dist.barrier()
        return report
    finally:
        dist.destroy_process_group()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_probe(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
