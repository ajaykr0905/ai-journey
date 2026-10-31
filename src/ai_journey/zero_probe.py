"""Execute CPU/Gloo collective conformance for ZeRO communication phases."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Sequence

import torch
import torch.distributed as dist


class ZeroProbeError(ValueError):
    """Raised when executed collective evidence is inconsistent."""


MAX_ZERO_PROBE_BYTES = 1024 * 1024


def _canonical_bytes(report: dict[str, object]) -> bytes:
    return json.dumps(
        report,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def seal_zero_probe_report(report: dict[str, object]) -> dict[str, object]:
    """Attach a canonical digest without mutating the input report."""

    if not isinstance(report, dict):
        raise TypeError("report must be a dictionary")
    unsigned = {key: value for key, value in report.items() if key != "integrity"}
    return {
        **unsigned,
        "integrity": {
            "algorithm": "sha256",
            "digest": hashlib.sha256(_canonical_bytes(unsigned)).hexdigest(),
        },
    }


def _numeric_vector(value: object, name: str, length: int) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise ZeroProbeError(f"{name} has the wrong shape")
    if any(
        isinstance(item, bool)
        or not isinstance(item, (int, float))
        or not math.isfinite(item)
        for item in value
    ):
        raise ZeroProbeError(f"{name} must contain finite numbers")
    return [float(item) for item in value]


def verify_zero_probe_report(report: dict[str, object]) -> dict[str, object]:
    """Verify the seal and recompute collective equivalence from rank inputs."""

    if not isinstance(report, dict):
        raise TypeError("report must be a dictionary")
    if report.get("schema_version") != 1:
        raise ZeroProbeError("unsupported probe schema")
    if report.get("kind") != "cpu_gloo_zero_collective_conformance":
        raise ZeroProbeError("unexpected probe kind")
    if report.get("executed") is not True:
        raise ZeroProbeError("probe must record executed=true")
    if report.get("performance_measured") is not False:
        raise ZeroProbeError("probe must not claim performance measurements")
    if report.get("backend") != "gloo" or report.get("device") != "cpu":
        raise ZeroProbeError("probe must remain CPU/Gloo evidence")
    try:
        expected_integrity = seal_zero_probe_report(report)["integrity"]
    except (TypeError, ValueError, OverflowError) as error:
        raise ZeroProbeError("probe report is not canonical") from error
    if report.get("integrity") != expected_integrity:
        raise ZeroProbeError("probe report integrity mismatch")

    world_size = report.get("world_size")
    chunk_elements = report.get("chunk_elements")
    ranks = report.get("ranks")
    if type(world_size) is not int or world_size <= 0:
        raise ZeroProbeError("world_size must be positive")
    if type(chunk_elements) is not int or chunk_elements <= 0:
        raise ZeroProbeError("chunk_elements must be positive")
    if not isinstance(ranks, list) or len(ranks) != world_size:
        raise ZeroProbeError("rank evidence must cover the world exactly")
    if not all(isinstance(item, dict) for item in ranks):
        raise ZeroProbeError("rank evidence entries must be objects")
    if [item.get("rank") for item in ranks] != list(range(world_size)):
        raise ZeroProbeError("rank evidence must be ordered and complete")

    vector_length = world_size * chunk_elements
    inputs = [
        _numeric_vector(item.get("input"), "rank input", vector_length)
        for item in ranks
    ]
    recomputed = [
        math.fsum(vector[index] for vector in inputs) for index in range(vector_length)
    ]
    expected = _numeric_vector(
        report.get("expected_sum"), "expected_sum", vector_length
    )
    if recomputed != expected:
        raise ZeroProbeError("expected sum does not match rank inputs")
    all_reduce = [
        _numeric_vector(item.get("all_reduce"), "all_reduce", vector_length)
        for item in ranks
    ]
    all_gather = [
        _numeric_vector(item.get("all_gather"), "all_gather", vector_length)
        for item in ranks
    ]
    shards = [
        _numeric_vector(item.get("reduce_scatter"), "reduce_scatter", chunk_elements)
        for item in ranks
    ]
    if (
        any(vector != expected for vector in (*all_reduce, *all_gather))
        or [value for shard in shards for value in shard] != expected
    ):
        raise ZeroProbeError("collective outputs do not reconstruct the expected sum")
    return report


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ZeroProbeError("duplicate probe report field")
        result[key] = value
    return result


def load_zero_probe_report(path: Path) -> dict[str, object]:
    """Load bounded CPU/Gloo evidence and verify every collective result."""

    if not isinstance(path, Path):
        raise TypeError("probe report path must be pathlib.Path")
    try:
        with path.open("rb") as stream:
            payload = stream.read(MAX_ZERO_PROBE_BYTES + 1)
        if len(payload) > MAX_ZERO_PROBE_BYTES:
            raise ZeroProbeError("probe report exceeds 1 MiB")
        report = json.loads(payload.decode("utf-8"), object_pairs_hook=_unique_object)
    except ZeroProbeError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError) as error:
        raise ZeroProbeError("unable to read probe report") from error
    return verify_zero_probe_report(report)


def _write_atomic(path: Path, report: dict[str, object]) -> None:
    payload = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if len(payload) > MAX_ZERO_PROBE_BYTES:
        raise ZeroProbeError("probe report exceeds 1 MiB")
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
            report = seal_zero_probe_report(unsigned)
            verify_zero_probe_report(report)
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
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--output", type=Path)
    target.add_argument("--verify-report", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.verify_report is not None:
        report = load_zero_probe_report(args.verify_report)
        print(
            f"verified_zero_probe={report['integrity']['digest']} "
            f"backend={report['backend']} world_size={report['world_size']} "
            "performance_measured=false"
        )
        return 0
    run_probe(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
