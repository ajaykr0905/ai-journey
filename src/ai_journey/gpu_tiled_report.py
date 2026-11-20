"""Sealed report of the tiled CUDA contract lab.

``build_report`` writes the tiled kernel text of each plan tile and the
cuBLAS call text, and it checks each text for each case shape without a
compiler. For each case it then draws the Day 74 operands, replays their fma
chain with ``fma_replay``, sums the float64 reference, and runs the block
simulator at each tile. A tile run must hold the replay's bits, count the
loads of ``TiledMatmulTraffic``, store each element of C once and log no
shared-memory hazard. Last, it estimates the declared occupancy of each tile
at each register count. ``seal_report`` adds a SHA-256 over the canonical
JSON of all other fields. The seal detects drift; it is not a signature, and
it proves nothing about a GPU.

The digests are the contract of a later NVIDIA run: a tiled CUDA output must
hash to the replay digest of its case, and a cuBLAS output, whose order of
additions is unknown, must stay inside the error bound of its case.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy

from .gpu_tiled_blocksim import run_tiled_blocks
from .gpu_tiled_config import GpuTiledPlan
from .gpu_tiled_cuda import (
    CUBLAS_HEADER,
    TF32_ASSUMPTION,
    GpuTiledError,
    check_cublas_source,
    check_tiled_source,
    cublas_source,
    kernel_name,
    launcher_name,
    shared_memory_bytes,
    source_sha256,
    tiled_cuda_source,
)
from .gpu_tiled_occupancy import estimate_occupancy
from .kernel_launch_cuda import CUDA_LANGUAGE
from .kernel_launch_io import canonical_sha256
from .kernel_launch_report import limits_mapping
from .matmul_metal_reference import (
    BOUND_SOURCE,
    CORRECTNESS_STREAM,
    ERROR_BOUND,
    GAMMA,
    REFERENCE,
    RELATIVE_ERROR,
    error_bound,
    float64_reference,
    fma_replay,
    matmul_inputs,
    matrix_sha256,
    max_relative_error,
)
from .matmul_model_occupancy import architecture_mapping, occupancy_mapping
from .memory_access_tiling import MatmulShape

SCHEMA_VERSION = 1
REPORT_KIND = "gpu_tiled_cuda"
DIGEST_FIELD = "report_sha256"
CUBLAS_FUNCTION = "sgemm_row_major"
FMA_RULE = (
    "sum = fma(As[ty][i], Bs[i][tx], sum) from +0.0, rounded once to float32 "
    "per step, for k = phase * T + i in increasing order"
)
ZERO_FILL = (
    "a tile slot outside A or B holds 0, and the inner loop stops at "
    "min(T, k - phase * T), so it reads no padded slot"
)
INPUTS = (
    "numpy.random.default_rng([seed, 0, index]) draws standard-normal float32 "
    "A[m, k] and then B[k, n], as the Day 74 Metal lab draws case index"
)
EVIDENCE_BOUNDARY: Mapping[str, object] = {
    "simulated_on_cpu": True,
    "gpu_executed": False,
    "cuda_compiled": False,
    "cublas_called": False,
    "nvidia_hardware_used": False,
    "timing_measured": False,
    "register_counts_measured": False,
    "tf32_assumption_checked": False,
    "statement": (
        "The block simulator ran on the CPU in numpy float32. No GPU ran, no "
        "CUDA or cuBLAS text was compiled, linked or called, and no time was "
        "measured. The occupancy uses declared register counts. That "
        "CUBLAS_DEFAULT_MATH keeps cublasSgemm off TF32 is an assumption from "
        "the cuBLAS documentation, which a later run must check on an NVIDIA "
        "host."
    ),
}


def rule_mapping() -> dict[str, object]:
    """Return the stated arithmetic, inputs and bound of every case."""

    return {
        "fma": FMA_RULE,
        "zero_fill": ZERO_FILL,
        "inputs": INPUTS,
        "replay": "matmul_metal_reference.fma_replay",
        "reference": REFERENCE,
        "error_bound": ERROR_BOUND,
        "gamma": GAMMA,
        "relative_error": RELATIVE_ERROR,
        "bound_source": BOUND_SOURCE,
    }


def sources_mapping(tiles: tuple[int, ...]) -> dict[str, object]:
    """Return each source text with its digest; none was compiled or run."""

    cublas = cublas_source()
    return {
        "language": CUDA_LANGUAGE,
        "compiled": False,
        "executed": False,
        "tiled": [
            {
                "tile": tile,
                "kernel": kernel_name(tile),
                "launcher": launcher_name(tile),
                "shared_memory_bytes": shared_memory_bytes(tile),
                "sha256": source_sha256(tiled_cuda_source(tile=tile)),
                "text": tiled_cuda_source(tile=tile),
            }
            for tile in tiles
        ],
        "cublas": {
            "function": CUBLAS_FUNCTION,
            "header": CUBLAS_HEADER,
            "called": False,
            "tf32_assumption": TF32_ASSUMPTION,
            "sha256": source_sha256(cublas),
            "text": cublas,
        },
    }


def _case_section(plan: GpuTiledPlan, index: int, shape: MatmulShape) -> dict:
    left, right = matmul_inputs(plan.seed, CORRECTNESS_STREAM, index, shape)
    replay = fma_replay(left, right)
    reference, magnitude = float64_reference(left, right)
    tiles = []
    for tile in plan.tiles:
        run = run_tiled_blocks(left, right, tile=tile)
        tiles.append(
            {
                "tile": tile,
                "cuda_check": check_tiled_source(
                    tiled_cuda_source(tile=tile), tile=tile, shape=shape
                ).to_mapping(),
                "run": run.to_mapping(),
                "matches_replay": bool(
                    numpy.array_equal(
                        run.output.view(numpy.uint32), replay.view(numpy.uint32)
                    )
                ),
            }
        )
    return {
        "index": index,
        "shape": shape.to_mapping(),
        "inputs_sha256": {"a": matrix_sha256(left), "b": matrix_sha256(right)},
        "replay_sha256": matrix_sha256(replay),
        "reference_sha256": matrix_sha256(reference),
        "error_bound": error_bound(shape.k),
        "replay_relative_error": max_relative_error(replay, reference, magnitude),
        "cublas_check": check_cublas_source(cublas_source(), shape=shape).to_mapping(),
        "tiles": tiles,
    }


def _occupancy_section(plan: GpuTiledPlan) -> dict[str, object]:
    return {
        "architecture": architecture_mapping(),
        "estimates": [
            {
                "tile": tile,
                **occupancy_mapping(
                    estimate_occupancy(tile=tile, registers_per_thread=count)
                ),
            }
            for tile in plan.tiles
            for count in plan.registers_per_thread
        ],
    }


def summarize(
    cases: list[Mapping[str, object]], occupancy: Mapping[str, object]
) -> dict[str, object]:
    """Count the runs and checks that held, out of all of them."""

    entries = [entry for case in cases for entry in case["tiles"]]
    runs = [entry["run"] for entry in entries]
    tiled = [entry["cuda_check"]["checks"] for entry in entries]
    cublas = [case["cublas_check"]["checks"] for case in cases]
    return {
        "cases": len(cases),
        "tiles": len(cases[0]["tiles"]),
        "block_runs": len(entries),
        "block_runs_matching_replay": sum(entry["matches_replay"] for entry in entries),
        "block_runs_matching_traffic": sum(run["matches_traffic"] for run in runs),
        "race_free_block_runs": sum(run["hazards"]["race_free"] for run in runs),
        "stored_once_block_runs": sum(run["stored_once"] for run in runs),
        "cuda_checks_passed": sum(sum(checks.values()) for checks in tiled),
        "cuda_checks": sum(len(checks) for checks in tiled),
        "cublas_checks_passed": sum(sum(checks.values()) for checks in cublas),
        "cublas_checks": sum(len(checks) for checks in cublas),
        "occupancy_estimates": len(occupancy["estimates"]),
    }


def build_report(plan: GpuTiledPlan) -> dict[str, object]:
    """Check, simulate and estimate every input of a plan; no GPU is used."""

    if type(plan) is not GpuTiledPlan:
        raise TypeError("plan must be GpuTiledPlan")
    cases = [
        _case_section(plan, index, shape) for index, shape in enumerate(plan.shapes)
    ]
    occupancy = _occupancy_section(plan)
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": REPORT_KIND,
        "plan": plan.to_mapping(),
        "limits": limits_mapping(),
        "rule": rule_mapping(),
        "sources": sources_mapping(plan.tiles),
        "cases": cases,
        "occupancy": occupancy,
        "summary": summarize(cases, occupancy),
        "evidence_boundary": dict(EVIDENCE_BOUNDARY),
    }


def report_digest(report: Mapping[str, object]) -> str:
    """Return the SHA-256 of the canonical report without its digest field."""

    if not isinstance(report, Mapping):
        raise TypeError("report must be a mapping")
    return canonical_sha256(
        {key: value for key, value in report.items() if key != DIGEST_FIELD}
    )


def seal_report(report: Mapping[str, object]) -> dict[str, object]:
    """Return a copy of an unsealed report with its SHA-256 digest added."""

    if not isinstance(report, Mapping):
        raise TypeError("report must be a mapping")
    if DIGEST_FIELD in report:
        raise GpuTiledError("report is already sealed")
    return {**report, DIGEST_FIELD: report_digest(report)}
