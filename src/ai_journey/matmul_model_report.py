"""Sealed run report of the matmul model lab.

``build_report`` runs every case of a plan through both thread mappings in
the canonical order and in each seeded permuted order. For each launch it
also totals the per-warp requests, cross-checks them with the memory_access
and SIMT labs, and lists the per-step addresses of two probe warps: the warps
of ``C[0, 0]`` and of ``C[m - 1, n - 1]``, each traced through the kernel
code. It then accounts FLOPs and bytes for every case and accounting shape,
places both traffic models on the roofline of each declared ceiling pair,
estimates occupancy for each declared register count, and generates and
checks the CUDA text. ``seal_report`` adds a SHA-256 over the canonical JSON
of all other fields. The seal detects drift; it is not a signature, and it
proves nothing about a GPU. The report holds no float: every ratio is an
exact fraction with a rounded decimal string.
"""

from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction
from hashlib import sha256

from .kernel_launch_buffer import values_sha256
from .kernel_launch_cuda import CUDA_LANGUAGE
from .kernel_launch_io import canonical_sha256
from .kernel_launch_reference import INPUT_LABEL
from .kernel_launch_report import _launch, _run_mapping, limits_mapping
from .matmul_model_access import (
    ACCESS_CLASSES,
    ELEMENT_BYTES,
    MatmulLaunch,
    analyze_launch,
    request_mapping,
    warp_of_element,
    warp_requests,
)
from .matmul_model_crosscheck import cross_check_launch
from .matmul_model_cuda import check_cuda_source, cuda_source
from .matmul_model_kernels import MAPPINGS, run_naive_matmul, thread_extent
from .matmul_model_occupancy import (
    architecture_mapping,
    estimate_occupancy,
    occupancy_mapping,
)
from .matmul_model_plan import MatmulCase, MatmulModelPlan
from .matmul_model_reference import (
    INPUT_STREAMS,
    SUMMATION_RULE,
    MatmulModelError,
    correctly_rounded_matmul,
    matmul_inputs,
    order_rounding,
    reference_matmul,
)
from .matmul_model_roofline import roofline
from .matmul_model_trace import trace_matches_model
from .matmul_model_traffic import account_matmul, ratio_mapping
from .memory_access_pattern import LINE_BYTES, SECTOR_BYTES, WARP_SIZE
from .memory_access_tiling import MatmulShape

SCHEMA_VERSION = 1
REPORT_KIND = "matmul_model"
DIGEST_FIELD = "report_sha256"
CUDA_ROUNDING = (
    "__fmul_rn and __fadd_rn: each product and each sum rounds to float32, "
    "ties to even, and CUDA never merges them into a fused multiply-add"
)
EVIDENCE_BOUNDARY: Mapping[str, object] = {
    "simulated_on_cpu": True,
    "pure_python": True,
    "gpu_executed": False,
    "cuda_compiled": False,
    "nvidia_hardware_used": False,
    "timing_measured": False,
    "ceilings_measured": False,
    "register_counts_measured": False,
    "statement": (
        "Every kernel ran thread by thread in a pure-Python simulator on the "
        "CPU. No GPU ran, no CUDA source was compiled, and no time was "
        "measured. The roofline uses declared ceilings and the occupancy uses "
        "declared register counts, so neither describes a measured device."
    ),
}


def rule_mapping() -> dict[str, object]:
    """Return the stated summation rule, the input streams and the CUDA form."""

    return {
        "summation": SUMMATION_RULE,
        "input_label": INPUT_LABEL,
        "input_streams": list(INPUT_STREAMS),
        "cuda_rounding": CUDA_ROUNDING,
    }


def geometry_mapping() -> dict[str, object]:
    """Return the declared warp and memory geometry of the access model."""

    return {
        "warp_size": WARP_SIZE,
        "sector_bytes": SECTOR_BYTES,
        "line_bytes": LINE_BYTES,
        "element_bytes": ELEMENT_BYTES,
        "access_classes": list(ACCESS_CLASSES),
        "cache_modeled": False,
    }


def _probes(
    launch: MatmulLaunch,
    a: tuple[float, ...],
    b: tuple[float, ...],
    reference: tuple[float, ...],
) -> list[dict[str, object]]:
    """List the per-step requests of the warps of the first and last element."""

    probes = []
    seen = set()
    shape = launch.shape
    for row, col in ((0, 0), (shape.m - 1, shape.n - 1)):
        ref = warp_of_element(launch, row, col)
        if ref in seen:
            continue
        seen.add(ref)
        requests = warp_requests(launch, ref)
        probes.append(
            {
                "element": [row, col],
                "warp": ref.to_mapping(),
                "active_lanes": requests.active_lanes,
                "steps": [
                    {
                        "step": i,
                        "a": request_mapping(a_load),
                        "b": request_mapping(b_load),
                    }
                    for i, (a_load, b_load) in enumerate(
                        zip(requests.a_loads, requests.b_loads)
                    )
                ],
                "c_store": request_mapping(requests.c_store),
                "trace_matches_model": trace_matches_model(
                    launch, ref, a, b, reference
                ),
            }
        )
    return probes


def _mapping_section(
    plan: MatmulModelPlan,
    case: MatmulCase,
    mapping: str,
    operands: tuple[tuple[float, ...], tuple[float, ...]],
    reference: tuple[float, ...],
    reference_sha256: str,
) -> dict[str, object]:
    a, b = operands
    launch = plan.launch(case, mapping)
    extent = thread_extent(case.shape, mapping)
    runs = [
        run_naive_matmul(
            a, b, case.shape, block=plan.block, mapping=mapping, order=order
        )
        for order in plan.orders
    ]
    access = analyze_launch(launch)
    return {
        "mapping": mapping,
        "kernel": runs[0].kernel,
        "extent": extent.to_list(),
        "ragged": extent.x % plan.block[0] != 0 or extent.y % plan.block[1] != 0,
        "launch": _launch(launch.config, extent),
        "runs": [_run_mapping(run, reference_sha256) for run in runs],
        "order_independent": len({run.output_sha256 for run in runs}) == 1
        and all(run.result.log.race_free for run in runs),
        "access": {
            **access.to_mapping(),
            "flops_per_transferred_byte": ratio_mapping(
                Fraction(case.shape.flops, access.transferred_bytes)
            ),
        },
        "cross_check": cross_check_launch(launch).to_mapping(),
        "probes": _probes(launch, a, b, reference),
    }


def _case_section(plan: MatmulModelPlan, case: MatmulCase) -> dict[str, object]:
    a, b = matmul_inputs(case.input_seed, case.shape)
    reference = reference_matmul(a, b, case.shape)
    exact = correctly_rounded_matmul(a, b, case.shape)
    reference_sha256 = values_sha256("float32", reference)
    return {
        "name": case.name,
        "shape": case.shape.to_mapping(),
        "inputs_sha256": {
            "a": values_sha256("float32", a),
            "b": values_sha256("float32", b),
        },
        "reference_sha256": reference_sha256,
        "order_rounding": {
            **order_rounding(reference, exact).to_mapping(),
            "correctly_rounded_sha256": values_sha256("float32", exact),
        },
        "mappings": [
            _mapping_section(plan, case, mapping, (a, b), reference, reference_sha256)
            for mapping in MAPPINGS
        ],
    }


def _accounting_section(
    plan: MatmulModelPlan, shape: MatmulShape, source: str
) -> dict[str, object]:
    accounting = account_matmul(shape)
    return {
        "source": source,
        **accounting.to_mapping(),
        "roofline": [
            {
                "ceilings": ceilings.to_mapping(),
                "no_reuse": roofline(
                    accounting.no_reuse.intensity, ceilings
                ).to_mapping(),
                "ideal_reuse": roofline(
                    accounting.ideal_reuse.intensity, ceilings
                ).to_mapping(),
            }
            for ceilings in plan.ceilings
        ],
    }


def _occupancy_section(plan: MatmulModelPlan) -> dict[str, object]:
    return {
        "architecture": architecture_mapping(),
        "threads_per_block": plan.threads_per_block,
        "estimates": [
            occupancy_mapping(
                estimate_occupancy(
                    threads_per_block=plan.threads_per_block,
                    registers_per_thread=count,
                )
            )
            for count in plan.registers_per_thread
        ],
    }


def _cuda_section(plan: MatmulModelPlan) -> dict[str, object]:
    text = cuda_source(block=plan.block)
    return {
        "language": CUDA_LANGUAGE,
        "compiled": False,
        "executed": False,
        "sha256": sha256(text.encode("ascii")).hexdigest(),
        "text": text,
        "structure_by_case": [
            {
                "case": case.name,
                **check_cuda_source(
                    text, shape=case.shape, block=plan.block
                ).to_mapping(),
            }
            for case in plan.cases
        ],
    }


def _summary(
    cases: list[dict[str, object]],
    accounting: list[dict[str, object]],
    occupancy: dict[str, object],
    cuda: dict[str, object],
) -> dict[str, object]:
    launches = [section for case in cases for section in case["mappings"]]
    runs = [run for section in launches for run in section["runs"]]
    probes = [probe for section in launches for probe in section["probes"]]
    points = [
        entry[model]
        for item in accounting
        for entry in item["roofline"]
        for model in ("no_reuse", "ideal_reuse")
    ]
    structures = cuda["structure_by_case"]
    return {
        "cases": len(cases),
        "launches": len(launches),
        "ragged_launches": sum(section["ragged"] for section in launches),
        "runs": len(runs),
        "runs_matching_reference": sum(run["matches_reference"] for run in runs),
        "memory_safe_runs": sum(
            run["out_of_bounds_reads"] == run["out_of_bounds_writes"] == 0
            for run in runs
        ),
        "race_free_runs": sum(
            run["write_conflicts"] == run["read_write_conflicts"] == 0 for run in runs
        ),
        "order_independent_launches": sum(
            section["order_independent"] for section in launches
        ),
        "cross_checked_launches": sum(
            section["cross_check"]["passed"] for section in launches
        ),
        "cross_checked_requests": sum(
            section["cross_check"]["memory_access_agreeing"] for section in launches
        ),
        "probes": len(probes),
        "probe_traces_matching_model": sum(
            probe["trace_matches_model"] for probe in probes
        ),
        "accounting_shapes": len(accounting),
        "roofline_points": len(points),
        "memory_bound_points": sum(point["bound"] == "memory" for point in points),
        "occupancy_estimates": len(occupancy["estimates"]),
        "cuda_checks_passed": sum(sum(item["checks"].values()) for item in structures),
        "cuda_checks": sum(len(item["checks"]) for item in structures),
    }


def build_report(plan: MatmulModelPlan) -> dict[str, object]:
    """Simulate and model every input of a validated plan; no GPU is touched."""

    if type(plan) is not MatmulModelPlan:
        raise TypeError("plan must be MatmulModelPlan")
    cases = [_case_section(plan, case) for case in plan.cases]
    accounting = [
        *(
            _accounting_section(plan, case.shape, f"case:{case.name}")
            for case in plan.cases
        ),
        *(
            _accounting_section(plan, shape, "declared")
            for shape in plan.accounting_shapes
        ),
    ]
    occupancy = _occupancy_section(plan)
    cuda = _cuda_section(plan)
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": REPORT_KIND,
        "plan": plan.to_mapping(),
        "limits": limits_mapping(),
        "rule": rule_mapping(),
        "geometry": geometry_mapping(),
        "cases": cases,
        "accounting": accounting,
        "occupancy": occupancy,
        "cuda_source": cuda,
        "summary": _summary(cases, accounting, occupancy, cuda),
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
        raise MatmulModelError("report is already sealed")
    return {**report, DIGEST_FIELD: report_digest(report)}
