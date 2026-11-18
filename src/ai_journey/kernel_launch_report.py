"""Sealed run report of the simulated kernel-launch lab.

``build_report`` runs every guarded kernel of a plan in the canonical order
and in each seeded permuted order, runs every unguarded demo in the same
orders, generates the CUDA source text and checks its structure. The report
holds the launch configurations, launched, useful and idle thread counts,
the misuse that the buffers recorded, the order-independence results and the
SHA-256 digests of every input and output. ``seal_report`` adds a SHA-256 over
the canonical JSON of all other fields. The seal detects drift; it is not a
signature, and it proves nothing about a GPU.
"""

from __future__ import annotations

from collections.abc import Mapping
from hashlib import sha256

from .kernel_launch_buffer import LaunchAccessLog, values_sha256
from .kernel_launch_config import (
    COMPUTE_CAPABILITY,
    LIMITS_SOURCE,
    MAX_BLOCK_DIM_X,
    MAX_BLOCK_DIM_Y,
    MAX_BLOCK_DIM_Z,
    MAX_GRID_DIM_X,
    MAX_GRID_DIM_Y,
    MAX_GRID_DIM_Z,
    MAX_THREADS_PER_BLOCK,
    WARP_SIZE,
    Dim3,
    KernelLaunchError,
    LaunchConfig,
)
from .kernel_launch_cuda import CUDA_LANGUAGE, check_cuda_source, cuda_source
from .kernel_launch_io import canonical_sha256
from .kernel_launch_kernels import (
    GRAYSCALE_WEIGHTS_PER_10000,
    KernelRun,
    run_rgb_to_grayscale,
    run_vector_add,
)
from .kernel_launch_plan import KernelLaunchPlan, UnguardedDemo
from .kernel_launch_reference import (
    GRAYSCALE_RULE,
    GRAYSCALE_WEIGHTS,
    GRAYSCALE_WEIGHTS_SOURCE,
    grayscale_rounding,
    reference_rgb_to_grayscale,
    reference_vector_add,
    rgb_image,
    rounded_sums,
    vector_add_inputs,
)

SCHEMA_VERSION = 1
REPORT_KIND = "kernel_launch_simulation"
DIGEST_FIELD = "report_sha256"
FIRST_RECORDS = 4
EVIDENCE_BOUNDARY: Mapping[str, object] = {
    "simulated_on_cpu": True,
    "pure_python": True,
    "gpu_executed": False,
    "cuda_compiled": False,
    "nvidia_hardware_used": False,
    "timing_measured": False,
    "statement": (
        "Every kernel ran thread by thread in a pure-Python simulator on the "
        "CPU. No GPU ran, no CUDA source was compiled, and no time was "
        "measured. The CUDA text is source for a later NVIDIA run."
    ),
}


def limits_mapping() -> dict[str, object]:
    """Return the named compute-capability limits that every launch obeys."""

    return {
        "compute_capability": COMPUTE_CAPABILITY,
        "warp_size": WARP_SIZE,
        "max_threads_per_block": MAX_THREADS_PER_BLOCK,
        "max_block_dim": [MAX_BLOCK_DIM_X, MAX_BLOCK_DIM_Y, MAX_BLOCK_DIM_Z],
        "max_grid_dim": [MAX_GRID_DIM_X, MAX_GRID_DIM_Y, MAX_GRID_DIM_Z],
        "source": LIMITS_SOURCE,
    }


def grayscale_rule_mapping() -> dict[str, object]:
    """Return the stated grayscale weights, their source and the uint8 rule."""

    return {
        "weights": list(GRAYSCALE_WEIGHTS),
        "weights_per_10000": list(GRAYSCALE_WEIGHTS_PER_10000),
        "rule": GRAYSCALE_RULE,
        "source": GRAYSCALE_WEIGHTS_SOURCE,
    }


def _launch(config: LaunchConfig, extent: Dim3) -> dict[str, object]:
    useful = config.threads_inside(extent)
    return {
        **config.geometry(),
        "useful_threads": useful,
        "idle_threads": config.launched_threads - useful,
    }


def _run_mapping(run: KernelRun, reference_sha256: str) -> dict[str, object]:
    return {
        "order": run.result.order.to_mapping(),
        "order_sha256": run.result.order_sha256,
        "threads_run": run.result.threads_run,
        "useful_threads": run.useful_threads,
        "output_sha256": run.output_sha256,
        "matches_reference": run.output_sha256 == reference_sha256,
        **run.result.log.counts(),
    }


def _kernel_section(
    runs: list[KernelRun], extent: Dim3, reference_sha256: str
) -> dict[str, object]:
    mappings = [_run_mapping(run, reference_sha256) for run in runs]
    return {
        "extent": extent.to_list(),
        "launch": _launch(runs[0].result.config, extent),
        "reference_sha256": reference_sha256,
        "runs": mappings,
        "order_independent": len({item["output_sha256"] for item in mappings}) == 1
        and all(run.result.log.race_free for run in runs),
    }


def _vector_add_section(plan: KernelLaunchPlan) -> dict[str, object]:
    case = plan.vector_add
    a, b = vector_add_inputs(case.input_seed, case.n)
    reference = values_sha256("float32", reference_vector_add(a, b))
    runs = [
        run_vector_add(a, b, threads_per_block=case.threads_per_block, order=order)
        for order in plan.orders
    ]
    return {
        "inputs_sha256": {
            "a": values_sha256("float32", a),
            "b": values_sha256("float32", b),
        },
        "rounded_sums": rounded_sums(a, b),
        **_kernel_section(runs, Dim3(case.n), reference),
    }


def _grayscale_section(plan: KernelLaunchPlan) -> dict[str, object]:
    case = plan.rgb_to_grayscale
    image = rgb_image(case.input_seed, case.width, case.height)
    reference = values_sha256(
        "uint8", reference_rgb_to_grayscale(image, case.width, case.height)
    )
    runs = [
        run_rgb_to_grayscale(
            image, width=case.width, height=case.height, block=case.block, order=order
        )
        for order in plan.orders
    ]
    return {
        "inputs_sha256": {"rgb": values_sha256("uint8", image)},
        "rounding": grayscale_rounding(image, case.width, case.height),
        **_kernel_section(runs, Dim3(case.width, case.height), reference),
    }


def _misuse(log: LaunchAccessLog) -> dict[str, object]:
    """Return the order-independent digests of everything a launch misused."""

    return {
        "out_of_bounds_sha256": canonical_sha256(
            [record.to_mapping() for record in log.canonical_out_of_bounds]
        ),
        "conflicts_sha256": canonical_sha256(
            {
                "write": [item.to_mapping() for item in log.write_conflicts],
                "read_write": [item.to_mapping() for item in log.read_write_conflicts],
            }
        ),
    }


def _demo_runs(
    plan: KernelLaunchPlan, demo: UnguardedDemo
) -> tuple[list[KernelRun], str]:
    if demo.kernel == "vector_add":
        (n,) = demo.extent
        a, b = vector_add_inputs(plan.vector_add.input_seed, n)
        reference = values_sha256("float32", reference_vector_add(a, b))
        runs = [
            run_vector_add(
                a,
                b,
                threads_per_block=plan.vector_add.threads_per_block,
                order=order,
                guarded=False,
            )
            for order in plan.orders
        ]
        return runs, reference
    width, height = demo.extent
    image = rgb_image(plan.rgb_to_grayscale.input_seed, width, height)
    reference = values_sha256("uint8", reference_rgb_to_grayscale(image, width, height))
    runs = [
        run_rgb_to_grayscale(
            image,
            width=width,
            height=height,
            block=plan.rgb_to_grayscale.block,
            order=order,
            guarded=False,
        )
        for order in plan.orders
    ]
    return runs, reference


def _demo_section(plan: KernelLaunchPlan, demo: UnguardedDemo) -> dict[str, object]:
    runs, reference = _demo_runs(plan, demo)
    first = runs[0]
    log = first.result.log
    config = first.result.config
    misuse = _misuse(log)
    writes = [
        record for record in log.canonical_out_of_bounds if record.access == "write"
    ]
    return {
        "kernel": first.kernel,
        "extent": first.extent.to_list(),
        "ragged": any(
            size % block
            for size, block in zip(first.extent.to_list(), config.block.to_list())
        ),
        "launch": config.geometry(),
        "orders_checked": len(runs),
        "out_of_bounds_reads": log.out_of_bounds_reads,
        "out_of_bounds_writes": log.out_of_bounds_writes,
        "write_conflicts": len(log.write_conflicts),
        "extra_writes": sum(len(item.writers) - 1 for item in log.write_conflicts),
        "read_write_conflicts": len(log.read_write_conflicts),
        **misuse,
        "first_out_of_bounds_writes": [
            record.to_mapping() for record in writes[:FIRST_RECORDS]
        ],
        "output_sha256": first.output_sha256,
        "in_bounds_output_matches_reference": first.output_sha256 == reference,
        "orders_agree": all(
            _misuse(run.result.log) == misuse
            and run.output_sha256 == first.output_sha256
            for run in runs
        ),
    }


def _cuda_section(plan: KernelLaunchPlan) -> dict[str, object]:
    block = plan.rgb_to_grayscale.block
    text = cuda_source(threads_per_block=plan.vector_add.threads_per_block, block=block)
    check = check_cuda_source(
        text,
        n=plan.vector_add.n,
        threads_per_block=plan.vector_add.threads_per_block,
        width=plan.rgb_to_grayscale.width,
        height=plan.rgb_to_grayscale.height,
        block=block,
    )
    return {
        "language": CUDA_LANGUAGE,
        "compiled": False,
        "executed": False,
        "sha256": sha256(text.encode("ascii")).hexdigest(),
        "text": text,
        "structure": check.to_mapping(),
    }


def _summary(
    kernels: tuple[dict[str, object], ...],
    demos: list[dict[str, object]],
    cuda: dict[str, object],
) -> dict[str, object]:
    runs = [run for section in kernels for run in section["runs"]]
    ragged = [demo for demo in demos if demo["ragged"]]
    aligned = [demo for demo in demos if not demo["ragged"]]
    checks = cuda["structure"]["checks"]
    return {
        "kernels": len(kernels),
        "runs": len(runs),
        "runs_matching_reference": sum(run["matches_reference"] for run in runs),
        "race_free_runs": sum(
            run["write_conflicts"] == 0 and run["read_write_conflicts"] == 0
            for run in runs
        ),
        "order_independent_kernels": sum(
            section["order_independent"] for section in kernels
        ),
        "unguarded_demos": len(demos),
        "ragged_demos_with_out_of_bounds_writes": sum(
            demo["out_of_bounds_writes"] > 0 for demo in ragged
        ),
        "ragged_demos": len(ragged),
        "aligned_demos_without_misuse": sum(
            demo["out_of_bounds_reads"] == demo["out_of_bounds_writes"] == 0
            and demo["write_conflicts"] == demo["read_write_conflicts"] == 0
            for demo in aligned
        ),
        "aligned_demos": len(aligned),
        "out_of_bounds_writes_caught": sum(
            demo["out_of_bounds_writes"] for demo in demos
        ),
        "write_conflicts_caught": sum(demo["write_conflicts"] for demo in demos),
        "cuda_checks_passed": sum(checks.values()),
        "cuda_checks": len(checks),
    }


def build_report(plan: KernelLaunchPlan) -> dict[str, object]:
    """Simulate every launch of a validated plan; no GPU is touched."""

    if type(plan) is not KernelLaunchPlan:
        raise TypeError("plan must be KernelLaunchPlan")
    vector = _vector_add_section(plan)
    gray = _grayscale_section(plan)
    demos = [_demo_section(plan, demo) for demo in plan.unguarded_demos]
    cuda = _cuda_section(plan)
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": REPORT_KIND,
        "plan": plan.to_mapping(),
        "limits": limits_mapping(),
        "grayscale_rule": grayscale_rule_mapping(),
        "vector_add": vector,
        "rgb_to_grayscale": gray,
        "unguarded_demos": demos,
        "cuda_source": cuda,
        "summary": _summary((vector, gray), demos, cuda),
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
        raise KernelLaunchError("report is already sealed")
    return {**report, DIGEST_FIELD: report_digest(report)}
