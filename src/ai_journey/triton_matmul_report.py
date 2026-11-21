"""Sealed comparison report of the Day 79 Triton port: correctness, lines and rates.

``build_report`` runs every case of a plan through a backend, both operand
families and every config, and then adds the sections that need no run: the
pinned text and its declared configs, the line counts of the Triton, CUDA and
Metal texts, the recorded Metal rates with the unmeasured sides, and the PyPI
facts of a CI install. ``seal_report`` adds a SHA-256 over the canonical JSON
of all other fields. The seal detects drift. It is not a signature, and it
proves nothing about a GPU.

Only a report from ``InterpreterBackend`` gets the interpreter label and its
evidence boundary. A report from any other backend, such as the NumPy
stand-in of the tests, is labeled synthetic, and the verifier keeps it so.
The report holds no interpreter time: the interpreter runs on the CPU, so a
time would say nothing about a GPU.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping

from .kernel_launch_io import canonical_sha256
from .kernel_metal_reference import FLOAT32_SENTINEL_BITS
from .matmul_metal_reference import (
    BOUND_SOURCE,
    ERROR_BOUND,
    GAMMA,
    REFERENCE,
    RELATIVE_ERROR,
)
from .triton_matmul_config import TritonMatmulPlan, interpreter_steps
from .triton_matmul_correctness import (
    EXACT,
    EXACT_INPUTS,
    NORMAL,
    NORMAL_INPUTS,
    check_case,
)
from .triton_matmul_kernel import (
    ACCUMULATOR,
    AUTOTUNE_KEY,
    INPUT_PRECISION,
    KERNEL_NAME,
    LANGUAGE,
    LAUNCHER_NAME,
    TRITON_SOURCE,
    TUTORIAL,
    TritonMatmulError,
    source_sha256,
)
from .triton_matmul_loc import loc_mapping
from .triton_matmul_performance import performance_mapping
from .triton_matmul_runtime import (
    ENVIRONMENT_FIELDS,
    INTERPRETER_BACKEND,
    InterpreterBackend,
    ci_install_mapping,
)

SCHEMA_VERSION = 1
REPORT_KIND = "triton_matmul_comparison"
DIGEST_FIELD = "report_sha256"
INTERPRETER_LABEL = "triton_cpu_interpreter_check"
SYNTHETIC_LABEL = "synthetic_backend"
INJECTED_BACKEND = "injected"
ORDER_RULE = (
    "tl.dot adds the products of one A tile and one B tile in an order that "
    "the compiler, or numpy.matmul in the CPU interpreter, chooses, and then "
    "adds them to a float32 accumulator, so no CPU replay names the output bits"
)
EXACT_RULE = (
    "each exact operand is an integer from -8 to 8, so each partial sum is an "
    "integer of at most 64 * K <= 2**19 in size and exact in float32 in any "
    "order: an exact output must hold the bits of the float64 reference"
)
GUARD_RULE = (
    "the buffer is C and then (Mp - 1) * N + Np - M * N sentinel slots, where "
    "Mp and Np are M and N padded to whole tiles; the tail must keep its "
    "sentinels, and each element of C must be finite"
)
_INTERPRETER_STATEMENT = (
    "The pinned Triton text ran in the Triton CPU interpreter with "
    "TRITON_INTERPRET=1, one program after another, as Python and NumPy code. "
    "No GPU compiled or ran it, nothing was tuned, and no time was reported. "
    "The CUDA texts and cuBLAS were not compiled or called. Each Metal rate is "
    "a copy of a recorded host-specific Apple GPU (Metal) measurement."
)
_SYNTHETIC_STATEMENT = (
    "A backend other than the Triton CPU interpreter produced the outputs, so "
    "the report shows only that the harness and the verifier agree. No GPU "
    "ran, nothing was tuned, and no time was reported."
)
INTERPRETER_BOUNDARY: Mapping[str, object] = {
    "triton_interpreter_on_cpu": True,
    "gpu_executed": False,
    "triton_compiled_for_gpu": False,
    "autotuned": False,
    "cuda_compiled": False,
    "cublas_called": False,
    "nvidia_or_amd_gpu_used": False,
    "interpreter_time_reported": False,
    "metal_rates_copied_from_recorded_reports": True,
    "statement": _INTERPRETER_STATEMENT,
}
SYNTHETIC_BOUNDARY: Mapping[str, object] = {
    **INTERPRETER_BOUNDARY,
    "triton_interpreter_on_cpu": False,
    "statement": _SYNTHETIC_STATEMENT,
}


def kernel_mapping(plan: TritonMatmulPlan) -> dict[str, object]:
    """Return the pinned text, its digest and the configs that a plan runs."""

    return {
        "language": LANGUAGE,
        "kernel": KERNEL_NAME,
        "launcher": LAUNCHER_NAME,
        "input_precision": INPUT_PRECISION,
        "accumulator": ACCUMULATOR,
        "sha256": source_sha256(),
        "text": TRITON_SOURCE,
        "autotune": {
            "key": list(AUTOTUNE_KEY),
            "configs": [config.to_mapping() for config in plan.configs],
            "tuned": False,
            "source": TUTORIAL,
        },
    }


def rule_mapping() -> dict[str, object]:
    """Return the stated inputs, reference, bound and rules of every case."""

    return {
        "normal_inputs": NORMAL_INPUTS,
        "exact_inputs": EXACT_INPUTS,
        "reference": REFERENCE,
        "error_bound": ERROR_BOUND,
        "gamma": GAMMA,
        "relative_error": RELATIVE_ERROR,
        "bound_source": BOUND_SOURCE,
        "order": ORDER_RULE,
        "exact": EXACT_RULE,
        "guard": GUARD_RULE,
        "sentinel_float32_bits": FLOAT32_SENTINEL_BITS,
    }


def summarize(
    plan: TritonMatmulPlan,
    cases: list[Mapping[str, object]],
    loc: Mapping[str, object],
) -> dict[str, object]:
    """Count the runs that held each rule, and name the largest bound ratio."""

    runs = [run for case in cases for run in case["runs"]]
    normal = [run for run in runs if run["family"] == NORMAL]
    exact = [run for run in runs if run["family"] == EXACT]
    triton = loc["texts"][0]["lines"]
    return {
        "cases": len(cases),
        "configs": len(plan.configs),
        "runs": len(runs),
        "normal_runs_within_bound": sum(run["bound_ratio"] <= 1 for run in normal),
        "exact_runs_with_exact_bits": sum(
            run["rounded_reference_differences"] == 0 for run in exact
        ),
        "max_bound_ratio": max(run["bound_ratio"] for run in normal),
        "normal_elements_off_the_rounded_reference": sum(
            run["rounded_reference_differences"] for run in normal
        ),
        "interpreter_steps": interpreter_steps(plan.shapes, plan.configs),
        "loc_texts": len(loc["texts"]),
        "triton_kernel_lines": triton["kernel"],
        "triton_host_lines": triton["host"],
    }


def _environment(backend: object) -> dict[str, str]:
    environment = getattr(backend, "environment", None)
    if (
        not isinstance(environment, Mapping)
        or tuple(environment) != ENVIRONMENT_FIELDS
        or not all(type(value) is str for value in environment.values())
    ):
        raise TritonMatmulError(
            f"a backend must name its environment: {', '.join(ENVIRONMENT_FIELDS)}"
        )
    return dict(environment)


def _load(load_average: Callable[[], tuple[float, float, float]]) -> list[float]:
    values = [float(value) for value in load_average()]
    if len(values) != 3 or not all(
        math.isfinite(value) and value >= 0 for value in values
    ):
        raise TritonMatmulError("a load average must be three finite values")
    return values


def build_report(
    plan: TritonMatmulPlan,
    backend: object,
    *,
    clock: Callable[[], float],
    load_average: Callable[[], tuple[float, float, float]],
) -> dict[str, object]:
    """Check every case of a plan on a backend and add the static sections."""

    if type(plan) is not TritonMatmulPlan:
        raise TypeError("plan must be TritonMatmulPlan")
    interpreter = type(backend) is InterpreterBackend
    if interpreter and backend.source_sha256 != source_sha256():
        raise TritonMatmulError("the backend did not compile the pinned text")
    environment = _environment(backend)
    recorded_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(clock()))
    before = _load(load_average)
    cases = [
        check_case(
            backend, seed=plan.seed, index=index, shape=shape, configs=plan.configs
        ).to_mapping()
        for index, shape in enumerate(plan.shapes)
    ]
    after = _load(load_average)
    loc = loc_mapping()
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": REPORT_KIND,
        "label": INTERPRETER_LABEL if interpreter else SYNTHETIC_LABEL,
        "backend": INTERPRETER_BACKEND if interpreter else INJECTED_BACKEND,
        "evidence_boundary": dict(
            INTERPRETER_BOUNDARY if interpreter else SYNTHETIC_BOUNDARY
        ),
        "plan": plan.to_mapping(),
        "plan_sha256": canonical_sha256(plan.to_mapping()),
        "environment": environment,
        "recorded_at": recorded_at,
        "host_load": {"before": before, "after": after},
        "kernel": kernel_mapping(plan),
        "rule": rule_mapping(),
        "correctness": {"cases": cases},
        "loc": loc,
        "performance": performance_mapping(),
        "ci_install": ci_install_mapping(),
        "summary": summarize(plan, cases, loc),
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
        raise TritonMatmulError("report is already sealed")
    return {**report, DIGEST_FIELD: report_digest(report)}
