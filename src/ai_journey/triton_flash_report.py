"""Sealed report of the Day 81 flash attention check: correctness and memory model.

``build_report`` runs every case of a plan through a backend, both input
families and every config, and then adds the sections that need no run: the
pinned text and its declared configs, the stated rules, the memory and traffic
model, and the PyPI facts of a CI install. ``seal_report`` adds a SHA-256 over
the canonical JSON of all other fields. The seal detects drift. It is not a
signature, and it proves nothing about a GPU.

Only a report from ``InterpreterBackend`` gets the interpreter label and its
evidence boundary. That backend takes only the host clock and the host load
average, so an interpreter report never holds an injected time. A report from
any other backend, such as the NumPy stand-in of the tests, is labeled
synthetic, and the verifier keeps it so. The report holds no interpreter
time, because a CPU interpreter time says nothing about a GPU.
"""

from __future__ import annotations

import math
import os
import time
from collections.abc import Callable, Mapping

from .kernel_launch_io import canonical_sha256
from .kernel_metal_reference import FLOAT32_SENTINEL_BITS
from .triton_flash_config import TritonFlashPlan, interpreter_steps, stored_values
from .triton_flash_correctness import check_case
from .triton_flash_kernel import (
    ACCUMULATOR,
    INPUT_PRECISION,
    KERNEL_NAME,
    LANGUAGE,
    LAUNCHER_NAME,
    TRITON_SOURCE,
    TritonFlashError,
    source_sha256,
)
from .triton_flash_memory import memory_mapping
from .triton_flash_reference import BOUND, INPUTS, REFERENCE
from .triton_flash_runtime import INTERPRETER_BACKEND, InterpreterBackend
from .triton_matmul_runtime import ENVIRONMENT_FIELDS, ci_install_mapping

SCHEMA_VERSION = 1
REPORT_KIND = "triton_flash_attention"
DIGEST_FIELD = "report_sha256"
INTERPRETER_LABEL = "triton_cpu_interpreter_check"
SYNTHETIC_LABEL = "synthetic_backend"
INJECTED_BACKEND = "injected"
GUARD_RULE = (
    "O, m and l hold rows = cdiv(N, BLOCK_M) * BLOCK_M rows per head, all "
    "starting as sentinels; rows past N must keep their sentinels, and every "
    "value in the first N rows must be finite, with l >= 1"
)
ORDER_RULE = (
    "tl.dot and tl.sum add in an order that the compiler, or NumPy in the CPU "
    "interpreter, chooses, so each run is held to the bound, not to bits"
)
_INTERPRETER_STATEMENT = (
    "The pinned Triton text ran in the Triton CPU interpreter with "
    "TRITON_INTERPRET=1, one program after another, as Python and NumPy code. "
    "No GPU compiled or ran it, nothing was tuned, and no time was reported. "
    "The memory section is a model of allocated intermediates, not a measured "
    "GPU footprint."
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
    "nvidia_gpu_used": False,
    "interpreter_time_reported": False,
    "memory_measured": False,
    "statement": _INTERPRETER_STATEMENT,
}
SYNTHETIC_BOUNDARY: Mapping[str, object] = {
    **INTERPRETER_BOUNDARY,
    "triton_interpreter_on_cpu": False,
    "statement": _SYNTHETIC_STATEMENT,
}


def kernel_mapping(plan: TritonFlashPlan) -> dict[str, object]:
    """Return the pinned text, its digest and the configs that a plan runs."""

    return {
        "language": LANGUAGE,
        "kernel": KERNEL_NAME,
        "launcher": LAUNCHER_NAME,
        "input_precision": INPUT_PRECISION,
        "accumulator": ACCUMULATOR,
        "grid": "(cdiv(N, BLOCK_M), BH)",
        "sha256": source_sha256(),
        "text": TRITON_SOURCE,
        "configs": [config.to_mapping() for config in plan.configs],
        "tuned": False,
    }


def rule_mapping() -> dict[str, object]:
    """Return the stated inputs, reference, bound and rules of every run."""

    return {
        "inputs": INPUTS,
        "reference": REFERENCE,
        "bound": BOUND,
        "order": ORDER_RULE,
        "guard": GUARD_RULE,
        "sentinel_float32_bits": FLOAT32_SENTINEL_BITS,
    }


def summarize(
    plan: TritonFlashPlan, cases: list[Mapping[str, object]]
) -> dict[str, object]:
    """Count the runs that held the bound, and name the largest ratios."""

    runs = [run for case in cases for run in case["runs"]]
    memory = memory_mapping()["rows"]
    return {
        "cases": len(cases),
        "configs": len(plan.configs),
        "runs": len(runs),
        "runs_within_bound": sum(
            run["output_bound_ratio"] <= 1 and run["lse_bound_ratio"] <= 1
            for run in runs
        ),
        "causal_runs": sum(
            len(case["runs"]) for case in cases if case["case"]["causal"]
        ),
        "max_output_bound_ratio": max(run["output_bound_ratio"] for run in runs),
        "max_lse_bound_ratio": max(run["lse_bound_ratio"] for run in runs),
        "interpreter_steps": interpreter_steps(plan.cases, plan.configs),
        "stored_values": stored_values(plan.cases, plan.configs),
        "largest_modeled_memory_ratio": max(row["memory_ratio"] for row in memory),
    }


def _environment(backend: object) -> dict[str, str]:
    environment = getattr(backend, "environment", None)
    if (
        not isinstance(environment, Mapping)
        or tuple(environment) != ENVIRONMENT_FIELDS
        or not all(type(value) is str for value in environment.values())
    ):
        raise TritonFlashError(
            f"a backend must name its environment: {', '.join(ENVIRONMENT_FIELDS)}"
        )
    return dict(environment)


def _load(load_average: Callable[[], tuple[float, float, float]]) -> list[float]:
    values = [float(value) for value in load_average()]
    if len(values) != 3 or not all(
        math.isfinite(value) and value >= 0 for value in values
    ):
        raise TritonFlashError("a load average must be three finite values")
    return values


def build_report(
    plan: TritonFlashPlan,
    backend: object,
    *,
    clock: Callable[[], float],
    load_average: Callable[[], tuple[float, float, float]],
) -> dict[str, object]:
    """Check every case of a plan on a backend and add the static sections."""

    if type(plan) is not TritonFlashPlan:
        raise TypeError("plan must be TritonFlashPlan")
    interpreter = type(backend) is InterpreterBackend
    if interpreter and backend.source_sha256 != source_sha256():
        raise TritonFlashError("the backend did not compile the pinned text")
    if interpreter and (clock is not time.time or load_average is not os.getloadavg):
        raise TritonFlashError(
            "an interpreter check records the host clock and the host load "
            "averages, so it takes no injected clock or load average"
        )
    environment = _environment(backend)
    recorded_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(clock()))
    before = _load(load_average)
    cases = [
        check_case(
            backend, seed=plan.seed, index=index, case=case, configs=plan.configs
        ).to_mapping()
        for index, case in enumerate(plan.cases)
    ]
    after = _load(load_average)
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
        "memory": memory_mapping(),
        "ci_install": ci_install_mapping(),
        "summary": summarize(plan, cases),
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
        raise TritonFlashError("report is already sealed")
    return {**report, DIGEST_FIELD: report_digest(report)}
