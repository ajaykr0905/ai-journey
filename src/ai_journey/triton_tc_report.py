"""Sealed report of the Day 82 Tensor Core lab: the model and the compile rule.

``build_report`` always holds the pinned kernel text, the declared config
and the speedup model. With a backend, it also compiles every variant for
sm_80 and sm_90, applies the instruction-class rule to each compile, and
keeps the digest and size of every TTGIR and PTX text. ``seal_report`` adds
a SHA-256 over the canonical JSON of all other fields. The seal detects
drift. It is not a signature.

The label says where the compile section came from. ``declared_model``: no
compile, only the model of declared inputs. ``triton_compiler_output``:
``CompilerBackend`` compiled the text with triton on this host, so the
report takes only the host clock and load averages. Any other backend gives
``synthetic_compile``. No label claims a GPU run: Triton compiles for
declared NVIDIA targets, nothing executes the PTX or cubin, and the model
times are model output, not measurements.
"""

from __future__ import annotations

import math
import os
import time
from collections.abc import Callable, Mapping
from pathlib import Path

from .kernel_launch_io import canonical_sha256
from .triton_tc_compile import (
    COMPILER_BACKEND,
    INJECTED_BACKEND,
    CompilerBackend,
    compile_all,
    environment_of,
)
from .triton_tc_config import config_sha256, read_config
from .triton_tc_kernel import TritonTcError, kernel_mapping
from .triton_tc_model import model_mapping
from .triton_tc_rule import check_compile, rule_summary

SCHEMA_VERSION = 1
REPORT_KIND = "triton_tc_lab"
DIGEST_FIELD = "report_sha256"
MODEL_LABEL = "declared_model"
COMPILER_LABEL = "triton_compiler_output"
SYNTHETIC_LABEL = "synthetic_compile"
LABELS = (MODEL_LABEL, COMPILER_LABEL, SYNTHETIC_LABEL)
_MODEL_STATEMENT = (
    "The report holds a speedup model of declared datasheet inputs. It is a "
    "model, not a measurement: no NVIDIA GPU ran, nothing was compiled, and "
    "no time was measured."
)
_COMPILER_STATEMENT = (
    "This host compiled the pinned text with triton for the declared NVIDIA "
    "targets sm_80 and sm_90 and read the TTGIR and PTX text. No NVIDIA GPU "
    "ran the kernel, no PTX or cubin was executed, and no time was measured. "
    "The model section is a model of declared inputs, not a measurement."
)
_SYNTHETIC_STATEMENT = (
    "A backend other than the triton compiler gave the compile texts, so the "
    "compile section shows only that the rule, the report and the verifier "
    "agree. No GPU ran, and the model section is not a measurement."
)
_BOUNDARY: Mapping[str, object] = {
    "nvidia_gpu_used": False,
    "ptx_executed": False,
    "cubin_executed": False,
    "time_measured": False,
    "model_is_measurement": False,
    "compiled_on_this_host": False,
    "statement": _MODEL_STATEMENT,
}
BOUNDARIES: Mapping[str, Mapping[str, object]] = {
    MODEL_LABEL: _BOUNDARY,
    COMPILER_LABEL: {
        **_BOUNDARY,
        "compiled_on_this_host": True,
        "statement": _COMPILER_STATEMENT,
    },
    SYNTHETIC_LABEL: {**_BOUNDARY, "statement": _SYNTHETIC_STATEMENT},
}


def _load(load_average: Callable[[], tuple[float, float, float]]) -> list[float]:
    values = [float(value) for value in load_average()]
    if len(values) != 3 or not all(
        math.isfinite(value) and value >= 0 for value in values
    ):
        raise TritonTcError("a load average must be three finite values")
    return values


def compile_section(backend: object) -> dict[str, object]:
    """Compile every variant for both targets and apply the rule to each."""

    environment = environment_of(backend)
    rows = []
    for compiled in compile_all(backend):
        row = check_compile(compiled)
        row["stages"] = compiled.to_mapping()["stages"]
        rows.append(row)
    return {
        "backend": COMPILER_BACKEND
        if type(backend) is CompilerBackend
        else INJECTED_BACKEND,
        "environment": environment,
        "rows": rows,
        "summary": rule_summary(rows),
    }


def summarize(sections: Mapping[str, object]) -> dict[str, object]:
    """Name the numbers that a reader checks first, from the other sections."""

    devices = sections["model"]["devices"]
    largest = {device["key"]: device["rows"][-1]["ratio"] for device in devices}
    compiled = sections["compile"]
    return {
        "kernel_sha256": sections["kernel"]["sha256"],
        "config_sha256": sections["config_sha256"],
        "per_clock_ratio": {
            device["key"]: device["per_clock_ratio"] for device in devices
        },
        "largest_size_ratio": largest,
        "memory_bound": {
            device["key"]: {
                str(row["size"]): row["memory_bound"]
                for row in device["rows"]
                if row["memory_bound"]
            }
            for device in devices
        },
        "compile": None if compiled is None else compiled["summary"],
    }


def build_report(
    config: Path,
    *,
    clock: Callable[[], float],
    load_average: Callable[[], tuple[float, float, float]],
    backend: object | None = None,
) -> dict[str, object]:
    """Build the unsealed report; with a backend, compile and check first."""

    if not isinstance(config, Path):
        raise TypeError("config must be pathlib.Path")
    compiler = type(backend) is CompilerBackend
    if compiler and (clock is not time.time or load_average is not os.getloadavg):
        raise TritonTcError(
            "a triton compile records the host clock and the host load "
            "averages, so it takes no injected clock or load average"
        )
    payload, declared = read_config(config)
    recorded_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(clock()))
    before = _load(load_average)
    compiled = None if backend is None else compile_section(backend)
    after = _load(load_average)
    if backend is None:
        label = MODEL_LABEL
    else:
        label = COMPILER_LABEL if compiler else SYNTHETIC_LABEL
    sections = {
        "kernel": kernel_mapping(),
        "config": declared.to_mapping(),
        "config_sha256": config_sha256(payload),
        "model": model_mapping(declared),
        "compile": compiled,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": REPORT_KIND,
        "label": label,
        "evidence_boundary": dict(BOUNDARIES[label]),
        **sections,
        "recorded_at": recorded_at,
        "host_load": {"before": before, "after": after},
        "summary": summarize(sections),
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
        raise TritonTcError("report is already sealed")
    return {**report, DIGEST_FIELD: report_digest(report)}
