"""Sealed report of the Day 80 Triton IR lab: what the compiler wrote, and why.

``build_report`` reads the fixture folder, checks the manifest, the kernel
source digest and the committed texts, and adds the PTX, TTIR and TTGIR
summaries, the stage map, the pipeline facts and the comparison of all 12
compiles. With a backend it also compiles every config again and compares
the digests. ``seal_report`` adds a SHA-256 over the canonical JSON of all
other fields. The seal detects drift. It is not a signature.

The label says where the texts came from. ``recorded_compiler_output``: the
host only read texts that the session owner compiled in a Linux container.
``triton_regeneration``: ``CompilerBackend`` compiled the text again on this
host, so the report holds the host clock and load averages only. Any other
backend gives ``synthetic_regeneration``. No label claims a GPU run: Triton
compiled for declared NVIDIA targets, and nothing executed the PTX or cubin.
"""

from __future__ import annotations

import math
import os
import time
from collections.abc import Callable, Mapping
from hashlib import sha256
from pathlib import Path

from .kernel_launch_io import canonical_sha256
from .triton_ir_compare import compare
from .triton_ir_manifest import (
    COMMITTED_CONFIG,
    COMMITTED_TARGET,
    TritonIrError,
    check_fixtures,
    check_source,
    read_manifest,
    require_recorded,
)
from .triton_ir_mlir import read_mlir, ttgir_summary, ttir_summary
from .triton_ir_ptx import read_ptx
from .triton_ir_runtime import CompilerBackend, regenerate
from .triton_ir_stage_map import pipeline_facts, stage_map
from .triton_matmul_kernel import KERNEL_NAME

SCHEMA_VERSION = 1
REPORT_KIND = "triton_ir_lab"
DIGEST_FIELD = "report_sha256"
RECORDED_LABEL = "recorded_compiler_output"
REGENERATION_LABEL = "triton_regeneration"
SYNTHETIC_LABEL = "synthetic_regeneration"
LABELS = (RECORDED_LABEL, REGENERATION_LABEL, SYNTHETIC_LABEL)
_RECORDED_STATEMENT = (
    "The stage texts are triton 3.9.0 compiler output for the declared NVIDIA "
    "targets sm_80 and sm_90, which the session owner recorded in a Linux "
    "aarch64 container. This host read the committed texts and the manifest "
    "only. No NVIDIA GPU ran the kernel, no PTX or cubin was executed, and no "
    "ptxas register report was read."
)
_REGENERATION_STATEMENT = (
    "This host compiled the pinned kernel text with triton for the declared "
    "NVIDIA targets sm_80 and sm_90 and compared each stage digest with the "
    "recorded dump. No NVIDIA GPU ran the kernel, no PTX or cubin was "
    "executed, and no ptxas register report was read."
)
_SYNTHETIC_STATEMENT = (
    "A backend other than the triton compiler produced the regenerated texts, "
    "so the regeneration section shows only that the harness and the "
    "verifier agree. No GPU ran, and no PTX or cubin was executed."
)
_BOUNDARY: Mapping[str, object] = {
    "compiler_output_only": True,
    "nvidia_gpu_used": False,
    "ptx_executed": False,
    "cubin_executed": False,
    "ptxas_register_report": False,
    "compiled_on_this_host": False,
    "statement": _RECORDED_STATEMENT,
}
BOUNDARIES: Mapping[str, Mapping[str, object]] = {
    RECORDED_LABEL: _BOUNDARY,
    REGENERATION_LABEL: {
        **_BOUNDARY,
        "compiled_on_this_host": True,
        "statement": _REGENERATION_STATEMENT,
    },
    SYNTHETIC_LABEL: {**_BOUNDARY, "statement": _SYNTHETIC_STATEMENT},
}


def _load(load_average: Callable[[], tuple[float, float, float]]) -> list[float]:
    values = [float(value) for value in load_average()]
    if len(values) != 3 or not all(
        math.isfinite(value) and value >= 0 for value in values
    ):
        raise TritonIrError("a load average must be three finite values")
    return values


def analyze(directory: Path) -> dict[str, object]:
    """Return every section that the fixture folder fixes, with no clock."""

    if not isinstance(directory, Path):
        raise TypeError("fixture folder must be pathlib.Path")
    payload, manifest = read_manifest(directory)
    require_recorded(payload)
    check_source(manifest)
    texts = check_fixtures(manifest, directory)
    record = manifest.record(COMMITTED_TARGET, COMMITTED_CONFIG)
    ttir = read_mlir(texts["ttir"].text)
    ttgir = read_mlir(texts["ttgir"].text)
    ptx = read_ptx(texts["ptx"].text)
    return {
        "manifest": manifest.to_mapping(),
        "manifest_sha256": sha256(payload).hexdigest(),
        "source": {"kernel": KERNEL_NAME, "sha256": manifest.source_sha256},
        "fixtures": [text.to_mapping() for text in texts.values()],
        "ptx": ptx.to_mapping(),
        "ttir": ttir_summary(ttir),
        "ttgir": ttgir_summary(ttgir),
        "stage_map": stage_map(ttir, ttgir, ptx),
        "pipeline": pipeline_facts(record, ttgir, ptx),
        "comparison": compare(manifest, texts["ttgir"].text, ptx),
        "_manifest": manifest,
    }


def summarize(sections: Mapping[str, object]) -> dict[str, object]:
    """Name the counts a reader checks first, from the other sections."""

    families = sections["ptx"]["families"]
    pipeline = sections["pipeline"]
    comparison = sections["comparison"]
    regeneration = sections["regeneration"]
    return {
        "source_sha256": sections["source"]["sha256"],
        "fixtures": len(sections["fixtures"]),
        "ptx_target": sections["ptx"]["target"],
        "ptx_instructions": sections["ptx"]["instructions"],
        "fma_rn_f32": families["fma.rn.f32"],
        "cp_async_all": sum(
            count for name, count in families.items() if name.startswith("cp.async")
        ),
        "mma": families["mma"],
        "ld_global": families["ld.global"],
        "fma_in_k_loop": pipeline["fma_in_k_loop"],
        "ring_buffers": pipeline["buffers_per_operand"],
        "shared_bytes": pipeline["shared_bytes"],
        "ttir_ops": sections["ttir"]["ops"],
        "ttgir_ops": sections["ttgir"]["ops"],
        "compiles": len(comparison["compiles"]),
        "rings_of_stages_minus_one": sum(
            row["ring_is_stages_minus_one"] for row in comparison["compiles"]
        ),
        "ttir_identical_across_targets": sum(
            row["identical"]["ttir"] for row in comparison["targets"]
        ),
        "regeneration": None if regeneration is None else regeneration["summary"],
    }


def build_report(
    directory: Path,
    *,
    clock: Callable[[], float],
    load_average: Callable[[], tuple[float, float, float]],
    backend: object | None = None,
) -> dict[str, object]:
    """Build the unsealed report; with a backend, compile and compare first."""

    compiler = type(backend) is CompilerBackend
    if compiler and (clock is not time.time or load_average is not os.getloadavg):
        raise TritonIrError(
            "a triton regeneration records the host clock and the host load "
            "averages, so it takes no injected clock or load average"
        )
    sections = analyze(directory)
    manifest = sections.pop("_manifest")
    recorded_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(clock()))
    before = _load(load_average)
    regeneration = None if backend is None else regenerate(manifest, backend)
    after = _load(load_average)
    if backend is None:
        label = RECORDED_LABEL
    else:
        label = REGENERATION_LABEL if compiler else SYNTHETIC_LABEL
    sections["regeneration"] = regeneration
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
        raise TritonIrError("report is already sealed")
    return {**report, DIGEST_FIELD: report_digest(report)}
