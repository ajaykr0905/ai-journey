"""Tests of the sealed Day 80 Triton IR report: sections, labels and seal.

No test imports triton. A regeneration report comes from a stand-in backend
here, so it is labeled synthetic.
"""

import os
import shutil
import time
from pathlib import Path

import pytest

from ai_journey.kernel_launch_io import canonical_sha256
from ai_journey.triton_ir_manifest import TritonIrError
from ai_journey.triton_ir_report import (
    BOUNDARIES,
    DIGEST_FIELD,
    RECORDED_LABEL,
    REPORT_KIND,
    SYNTHETIC_LABEL,
    build_report,
    report_digest,
    seal_report,
)
from ai_journey.triton_ir_runtime import CompilerBackend
from test_triton_ir_runtime import FixtureBackend

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "day80_triton_ir"
FIELDS = {
    "schema_version",
    "kind",
    "label",
    "evidence_boundary",
    "manifest",
    "manifest_sha256",
    "source",
    "fixtures",
    "ptx",
    "ttir",
    "ttgir",
    "stage_map",
    "pipeline",
    "comparison",
    "regeneration",
    "recorded_at",
    "host_load",
    "summary",
}


def recorded():
    return build_report(
        FIXTURES, clock=lambda: 1795046400.0, load_average=lambda: (1.5, 2.0, 2.5)
    )


def test_a_recorded_report_holds_every_section_and_names_its_boundary():
    report = recorded()
    assert set(report) == FIELDS
    assert report["kind"] == REPORT_KIND
    assert report["label"] == RECORDED_LABEL
    boundary = report["evidence_boundary"]
    assert boundary == dict(BOUNDARIES[RECORDED_LABEL])
    assert boundary["nvidia_gpu_used"] is False
    assert boundary["ptx_executed"] is False
    assert boundary["ptxas_register_report"] is False
    assert boundary["compiled_on_this_host"] is False
    assert report["regeneration"] is None
    assert report["recorded_at"] == "2026-11-19T00:00:00Z"
    assert report["host_load"] == {"before": [1.5, 2.0, 2.5], "after": [1.5, 2.0, 2.5]}


def test_the_summary_of_the_recorded_report():
    assert recorded()["summary"] == {
        "source_sha256": (
            "9eb35e273b0fd229e83e223861ef7ce13c4f15ea46d2694baf13920851a31d56"
        ),
        "fixtures": 3,
        "ptx_target": "sm_80",
        "ptx_instructions": 3097,
        "fma_rn_f32": 1024,
        "cp_async_all": 252,
        "mma": 0,
        "ld_global": 0,
        "fma_in_k_loop": 1024,
        "ring_buffers": 4,
        "shared_bytes": 49152,
        "ttir_ops": 97,
        "ttgir_ops": 220,
        "compiles": 12,
        "rings_of_stages_minus_one": 12,
        "ttir_identical_across_targets": 6,
        "regeneration": None,
    }


def test_the_report_is_the_same_for_the_same_inputs():
    assert canonical_sha256(recorded()) == canonical_sha256(recorded())


def test_a_stand_in_backend_gives_a_synthetic_regeneration():
    report = build_report(
        FIXTURES,
        clock=lambda: 0.0,
        load_average=lambda: (0.0, 0.0, 0.0),
        backend=FixtureBackend(),
    )
    assert report["label"] == SYNTHETIC_LABEL
    assert report["evidence_boundary"]["compiled_on_this_host"] is False
    assert report["regeneration"]["backend"] == "injected"
    assert report["summary"]["regeneration"]["texts_matched"] == 3


def test_a_triton_regeneration_takes_only_the_host_clock(monkeypatch):
    backend = object.__new__(CompilerBackend)
    with pytest.raises(TritonIrError, match="host clock"):
        build_report(
            FIXTURES, clock=lambda: 0.0, load_average=os.getloadavg, backend=backend
        )
    with pytest.raises(TritonIrError, match="host clock"):
        build_report(
            FIXTURES, clock=time.time, load_average=lambda: (0, 0, 0), backend=backend
        )


def test_a_bad_load_average_or_folder_is_refused(tmp_path):
    with pytest.raises(TritonIrError, match="three finite"):
        build_report(FIXTURES, clock=lambda: 0.0, load_average=lambda: (1.0, 2.0))
    with pytest.raises(TritonIrError, match="three finite"):
        build_report(
            FIXTURES, clock=lambda: 0.0, load_average=lambda: (1.0, float("nan"), 1.0)
        )
    with pytest.raises(TypeError, match="pathlib.Path"):
        build_report(str(FIXTURES), clock=lambda: 0.0, load_average=lambda: (0, 0, 0))
    folder = tmp_path / "other"
    shutil.copytree(FIXTURES, folder)
    manifest = folder / "index.json"
    manifest.write_bytes(manifest.read_bytes() + b"\n")
    with pytest.raises(TritonIrError, match="not the recorded"):
        build_report(folder, clock=lambda: 0.0, load_average=lambda: (0, 0, 0))


def test_the_seal_covers_every_other_field():
    report = recorded()
    sealed = seal_report(report)
    assert sealed[DIGEST_FIELD] == report_digest(report) == canonical_sha256(report)
    assert report_digest(sealed) == sealed[DIGEST_FIELD]
    with pytest.raises(TritonIrError, match="already sealed"):
        seal_report(sealed)
    changed = {**sealed, "label": SYNTHETIC_LABEL}
    assert report_digest(changed) != sealed[DIGEST_FIELD]
    with pytest.raises(TypeError):
        seal_report([])
