"""Tests of the sealed Day 82 report, with a SYNTHETIC compile backend.

``SyntheticTcBackend`` writes PTX and TTGIR texts by hand for each variant
and target: a header and one tile product of the expected instruction. No
compiler wrote them, and no test presents them as compiler output. It picks
``mma.sync`` for fp32-tf32 on sm_90 on purpose, so the record-not-fail path
of the rule runs. No test imports triton.
"""

from pathlib import Path

import pytest

from ai_journey.kernel_launch_io import canonical_sha256
from ai_journey.triton_tc_compile import (
    COMPILE_ORDER,
    ENVIRONMENT_FIELDS,
    CompiledVariant,
    CompilerBackend,
)
from ai_journey.triton_tc_kernel import PINNED_SHA256, TritonTcError
from ai_journey.triton_tc_report import (
    BOUNDARIES,
    DIGEST_FIELD,
    MODEL_LABEL,
    SYNTHETIC_LABEL,
    build_report,
    report_digest,
    seal_report,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "day-82-triton-tc.json"
OPERAND = {"fp32-tf32": "tf32", "fp16": "f16", "bf16": "bf16"}
ELEMENT = {"fp32-ieee": "f32", "fp32-tf32": "f32", "fp16": "f16", "bf16": "bf16"}


def synthetic_texts(variant, target):
    """Return SYNTHETIC hand-built (ttgir, ptx) texts with one tile product."""

    capability = target.removeprefix("sm_")
    if variant == "fp32-ieee":
        lines, version = ["fma.rn.f32 %f1, %f2, %f3, %f1;"] * 4096, None
    else:
        kind = OPERAND[variant]
        k = 8 if kind == "tf32" else 16
        if target == "sm_90" and variant != "fp32-tf32":
            opcode = f"wgmma.mma_async.sync.aligned.m64n128k{k}.f32.{kind}.{kind}"
            lines, version = [f"{opcode} x;"] * (64 // k), 3
        else:
            opcode = f"mma.sync.aligned.m16n8k{k}.row.col.f32.{kind}.{kind}.f32"
            lines, version = [f"{opcode} x;"] * (1024 // k), 2
    ptx = "\n".join(
        [
            "// SYNTHETIC PTX written by hand for a test. Not compiler output.",
            ".version 8.8",
            ".target " + ("sm_90a" if target == "sm_90" else "sm_80"),
            ".address_size 64",
            "{",
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r1, %r2, %r3, %r4}, [%r5];",
            *lines,
            "ret;",
            "}",
            "",
        ]
    )
    encodings = [
        "#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, "
        "order = [1, 0]}>",
        "#smem = #ttg.shared_memory",
    ]
    if version is not None:
        encodings.insert(
            0,
            f"#mma = #ttg.nvidia_mma<{{versionMajor = {version}, versionMinor = 0, "
            "warpsPerCTA = [4, 1], instrShape = [16, 8]}>",
        )
    ttgir = "\n".join(
        [
            *encodings,
            'module attributes {"ttg.num-warps" = 4 : i32, '
            f'ttg.target = "cuda:{capability}", "ttg.threads-per-warp" = 32 : i32}} {{',
            "  tt.func public @tc_matmul_kernel() {",
            "    %a = ttg.local_alloc : () -> "
            f"!ttg.memdesc<2x128x32x{ELEMENT[variant]}, #shared, #smem, mutable>",
            "    tt.return",
            "  }",
            "}",
            "",
        ]
    )
    return ttgir, ptx


class SyntheticTcBackend:
    """Return SYNTHETIC compiles for every variant, and record the requests."""

    name = "synthetic"
    environment = dict.fromkeys(ENVIRONMENT_FIELDS, "synthetic")

    def __init__(self):
        self.calls = []

    def compile(self, variant, target):
        self.calls.append((variant.name, target))
        ttgir, ptx = synthetic_texts(variant.name, target)
        return CompiledVariant(variant.name, target, ttgir, ptx, 65536, 4, 3)


def unsealed(backend=None):
    return build_report(
        CONFIG,
        clock=lambda: 0.0,
        load_average=lambda: (1.0, 2.0, 3.0),
        backend=backend,
    )


def test_a_model_report_holds_the_model_and_no_compile():
    report = unsealed()

    assert report["label"] == MODEL_LABEL
    assert report["evidence_boundary"] == BOUNDARIES[MODEL_LABEL]
    assert report["evidence_boundary"]["nvidia_gpu_used"] is False
    assert report["evidence_boundary"]["model_is_measurement"] is False
    assert report["compile"] is None
    assert report["kernel"]["sha256"] == PINNED_SHA256
    assert report["recorded_at"] == "1970-01-01T00:00:00Z"
    assert report["host_load"] == {"before": [1.0, 2.0, 3.0], "after": [1.0, 2.0, 3.0]}
    summary = report["summary"]
    assert summary["compile"] is None
    assert summary["memory_bound"] == {
        "a100-sxm4-80gb": {"512": ["fp16", "bf16"]},
        "h100-sxm5-80gb": {
            "512": ["tf32", "fp16", "bf16"],
            "1024": ["fp16", "bf16"],
        },
    }
    assert summary["largest_size_ratio"]["a100-sxm4-80gb"]["fp32"]["decimal"] == (
        "16.000000"
    )


def test_a_synthetic_compile_report_checks_all_eight_compiles():
    backend = SyntheticTcBackend()
    report = unsealed(backend)
    section = report["compile"]

    assert backend.calls == list(COMPILE_ORDER)
    assert report["label"] == SYNTHETIC_LABEL
    assert report["evidence_boundary"]["compiled_on_this_host"] is False
    assert section["backend"] == "injected"
    assert section["summary"]["all_accepted"] is True
    assert section["summary"]["mma_sync_on_sm_90"] == ["fp32-tf32"]
    assert section["summary"]["observed"]["bf16@sm_90"] == "wgmma"
    row = section["rows"][0]
    assert (row["variant"], row["target"], row["observed"]) == (
        "fp32-ieee",
        "sm_80",
        "scalar_fma",
    )
    assert set(row["stages"]) == {"ttgir", "ptx"}
    assert row["stages"]["ptx"]["file"] == "fp32-ieee/sm_80/ptx.ptx"


def test_the_seal_covers_every_field_but_itself():
    report = seal_report(unsealed())

    assert report[DIGEST_FIELD] == report_digest(report)
    assert report[DIGEST_FIELD] == canonical_sha256(
        {key: value for key, value in report.items() if key != DIGEST_FIELD}
    )
    with pytest.raises(TritonTcError, match="already sealed"):
        seal_report(report)
    with pytest.raises(TypeError, match="pathlib.Path"):
        build_report(str(CONFIG), clock=lambda: 0.0, load_average=lambda: (0, 0, 0))


def test_a_triton_compile_takes_only_the_host_clock_and_load():
    backend = CompilerBackend.__new__(CompilerBackend)
    with pytest.raises(TritonTcError, match="no injected clock"):
        unsealed(backend)
