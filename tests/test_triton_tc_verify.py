"""Tamper tests of the independent Day 82 report verifier.

Each test changes one field of a sealed report and seals it again, so only
the recomputation can catch the change. The compile reports come from the
SYNTHETIC backend of ``test_triton_tc_report``: hand-built texts, not
compiler output. One test reads the real Day 80 PTX fixture to show that
the verifier's own recount agrees with the lab analyzer on compiler output.
No test imports triton.
"""

import copy
import json
from hashlib import sha256
from pathlib import Path

import pytest

from ai_journey.kernel_launch_io import canonical_json_bytes
from ai_journey.triton_tc_compile import compile_all, stage_path, write_dump
from ai_journey.triton_tc_ptx import analyze_ptx
from ai_journey.triton_tc_report import DIGEST_FIELD, seal_report
from ai_journey.triton_tc_verify import (
    TritonTcVerificationError,
    load_report,
    recount_ptx,
    verify_report,
)
from test_triton_tc_report import SyntheticTcBackend, unsealed

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "day-82-triton-tc.json"
DAY80_PTX = ROOT / "tests/fixtures/day80_triton_ir/sm_80/m32_n64_k32_g8_w2_s5/ptx.ptx"
MODEL = seal_report(unsealed())
SYNTHETIC = seal_report(unsealed(SyntheticTcBackend()))


def reseal(report, edit):
    body = copy.deepcopy({k: v for k, v in report.items() if k != DIGEST_FIELD})
    edit(body)
    return seal_report(body)


def dump(folder):
    backend = SyntheticTcBackend()
    write_dump(folder, compile_all(backend), backend.environment)
    return folder


def row(report, variant, target):
    (found,) = [
        item
        for item in report["compile"]["rows"]
        if (item["variant"], item["target"]) == (variant, target)
    ]
    return found


def a100_point(report, size, path):
    device = report["model"]["devices"][0]
    (item,) = [entry for entry in device["rows"] if entry["size"] == size]
    return item["points"][path]


def test_a_model_report_and_a_synthetic_compile_report_verify(tmp_path):
    verify_report(MODEL, CONFIG)
    verify_report(SYNTHETIC, CONFIG)
    verify_report(SYNTHETIC, CONFIG, dump(tmp_path))
    path = tmp_path / "report.json"
    path.write_bytes(canonical_json_bytes(SYNTHETIC) + b"\n")
    assert load_report(path, CONFIG, tmp_path)["label"] == "synthetic_compile"


def test_an_unsealed_change_breaks_the_seal():
    report = copy.deepcopy(MODEL)
    report["summary"]["config_sha256"] = "0" * 64
    with pytest.raises(TritonTcVerificationError, match="digest does not match"):
        verify_report(report, CONFIG)


@pytest.mark.parametrize(
    ("edit", "path"),
    [
        (lambda r: r["evidence_boundary"].update(nvidia_gpu_used=True), "evidence"),
        (lambda r: r["evidence_boundary"].update(model_is_measurement=True), "evid"),
        (lambda r: r.update(label="triton_compiler_output"), "evidence_boundary"),
        (lambda r: r["kernel"].update(sha256="1" * 64), "kernel section"),
        (lambda r: r["kernel"]["config"].update(num_stages=4), "kernel section"),
        (
            lambda r: r["config"]["devices"][1][
                "memory_bandwidth_gb_per_second"
            ].update(value="3900"),
            "config",
        ),
        (lambda r: r.update(config_sha256="2" * 64), "config_sha256"),
        (
            lambda r: a100_point(r, 512, "bf16").update(bound="compute"),
            "model.devices",
        ),
        (
            lambda r: a100_point(r, 512, "fp32")["time_us"].update(decimal="13.7"),
            "model.devices",
        ),
        (
            lambda r: r["model"]["devices"][1]["per_clock_ratio"]["fp32"].update(
                numerator=15
            ),
            "model.devices",
        ),
        (lambda r: r["model"]["rule"].pop(), "model.rule"),
        (lambda r: r["summary"]["memory_bound"].update({"a100-sxm4-80gb": {}}), "summ"),
        (lambda r: r.update(recorded_at="2026-11-26"), "recorded_at"),
        (lambda r: r["host_load"].update(after=[1.0, 2.0]), "host_load"),
    ],
)
def test_a_model_report_edit_is_caught_by_recomputation(edit, path):
    with pytest.raises(TritonTcVerificationError, match=path):
        verify_report(reseal(MODEL, edit), CONFIG)


def mma_form(report):
    return row(report, "bf16", "sm_80")["ptx"]["mma_forms"][0]


@pytest.mark.parametrize(
    ("edit", "path"),
    [
        (lambda r: row(r, "bf16", "sm_90").update(observed="mma_sync"), "rows"),
        (lambda r: row(r, "fp16", "sm_80")["checks"].update({"class": False}), "rows"),
        (lambda r: row(r, "fp16", "sm_80").update(accepted=False), "rows"),
        (lambda r: row(r, "fp32-tf32", "sm_90").update(note=None), "rows"),
        (lambda r: mma_form(r).update(a_type="f16"), "mma_forms"),
        (lambda r: mma_form(r).update(count=63), "count its MMA forms"),
        (
            lambda r: row(r, "bf16", "sm_80")["ptx"]["classes"].update(fma_rn_f32=1),
            "rows",
        ),
        (lambda r: row(r, "bf16", "sm_80").update(shared_bytes=1024), "rows"),
        (
            lambda r: row(r, "bf16", "sm_80")["ttgir"]["buffers"][0].update(bytes=1),
            "buffers bytes",
        ),
        (lambda r: row(r, "bf16", "sm_80")["ttgir"].update(mma_versions=[3]), "rows"),
        (lambda r: r["compile"]["summary"].update(accepted=7), "compile.summary"),
        (lambda r: r["compile"].update(backend="triton_compiler"), "compile.backend"),
        (lambda r: r["compile"]["rows"].reverse(), "order"),
        (lambda r: r.update(compile=None), "compile"),
    ],
)
def test_a_compile_report_edit_is_caught_by_the_rule_recount(edit, path):
    with pytest.raises(TritonTcVerificationError, match=path):
        verify_report(reseal(SYNTHETIC, edit), CONFIG)


def test_a_consistent_change_to_one_dumped_text_is_caught_by_the_text_recount(
    tmp_path,
):
    folder = dump(tmp_path)
    text = folder / stage_path("bf16", "sm_80", "ptx")
    changed = text.read_text("ascii").replace(
        "ret;", "fma.rn.f32 %f1, %f1, %f1, %f1;\nret;"
    )
    text.write_text(changed, encoding="ascii")
    index = json.loads((folder / "index.json").read_text("ascii"))
    record = index["compiles"][6]["stages"]["ptx"]
    record.update(
        sha256=sha256(changed.encode()).hexdigest(),
        lines=len(changed.splitlines()),
        bytes=len(changed),
    )
    (folder / "index.json").write_text(json.dumps(index), encoding="ascii")

    def follow(report):
        row(report, "bf16", "sm_80")["stages"]["ptx"] = dict(record)

    report = reseal(SYNTHETIC, follow)
    verify_report(report, CONFIG)
    with pytest.raises(TritonTcVerificationError, match=r"dump\[bf16@sm_80\]\.ptx"):
        verify_report(report, CONFIG, folder)


def test_a_dumped_text_that_differs_from_its_digest_is_refused(tmp_path):
    folder = dump(tmp_path)
    text = folder / stage_path("fp16", "sm_90", "ttgir")
    text.write_text(text.read_text("ascii") + "\n", encoding="ascii")
    with pytest.raises(TritonTcVerificationError, match="does not match its digest"):
        verify_report(SYNTHETIC, CONFIG, folder)
    with pytest.raises(TritonTcVerificationError, match="no dump folder"):
        verify_report(MODEL, CONFIG, folder)


def test_the_verifier_recount_agrees_with_the_lab_on_real_compiler_output():
    """The Day 80 fixture is real triton 3.9.0 PTX of the Day 79 kernel."""

    text = DAY80_PTX.read_text(encoding="ascii")
    lab = analyze_ptx(text).to_mapping()
    recount = recount_ptx(text)

    assert recount["classes"] == lab["classes"]
    assert recount["instructions"] == lab["instructions"] == 3097
    assert (recount["target"], recount["version"]) == (lab["target"], lab["version"])
    assert recount["ldmatrix_forms"] == lab["ldmatrix_forms"]
