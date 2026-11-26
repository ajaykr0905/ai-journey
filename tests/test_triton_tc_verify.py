"""Tamper tests of the independent Day 82 report verifier.

Each test changes one field of a sealed report and seals it again, so only
the recomputation can catch the change. The compile reports come from the
SYNTHETIC backend of ``test_triton_tc_report``: hand-built texts, not
compiler output. ``DottedSyntheticBackend`` adds one hand-written ``tt.dot``
line, so the dot fields of a row are in the tests too. Two tests read the
real Day 80 fixture to show that the verifier's own recount rebuilds the
lab's record of compiler output. No test imports triton.
"""

import copy
import dataclasses
import json
from hashlib import sha256
from pathlib import Path

import pytest

from ai_journey.kernel_launch_io import canonical_json_bytes
from ai_journey.triton_tc_compile import (
    CompiledVariant,
    compile_all,
    stage_path,
    write_dump,
)
from ai_journey.triton_tc_ptx import analyze_ptx
from ai_journey.triton_tc_report import DIGEST_FIELD, seal_report
from ai_journey.triton_tc_rule import check_compile
from ai_journey.triton_tc_verify import (
    TritonTcVerificationError,
    load_report,
    recount_ptx,
    recount_ttgir,
    verify_report,
)
from test_triton_tc_ptx import SYNTHETIC_COMMENTS
from test_triton_tc_report import SyntheticTcBackend, synthetic_texts, unsealed
from test_triton_tc_ttgir import (
    SYNTHETIC_AMPERE_BF16,
    SYNTHETIC_BLOCKED_FP32,
    SYNTHETIC_HOPPER_TF32,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "day-82-triton-tc.json"
DAY80 = ROOT / "tests/fixtures/day80_triton_ir/sm_80/m32_n64_k32_g8_w2_s5"
DAY80_PTX = DAY80 / "ptx.ptx"
# One SYNTHETIC tt.dot line, written by hand for the bf16 sm_80 TTGIR.
DOT_LINE = (
    "    %z = tt.dot %x, %y, %acc : tensor<128x32xbf16, #ttg.dot_op<{opIdx = 0, "
    "parent = #mma, kWidth = 8}>> * tensor<32x128xbf16, #ttg.dot_op<{opIdx = 1, "
    "parent = #mma, kWidth = 8}>> -> tensor<128x128xf32, #mma>"
)


class DottedSyntheticBackend(SyntheticTcBackend):
    """The SYNTHETIC texts, with ``DOT_LINE`` in the bf16 sm_80 TTGIR."""

    def compile(self, variant, target):
        item = super().compile(variant, target)
        if item.key != ("bf16", "sm_80"):
            return item
        ttgir = item.ttgir.replace("    tt.return", DOT_LINE + "\n    tt.return")
        return dataclasses.replace(item, ttgir=ttgir)


MODEL = seal_report(unsealed())
SYNTHETIC = seal_report(unsealed(SyntheticTcBackend()))
DOTTED = seal_report(unsealed(DottedSyntheticBackend()))


def reseal(report, edit):
    body = copy.deepcopy({k: v for k, v in report.items() if k != DIGEST_FIELD})
    edit(body)
    return seal_report(body)


def dump(folder, backend_type=SyntheticTcBackend):
    backend = backend_type()
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
        (lambda r: r["kernel"].update(sha256="1" * 64), "kernel.sha256"),
        (lambda r: r["kernel"]["config"].update(num_stages=4), "kernel.config"),
        (
            lambda r: r["kernel"]["variants"][2].update(input_precision="tf32"),
            r"kernel.variants\[2\]",
        ),
        (lambda r: r["kernel"].update(plan_file="other.py"), "kernel.plan_file"),
        (lambda r: r["kernel"].update(text="import triton\n"), "kernel.text"),
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


def test_the_verifier_recount_reads_synthetic_comments_and_labels_as_the_lab():
    """SYNTHETIC text: C comment rules, a string literal and a lowercase label."""

    lab = analyze_ptx(SYNTHETIC_COMMENTS).to_mapping()
    recount = recount_ptx(SYNTHETIC_COMMENTS)

    assert recount["classes"] == lab["classes"]
    assert recount["instructions"] == lab["instructions"] == 4
    assert (recount["target"], recount["version"]) == ("sm_80", "8.8")
    for tail in ("/* never closed\n", "*/\n"):
        with pytest.raises(TritonTcVerificationError, match="block comment"):
            recount_ptx(SYNTHETIC_COMMENTS + tail)


def ttgir(report, variant="bf16", target="sm_80"):
    return row(report, variant, target)["ttgir"]


@pytest.mark.parametrize(
    "edit",
    [
        lambda r: ttgir(r)["mma"][0]["parameters"].update(instrShape=[16, 16]),
        lambda r: ttgir(r)["mma"][0]["parameters"].update(warpsPerCTA=[2, 2]),
        lambda r: ttgir(r)["ops"].update({"ttg.local_load": 2}),
        lambda r: ttgir(r)["dot_operands"].append(
            {
                "alias": None,
                "kind": "dot_op",
                "parameters": {"opIdx": 0},
                "parent_kind": None,
            }
        ),
        lambda r: ttgir(r)["shared"][0]["parameters"].update(vec=4),
        lambda r: row(r, "bf16", "sm_80")["ptx"].update(version="9.0"),
        lambda r: row(r, "bf16", "sm_80")["ptx"].update(
            ldmatrix_forms={"ldmatrix.sync.aligned.m8n8.x2.shared.b16": 1}
        ),
    ],
)
def test_a_text_field_edit_is_caught_only_with_the_dump(edit, tmp_path):
    """Without the texts the row agrees with itself; the recount finds it."""

    report = reseal(SYNTHETIC, edit)
    verify_report(report, CONFIG)
    with pytest.raises(TritonTcVerificationError, match=r"dump\[bf16@sm_80\]"):
        verify_report(report, CONFIG, dump(tmp_path))


@pytest.mark.parametrize(
    ("field", "value"),
    [("layouts", ["col", "row"]), ("c_type", "f16"), ("shape", "m16n8k8")],
)
def test_an_mma_form_must_agree_with_the_opcode_grammar(field, value):
    report = reseal(SYNTHETIC, lambda r: mma_form(r).update({field: value}))
    with pytest.raises(TritonTcVerificationError, match="mma_forms"):
        verify_report(report, CONFIG)


def bf16(report):
    return row(report, "bf16", "sm_80")


def test_the_dotted_synthetic_report_verifies_with_its_dump(tmp_path):
    record = bf16(DOTTED)["ttgir"]

    assert record["ops"]["tt.dot"] == len(record["dots"]) == 1
    assert record["dots"][0]["result_kind"] == "nvidia_mma"
    assert [item["parent_kind"] for item in record["dot_operands"]] == [
        "nvidia_mma",
        "nvidia_mma",
    ]
    verify_report(DOTTED, CONFIG, dump(tmp_path, DottedSyntheticBackend))


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda r: bf16(r).update(gpu_time_us=1.0), "fields do not match"),
        (lambda r: bf16(r)["ptx"].update(measured=True), "fields do not match"),
        (lambda r: bf16(r)["ttgir"].update(occupancy=1), "fields do not match"),
        (lambda r: bf16(r)["stages"]["ptx"].update(run=1), "fields do not match"),
        (lambda r: bf16(r)["ptx"]["classes"].update(hmma=1), "classes"),
        (lambda r: bf16(r)["ttgir"]["ops"].update({"tt.load": 2}), "ops"),
        (lambda r: mma_form(r).update(extra="x"), "mma_forms"),
        (lambda r: mma_form(r).update(m=32), "mma_forms"),
        (lambda r: mma_form(r).update(n=16), "mma_forms"),
        (lambda r: mma_form(r).update(opcode="mma.sync.aligned.x"), "mma_forms"),
        (lambda r: bf16(r)["ptx"]["ldmatrix_forms"].clear(), "ldmatrix"),
        (
            lambda r: bf16(r)["ttgir"]["buffers"][0].update(element_bits=8),
            "element_bits",
        ),
        (
            lambda r: bf16(r)["ttgir"]["mma"][0]["parameters"].update(versionMajor=3),
            "mma_versions",
        ),
        (lambda r: bf16(r)["ttgir"]["dots"].clear(), "dots"),
        (lambda r: bf16(r)["ttgir"]["buffers"].clear(), "buffers"),
        (lambda r: bf16(r)["ttgir"]["dot_operands"][0].update(kind="blocked"), "dot"),
        (lambda r: bf16(r)["ttgir"]["shared"][0].update(kind="blocked"), "shared"),
    ],
)
def test_a_row_field_outside_its_record_is_caught_without_the_dump(edit, message):
    with pytest.raises(TritonTcVerificationError, match=message):
        verify_report(reseal(DOTTED, edit), CONFIG)


@pytest.mark.parametrize(
    "edit",
    [
        lambda r: bf16(r)["ttgir"].update(num_warps=8),
        lambda r: bf16(r)["ttgir"]["dots"][0].update(input_precision="tf32"),
        lambda r: bf16(r)["ttgir"]["dots"][0].update(result_kind="blocked"),
        lambda r: bf16(r)["ttgir"]["dots"][0].update(line=1),
        lambda r: bf16(r)["ttgir"]["buffers"][0].update(encoding="#shared9"),
        lambda r: bf16(r)["ttgir"]["buffers"][0].update(memory=None),
        lambda r: bf16(r)["ttgir"]["buffers"][0].update(line=1),
        lambda r: bf16(r)["ttgir"]["mma"][0].update(kind="mma"),
        lambda r: bf16(r)["ttgir"]["mma"][0]["parameters"].update(CTAOrder=[0, 1]),
        lambda r: bf16(r)["ttgir"]["shared"][0]["parameters"].update(order=[0, 1]),
        lambda r: bf16(r)["ttgir"]["shared"][0].update(kind="nvmma_shared"),
        lambda r: bf16(r)["ttgir"]["dot_operands"][0].update(parent_kind="blocked"),
        lambda r: bf16(r)["ttgir"]["dot_operands"][1]["parameters"].update(kWidth=4),
    ],
)
def test_every_text_field_of_a_row_is_rebuilt_from_the_dump(edit, tmp_path):
    """Without the texts the row agrees with itself; the rebuild finds it."""

    report = reseal(DOTTED, edit)
    verify_report(report, CONFIG)
    with pytest.raises(TritonTcVerificationError, match=r"dump\[bf16@sm_80\]"):
        verify_report(report, CONFIG, dump(tmp_path, DottedSyntheticBackend))


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda index: index.update(kernel_sha256="3" * 64), "kernel_sha256"),
        (lambda index: index["config"].update(num_stages=4), "config"),
        (lambda index: index["environment"].update(triton="3.8.0"), "environment"),
        (lambda index: index["compiles"][6].update(num_warps=8), "num_warps"),
        (lambda index: index.update(measured=True), "fields do not match"),
    ],
)
def test_the_dump_index_must_name_the_text_config_and_environment_of_the_report(
    edit, message, tmp_path
):
    folder = dump(tmp_path)
    index = json.loads((folder / "index.json").read_text("ascii"))
    edit(index)
    (folder / "index.json").write_text(json.dumps(index), encoding="ascii")
    with pytest.raises(TritonTcVerificationError, match=message):
        verify_report(SYNTHETIC, CONFIG, folder)


def test_a_consistent_change_of_a_dumped_text_size_is_caught_by_its_text(tmp_path):
    folder = dump(tmp_path)
    index = json.loads((folder / "index.json").read_text("ascii"))
    index["compiles"][6]["stages"]["ptx"]["lines"] += 1
    (folder / "index.json").write_text(json.dumps(index), encoding="ascii")
    report = reseal(
        SYNTHETIC,
        lambda r: bf16(r)["stages"]["ptx"].update(
            lines=index["compiles"][6]["stages"]["ptx"]["lines"]
        ),
    )
    verify_report(report, CONFIG)
    with pytest.raises(TritonTcVerificationError, match="lines and bytes"):
        verify_report(report, CONFIG, folder)


def lab_record(variant, target, ttgir_text, ptx_text):
    return check_compile(
        CompiledVariant(variant, target, ttgir_text, ptx_text, 65536, 4, 3)
    )


def test_the_verifier_rebuilds_the_lab_record_of_real_compiler_output():
    """The Day 80 fixture is real triton 3.9.0 output of the Day 79 kernel."""

    ttgir_text = (DAY80 / "ttgir.mlir").read_text(encoding="ascii")
    ptx_text = DAY80_PTX.read_text(encoding="ascii")
    record = lab_record("fp32-ieee", "sm_80", ttgir_text, ptx_text)

    assert recount_ptx(ptx_text) == record["ptx"]
    assert recount_ttgir(ttgir_text) == record["ttgir"]
    assert record["ttgir"]["dots"][0]["result_kind"] == "blocked"


@pytest.mark.parametrize(
    ("variant", "target", "text"),
    [
        ("bf16", "sm_80", SYNTHETIC_AMPERE_BF16),
        ("fp32-tf32", "sm_90", SYNTHETIC_HOPPER_TF32),
        ("fp32-ieee", "sm_80", SYNTHETIC_BLOCKED_FP32),
    ],
    ids=["ampere-bf16", "hopper-tf32", "blocked-fp32"],
)
def test_the_verifier_rebuilds_the_lab_record_of_synthetic_modules(
    variant, target, text
):
    """SYNTHETIC modules: MMA v2 and v3, nvmma_shared and warp_group_dot."""

    ptx_text = synthetic_texts(variant, target)[1]

    assert recount_ttgir(text) == lab_record(variant, target, text, ptx_text)["ttgir"]
