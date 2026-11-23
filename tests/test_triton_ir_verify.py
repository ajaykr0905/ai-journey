"""Tamper tests of the independent Day 80 report verifier.

Each test changes one field of a sealed report and seals it again, so only
the recomputation from the fixtures can catch the change. No test imports
triton.
"""

import copy
import shutil
from pathlib import Path

import pytest

from ai_journey.kernel_launch_io import canonical_json_bytes
from ai_journey.triton_ir_manifest import TritonIrError
from ai_journey.triton_ir_report import (
    DIGEST_FIELD,
    REGENERATION_LABEL,
    SYNTHETIC_LABEL,
    build_report,
    seal_report,
)
from ai_journey.triton_ir_verify import (
    TritonIrVerificationError,
    load_report,
    verify_report,
)
from test_triton_ir_runtime import FixtureBackend

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "day80_triton_ir"


def unsealed(backend=None):
    return build_report(
        FIXTURES,
        clock=lambda: 0.0,
        load_average=lambda: (1.0, 2.0, 3.0),
        backend=backend,
    )


RECORDED = seal_report(unsealed())
SYNTHETIC = seal_report(unsealed(FixtureBackend()))


def reseal(report, edit):
    body = copy.deepcopy({k: v for k, v in report.items() if k != DIGEST_FIELD})
    edit(body)
    return seal_report(body)


def test_both_labels_verify():
    verify_report(RECORDED, FIXTURES)
    verify_report(SYNTHETIC, FIXTURES)


def test_an_unsealed_change_breaks_the_seal():
    report = copy.deepcopy(RECORDED)
    report["summary"]["mma"] = 1
    with pytest.raises(TritonIrVerificationError, match="digest does not match"):
        verify_report(report, FIXTURES)


def statement(report, key):
    return next(row for row in report["stage_map"]["statements"] if row["key"] == key)


@pytest.mark.parametrize(
    ("edit", "path"),
    [
        (lambda r: r["evidence_boundary"].update(nvidia_gpu_used=True), "evidence"),
        (lambda r: r["evidence_boundary"].update(ptx_executed=True), "evidence"),
        (lambda r: r["manifest"].update(triton="3.8.0"), "manifest"),
        (lambda r: r.update(manifest_sha256="0" * 64), "manifest_sha256"),
        (lambda r: r["source"].update(sha256="1" * 64), "source"),
        (lambda r: r["fixtures"][2].update(lines=4540), "fixtures"),
        (lambda r: r["ptx"]["families"].update(mma=1), "ptx.families.mma"),
        (lambda r: r["ptx"]["families"].update(**{"fma.rn.f32": 1023}), "ptx.families"),
        (lambda r: r["ptx"]["opcodes"].pop("ret"), "ptx.opcodes"),
        (lambda r: r["ptx"].update(target="sm_90"), "ptx.target"),
        (lambda r: r["ptx"]["registers"][1].update(declared=128), "ptx.registers"),
        (lambda r: r["ptx"]["loops"][0].update(last_line=3870), "ptx.loops"),
        (lambda r: r["ptx"]["shared"][0].update(elements=49152), "ptx.shared"),
        (lambda r: r["ptx"].update(predicated=0), "ptx.predicated"),
        (lambda r: r["ttir"].update(tt_load_masked=0), "ttir.tt_load_masked"),
        (lambda r: r["ttir"].update(ops_with_source_line=97), "ttir.ops_with"),
        (lambda r: r["ttgir"].update(mma_encoding=True), "ttgir.mma_encoding"),
        (lambda r: r["ttgir"].update(async_wait=[4, 0]), "ttgir.async_wait"),
        (
            lambda r: r["ttgir"].update(local_alloc=["3x32x32", "3x32x64"]),
            "ttgir.local",
        ),
        (lambda r: r["ttgir"]["attributes"].update(**{"ttg.num-warps": "4"}), "ttgir"),
        (lambda r: statement(r, "dot")["ptx"].update(mma=1), "stage_map"),
        (
            lambda r: statement(r, "store")["ttgir"].pop("ttg.convert_layout"),
            "stage_map",
        ),
        (lambda r: statement(r, "a_load").update(line=41), "stage_map"),
        (lambda r: r["stage_map"]["lines"].pop(), "stage_map.lines"),
        (lambda r: r["pipeline"].update(buffers_per_operand=5), "pipeline"),
        (lambda r: r["pipeline"].update(wait_group_in_k_loop=[4]), "pipeline"),
        (lambda r: r["pipeline"].update(fma_matches_tile=False), "pipeline"),
        (lambda r: r["comparison"]["compiles"][0].update(shared_bytes=1), "comparison"),
        (
            lambda r: r["comparison"]["targets"][0]["identical"].update(ttir=False),
            "comparison.targets",
        ),
        (
            lambda r: r["comparison"]["targets"][5].update(
                ttgir_is_retargeted_sm_80=False
            ),
            "comparison.targets",
        ),
        (lambda r: r["comparison"]["needs_full_dumps"].pop(), "comparison.needs"),
        (lambda r: r["summary"].update(cp_async_all=240), "summary.cp_async_all"),
        (lambda r: r["summary"].update(regeneration={}), "summary.regeneration"),
    ],
)
def test_a_resealed_change_of_a_recorded_report_is_caught(edit, path):
    with pytest.raises(TritonIrVerificationError, match=path.replace(".", r"\.")):
        verify_report(reseal(RECORDED, edit), FIXTURES)


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda r: r.update(label=SYNTHETIC_LABEL), "evidence_boundary"),
        (lambda r: r.update(regeneration={"backend": "x"}), "holds no regeneration"),
        (lambda r: r.update(recorded_at="2026-11-23 10:00"), "UTC time"),
        (lambda r: r["host_load"].update(before=[1.0, 2.0]), "three finite"),
        (lambda r: r["host_load"].update(after=[1, 2, 3]), "three finite"),
        (lambda r: r.update(kind="triton_matmul_comparison"), "kind"),
        (lambda r: r.update(label="gpu_run"), "not a Day 80 label"),
        (lambda r: r.pop("pipeline"), "fields do not match"),
        (lambda r: r["ptx"].pop("loops"), "fields do not match"),
    ],
)
def test_a_label_time_or_schema_change_is_caught(edit, message):
    with pytest.raises(TritonIrVerificationError, match=message):
        verify_report(reseal(RECORDED, edit), FIXTURES)


def regeneration_row(report):
    return report["regeneration"]["compiles"][5]


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (
            lambda r: regeneration_row(r)["stages"]["llir"].update(match=True),
            "regeneration",
        ),
        (
            lambda r: regeneration_row(r)["stages"]["ptx"].update(
                recorded_sha256="0" * 64
            ),
            "regeneration",
        ),
        (lambda r: regeneration_row(r)["cubin"].update(match=True), "regeneration"),
        (lambda r: regeneration_row(r)["metadata"].update(match=False), "regeneration"),
        (lambda r: r["regeneration"].update(backend="triton_compiler"), "regeneration"),
        (lambda r: r["regeneration"]["compiles"].pop(), "one row for each"),
        (lambda r: r["regeneration"]["summary"].update(texts_matched=48), "summary"),
        (
            lambda r: regeneration_row(r)["stages"]["ttir"].update(
                regenerated_sha256="x"
            ),
            "must be hex",
        ),
        (lambda r: r["regeneration"]["environment"].pop("triton"), "environment"),
        (lambda r: r.update(regeneration=None), "needs its section"),
    ],
)
def test_a_change_of_a_synthetic_regeneration_is_caught(edit, message):
    with pytest.raises(TritonIrVerificationError, match=message):
        verify_report(reseal(SYNTHETIC, edit), FIXTURES)


def test_a_synthetic_report_cannot_pass_as_a_triton_regeneration():
    def promote(report):
        report["label"] = REGENERATION_LABEL
        report["evidence_boundary"]["compiled_on_this_host"] = True

    with pytest.raises(TritonIrVerificationError, match="evidence_boundary"):
        verify_report(reseal(SYNTHETIC, promote), FIXTURES)


def test_a_changed_fixture_breaks_verification(tmp_path):
    folder = tmp_path / "fixtures"
    shutil.copytree(FIXTURES, folder)
    path = folder / "sm_80" / "m32_n64_k32_g8_w2_s5" / "ttgir.mlir"
    path.write_text(path.read_text().replace("num = 6", "num = 4"))
    with pytest.raises(TritonIrVerificationError, match="ttgir digest"):
        verify_report(RECORDED, folder)
    (folder / "index.json").write_bytes(b"{}")
    with pytest.raises(TritonIrVerificationError, match="not the recorded"):
        verify_report(RECORDED, folder)


def test_load_report_needs_canonical_bytes(tmp_path):
    path = tmp_path / "report.json"
    path.write_bytes(canonical_json_bytes(RECORDED) + b"\n")
    assert load_report(path, FIXTURES) == RECORDED
    path.write_bytes(canonical_json_bytes(RECORDED))
    with pytest.raises(TritonIrVerificationError, match="not canonical"):
        load_report(path, FIXTURES)
    path.write_bytes(b"[]\n")
    with pytest.raises(TritonIrVerificationError, match="JSON object"):
        load_report(path, FIXTURES)
    path.write_bytes(b'{"a": 1, "a": 2}\n')
    with pytest.raises(TritonIrVerificationError, match="duplicate"):
        load_report(path, FIXTURES)
    with pytest.raises(TritonIrError, match="readable"):
        load_report(tmp_path / "missing.json", FIXTURES)
    with pytest.raises(TypeError, match="pathlib.Path"):
        verify_report(RECORDED, str(FIXTURES))


def test_a_synthetic_report_verifies_after_a_canonical_file_round_trip(tmp_path):
    path = tmp_path / "synthetic.json"
    path.write_bytes(canonical_json_bytes(SYNTHETIC) + b"\n")
    assert load_report(path, FIXTURES) == SYNTHETIC
