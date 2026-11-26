"""Tests of the PTX instruction rules on SYNTHETIC snippets written by hand.

Every ``SYNTHETIC_*`` text below was written by hand for these tests. No
compiler wrote it, and no test presents it as compiler output: each one only
has the line forms that the rules must read. The last test reads the one
real PTX text in the repository, the recorded Day 80 compile of the Day 79
float32 kernel, and says so.
"""

from fractions import Fraction
from pathlib import Path

import pytest

from ai_journey.triton_ir_ptx import read_ptx
from ai_journey.triton_tc_kernel import TC_CONFIG, TcConfig, TritonTcError
from ai_journey.triton_tc_ptx import (
    CLASSES,
    analyze_ptx,
    classify,
    parse_mma,
    read_instructions,
    static_tile_products,
)

ROOT = Path(__file__).resolve().parents[1]
DAY80_PTX = ROOT / "tests/fixtures/day80_triton_ir/sm_80/m32_n64_k32_g8_w2_s5/ptx.ptx"
HEADER_80 = """\
// SYNTHETIC PTX written by hand for a unit test. It is not compiler output.
.version 8.8
.target sm_80
.address_size 64

.visible .entry synthetic_kernel(
\t.param .u64 .ptr .global .align 1 synthetic_kernel_param_0
)
.reqntid 128, 1, 1
{
\t.reg .pred \t%p<4>;
\t.reg .b32 \t%r<64>;
\t.reg .f32 \t%f<64>;
"""
SYNTHETIC_MMA_SYNC_BF16 = (
    HEADER_80
    + """\
$L__BB0_1:
\tcp.async.cg.shared.global [ %r1 + 0 ], [ %rd1 + 0 ], 0x10, %r2;
\tcp.async.cg.shared.global [ %r1 + 2048 ], [ %rd2 + 0 ], 0x10, %r2;
\tcp.async.commit_group;
\tcp.async.wait_group 1;
\tbar.sync \t0;
\tldmatrix.sync.aligned.m8n8.x4.shared.b16 { %r3, %r4, %r5, %r6 }, [ %r7 + 0 ];
\tldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 { %r8, %r9, %r10, %r11 }, [ %r12 + 0 ];
\t// begin inline asm
\tmma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %f1, %f2, %f3, %f4 }, \
{ %r3, %r4, %r5, %r6 }, { %r8, %r9 }, { %f1, %f2, %f3, %f4 };
\t// end inline asm
\t// begin inline asm
\tmma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %f5, %f6, %f7, %f8 }, \
{ %r3, %r4, %r5, %r6 }, { %r10, %r11 }, { %f5, %f6, %f7, %f8 };
\t// end inline asm
\t@%p1 bra \t$L__BB0_1;
\tret;
}
"""
)
SYNTHETIC_MMA_SYNC_TF32 = (
    HEADER_80
    + """\
\tcvt.rna.tf32.f32 %r20, %f9;
\tcvt.rna.tf32.f32 %r21, %f10;
\tmma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 { %f1, %f2, %f3, %f4 }, \
{ %r20, %r21, %r22, %r23 }, { %r24, %r25 }, { %f1, %f2, %f3, %f4 };
\tret;
}
"""
)
SYNTHETIC_WGMMA_BF16 = """\
// SYNTHETIC PTX written by hand for a unit test. It is not compiler output.
.version 9.4
.target sm_90a
.address_size 64
.visible .entry synthetic_kernel(
\t.param .u64 synthetic_kernel_param_0
)
{
\twgmma.fence.sync.aligned;
\t// begin inline asm
\t{
\t.reg .pred p;
\tsetp.ne.b32 p, %r20, 0;
\twgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%f1, %f2}, %rd4, %rd5, \
p, 1, 1, 0, 1;
\t}
\t// end inline asm
\t{ .reg .pred p; setp.ne.b32 p, %r21, 0; \
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%f1, %f2}, %rd6, %rd7, \
p, 1, 1, 0, 1; }
\twgmma.commit_group.sync.aligned;
\twgmma.wait_group.sync.aligned 0;
\tret;
}
"""
SYNTHETIC_SCALAR_FMA = (
    HEADER_80
    + """\
\tfma.rn.f32 \t%f5, %f1, %f2, %f5;
\t@%p2 fma.rn.f32 %f6, %f1, %f3, %f6; // a guarded FMA counts too
\t@!%p3 fma.rn.f32 %f7, %f1, %f4, %f7;
\tfma.rn.ftz.f32 %f8, %f1, %f4, %f8; /* flush-to-zero is another opcode */
\tmad.lo.s32 %r1, %r2, %r3, %r4;
LAB_WAIT:
\tret;
}
"""
)


def test_the_rules_give_each_class_of_the_synthetic_opcodes():
    rows = {
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32": "mma_sync",
        "wgmma.mma_async.sync.aligned.m64n128k8.f32.tf32.tf32": "wgmma_mma_async",
        "wgmma.fence.sync.aligned": "wgmma_fence",
        "wgmma.commit_group.sync.aligned": "wgmma_commit_group",
        "wgmma.wait_group.sync.aligned": "wgmma_wait_group",
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16": "ldmatrix",
        "stmatrix.sync.aligned.m8n8.x4.shared.b16": "stmatrix",
        "cp.async.cg.shared.global": "cp_async",
        "cp.async.ca.shared.global": "cp_async",
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile": "cp_async_bulk",
        "cp.async.commit_group": "cp_async_commit_group",
        "cp.async.wait_group": "cp_async_wait_group",
        "cp.async.wait_all": "cp_async_wait_group",
        "cvt.rna.tf32.f32": "cvt_tf32",
        "fma.rn.f32": "fma_rn_f32",
        "fma.rn.ftz.f32": None,
        "fma.rn.f32x2": None,
        "mad.lo.s32": None,
    }
    assert {opcode: classify(opcode) for opcode in rows} == rows
    assert set(rows.values()) - {None} == set(CLASSES)


def test_a_synthetic_mma_sync_text_counts_its_bf16_forms_and_async_copies():
    analysis = analyze_ptx(SYNTHETIC_MMA_SYNC_BF16)

    assert (analysis.version, analysis.target) == ("8.8", "sm_80")
    assert {name: count for name, count in analysis.classes.items() if count} == {
        "mma_sync": 2,
        "ldmatrix": 2,
        "cp_async": 2,
        "cp_async_commit_group": 1,
        "cp_async_wait_group": 1,
    }
    (form,) = analysis.mma_forms
    assert form.to_mapping() == {
        "opcode": "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32",
        "family": "mma_sync",
        "shape": "m16n8k16",
        "m": 16,
        "n": 8,
        "k": 16,
        "d_type": "f32",
        "a_type": "bf16",
        "b_type": "bf16",
        "c_type": "f32",
        "layouts": ["row", "col"],
        "count": 2,
        "multiply_adds": 2048,
    }
    assert analysis.ldmatrix_forms == {
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16": 1,
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16": 1,
    }
    assert analysis.tensor_core_classes == ("mma_sync",)
    assert analysis.instructions == 11


def test_a_synthetic_tf32_text_names_the_m16n8k8_shape_and_its_conversions():
    analysis = analyze_ptx(SYNTHETIC_MMA_SYNC_TF32)
    (form,) = analysis.mma_forms

    assert (form.shape, form.a_type, form.b_type, form.d_type) == (
        "m16n8k8",
        "tf32",
        "tf32",
        "f32",
    )
    assert analysis.classes["cvt_tf32"] == 2
    assert form.multiply_adds == 1024


def test_a_synthetic_wgmma_text_counts_each_statement_inside_inline_asm():
    analysis = analyze_ptx(SYNTHETIC_WGMMA_BF16)

    assert analysis.target == "sm_90a"
    assert analysis.tensor_core_classes == ("wgmma_mma_async",)
    assert {name: count for name, count in analysis.classes.items() if count} == {
        "wgmma_mma_async": 2,
        "wgmma_fence": 1,
        "wgmma_commit_group": 1,
        "wgmma_wait_group": 1,
    }
    (form,) = analysis.mma_forms
    assert (form.family, form.shape, form.a_type, form.c_type, form.layouts) == (
        "wgmma_mma_async",
        "m64n128k16",
        "bf16",
        None,
        (),
    )
    opcodes = [opcode for _, opcode in read_instructions(SYNTHETIC_WGMMA_BF16)]
    assert opcodes.count("setp.ne.b32") == 2
    assert "p," not in opcodes and ".reg" not in opcodes


def test_a_synthetic_scalar_text_counts_only_the_exact_fma_rn_f32_opcode():
    analysis = analyze_ptx(SYNTHETIC_SCALAR_FMA)

    assert analysis.classes["fma_rn_f32"] == 3
    assert analysis.tensor_core_classes == ()
    assert analysis.mma_forms == ()
    opcodes = [opcode for _, opcode in read_instructions(SYNTHETIC_SCALAR_FMA)]
    assert opcodes == [
        "fma.rn.f32",
        "fma.rn.f32",
        "fma.rn.f32",
        "fma.rn.ftz.f32",
        "mad.lo.s32",
        "ret",
    ]


def test_static_tile_products_of_synthetic_counts_follow_the_instruction_shapes():
    def products(text, config=TC_CONFIG):
        return static_tile_products(analyze_ptx(text), config)

    # 128 x 128 x 32 = 524288 multiply-adds; 4 warps, one warpgroup.
    sixty_four = SYNTHETIC_MMA_SYNC_BF16.replace(
        "\t@%p1 bra",
        "\tmma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 x;\n" * 62
        + "\t@%p1 bra",
    )
    assert products(sixty_four)["mma_sync"] == 1
    assert products(SYNTHETIC_MMA_SYNC_BF16)["mma_sync"] == Fraction(1, 32)
    four = SYNTHETIC_WGMMA_BF16.replace(
        "\twgmma.commit",
        "\twgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 x;\n" * 2
        + "\twgmma.commit",
    )
    assert products(four)["wgmma_mma_async"] == 1
    assert products(SYNTHETIC_SCALAR_FMA, TcConfig(16, 16, 16, 1, 1)) == {
        "fma_rn_f32": Fraction(96, 4096),
        "mma_sync": 0,
        "wgmma_mma_async": 0,
    }
    with pytest.raises(TypeError, match="PtxAnalysis"):
        static_tile_products({}, TC_CONFIG)


@pytest.mark.parametrize(
    ("opcode", "message"),
    [
        ("mma.sync.aligned.row.col.f32.bf16.bf16.f32", "one mNnNkN shape"),
        ("mma.sync.aligned.m16n8k16.row.col.f32.bf16.f32", "4 operand types"),
        ("wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16", "3 operand types"),
        ("fma.rn.f32", "not an MMA opcode"),
    ],
)
def test_a_synthetic_mma_opcode_without_a_shape_or_types_is_refused(opcode, message):
    with pytest.raises(TritonTcError, match=message):
        parse_mma(opcode)


def test_a_synthetic_text_must_declare_one_target_and_one_version():
    with pytest.raises(TritonTcError, match=r"\.target once"):
        analyze_ptx(SYNTHETIC_SCALAR_FMA.replace(".target sm_80\n", ""))
    with pytest.raises(TritonTcError, match=r"\.target once"):
        analyze_ptx(SYNTHETIC_SCALAR_FMA + ".target sm_90a\n")
    with pytest.raises(TritonTcError, match=r"\.version once"):
        analyze_ptx(SYNTHETIC_SCALAR_FMA.replace(".version 8.8\n", ""))
    with pytest.raises(TritonTcError, match="must be a string"):
        analyze_ptx(b".target sm_80")


def test_the_recorded_day_80_compile_of_the_day_79_kernel_reads_as_scalar_fma():
    """This text is real triton 3.9.0 output for the Day 79 float32 kernel."""

    text = DAY80_PTX.read_text(encoding="ascii")
    analysis = analyze_ptx(text)

    assert len(read_ptx(text).instructions) == analysis.instructions == 3097
    assert analysis.target == "sm_80"
    assert analysis.tensor_core_classes == ()
    assert analysis.classes["fma_rn_f32"] == 1024
    assert analysis.classes["cp_async"] == 240
    # 32 x 64 x 32 tile, 2 warps: 1024 FMAs per thread is one tile product.
    products = static_tile_products(analysis, TcConfig(32, 64, 32, 2, 5))
    assert products == {"fma_rn_f32": 1, "mma_sync": 0, "wgmma_mma_async": 0}
