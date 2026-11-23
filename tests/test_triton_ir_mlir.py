"""Tests of the TTIR and TTGIR readers on small synthetic texts and on the fixtures.

The committed TTIR and TTGIR are the triton 3.9.0 output for ``sm_80`` and
the config ``m32_n64_k32_g8_w2_s5``. Their counts are pinned here. No test
imports triton.
"""

from pathlib import Path

import pytest

from ai_journey.triton_ir_manifest import TritonIrError
from ai_journey.triton_ir_mlir import read_mlir, ttgir_summary, ttir_summary

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "day80_triton_ir"
CONFIG = FIXTURES / "sm_80" / "m32_n64_k32_g8_w2_s5"
SOURCE = '"<ai_journey.triton_matmul_kernel.TRITON_SOURCE sha256=abc>"'

TTIR = f"""#loc = loc({SOURCE}:3:1)
module {{
  tt.func public @k(%x: !tt.ptr<f32> loc("x"(#loc)), %n: i32 loc("n"(#loc))) attributes {{noinline = false}} {{
    %c0 = arith.constant 0 : i32 loc(#loc5)
    %r = tt.make_range {{end = 16 : i32, start = 0 : i32}} : tensor<16xi32> loc(#loc6)
    %s = tt.splat %n : i32 -> tensor<16xi32> loc(#loc7)
    %m = arith.cmpi slt, %r, %s : tensor<16xi32> loc(#loc7)
    %p = tt.splat %x : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>> loc(#loc8)
    %acc = scf.for %i = %c0 to %n step %c0 iter_args(%a = %c0) -> (i32)  : i32 {{
      %v = tt.load %p, %m, %c0 : tensor<16x!tt.ptr<f32>> loc(#loc9)
      %w = tt.load %p : tensor<16x!tt.ptr<f32>> loc(#loc9)
      scf.yield %a : i32 loc(#loc4)
    }} loc(#loc10)
    tt.store %p, %r, %m : tensor<16x!tt.ptr<f32>> loc(#loc11)
    tt.return loc(#loc)
  }} loc(#loc)
}} loc(#loc)
#loc1 = loc("/site-packages/triton/language/standard.py":43:13)
#loc2 = loc({SOURCE}:5:9)
#loc3 = loc({SOURCE}:6:9)
#loc4 = loc({SOURCE}:7:5)
#loc5 = loc(unknown)
#loc6 = loc("r"(#loc2))
#loc7 = loc(callsite(#loc1 at #loc3))
#loc8 = loc("p"(#loc12))
#loc9 = loc({SOURCE}:8:13)
#loc10 = loc("acc"(#loc4))
#loc11 = loc({SOURCE}:9:5)
#loc12 = loc({SOURCE}:6:20)
"""

TTGIR = """#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#loc = loc(unknown)
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @k(%x: !tt.ptr<f32> loc(#loc)) {
    %a = ttg.local_alloc : () -> !ttg.memdesc<3x16x16xf32, #shared, #smem, mutable> loc(#loc)
    %b = ttg.async_copy_global_to_local %x, %a : tensor<16x!tt.ptr<f32>, #blocked> -> <16xf32, #shared, #smem, mutable> loc(#loc)
    %c = ttg.async_commit_group tokens %b loc(#loc)
    %d = ttg.async_wait %c {num = 2 : i32} loc(#loc)
    %e = tt.dot %a, %a, %a : tensor<16x16xf32, #ttg.dot_op<{opIdx = 0, parent = #mma}>> * tensor<16x16xf32, #ttg.dot_op<{opIdx = 1, parent = #mma}>> -> tensor<16x16xf32, #mma> loc(#loc)
    tt.return loc(#loc)
  } loc(#loc)
} loc(#loc)
"""


def test_a_small_ttir_module_and_its_locations():
    module = read_mlir(TTIR)
    assert module.function == "k"
    assert module.attributes == ()
    lines = {op.line: op.source_line for op in module.ops}
    names = {op.line: op.name for op in module.ops}
    by_name = {}
    for line, name in names.items():
        by_name.setdefault(name, []).append(lines[line])
    assert by_name["arith.constant"] == [None]
    # A named location takes the line of the location it wraps.
    assert by_name["tt.make_range"] == [5]
    # A callsite takes the line of the call, not the line in standard.py.
    assert by_name["arith.cmpi"] == [6]
    # A region op takes the loc after its closing brace.
    assert by_name["scf.for"] == [7]
    assert by_name["tt.load"] == [8, 8]
    assert by_name["tt.func"] == [3]
    summary = ttir_summary(module)
    assert summary["tt_load"] == 2
    assert summary["tt_load_masked"] == 1
    assert summary["tt_load_with_other"] == 1
    assert summary["tt_store_masked"] == 1
    assert summary["tt_make_range"] == [{"start": 0, "end": 16}]
    assert summary["tt_splat"] == 2
    assert summary["scf_for"] == 1
    assert summary["locations"] == 13


def test_a_small_ttgir_module_names_its_layouts():
    summary = ttgir_summary(read_mlir(TTGIR))
    assert (summary["num_warps"], summary["threads_per_warp"], summary["target"]) == (
        4,
        32,
        "cuda:80",
    )
    assert summary["encodings"]["mma"]["kind"] == "ttg.nvidia_mma"
    assert summary["mma_encoding"] is True
    assert summary["dot_operands"] == [
        {"op_idx": 0, "parent": "#mma", "parent_kind": "ttg.nvidia_mma"},
        {"op_idx": 1, "parent": "#mma", "parent_kind": "ttg.nvidia_mma"},
    ]
    assert summary["local_alloc"] == ["3x16x16"]
    assert summary["async_wait"] == [2]
    assert summary["async_copy_global_to_local"] == 1
    assert summary["inline_encodings"] == {"dot_op": 2}


def test_ttir_refuses_a_layout_and_ttgir_needs_its_attributes():
    with pytest.raises(TritonIrError, match="no GPU layout"):
        ttir_summary(read_mlir(TTGIR))
    with pytest.raises(TritonIrError, match="must name ttg.num-warps"):
        ttgir_summary(read_mlir(TTGIR.replace('"ttg.num-warps" = 4 : i32, ', "")))


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (
            lambda t: t.replace("module {", "module {\nmodule {", 1),
            "exactly one module",
        ),
        (lambda t: t.replace("loc(#loc9)\n", "\n", 1), "no trailing loc"),
        (lambda t: t.replace("} loc(#loc10)", "}"), "closes a region with no loc"),
        (lambda t: t.replace("#loc12 = loc", "#loc13 = loc"), "#loc12 is not defined"),
        (lambda t: t.replace("#loc12 = loc", "#loc11 = loc"), "defined twice"),
        (
            lambda t: t.replace('loc("p"(#loc12))', 'loc("p"(#loc8))'),
            "refers to itself",
        ),
        (lambda t: t.replace("loc(unknown)", "loc(fused[#loc1])"), "unknown form"),
        (lambda t: t.replace("    tt.return", "    !bad\n    tt.return"), "not an op"),
        (
            lambda t: t.replace("  }} loc(#loc)\n".replace("}}", "}"), ""),
            "never closes",
        ),
        (lambda t: t + "tt.return loc(#loc)\n", "outside the module"),
        (lambda t: t + "\u00b5", "ASCII"),
    ],
)
def test_a_text_that_is_not_one_module_is_refused(edit, message):
    text = edit(TTIR)
    assert text != TTIR
    with pytest.raises(TritonIrError, match=message):
        read_mlir(text)


def committed(stage):
    return read_mlir((CONFIG / f"{stage}.mlir").read_text(encoding="ascii"))


def test_the_committed_ttir_counts():
    summary = ttir_summary(committed("ttir"))
    assert summary["function"] == "matmul_kernel"
    assert (summary["ops"], summary["locations"], summary["ops_with_source_line"]) == (
        97,
        109,
        92,
    )
    assert summary["tt_load"] == summary["tt_load_masked"] == 2
    assert summary["tt_load_with_other"] == 2
    assert summary["tt_store"] == summary["tt_store_masked"] == 1
    assert summary["tt_dot"] == summary["scf_for"] == 1
    assert summary["tt_make_range"] == [
        {"start": 0, "end": 32},
        {"start": 0, "end": 64},
    ]
    assert summary["tt_splat"] == 19
    assert summary["histogram"]["tt.broadcast"] == 12
    assert summary["histogram"]["tt.addptr"] == 8
    assert not [name for name in summary["histogram"] if name.startswith("ttg.")]


def test_the_committed_ttgir_counts():
    summary = ttgir_summary(committed("ttgir"))
    assert summary["attributes"] == {
        "ttg.num-ctas": "1",
        "ttg.num-warps": "2",
        "ttg.target": "cuda:80",
        "ttg.threads-per-warp": "32",
        "ttng.two-ctas": "false",
    }
    kinds = {name: value["kind"] for name, value in summary["encodings"].items()}
    assert kinds == {
        "blocked": "ttg.blocked",
        "blocked1": "ttg.blocked",
        "blocked2": "ttg.blocked",
        "shared": "ttg.swizzled_shared",
        "smem": "ttg.shared_memory",
    }
    assert summary["encodings"]["blocked"]["parameters"] == (
        "<{sizePerThread = [4, 4], threadsPerWarp = [2, 16], warpsPerCTA = [2, 1], "
        "order = [1, 0]}>"
    )
    assert summary["mma_encoding"] is False
    assert summary["dot_operands"] == [
        {"op_idx": 0, "parent": "#blocked", "parent_kind": "ttg.blocked"},
        {"op_idx": 1, "parent": "#blocked", "parent_kind": "ttg.blocked"},
    ]
    assert summary["inline_encodings"] == {"dot_op": 4, "slice": 15}
    # num_stages = 5 gives a ring of 4 buffers for each operand.
    assert summary["local_alloc"] == ["4x32x32", "4x32x64"]
    assert summary["async_copy_global_to_local"] == 10
    assert summary["async_commit_group"] == 10
    assert summary["async_wait"] == [6, 0]
    assert summary["local_load"] == 2
    assert summary["convert_layout"] == 1
    assert summary["tt_load"] == 0
    assert (summary["ops"], summary["locations"]) == (220, 107)
