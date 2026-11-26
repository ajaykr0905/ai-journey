"""Tests of the TTGIR encoding reader on SYNTHETIC modules written by hand.

Every ``SYNTHETIC_*`` module below was written by hand for these tests, in
the line forms of a TTGIR text. No compiler wrote it, and no test presents
it as compiler output. The swizzle tables are computed by hand from the
stated rule. The last test reads the one real TTGIR text in the repository,
the recorded Day 80 compile of the Day 79 float32 kernel, and says so.
"""

from pathlib import Path

import pytest

from ai_journey.triton_tc_kernel import TritonTcError
from ai_journey.triton_tc_ttgir import (
    analyze_ttgir,
    element_bits,
    parse_memdesc,
    parse_parameters,
    shared_bank,
    swizzle_table,
    swizzled_column,
)

ROOT = Path(__file__).resolve().parents[1]
DAY80_TTGIR = (
    ROOT / "tests/fixtures/day80_triton_ir/sm_80/m32_n64_k32_g8_w2_s5/ttgir.mlir"
)
SYNTHETIC_AMPERE_BF16 = """\
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], \
warpsPerCTA = [4, 1], order = [1, 0]}>
#loc = loc("synthetic":1:1)
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], \
instrShape = [16, 8]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, \
ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @synthetic_kernel(%arg0: !tt.ptr<bf16>) attributes {noinline = false} {
    %a = ttg.local_alloc : () -> !ttg.memdesc<2x128x32xbf16, #shared, #smem, mutable> \
loc(#loc)
    %b = ttg.local_alloc : () -> !ttg.memdesc<2x32x128xbf16, #shared1, #smem, mutable> \
loc(#loc)
    %0:2 = scf.for %k = %c0 to %c4 step %c1 iter_args(%acc = %cst, %i = %c0) -> \
(tensor<128x128xf32, #mma>, i32) : i32 {
      %x = ttg.local_load %a0 : !ttg.memdesc<128x32xbf16, #shared, #smem, mutable> -> \
tensor<128x32xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> loc(#loc)
      %y = ttg.local_load %b0 : !ttg.memdesc<32x128xbf16, #shared1, #smem, mutable> -> \
tensor<32x128xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> loc(#loc)
      %z = tt.dot %x, %y, %acc : tensor<128x32xbf16, #ttg.dot_op<{opIdx = 0, \
parent = #mma, kWidth = 8}>> * tensor<32x128xbf16, #ttg.dot_op<{opIdx = 1, \
parent = #mma, kWidth = 8}>> -> tensor<128x128xf32, #mma> loc(#loc)
      scf.yield %z, %i : tensor<128x128xf32, #mma>, i32 loc(#loc)
    } loc(#loc)
    tt.return loc(#loc)
  } loc(#loc)
} loc(#loc)
"""
SYNTHETIC_HOPPER_TF32 = """\
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], \
instrShape = [16, 128, 8]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, \
elementBitWidth = 32}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, \
elementBitWidth = 32}>
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, \
ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @synthetic_kernel() {
    %a = ttg.local_alloc : () -> !ttg.memdesc<2x128x32xf32, #shared, #smem, mutable>
    %b = ttg.local_alloc : () -> !ttg.memdesc<2x32x128xf32, #shared1, #smem, mutable>
    %bar = ttg.local_alloc : () -> !ttg.memdesc<2xi64, #shared2, #smem, mutable>
    ttng.fence_async_shared {bCluster = false}
    %z = ttng.warp_group_dot %x, %y, %acc {inputPrecision = tf32, isAsync = true} : \
!ttg.memdesc<128x32xf32, #shared, #smem> * !ttg.memdesc<32x128xf32, #shared1, #smem> \
-> tensor<128x128xf32, #mma>
    %w:3 = ttng.warp_group_dot_wait %z, %x, %y {pendings = 1 : i32} : \
tensor<128x128xf32, #mma>, !ttg.memdesc<128x32xf32, #shared, #smem>, \
!ttg.memdesc<32x128xf32, #shared1, #smem>
    tt.return
  }
}
"""
SYNTHETIC_BLOCKED_FP32 = """\
#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 32], \
warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", \
"ttg.threads-per-warp" = 32 : i32} {
  tt.func public @synthetic_kernel() {
    %a = ttg.local_alloc : () -> !ttg.memdesc<2x128x32xf32, #shared, #smem, mutable>
    %z = tt.dot %x, %y, %acc : tensor<128x32xf32, #ttg.dot_op<{opIdx = 0, \
parent = #blocked}>> * tensor<32x128xf32, #ttg.dot_op<{opIdx = 1, \
parent = #blocked}>> -> tensor<128x128xf32, #blocked>
    tt.return
  }
}
"""


def unwrap(text):
    """Join the backslash-continued lines of a synthetic module."""

    return text.replace("\\\n", "")


def test_a_synthetic_ampere_module_names_mma_v2_and_its_swizzled_operands():
    analysis = analyze_ttgir(unwrap(SYNTHETIC_AMPERE_BF16))
    mapping = analysis.to_mapping()

    assert (analysis.target, analysis.num_warps, analysis.threads_per_warp) == (
        "cuda:80",
        4,
        32,
    )
    assert mapping["mma"] == [
        {
            "alias": "mma",
            "kind": "nvidia_mma",
            "parameters": {
                "versionMajor": 2,
                "versionMinor": 0,
                "warpsPerCTA": [2, 2],
                "instrShape": [16, 8],
            },
        }
    ]
    assert analysis.mma_versions == (2,)
    assert [item["parameters"] for item in mapping["shared"]] == [
        {"vec": 8, "perPhase": 2, "maxPhase": 4, "order": [1, 0]},
        {"vec": 8, "perPhase": 1, "maxPhase": 8, "order": [1, 0]},
    ]
    assert [
        (item["parameters"], item["parent_kind"]) for item in mapping["dot_operands"]
    ] == [
        ({"opIdx": 0, "parent": "#mma", "kWidth": 8}, "nvidia_mma"),
        ({"opIdx": 1, "parent": "#mma", "kWidth": 8}, "nvidia_mma"),
    ]
    assert mapping["dots"] == [
        {
            "op": "tt.dot",
            "line": 14,
            "result_encoding": "#mma",
            "input_precision": None,
            "result_kind": "nvidia_mma",
        }
    ]
    assert [(item.shape, item.element, item.bytes) for item in analysis.buffers] == [
        ((2, 128, 32), "bf16", 16384),
        ((2, 32, 128), "bf16", 16384),
    ]
    assert mapping["buffer_bytes"] == 32768
    assert {name: count for name, count in analysis.ops.items() if count} == {
        "tt.dot": 1,
        "ttg.local_alloc": 2,
        "ttg.local_load": 2,
        "scf.for": 1,
    }


def test_a_synthetic_hopper_module_names_mma_v3_and_nvmma_shared_swizzles():
    analysis = analyze_ttgir(unwrap(SYNTHETIC_HOPPER_TF32))
    mapping = analysis.to_mapping()

    assert analysis.target == "cuda:90"
    assert analysis.mma_versions == (3,)
    assert mapping["mma"][0]["parameters"]["instrShape"] == [16, 128, 8]
    assert [item["parameters"] for item in mapping["shared"]] == [
        {"swizzlingByteWidth": 128, "transposed": False, "elementBitWidth": 32},
        {"swizzlingByteWidth": 128, "transposed": True, "elementBitWidth": 32},
        {"vec": 1, "perPhase": 1, "maxPhase": 1, "order": [0]},
    ]
    (dot,) = mapping["dots"]
    assert (dot["op"], dot["input_precision"], dot["result_kind"]) == (
        "ttng.warp_group_dot",
        "tf32",
        "nvidia_mma",
    )
    assert analysis.ops["ttng.warp_group_dot_wait"] == 1
    assert analysis.ops["ttng.fence_async_shared"] == 1
    assert [(item.element, item.bytes) for item in analysis.buffers] == [
        ("f32", 32768),
        ("f32", 32768),
        ("i64", 16),
    ]
    assert mapping["dot_operands"] == []


def test_a_synthetic_blocked_dot_has_no_mma_encoding():
    analysis = analyze_ttgir(unwrap(SYNTHETIC_BLOCKED_FP32))

    assert analysis.mma == ()
    assert analysis.mma_versions == ()
    assert [item.parameters["parent"] for item in analysis.dot_operands] == [
        "#blocked",
        "#blocked",
    ]
    assert analysis.to_mapping()["dots"][0]["result_kind"] == "blocked"


def test_a_synthetic_module_must_be_one_module_with_its_attributes():
    text = unwrap(SYNTHETIC_BLOCKED_FP32)
    with pytest.raises(TritonTcError, match="exactly one module"):
        analyze_ttgir(text.replace("module attributes", "modul attributes"))
    with pytest.raises(TritonTcError, match="ttg.target"):
        analyze_ttgir(text.replace('ttg.target = "cuda:80", ', ""))
    with pytest.raises(TritonTcError, match="must be a string"):
        analyze_ttgir(text.encode())


def test_encoding_parameters_and_memdesc_types_parse_by_their_own_rules():
    assert parse_parameters(
        "<{register = [[0, 1], [0, 2]], block = [], transposed = false}>"
    ) == {"register": [[0, 1], [0, 2]], "block": [], "transposed": False}
    assert parse_parameters("<{vec = 8, 4 : i32}>") == {
        "vec": 8,
        "unparsed": ["4 : i32"],
    }
    assert parse_parameters("") == {}
    with pytest.raises(TritonTcError, match="are not"):
        parse_parameters("<vec = 8>")
    assert parse_memdesc("3x128x32xbf16, #shared, #smem, mutable") == (
        (3, 128, 32),
        "bf16",
        "#shared",
        "#smem",
    )
    with pytest.raises(TritonTcError, match="static shape"):
        parse_memdesc("?x32xf16, #shared, #smem")
    assert [element_bits(name) for name in ("f32", "bf16", "i64", "f8E4M3FN")] == [
        32,
        16,
        64,
        8,
    ]
    with pytest.raises(TritonTcError, match="no known width"):
        element_bits("complex")


def test_hand_computed_swizzles_follow_the_xor_rule():
    # vec 1: row r stores logical column c at column c XOR r.
    assert swizzle_table(4, 4, 1, 1, 4) == [
        [0, 1, 2, 3],
        [1, 0, 3, 2],
        [2, 3, 0, 1],
        [3, 2, 1, 0],
    ]
    # perPhase 2: two rows share one phase.
    assert [row[:2] for row in swizzle_table(4, 4, 1, 2, 4)] == [
        [0, 1],
        [0, 1],
        [1, 0],
        [1, 0],
    ]
    # maxPhase 2: the phase repeats every two rows.
    assert [row[0] for row in swizzle_table(4, 4, 1, 1, 2)] == [0, 1, 0, 1]
    # vec 2: whole groups of two move; elements keep their order inside.
    assert swizzle_table(2, 8, 2, 1, 4)[1] == [2, 3, 0, 1, 6, 7, 4, 5]
    # maxPhase 1 is no swizzle.
    assert swizzle_table(3, 4, 1, 1, 1) == [[0, 1, 2, 3]] * 3
    for column in range(64):
        stored = swizzled_column(5, column, 8, 1, 8)
        assert swizzled_column(5, stored, 8, 1, 8) == column


def test_a_hand_computed_swizzle_spreads_one_column_over_eight_banks():
    """Eight rows of a 64-element fp16 tile: each row is 128 bytes, 32 banks."""

    def banks(max_phase):
        return [
            shared_bank(
                row,
                0,
                columns=64,
                element_bytes=2,
                vec=8,
                per_phase=1,
                max_phase=max_phase,
            )
            for row in range(8)
        ]

    assert banks(1) == [0] * 8
    assert banks(8) == [0, 4, 8, 12, 16, 20, 24, 28]


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ((4, 6, 2, 1, 2), "power of two"),
        ((4, 4, 1, 1, 8), "out of its row"),
        ((4, 4, 0, 1, 1), "vec must be an integer of at least 1"),
    ],
)
def test_a_swizzle_table_refuses_a_group_that_leaves_its_row(arguments, message):
    with pytest.raises(TritonTcError, match=message):
        swizzle_table(*arguments)
    with pytest.raises(TritonTcError, match="row must be"):
        swizzled_column(-1, 0, 1, 1, 1)


def test_the_recorded_day_80_ttgir_of_the_day_79_kernel_has_a_blocked_dot():
    """This text is real triton 3.9.0 output for the Day 79 float32 kernel."""

    analysis = analyze_ttgir(DAY80_TTGIR.read_text(encoding="ascii"))

    assert (analysis.target, analysis.num_warps) == ("cuda:80", 2)
    assert analysis.mma == ()
    assert [item.parameters for item in analysis.shared] == [
        {"vec": 1, "perPhase": 1, "maxPhase": 1, "order": [1, 0]}
    ]
    assert [item.shape for item in analysis.buffers] == [(4, 32, 32), (4, 32, 64)]
    assert analysis.to_mapping()["dots"][0]["result_kind"] == "blocked"
