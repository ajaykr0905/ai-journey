"""Tests of the instruction-class rule on SYNTHETIC compiles built by hand.

``synthetic_compile`` writes a PTX text and a TTGIR module by hand: a header,
the module attributes and a chosen number of copies of one instruction line.
No compiler wrote any of them, and no test presents them as compiler output.
The counts are picked so that one text holds one 128 x 128 x 32 tile product
of its class with 4 warps: 64 ``mma.sync`` m16n8k16 per warp, 4 ``wgmma``
m64n128k16 per warpgroup, or 4096 ``fma.rn.f32`` per thread.
"""

import pytest

from ai_journey.triton_tc_compile import CompiledVariant
from ai_journey.triton_tc_kernel import TritonTcError
from ai_journey.triton_tc_rule import (
    CHECKS,
    allowed_classes,
    check_compile,
    observed_class,
    rule_summary,
)

MMA = {
    ("bf16", "mma_sync"): "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 x;",
    ("f16", "mma_sync"): "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 x;",
    ("tf32", "mma_sync"): "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 x;",
    ("bf16", "wgmma"): "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 x;",
}
ONE_TILE = {"mma_sync": 64, "wgmma": 4, "fma": 4096}


def synthetic_ptx(target, lines):
    """A SYNTHETIC PTX text written by hand for a rule test."""

    ptx_target = "sm_90a" if target == "sm_90" else target
    return "\n".join(
        [
            "// SYNTHETIC PTX written by hand for a unit test. Not compiler output.",
            ".version 8.8",
            f".target {ptx_target}",
            ".address_size 64",
            "{",
            *lines,
            "ret;",
            "}",
            "",
        ]
    )


def synthetic_ttgir(target, version, buffer="2x128x32xbf16"):
    """A SYNTHETIC TTGIR module written by hand for a rule test."""

    capability = target.removeprefix("sm_")
    encodings = ["#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, "]
    encodings[0] += "maxPhase = 8, order = [1, 0]}>"
    if version is not None:
        encodings.append(
            f"#mma = #ttg.nvidia_mma<{{versionMajor = {version}, versionMinor = 0, "
            "warpsPerCTA = [4, 1], instrShape = [16, 8]}>"
        )
    return "\n".join(
        [
            *encodings,
            "#smem = #ttg.shared_memory",
            'module attributes {"ttg.num-warps" = 4 : i32, '
            f'ttg.target = "cuda:{capability}", "ttg.threads-per-warp" = 32 : i32}} {{',
            "  tt.func public @synthetic_kernel() {",
            "    %a = ttg.local_alloc : () -> "
            f"!ttg.memdesc<{buffer}, #shared, #smem, mutable>",
            "    tt.return",
            "  }",
            "}",
            "",
        ]
    )


def synthetic_compile(
    variant,
    target,
    *,
    instruction,
    copies,
    version,
    ptx_target=None,
    shared=32768,
    buffer="2x128x32xbf16",
):
    return CompiledVariant(
        variant=variant,
        target=target,
        ttgir=synthetic_ttgir(target, version, buffer),
        ptx=synthetic_ptx(ptx_target or target, [instruction] * copies),
        shared_bytes=shared,
        num_warps=4,
        num_stages=3,
    )


def bf16_on(target, family="mma_sync", **change):
    arguments = {
        "instruction": MMA[("bf16", family)],
        "copies": ONE_TILE[family],
        "version": 2 if family == "mma_sync" else 3,
        **change,
    }
    return synthetic_compile("bf16", target, **arguments)


def failed(row):
    return [name for name in CHECKS if not row["checks"][name]]


def test_the_rule_table_names_the_expected_class_first():
    assert allowed_classes("fp32-ieee", "sm_80") == ("scalar_fma",)
    assert allowed_classes("fp32-ieee", "sm_90") == ("scalar_fma",)
    for variant in ("fp32-tf32", "fp16", "bf16"):
        assert allowed_classes(variant, "sm_80") == ("mma_sync",)
        assert allowed_classes(variant, "sm_90") == ("wgmma", "mma_sync")
    with pytest.raises(TritonTcError, match="target must be one of"):
        allowed_classes("bf16", "sm_100")


def test_the_observed_class_comes_from_the_ptx_counts():
    def counts(sync=0, group=0, fma=0):
        return {"mma_sync": sync, "wgmma_mma_async": group, "fma_rn_f32": fma}

    assert observed_class(counts(sync=1)) == "mma_sync"
    assert observed_class(counts(group=1, fma=9)) == "wgmma"
    assert observed_class(counts(sync=1, group=1)) == "mixed"
    assert observed_class(counts(fma=1)) == "scalar_fma"
    assert observed_class(counts()) == "none"


def test_a_synthetic_bf16_mma_sync_compile_on_sm_80_is_accepted():
    row = check_compile(bf16_on("sm_80"))

    assert (row["expected"], row["observed"], row["accepted"]) == (
        "mma_sync",
        "mma_sync",
        True,
    )
    assert row["note"] is None
    assert row["tile_products"]["mma_sync"] == {"numerator": 1, "denominator": 1}
    assert row["ptx"]["mma_forms"][0]["shape"] == "m16n8k16"
    assert row["ttgir"]["mma_versions"] == [2]
    assert (row["largest_buffer_bytes"], row["shared_bytes"]) == (16384, 32768)


def test_a_synthetic_sm_90_compile_takes_wgmma_and_records_mma_sync():
    expected = check_compile(bf16_on("sm_90", "wgmma"))
    recorded = check_compile(bf16_on("sm_90"))

    assert (expected["observed"], expected["accepted"], expected["note"]) == (
        "wgmma",
        True,
        None,
    )
    assert (recorded["observed"], recorded["accepted"]) == ("mma_sync", True)
    assert recorded["note"] == (
        "triton chose mma.sync on sm_90; the rule accepts and records it"
    )
    summary = rule_summary([expected, recorded])
    assert summary["mma_sync_on_sm_90"] == ["bf16"]
    assert summary["observed"] == {"bf16@sm_90": "mma_sync"}
    assert (summary["accepted"], summary["all_accepted"]) == (2, True)


def test_a_synthetic_fp32_ieee_compile_must_use_scalar_fma_only():
    scalar = synthetic_compile(
        "fp32-ieee",
        "sm_80",
        instruction="fma.rn.f32 %f1, %f2, %f3, %f1;",
        copies=ONE_TILE["fma"],
        version=None,
        buffer="2x128x32xf32",
    )
    row = check_compile(scalar)
    assert (row["observed"], row["accepted"]) == ("scalar_fma", True)
    assert row["ptx"]["mma_forms"] == []

    tensor = synthetic_compile(
        "fp32-ieee",
        "sm_90",
        instruction=MMA[("tf32", "mma_sync")],
        copies=128,
        version=2,
    )
    rejected = check_compile(tensor)
    assert rejected["accepted"] is False
    # The encoding agrees with the PTX, so only the class and the types fail.
    assert failed(rejected) == ["class", "operand_types"]


@pytest.mark.parametrize(
    ("compiled", "checks"),
    [
        (bf16_on("sm_80", instruction=MMA[("f16", "mma_sync")]), ["operand_types"]),
        (bf16_on("sm_80", version=3), ["mma_encoding"]),
        (bf16_on("sm_80", copies=63), ["tile_products"]),
        (bf16_on("sm_80", shared=16383), ["shared_fits"]),
        (bf16_on("sm_90", "wgmma", ptx_target="sm_80"), ["target"]),
        (bf16_on("sm_80", "wgmma", version=3), ["class"]),
    ],
)
def test_a_synthetic_compile_fails_exactly_the_check_that_it_breaks(compiled, checks):
    row = check_compile(compiled)

    assert row["accepted"] is False
    assert failed(row) == checks


def test_a_synthetic_tensor_core_text_must_not_compute_the_dot_with_fma():
    compiled = synthetic_compile(
        "fp16",
        "sm_80",
        instruction=MMA[("f16", "mma_sync")],
        copies=ONE_TILE["mma_sync"],
        version=2,
    )
    fma = "fma.rn.f32 %f1, %f2, %f3, %f1;\n" * ONE_TILE["fma"]
    doubled = CompiledVariant(
        **{**compiled.__dict__, "ptx": compiled.ptx.replace("ret;", fma + "ret;")}
    )
    assert check_compile(compiled)["accepted"] is True
    assert failed(check_compile(doubled)) == ["no_scalar_dot"]
    with pytest.raises(TypeError, match="CompiledVariant"):
        check_compile({"variant": "fp16"})
